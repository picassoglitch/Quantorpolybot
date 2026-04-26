"""Async Ollama client — tiered three-model architecture.

Three named roles, each with its own model, timeout, and temperature:

  - fast      (small model, ~3B, low temp): event_sniper entries,
                scalping re-scores, anything latency-sensitive.
  - deep      (mid model, ~7B, moderate temp): signal pipeline,
                longshot, full evidence synthesis. Also the default
                for legacy callers (generate_json / generate_text).
  - validator (strict model, low temp): cross-check on high-stakes
                entries (>= validator_high_stakes_usd).

Per-call telemetry is persisted to the ``ollama_stats`` table so the
dashboard can plot per-model latency and timeout rates. Three
class-level queue counters track in-flight calls per tier — callers
can read ``OllamaClient.fast_queue_saturated()`` to decide whether to
skip Ollama entirely (event_sniper uses this to fall back to the
keyword heuristic under GPU pressure).

Uses raw httpx so we don't need the sync ollama lib in the hot path.
Returns parsed-JSON dicts (or None on failure).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import weakref
from contextlib import suppress
from typing import Any

import httpx
from loguru import logger

from core.utils.config import env, get_config

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# After this many consecutive failures we enter a cooldown window during
# which calls return None immediately instead of hammering a broken Ollama.
_FAIL_THRESHOLD = 5
_COOLDOWN_SECONDS = 60.0


def _ollama_cfg() -> dict[str, Any]:
    return get_config().get("ollama") or {}


# ---- Shared httpx client (module-level, not per-call) --------------------
# Creating a new AsyncClient for every Ollama call re-establishes TCP +
# does DNS resolution under event-loop contention, which blew the
# effective budget past the 10s fast-tier timeout even though Ollama
# itself responded in ~5s. A module-level singleton with keep-alive
# connections removes that overhead and — critically — fixes the socket
# bind so we never wait on pool exhaustion silently.
_SHARED_CLIENT: httpx.AsyncClient | None = None
_CLIENT_LOCK = asyncio.Lock()

# Epoch second of the last successful /api/generate response. Read by
# the event-loop watchdog to decide when "Ollama is silent" escalates
# from WARN to ERROR.
LAST_SUCCESS_TS: float = 0.0

# Process-wide in-flight cap. Ollama on a single-GPU box (8 GB VRAM,
# one model loaded) can only physically decode ~1-2 requests at a time;
# beyond that, extra concurrent HTTP requests either queue inside Ollama
# (wasting client-side timeouts) or — worse, on Windows — pile up in the
# kernel accept backlog and surface as httpx ConnectTimeouts even though
# the server is healthy. A module-level semaphore forces client-side
# serialization that matches the GPU's real throughput. Read from config
# so it scales on larger machines without a code change.
_GLOBAL_SEM: asyncio.Semaphore | None = None
_SEM_LOCK = asyncio.Lock()


async def _get_global_semaphore() -> asyncio.Semaphore:
    global _GLOBAL_SEM
    if _GLOBAL_SEM is not None:
        return _GLOBAL_SEM
    async with _SEM_LOCK:
        if _GLOBAL_SEM is None:
            # Default 2: one call decoding on the GPU while the next is
            # uploading its prompt. Higher values just queue internally.
            n = int(_ollama_cfg().get("max_concurrent_calls", 2))
            if n < 1:
                n = 1
            _GLOBAL_SEM = asyncio.Semaphore(n)
    return _GLOBAL_SEM


def _build_shared_client() -> httpx.AsyncClient:
    # connect=30 covers the worst case we actually see in the wild:
    # Ollama's HTTP listener on Windows briefly stalls while it's
    # mmap'ing a cold GGUF off disk (~5 GB for qwen2.5:7b-q4), during
    # which new SYN packets sit unanswered. 10s was too tight and
    # caused spurious ConnectTimeouts on the first post-swap call.
    # The per-tier ``asyncio.wait_for`` still bounds a stuck call
    # (25s fast, 60s deep/validator). read must be wider than the deep
    # tier's 60s budget so the transport doesn't raise ReadTimeout
    # before wait_for fires. Tight pool (4/2) + the module semaphore
    # prevent runaway concurrency masking a real GPU queue backup.
    timeout = httpx.Timeout(connect=30.0, read=90.0, write=5.0, pool=5.0)
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
    return httpx.AsyncClient(timeout=timeout, limits=limits)


async def _get_shared_client() -> httpx.AsyncClient:
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None and not _SHARED_CLIENT.is_closed:
        return _SHARED_CLIENT
    async with _CLIENT_LOCK:
        if _SHARED_CLIENT is None or _SHARED_CLIENT.is_closed:
            _SHARED_CLIENT = _build_shared_client()
    return _SHARED_CLIENT


async def reset_shared_client() -> None:
    """Close the shared client + pool and force a fresh one on next call.
    Called by the watchdog when event-loop contention is spiking — dumps
    any half-open connections that might be stuck on a silent socket."""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None and not _SHARED_CLIENT.is_closed:
        with suppress(Exception):
            await _SHARED_CLIENT.aclose()
    _SHARED_CLIENT = None


async def aclose_shared_client() -> None:
    """Called at shutdown from run.py. Safe to invoke repeatedly."""
    await reset_shared_client()


class OllamaClient:
    # Track every live instance so the Settings page can reset back-off
    # state across the whole process when host/model change.
    _instances: "weakref.WeakSet[OllamaClient]" = weakref.WeakSet()

    # Process-wide queue depth counters. Incremented on call start,
    # decremented in a finally block so exceptions never leak a slot.
    pending_fast: int = 0
    pending_deep: int = 0
    pending_validator: int = 0

    def __init__(self) -> None:
        # Legacy keys still honored so existing callers don't regress.
        legacy_timeout = float(
            get_config().get("signals", "ollama_timeout_seconds", default=60)
        )
        self._default_timeout = legacy_timeout
        self._consecutive_failures = 0
        self._cooldown_until = 0.0
        # Per-model set so switching via the dashboard re-arms the warning
        # for whichever model is currently missing.
        self._missing_models_warned: set[str] = set()
        OllamaClient._instances.add(self)

    # ---- Global state management -------------------------------------

    @classmethod
    def _reset_global_cooldowns(cls) -> None:
        """Clear back-off state on every live client. Called by the
        Settings page after the user updates OLLAMA_* env vars so the new
        config gets tried immediately."""
        for inst in list(cls._instances):
            inst._consecutive_failures = 0
            inst._cooldown_until = 0.0
            inst._missing_models_warned = set()

    @classmethod
    def queue_depths(cls) -> dict[str, int]:
        return {
            "fast": cls.pending_fast,
            "deep": cls.pending_deep,
            "validator": cls.pending_validator,
        }

    @classmethod
    def fast_queue_saturated(cls) -> bool:
        """True when the fast-tier queue is at/above the configured alert.
        Callers that can degrade gracefully (event_sniper -> heuristic)
        check this before issuing a call."""
        alert = int(_ollama_cfg().get("queue_depth_alert", 5))
        return cls.pending_fast >= alert

    # ---- Config / model resolution -----------------------------------

    @property
    def host(self) -> str:
        # Read at call time so Settings-page changes take effect without
        # a process restart. OLLAMA_HOST env overrides the YAML value.
        return env(
            "OLLAMA_HOST",
            str(_ollama_cfg().get("host") or "http://localhost:11434"),
        ).rstrip("/")

    @property
    def fast_model(self) -> str:
        return str(_ollama_cfg().get("fast_model") or self.deep_model)

    @property
    def deep_model(self) -> str:
        # OLLAMA_MODEL env may override the deep tier but only when it
        # names a fully-qualified Ollama tag (e.g. "qwen2.5:7b-...").
        # A bare name like "qwen2.5" in .env used to silently downgrade
        # the deep tier to a different (often non-existent) model —
        # that's what caused the "qwen2.5 plain" regression.
        cfg = _ollama_cfg()
        cfg_value = str(cfg.get("deep_model") or cfg.get("model") or "mistral")
        env_value = env("OLLAMA_MODEL", "")
        if env_value and ":" in env_value:
            return env_value
        return cfg_value

    @property
    def validator_model(self) -> str:
        return str(_ollama_cfg().get("validator_model") or self.deep_model)

    # Legacy alias so old callers (health check, prompt evolution) keep
    # seeing `.model` as the deep-tier model.
    @property
    def model(self) -> str:
        return self.deep_model

    def _model_for(self, call_type: str) -> str:
        if call_type == "fast":
            return self.fast_model
        if call_type == "validator":
            return self.validator_model
        return self.deep_model

    def _timeout_for(self, call_type: str) -> float:
        cfg = _ollama_cfg()
        if call_type == "fast":
            return float(cfg.get("fast_timeout_seconds", 25))
        if call_type == "validator":
            return float(cfg.get("validator_timeout_seconds", 60))
        if call_type == "deep":
            return float(cfg.get("deep_timeout_seconds", 60))
        return self._default_timeout

    def _temperature_for(self, call_type: str) -> float:
        cfg = _ollama_cfg()
        if call_type == "fast":
            return float(cfg.get("fast_temperature", 0.2))
        if call_type == "validator":
            return float(cfg.get("validator_temperature", 0.1))
        if call_type == "deep":
            return float(cfg.get("deep_temperature", 0.3))
        return 0.4  # generate_text default

    # ---- Back-off plumbing -------------------------------------------

    def _in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._cooldown_until = 0.0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _FAIL_THRESHOLD:
            self._cooldown_until = time.monotonic() + _COOLDOWN_SECONDS
            logger.warning(
                "[ollama] {} consecutive failures; pausing calls for {:.0f}s",
                self._consecutive_failures,
                _COOLDOWN_SECONDS,
            )

    def _explain_failure(self, e: Exception, model: str) -> None:
        # 404 from /api/generate almost always means the model isn't pulled.
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
            if model not in self._missing_models_warned:
                logger.warning(
                    "[ollama] /api/generate returned 404 — model '{}' is not "
                    "available locally. Run `ollama pull {}` to install it. "
                    "Further 404s for this model will be suppressed.",
                    model, model,
                )
                self._missing_models_warned.add(model)
            return
        # Enrich the generic failure log. TimeoutError / ReadTimeout carry
        # an empty ``str(e)`` which previously rendered as just ":" — the
        # type name is what tells us "GPU was too slow" vs "connection
        # refused" vs "HTTP 500". For HTTPStatusError, surface the status.
        if isinstance(e, httpx.HTTPStatusError):
            detail = f"HTTP {e.response.status_code} {e.response.reason_phrase}"
        else:
            detail = str(e) or type(e).__name__
        logger.warning(
            "[ollama] request failed ({}) {}: {}", model, type(e).__name__, detail,
        )

    # ---- Core call path ----------------------------------------------

    @classmethod
    def _inc(cls, call_type: str) -> None:
        if call_type == "fast":
            cls.pending_fast += 1
        elif call_type == "validator":
            cls.pending_validator += 1
        elif call_type == "deep":
            cls.pending_deep += 1

    @classmethod
    def _dec(cls, call_type: str) -> None:
        if call_type == "fast":
            cls.pending_fast = max(0, cls.pending_fast - 1)
        elif call_type == "validator":
            cls.pending_validator = max(0, cls.pending_validator - 1)
        elif call_type == "deep":
            cls.pending_deep = max(0, cls.pending_deep - 1)

    async def _generate(
        self,
        prompt: str,
        *,
        call_type: str,
        response_format: str = "json",
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Single Ollama /api/generate call with stats logging.

        Returns (parsed_body, meta) where ``meta`` always contains at
        least {"raw_text", "latency_ms", "success", "error"} so the
        caller (or validator) can inspect what came back.
        """
        model = self._model_for(call_type)
        timeout = self._timeout_for(call_type)
        temperature = self._temperature_for(call_type)
        meta: dict[str, Any] = {
            "model": model,
            "call_type": call_type,
            "latency_ms": 0.0,
            "success": False,
            "error": "",
            "raw_text": "",
            "tokens_in": 0,
            "tokens_out": 0,
        }
        if self._in_cooldown():
            meta["error"] = "cooldown"
            await self._log_stat(meta)
            return None, meta

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Explicit options so the server doesn't fall back to its
            # default context/output sizes. num_predict in particular
            # caps runaway generations that would otherwise exceed the
            # tier's timeout budget.
            "options": {
                "temperature": temperature,
                "top_p": float(_ollama_cfg().get("top_p", 0.9)),
                "num_ctx": int(_ollama_cfg().get("num_ctx", 4096)),
                "num_predict": int(_ollama_cfg().get("num_predict", 300)),
            },
        }
        if response_format == "json":
            body["format"] = "json"

        url = f"{self.host}/api/generate"
        start = time.perf_counter()
        OllamaClient._inc(call_type)
        client = await _get_shared_client()
        sem = await _get_global_semaphore()
        try:
            # Serialize against Ollama's real throughput (see _GLOBAL_SEM
            # docs). The wait here is bounded by the tier timeout — if a
            # slot doesn't free up in time, we fail fast rather than pile
            # up TCP connections the server can't accept.
            async with sem:
                # Per-call wait_for wraps the shared client's post so each
                # tier keeps its own budget — the shared client's read
                # timeout is set to the widest tier (deep/validator).
                r = await asyncio.wait_for(
                    client.post(url, json=body), timeout=timeout,
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            meta["latency_ms"] = (time.perf_counter() - start) * 1000.0
            meta["error"] = type(e).__name__
            logger.debug(
                "[ollama] tier={} model={} latency_ms={:.0f} success=false error={}",
                call_type, model, meta["latency_ms"], type(e).__name__,
            )
            self._explain_failure(e, model)
            self._record_failure()
            await self._log_stat(meta)
            return None, meta
        finally:
            OllamaClient._dec(call_type)

        meta["latency_ms"] = (time.perf_counter() - start) * 1000.0
        meta["success"] = True
        # Ollama returns prompt_eval_count / eval_count when available.
        meta["tokens_in"] = int(data.get("prompt_eval_count") or 0)
        meta["tokens_out"] = int(data.get("eval_count") or 0)
        self._record_success()
        global LAST_SUCCESS_TS
        LAST_SUCCESS_TS = time.time()

        text = (data.get("response") or "").strip()
        meta["raw_text"] = text
        parsed = self._extract_json(text) if response_format == "json" else None
        logger.debug(
            "[ollama] tier={} model={} latency_ms={:.0f} success=true "
            "tokens_in={} tokens_out={}",
            call_type, model, meta["latency_ms"],
            meta["tokens_in"], meta["tokens_out"],
        )
        await self._log_stat(meta)
        return parsed, meta

    @staticmethod
    async def _log_stat(meta: dict[str, Any]) -> None:
        """Persist one call's telemetry. Best-effort — a logging failure
        must never fail the call."""
        try:
            # Deferred import: avoids a circular dep at module load time
            # and keeps OllamaClient importable for unit tests that don't
            # have a DB configured.
            from core.utils.db import execute
            from core.utils.helpers import now_ts

            await execute(
                """INSERT INTO ollama_stats
                   (model, call_type, latency_ms, success, tokens_in,
                    tokens_out, called_at, error)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    meta.get("model") or "",
                    meta.get("call_type") or "",
                    float(meta.get("latency_ms") or 0.0),
                    1 if meta.get("success") else 0,
                    int(meta.get("tokens_in") or 0),
                    int(meta.get("tokens_out") or 0),
                    now_ts(),
                    str(meta.get("error") or ""),
                ),
            )
        except Exception as e:
            logger.debug("[ollama] stats log failed: {}", e)

    # ---- Public tiered API -------------------------------------------

    async def fast_score(
        self, prompt: str, *, context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Small-model JSON scoring. Target <5s. Used by event_sniper
        and scalping re-scores."""
        parsed, _ = await self._generate(prompt, call_type="fast")
        return parsed

    async def deep_score(
        self, prompt: str, *, context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Mid-model JSON scoring. Target 10-30s. Used by signal
        pipeline and longshot."""
        parsed, _ = await self._generate(prompt, call_type="deep")
        return parsed

    async def validate(
        self, prompt: str, *, context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Cross-validation model. Low temperature, no exploration —
        used by shadow.open_position on entries >= high-stakes threshold."""
        parsed, _ = await self._generate(prompt, call_type="validator")
        return parsed

    # ---- Legacy API (kept for backward compat) -----------------------

    async def generate_json(self, prompt: str) -> dict[str, Any] | None:
        """Legacy entry point. Routes to the deep tier — callers that
        haven't been migrated to fast/validator get sensible defaults."""
        return await self.deep_score(prompt)

    async def generate_text(self, prompt: str) -> str:
        """Freeform (non-JSON) generation. Used by prompt evolution.
        Always goes to the deep tier at a slightly warmer temperature."""
        if self._in_cooldown():
            return ""
        url = f"{self.host}/api/generate"
        model = self.deep_model
        meta: dict[str, Any] = {
            "model": model,
            "call_type": "generate_text",
            "latency_ms": 0.0,
            "success": False,
            "error": "",
            "tokens_in": 0,
            "tokens_out": 0,
        }
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature_for("generate_text"),
                "top_p": float(_ollama_cfg().get("top_p", 0.9)),
                "num_ctx": int(_ollama_cfg().get("num_ctx", 4096)),
                "num_predict": int(_ollama_cfg().get("num_predict", 300)),
            },
        }
        start = time.perf_counter()
        client = await _get_shared_client()
        sem = await _get_global_semaphore()
        try:
            async with sem:
                r = await asyncio.wait_for(
                    client.post(url, json=body), timeout=self._timeout_for("deep"),
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            meta["latency_ms"] = (time.perf_counter() - start) * 1000.0
            meta["error"] = type(e).__name__
            self._explain_failure(e, model)
            self._record_failure()
            await self._log_stat(meta)
            return ""
        meta["latency_ms"] = (time.perf_counter() - start) * 1000.0
        meta["success"] = True
        meta["tokens_in"] = int(data.get("prompt_eval_count") or 0)
        meta["tokens_out"] = int(data.get("eval_count") or 0)
        self._record_success()
        global LAST_SUCCESS_TS
        LAST_SUCCESS_TS = time.time()
        await self._log_stat(meta)
        return (data.get("response") or "").strip()

    # ---- Health ------------------------------------------------------
    # Both probes open a short-lived client rather than reusing the
    # shared one. They're called at most once per minute (health check
    # scheduler / dashboard render), and the shared client is loop-bound
    # to the main feed loop — the dashboard runs on its own loop.

    async def warmup(self, *, timeout: float = 120.0) -> bool:
        """Force a model load + one-token generation before lanes go live.
        Ollama on Windows stalls its HTTP listener for several seconds
        the first time a model is swapped into VRAM; letting lanes kick
        off traffic during that window triggers spurious ConnectTimeouts
        on the first batch of real calls. We run this at boot with a
        generous wait_for so the model is already resident by the time
        the signal pipeline or lane scans dispatch. Returns True on a
        successful generation; logs + returns False on any error (boot
        still proceeds — the per-call retry path handles a degraded
        Ollama afterward)."""
        model = self.deep_model
        body = {
            "model": model,
            "prompt": "ping",
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0.0},
        }
        url = f"{self.host}/api/generate"
        t0 = time.perf_counter()
        try:
            # Bypass the shared client + semaphore — warmup runs before
            # any other caller and should use a fresh, generously-timed
            # client so a cold-disk mmap doesn't abort it.
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30.0, read=timeout, write=5.0, pool=5.0),
            ) as client:
                r = await asyncio.wait_for(
                    client.post(url, json=body), timeout=timeout,
                )
                r.raise_for_status()
        except Exception as e:
            logger.warning(
                "[ollama] warmup failed ({}) {}: {} (after {:.1f}s)",
                model, type(e).__name__, str(e) or type(e).__name__,
                time.perf_counter() - t0,
            )
            return False
        elapsed = time.perf_counter() - t0
        global LAST_SUCCESS_TS
        LAST_SUCCESS_TS = time.time()
        logger.info(
            "[ollama] warmup OK ({}): model resident in {:.1f}s", model, elapsed,
        )
        return True

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.host}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def running_models(self) -> list[dict[str, Any]]:
        """Query /api/ps for models currently loaded in VRAM. Best-effort
        — returns [] on any failure so the dashboard can still render."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.host}/api/ps")
                r.raise_for_status()
                data = r.json()
                return list(data.get("models") or [])
        except Exception:
            return []

    # ---- Parser ------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        # Try direct parse first (Ollama format=json should give us clean JSON).
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
