"""Tiered Ollama client — hard thread-isolation edition.

Three named roles, each with its own model, timeout, and temperature:

  - fast      (small model, ~3B, low temp): event_sniper entries,
                scalping re-scores, anything latency-sensitive.
  - deep      (mid model, ~7B, moderate temp): signal pipeline,
                longshot, full evidence synthesis. Also the default
                for legacy callers (generate_json / generate_text).
  - validator (strict model, low temp): cross-check on high-stakes
                entries (>= validator_high_stakes_usd).

Architecture (April 2026 hard-isolation rebuild):

  1. Every Ollama HTTP call runs on a per-tier ``ThreadPoolExecutor``
     in ``core.signals.ollama_executor``. The asyncio loop never
     touches a socket — it only awaits a ``Future`` via
     ``asyncio.wrap_future``. This removes the entire class of subtle
     async-httpx blocking issues (connection-pool locks, sync DNS on
     Windows, response parsing, cancellation propagation) that
     defeated the per-tier semaphore approach.

  2. ``asyncio.wait_for`` enforces a hard wall-clock budget on the
     loop side. On timeout the asyncio future is cancelled; the worker
     thread is abandoned (Python can't pre-empt sync code from another
     thread), but the sync httpx client has its own read-timeout
     ceiling so the worker self-terminates within a bounded window.

  3. A process-wide circuit breaker per tier opens after N consecutive
     failures (default 5) and auto-recovers via a half-open probe. When
     a tier's circuit is open, calls return None instantly — no thread
     dispatch, no log spam.

  4. Per-call ``request_id`` (uuid4 hex[:8]) flows into every log line
     so operator triage can follow one call across the asyncio side
     and the worker thread.

Returns parsed-JSON dicts (or None on failure).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
import weakref
from typing import Any

import httpx
from loguru import logger

from core.signals import ollama_executor
from core.utils.config import env, get_config

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_TIERS: tuple[str, ...] = ("fast", "deep", "validator")

# Epoch second of the last successful /api/generate response. Read by
# the event-loop watchdog to decide when "Ollama is silent" escalates
# from WARN to ERROR.
LAST_SUCCESS_TS: float = 0.0


def _ollama_cfg() -> dict[str, Any]:
    return get_config().get("ollama") or {}


# ============================================================================
# Process-wide circuit breaker
# ============================================================================
#
# The previous per-instance / per-tier cooldown wasn't enough during the
# April 2026 soak: a transient Ollama stall produced timeouts on every
# lane simultaneously, each lane's own client back-off counter rose
# independently, and the watchdog spent the whole window in degraded
# mode while the lanes hammered the broken server.
#
# A process-wide breaker fixes that: once any tier hits the failure
# threshold, every OllamaClient in the process treats that tier as
# disabled for the cooldown window. One call gets through after the
# window (half-open probe); success closes; failure re-opens for a
# longer window.
#
# All circuit state is read and written from the asyncio loop thread —
# no cross-thread access — so no lock is needed.


class _CircuitState:
    __slots__ = (
        "consecutive_failures",
        "open_until",
        "is_half_open",
        "open_count",
    )

    def __init__(self) -> None:
        self.consecutive_failures: int = 0
        # Monotonic timestamp; calls before this return None instantly.
        self.open_until: float = 0.0
        # When True, the next call is a probe — only one in-flight
        # probe at a time (guarded by ``_probe_in_flight`` below).
        self.is_half_open: bool = False
        # Number of times the breaker has opened in this process —
        # surfaced in logs to make repeated outages obvious in triage.
        self.open_count: int = 0


_circuit: dict[str, _CircuitState] = {t: _CircuitState() for t in _TIERS}
# Half-open probe gate: only one probe per tier at a time, so a
# stampede of callers post-recovery doesn't all dispatch.
_probe_in_flight: dict[str, bool] = {t: False for t in _TIERS}


def _circuit_threshold() -> int:
    return int(_ollama_cfg().get("circuit_open_threshold", 5))


def _circuit_open_seconds() -> float:
    return float(_ollama_cfg().get("circuit_open_seconds", 60.0))


def circuit_is_open(tier: str) -> bool:
    """Public read for callers that want to short-circuit BEFORE
    incurring a heuristic-fallback build (e.g. dashboard health card)."""
    state = _circuit.get(tier)
    if state is None:
        return False
    return time.monotonic() < state.open_until and not state.is_half_open


def _record_circuit_success(tier: str) -> None:
    state = _circuit[tier]
    if state.consecutive_failures > 0 or state.is_half_open or state.open_until > 0:
        logger.info(
            "[ollama] circuit CLOSED tier={} (was open_count={})",
            tier, state.open_count,
        )
    state.consecutive_failures = 0
    state.open_until = 0.0
    state.is_half_open = False
    _probe_in_flight[tier] = False


def _record_circuit_failure(tier: str, error: str) -> None:
    state = _circuit[tier]
    state.consecutive_failures += 1
    threshold = _circuit_threshold()
    cooldown = _circuit_open_seconds()
    # Half-open probe failed — re-open for a longer window so we don't
    # pin Ollama under load right after recovery.
    if state.is_half_open:
        state.open_until = time.monotonic() + cooldown * 2
        state.open_count += 1
        state.is_half_open = False
        _probe_in_flight[tier] = False
        logger.warning(
            "[ollama] circuit RE-OPENED tier={} (probe failed: {}); "
            "cooldown={:.0f}s open_count={}",
            tier, error, cooldown * 2, state.open_count,
        )
        return
    if state.consecutive_failures >= threshold and state.open_until == 0.0:
        state.open_until = time.monotonic() + cooldown
        state.open_count += 1
        logger.warning(
            "[ollama] circuit OPENED tier={} after {} consecutive failures; "
            "cooldown={:.0f}s open_count={}",
            tier, state.consecutive_failures, cooldown, state.open_count,
        )


def _circuit_admit(tier: str) -> tuple[bool, bool]:
    """Decide whether this call may proceed.

    Returns ``(admit, is_probe)``. When ``admit`` is False, the caller
    must short-circuit to None without dispatching. ``is_probe`` is
    True iff the breaker is half-open and this call was selected as
    the recovery probe — the caller still dispatches normally; the
    flag affects how we record success/failure afterwards.
    """
    state = _circuit.get(tier)
    if state is None:
        return True, False
    now = time.monotonic()
    if state.open_until == 0.0:
        return True, False
    if now < state.open_until:
        # Still inside the cooldown window: deny.
        return False, False
    # Window elapsed — admit ONE caller as a probe; everyone else
    # waits for the probe outcome.
    if state.is_half_open and _probe_in_flight[tier]:
        return False, False
    state.is_half_open = True
    _probe_in_flight[tier] = True
    logger.info(
        "[ollama] circuit HALF-OPEN tier={} probing recovery", tier,
    )
    return True, True


def reset_circuits_for_tests() -> None:
    """Drop circuit state. Tests should call this in an autouse
    fixture if they exercise the breaker."""
    for t in _TIERS:
        _circuit[t] = _CircuitState()
        _probe_in_flight[t] = False


# ============================================================================
# Backwards-compat surface (warmup / healthy / running_models still
# need a short-lived async client; the executor is for hot-path calls)
# ============================================================================


async def reset_shared_client() -> None:
    """Compatibility shim: the watchdog still calls this on contention.
    With the executor architecture there's no shared async client to
    reset, but the executor's per-thread sync clients aren't the
    problem either — the failure mode that motivated this hook
    (entangled async pool state) no longer exists. We call the
    executor's reset helper as a heavy hammer that drops every per-
    thread sync client too, just in case."""
    ollama_executor.reset_for_tests()


async def aclose_shared_client() -> None:
    """Called from run.py at shutdown. Tear down the executor pools so
    no zombie threads outlive the asyncio loop."""
    ollama_executor.shutdown(wait=True)


class OllamaClient:
    # Track every live instance so the Settings page can reset back-off
    # state across the whole process when host/model change.
    _instances: "weakref.WeakSet[OllamaClient]" = weakref.WeakSet()

    # Process-wide queue depth counters (legacy: still read by the
    # watchdog and dashboard). Now a thin wrapper over the executor's
    # in-flight tracker so the numbers match what's really running on
    # threads.
    pending_fast: int = 0
    pending_deep: int = 0
    pending_validator: int = 0

    def __init__(self) -> None:
        # Legacy keys still honored so existing callers don't regress.
        legacy_timeout = float(
            get_config().get("signals", "ollama_timeout_seconds", default=60)
        )
        self._default_timeout = legacy_timeout
        # Per-instance per-tier cooldown was superseded by the module-
        # level circuit breaker above, but we keep the counter for the
        # dashboard's existing per-client view.
        self._consecutive_failures: dict[str, int] = {t: 0 for t in _TIERS}
        # Per-model set so switching via the dashboard re-arms the warning
        # for whichever model is currently missing.
        self._missing_models_warned: set[str] = set()
        OllamaClient._instances.add(self)

    # ---- Global state management -------------------------------------

    @classmethod
    def _reset_global_cooldowns(cls) -> None:
        """Clear back-off state on every live client AND the module-
        level circuit breaker. Called by the Settings page after the
        user updates OLLAMA_* env vars so the new config gets tried
        immediately."""
        for inst in list(cls._instances):
            inst._consecutive_failures = {t: 0 for t in _TIERS}
            inst._missing_models_warned = set()
        reset_circuits_for_tests()

    @classmethod
    def queue_depths(cls) -> dict[str, int]:
        """Return per-tier in-flight counts.

        We consult two sources and return the max for each tier:

          1. The class-level ``pending_*`` counters, which the executor
             increments/decrements on every dispatch (see
             ``ollama_executor._do_post``). These reflect real running
             calls in production.
          2. Tests that simulate saturation by setting ``pending_fast =
             99`` directly. Reading the executor would clobber the test
             override; we honor whichever value is larger.

        Both sources are read without mutating the class state, so a
        test override survives subsequent calls."""
        snap = ollama_executor.in_flight_snapshot()
        per_tier = snap.per_tier_in_flight
        return {
            "fast": max(cls.pending_fast, per_tier.get("fast", 0)),
            "deep": max(cls.pending_deep, per_tier.get("deep", 0)),
            "validator": max(
                cls.pending_validator, per_tier.get("validator", 0),
            ),
        }

    @classmethod
    def fast_queue_saturated(cls) -> bool:
        """True when the fast-tier queue is at/above the configured alert.
        Callers that can degrade gracefully (event_sniper -> heuristic)
        check this before issuing a call."""
        depth = cls.queue_depths()["fast"]
        alert = int(_ollama_cfg().get("queue_depth_alert", 5))
        return depth >= alert

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
            return float(cfg.get("fast_timeout_seconds", 10))
        if call_type == "validator":
            return float(cfg.get("validator_timeout_seconds", 20))
        if call_type == "deep":
            return float(cfg.get("deep_timeout_seconds", 20))
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

    # ---- Back-off plumbing (legacy per-instance counters; the real
    # circuit lives at module scope) ----------------------------------

    def _in_cooldown(self, tier: str = "deep") -> bool:
        return circuit_is_open(tier)

    def _record_success(self, tier: str = "deep") -> None:
        self._consecutive_failures[tier] = 0
        _record_circuit_success(tier)

    def _record_failure(self, tier: str = "deep", *, error: str = "") -> None:
        self._consecutive_failures[tier] = self._consecutive_failures.get(tier, 0) + 1
        _record_circuit_failure(tier, error or "unknown")

    def _explain_failure(self, error: str, model: str) -> None:
        # 404 from /api/generate almost always means the model isn't pulled.
        if error.startswith("HTTPStatusError:404"):
            if model not in self._missing_models_warned:
                logger.warning(
                    "[ollama] /api/generate returned 404 — model '{}' is not "
                    "available locally. Run `ollama pull {}` to install it. "
                    "Further 404s for this model will be suppressed.",
                    model, model,
                )
                self._missing_models_warned.add(model)
            return
        logger.warning("[ollama] request failed ({}) {}", model, error)

    # ---- Class-level legacy queue helpers (kept for tests) -----------
    # These mirror what the executor tracks. Tests poke ``pending_fast``
    # directly to simulate saturation; we honor that by not refreshing
    # from the executor when the test sets a value > the executor's view.

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

    @classmethod
    def _queue_for(cls, call_type: str) -> int:
        # Honors both the class counter (test-set saturation) and the
        # executor's real in-flight view; same semantics as
        # ``queue_depths`` but for a single tier.
        return cls.queue_depths().get(call_type, 0)

    # ---- Core call path ----------------------------------------------

    async def _generate(
        self,
        prompt: str,
        *,
        call_type: str,
        response_format: str = "json",
        tag: str = "",
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Single Ollama /api/generate call with stats logging.

        ``tag`` is an opaque caller-supplied label (typically a market_id
        or "pipeline:<feed_id>") that flows into the per-call log line so
        operator triage can map a slow latency back to the originating
        market without grep'ing across modules. It does NOT affect the
        request body or the stats row.

        Returns (parsed_body, meta) where ``meta`` always contains at
        least {"raw_text", "latency_ms", "success", "error",
        "request_id"} so the caller (or validator) can inspect what
        came back.
        """
        request_id = uuid.uuid4().hex[:8]
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
            "request_id": request_id,
        }

        admit, is_probe = _circuit_admit(call_type)
        if not admit:
            meta["error"] = "circuit_open"
            logger.debug(
                "[ollama] tier={} req={} skipped (circuit open) tag={}",
                call_type, request_id, tag or "-",
            )
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
        # Sync-side timeout is generous (~3x the asyncio budget) so the
        # asyncio wait_for is the operative deadline. Worst case the
        # worker thread eventually unblocks even if asyncio gave up.
        sync_timeout = max(timeout * 3.0, timeout + 30.0)
        loop = asyncio.get_running_loop()
        cf_future, _rid = ollama_executor.submit_generate(
            tier=call_type, url=url, body=body,
            sync_timeout=sync_timeout, request_id=request_id,
        )
        queue_depth_at_dispatch = OllamaClient.queue_depths().get(call_type, 0)
        aio_future = asyncio.wrap_future(cf_future, loop=loop)
        start = time.perf_counter()
        try:
            # ``wait_for`` cancels the asyncio future on timeout. The
            # underlying concurrent.futures.Future may keep running on
            # the worker thread (Python can't pre-empt sync code) but
            # the executor's sync-side timeout is bounded, so the
            # worker self-terminates within sync_timeout seconds.
            result: ollama_executor.GenerateResult = await asyncio.wait_for(
                aio_future, timeout=timeout,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            meta["latency_ms"] = elapsed_ms
            meta["error"] = "TimeoutError"
            logger.warning(
                "[ollama] tier={} req={} model={} tag={} latency_ms={:.0f} "
                "TIMEOUT (asyncio.wait_for >{:.1f}s)",
                call_type, request_id, model, tag or "-", elapsed_ms, timeout,
            )
            self._record_failure(call_type, error="TimeoutError")
            await self._log_stat(meta)
            return None, meta
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            meta["latency_ms"] = elapsed_ms
            meta["error"] = type(e).__name__
            logger.warning(
                "[ollama] tier={} req={} model={} tag={} unexpected: {}",
                call_type, request_id, model, tag or "-", e,
            )
            self._record_failure(call_type, error=type(e).__name__)
            await self._log_stat(meta)
            return None, meta

        meta["latency_ms"] = result.latency_ms
        if result.data is None:
            meta["error"] = result.error or "EmptyResponse"
            logger.debug(
                "[ollama] tier={} req={} model={} tag={} latency_ms={:.0f} "
                "queue_depth={} success=false error={}",
                call_type, request_id, model, tag or "-", result.latency_ms,
                queue_depth_at_dispatch, meta["error"],
            )
            self._explain_failure(meta["error"], model)
            self._record_failure(call_type, error=meta["error"])
            await self._log_stat(meta)
            return None, meta

        meta["success"] = True
        meta["tokens_in"] = int(result.data.get("prompt_eval_count") or 0)
        meta["tokens_out"] = int(result.data.get("eval_count") or 0)
        self._record_success(call_type)
        global LAST_SUCCESS_TS
        LAST_SUCCESS_TS = time.time()

        text = (result.data.get("response") or "").strip()
        meta["raw_text"] = text
        parsed = self._extract_json(text) if response_format == "json" else None
        logger.debug(
            "[ollama] tier={} req={} model={} tag={} latency_ms={:.0f} "
            "queue_depth={} success=true tokens_in={} tokens_out={} "
            "circuit_probe={}",
            call_type, request_id, model, tag or "-", result.latency_ms,
            queue_depth_at_dispatch, meta["tokens_in"], meta["tokens_out"],
            "yes" if is_probe else "no",
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
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        tag: str = "",
    ) -> dict[str, Any] | None:
        """Small-model JSON scoring. Target <5s. Used by event_sniper
        and scalping re-scores."""
        parsed, _ = await self._generate(prompt, call_type="fast", tag=tag)
        return parsed

    async def deep_score(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        tag: str = "",
    ) -> dict[str, Any] | None:
        """Mid-model JSON scoring. Target 10-30s. Used by signal
        pipeline and longshot."""
        parsed, _ = await self._generate(prompt, call_type="deep", tag=tag)
        return parsed

    async def validate(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        tag: str = "",
    ) -> dict[str, Any] | None:
        """Cross-validation model. Low temperature, no exploration —
        used by shadow.open_position on entries >= high-stakes threshold."""
        parsed, _ = await self._generate(prompt, call_type="validator", tag=tag)
        return parsed

    # ---- Legacy API (kept for backward compat) -----------------------

    async def generate_json(
        self, prompt: str, *, tag: str = "",
    ) -> dict[str, Any] | None:
        """Legacy entry point. Routes to the deep tier — callers that
        haven't been migrated to fast/validator get sensible defaults."""
        return await self.deep_score(prompt, tag=tag)

    async def generate_text(self, prompt: str, *, tag: str = "") -> str:
        """Freeform (non-JSON) generation. Used by prompt evolution.
        Always goes to the deep tier at a slightly warmer temperature."""
        # Routes through the same _generate path so we get the executor,
        # circuit breaker, and structured logs for free.
        request_id = uuid.uuid4().hex[:8]
        admit, _is_probe = _circuit_admit("deep")
        if not admit:
            return ""
        url = f"{self.host}/api/generate"
        model = self.deep_model
        timeout = self._timeout_for("deep")
        sync_timeout = max(timeout * 3.0, timeout + 30.0)
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
        loop = asyncio.get_running_loop()
        cf_future, _rid = ollama_executor.submit_generate(
            tier="deep", url=url, body=body,
            sync_timeout=sync_timeout, request_id=request_id,
        )
        aio_future = asyncio.wrap_future(cf_future, loop=loop)
        try:
            result: ollama_executor.GenerateResult = await asyncio.wait_for(
                aio_future, timeout=timeout,
            )
        except asyncio.TimeoutError:
            self._record_failure("deep", error="TimeoutError")
            return ""
        except Exception as e:
            self._record_failure("deep", error=type(e).__name__)
            return ""
        if result.data is None:
            self._explain_failure(result.error, model)
            self._record_failure("deep", error=result.error)
            return ""
        self._record_success("deep")
        global LAST_SUCCESS_TS
        LAST_SUCCESS_TS = time.time()
        return (result.data.get("response") or "").strip()

    # ---- Health ------------------------------------------------------
    # Both probes open a short-lived async client. They're called at
    # most once per minute (health check scheduler / dashboard render),
    # latency is single-digit ms on a healthy server, and they don't
    # share state with the hot-path executor — so the async path here
    # is a deliberate keep-it-simple choice.

    async def warmup(self, *, timeout: float = 120.0) -> bool:
        """Force a model load + one-token generation before lanes go live.
        Routes through the executor so the loop stays clean even if the
        model swap takes minutes (cold mmap of a 5 GB GGUF). Returns
        True on a successful generation; logs + returns False on any
        error."""
        model = self.deep_model
        body = {
            "model": model,
            "prompt": "ping",
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0.0},
        }
        url = f"{self.host}/api/generate"
        loop = asyncio.get_running_loop()
        request_id = uuid.uuid4().hex[:8]
        # Warmup uses the deep tier's executor (the deep model is what
        # we're loading), with a generous sync_timeout because cold
        # mmap on Windows can take 30-60s.
        cf_future, _rid = ollama_executor.submit_generate(
            tier="deep", url=url, body=body,
            sync_timeout=max(timeout, 120.0), request_id=request_id,
        )
        aio_future = asyncio.wrap_future(cf_future, loop=loop)
        t0 = time.perf_counter()
        try:
            result: ollama_executor.GenerateResult = await asyncio.wait_for(
                aio_future, timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[ollama] warmup timeout ({}, req={}) after {:.1f}s",
                model, request_id, time.perf_counter() - t0,
            )
            return False
        except Exception as e:
            logger.warning(
                "[ollama] warmup failed ({}, req={}): {}",
                model, request_id, e,
            )
            return False
        if result.data is None:
            logger.warning(
                "[ollama] warmup error ({}, req={}): {} (after {:.1f}s)",
                model, request_id, result.error, time.perf_counter() - t0,
            )
            return False
        elapsed = time.perf_counter() - t0
        global LAST_SUCCESS_TS
        LAST_SUCCESS_TS = time.time()
        logger.info(
            "[ollama] warmup OK ({}, req={}): model resident in {:.1f}s",
            model, request_id, elapsed,
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
