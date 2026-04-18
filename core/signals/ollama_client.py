"""Async Ollama client. Uses raw httpx so we don't need the sync ollama lib
in the hot path. Returns a strict-JSON parsed dict.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx
from loguru import logger

from core.utils.config import env, get_config

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# After this many consecutive failures we enter a cooldown window during
# which calls return None immediately instead of hammering a broken Ollama.
_FAIL_THRESHOLD = 5
_COOLDOWN_SECONDS = 60.0


class OllamaClient:
    def __init__(self) -> None:
        self.host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = env("OLLAMA_MODEL", "mistral")
        timeout = float(get_config().get("signals", "ollama_timeout_seconds", default=60))
        self._timeout = timeout
        self._consecutive_failures = 0
        self._cooldown_until = 0.0
        self._missing_model_warned = False

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

    def _explain_failure(self, e: Exception) -> None:
        # 404 from /api/generate almost always means the model isn't pulled.
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
            if not self._missing_model_warned:
                logger.warning(
                    "[ollama] /api/generate returned 404 — model '{}' is not "
                    "available locally. Run `ollama pull {}` to install it. "
                    "Further 404s will be suppressed.",
                    self.model, self.model,
                )
                self._missing_model_warned = True
            return
        # Otherwise log every failure (transient network / timeout / etc).
        logger.warning("[ollama] request failed: {}", e)

    async def generate_json(self, prompt: str) -> dict[str, Any] | None:
        """Call Ollama and return a parsed JSON dict, or None on failure."""
        if self._in_cooldown():
            return None
        url = f"{self.host}/api/generate"
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(url, json=body)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            self._explain_failure(e)
            self._record_failure()
            return None
        self._record_success()
        text = (data.get("response") or "").strip()
        return self._extract_json(text)

    async def generate_text(self, prompt: str) -> str:
        if self._in_cooldown():
            return ""
        url = f"{self.host}/api/generate"
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(url, json=body)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            self._explain_failure(e)
            self._record_failure()
            return ""
        self._record_success()
        return (data.get("response") or "").strip()

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.host}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

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
