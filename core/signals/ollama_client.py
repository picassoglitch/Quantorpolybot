"""Async Ollama client. Uses raw httpx so we don't need the sync ollama lib
in the hot path. Returns a strict-JSON parsed dict.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
from loguru import logger

from core.utils.config import env, get_config

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class OllamaClient:
    def __init__(self) -> None:
        self.host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.model = env("OLLAMA_MODEL", "mistral")
        timeout = float(get_config().get("signals", "ollama_timeout_seconds", default=60))
        self._timeout = timeout

    async def generate_json(self, prompt: str) -> dict[str, Any] | None:
        """Call Ollama and return a parsed JSON dict, or None on failure."""
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
            logger.warning("[ollama] request failed: {}", e)
            return None
        text = (data.get("response") or "").strip()
        return self._extract_json(text)

    async def generate_text(self, prompt: str) -> str:
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
                return (data.get("response") or "").strip()
        except Exception as e:
            logger.warning("[ollama] text request failed: {}", e)
            return ""

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
