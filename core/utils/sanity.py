"""One-shot startup sanity pings.

Hits the three external dependencies we care about — Polymarket's
gamma-api, the CLOB REST host, and the local Ollama server — with
short, individual timeouts. Logs OK/latency per endpoint, never
raises, never crashes the process. A FAIL here is informational so
the operator sees the issue at boot; the feed/signal loops have their
own retries and will pick up once the endpoint recovers.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from loguru import logger

from core.utils.config import env, get_config


async def _ping(name: str, url: str, timeout: float) -> bool:
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        ok = 200 <= r.status_code < 500  # 4xx still proves reachability
        level = logger.info if ok else logger.warning
        level(
            "[sanity] {} {} (status={}, latency={:.0f}ms)",
            name, "OK" if ok else "FAIL", r.status_code, latency_ms,
        )
        return ok
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.warning(
            "[sanity] {} FAIL ({}): {} (after {:.0f}ms)",
            name, type(e).__name__, e, latency_ms,
        )
        return False


async def run_startup_checks() -> None:
    cfg = get_config()
    gamma = cfg.get(
        "markets", "gamma_url",
        default="https://gamma-api.polymarket.com/markets",
    )
    clob = env("POLY_HOST", "https://clob.polymarket.com").rstrip("/")
    ollama = env(
        "OLLAMA_HOST",
        str(cfg.get("ollama", "host", default="http://localhost:11434")),
    ).rstrip("/")

    try:
        results = await asyncio.gather(
            _ping("gamma-api", f"{gamma}?limit=1", timeout=5.0),
            _ping("clob", f"{clob}/", timeout=5.0),
            _ping("ollama", f"{ollama}/api/tags", timeout=2.0),
            return_exceptions=True,
        )
    except Exception as e:
        logger.warning("[sanity] startup checks dispatch failed: {}", e)
        return

    passed = sum(1 for r in results if r is True)
    logger.info("[sanity] {}/{} endpoints reachable", passed, len(results))
