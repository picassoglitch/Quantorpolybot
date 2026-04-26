"""News enricher — drains unenriched ``feed_items`` through a local
Ollama model and fills ``enriched_json`` + ``market_relevance``.

Runs as its own supervised feed-manager task so one slow Ollama cycle
never blocks ingestion. Graceful degradation: on parse failure or
Ollama unreachable, the row is marked with ``market_relevance=0`` and
a logged warning — never left permanently unenriched.

TODO: full-article fetching (trafilatura). Currently enriches on
title + summary only, which is enough to assign tickers/topics for
85% of headlines but will miss context buried in long articles.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from core.feeds.news_store import mark_enriched, pending_enrichment
from core.i18n import t
from core.utils.config import env, get_config
from core.utils.watchdog import is_degraded

# Enrichment is a BACKGROUND metadata layer. The scoring pipeline
# (trading hot path) shares the same local Ollama, so the enricher must
# yield hard when the watchdog flags contention — otherwise its batches
# starve scoring and the bot misses entries. Multiplier is intentionally
# aggressive (vs x2-x3 on other feeds) because the contended resource
# IS Ollama, not the event loop.
DEGRADED_POLL_MULTIPLIER = 10

TOPIC_ENUM = [
    "regulation", "macro", "hack", "listing", "earnings",
    "geopolitics", "election", "fed", "etf", "polymarket_direct",
]

_PROMPT_TMPL = """You classify prediction-market-relevant news.

Title: {title}
Summary: {summary}
Source: {source} (trust={weight:.2f})

Return ONLY compact JSON with EXACTLY these keys:
{{
  "tickers": ["BTC", "ETH", ...],
  "topics": [one or more of {topics}],
  "sentiment": "bullish" | "bearish" | "neutral",
  "market_relevance": 0.0 to 1.0
}}

market_relevance is how likely this news moves a Polymarket market in the next 24h.
- 0.0-0.2: generic/off-topic.
- 0.3-0.5: context, slow-moving.
- 0.6-0.8: directly relevant (e.g. ETF decision, macro print, regulatory action).
- 0.9-1.0: market-moving within the hour.
Return nothing but the JSON object."""


class NewsEnricher:
    component = "feed.news_enrich"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._inflight = 0

    async def run(self) -> None:
        cfg = get_config().get("feeds", "news_enrich") or {}
        if not cfg.get("enabled", True):
            logger.info("[news-enrich] disabled")
            return
        # Default to a small model that shares VRAM with the scoring
        # path (qwen2.5:3b is already loaded) — running an 8b model here
        # while scoring hits 3b/7b evicts scoring's model on every
        # enrichment batch and melts Ollama throughput.
        model = env("OLLAMA_NEWS_MODEL", cfg.get("model", "qwen2.5:3b-instruct-q4_K_M"))
        base = env("OLLAMA_BASE_URL", env("OLLAMA_HOST", "http://localhost:11434"))
        concurrency = int(cfg.get("concurrency", 1))
        batch = int(cfg.get("batch_size", 5))
        idle_poll = float(cfg.get("idle_poll_seconds", 10))
        busy_poll = float(cfg.get("busy_poll_seconds", 3.0))

        logger.info(t("news.enricher.starting", model=model, concurrency=concurrency))
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=60.0, base_url=base) as client:
            while not self._stop.is_set():
                # Watchdog-aware throttle: when the scoring hot path is
                # struggling, pause enrichment entirely rather than
                # fighting for Ollama time slices.
                if is_degraded():
                    await self._sleep(idle_poll * DEGRADED_POLL_MULTIPLIER)
                    continue
                try:
                    rows = await pending_enrichment(limit=batch)
                    if not rows:
                        await self._sleep(idle_poll)
                        continue
                    await asyncio.gather(*[
                        self._one(client, sem, model, r) for r in rows
                    ])
                    await self._sleep(busy_poll)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "[news-enrich] cycle error ({}): {}", type(e).__name__, e,
                    )
                    await self._sleep(idle_poll)
        logger.info(t("news.enricher.stopped"))

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        model: str,
        row: dict[str, Any],
    ) -> None:
        async with sem:
            fid = int(row["id"])
            prompt = _PROMPT_TMPL.format(
                title=(row.get("title") or "")[:300],
                summary=(row.get("summary") or "")[:800],
                source=row.get("source") or "",
                weight=float(row.get("source_weight") or 0.5),
                topics=", ".join(TOPIC_ENUM),
            )
            try:
                resp = await client.post(
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                raw = payload.get("response") or ""
                parsed = _parse_json(raw)
            except Exception as e:
                logger.warning(
                    "[news-enrich] ollama call failed id={} ({}): {}",
                    fid, type(e).__name__, e,
                )
                parsed = None

            if not parsed:
                logger.warning(t("news.enricher.parse_fail", id=fid))
                await mark_enriched(fid, {"error": "parse_fail"}, 0.0)
                return

            normalised = _normalise_fields(parsed)
            await mark_enriched(
                fid, normalised, float(normalised.get("market_relevance") or 0.0),
            )


def _parse_json(text: str) -> dict[str, Any] | None:
    """Ollama format=json sometimes wraps JSON in markdown fences or
    pads with whitespace. Try direct parse, then best-effort extract."""
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _normalise_fields(obj: dict[str, Any]) -> dict[str, Any]:
    tickers = obj.get("tickers") or []
    if not isinstance(tickers, list):
        tickers = []
    topics = obj.get("topics") or []
    if not isinstance(topics, list):
        topics = []
    topics = [t for t in topics if isinstance(t, str) and t in TOPIC_ENUM]
    sentiment = str(obj.get("sentiment") or "neutral").lower().strip()
    if sentiment not in ("bullish", "bearish", "neutral"):
        sentiment = "neutral"
    try:
        rel = float(obj.get("market_relevance") or 0.0)
    except (TypeError, ValueError):
        rel = 0.0
    rel = max(0.0, min(1.0, rel))
    return {
        "tickers": [str(x).upper()[:10] for x in tickers if x][:10],
        "topics": topics[:5],
        "sentiment": sentiment,
        "market_relevance": rel,
    }


async def enrich_batch(items: list[dict[str, Any]], concurrency: int = 4) -> int:
    """One-shot enrichment used by tests / manual backfills. Returns the
    number of items successfully marked."""
    if not items:
        return 0
    model = env("OLLAMA_NEWS_MODEL", "llama3:8b-instruct-q8_0")
    base = env("OLLAMA_BASE_URL", env("OLLAMA_HOST", "http://localhost:11434"))
    sem = asyncio.Semaphore(concurrency)
    enricher = NewsEnricher()
    count = 0
    async with httpx.AsyncClient(timeout=60.0, base_url=base) as client:
        async def _go(r: dict[str, Any]) -> None:
            nonlocal count
            await enricher._one(client, sem, model, r)
            count += 1
        await asyncio.gather(*[_go(r) for r in items])
    return count
