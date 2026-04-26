"""Polymarket per-market related-news scraper.

For each active market in our DB, hits gamma-api /markets/{id} and
extracts the `relatedNews` field (or any URLs embedded in the market
description). Each linked article becomes a feed_item with
``meta.linked_market_id`` set so the signal pipeline can bypass its
keyword pre-filter — these news items are by definition relevant to
that specific market.

Polite to Gamma: paces requests via ``request_delay_seconds`` and caps
per-cycle volume via ``max_markets_per_cycle``.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Iterable

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts
from core.utils.watchdog import is_degraded

GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets/{market_id}"
_HEADERS = {
    "User-Agent": "NexoPolyBot/0.1 (+https://github.com/local)",
    "Accept": "application/json",
}
# Crude URL extractor for description bodies that don't carry a
# structured relatedNews field. Avoids capturing trailing punctuation.
_URL_RE = re.compile(r"https?://[^\s)\]<>\"']+", re.IGNORECASE)


class PolymarketNewsFeed:
    component = "feed.polymarket_news"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        # Round-robin cursor across the active-market list so each
        # cycle picks up where the previous one left off.
        self._cursor: int = 0

    async def run(self) -> None:
        cfg = get_config().get("feeds", "polymarket_news") or {}
        if not cfg.get("enabled", True):
            logger.info("[poly_news] disabled")
            return
        poll = int(cfg.get("poll_seconds", 600))
        per_cycle = int(cfg.get("max_markets_per_cycle", 80))
        delay = float(cfg.get("request_delay_seconds", 0.25))
        backoff = Backoff(base=5, cap=300)

        logger.info("[poly_news] starting; poll={}s per_cycle={}", poll, per_cycle)
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
            while not self._stop.is_set():
                # When the event loop is under pressure, polymarket_news is
                # the lowest-value feed (per-market Gamma lookups). Skip
                # the cycle entirely rather than throttle — frees the pool
                # for order/signal traffic. Watchdog DEGRADED flip-off
                # happens after ~3min of clean ticks.
                if is_degraded():
                    logger.debug("[poly_news] degraded — skipping cycle")
                    await self._sleep(poll)
                    continue
                try:
                    new = await self._poll_cycle(client, per_cycle, delay)
                    if new:
                        logger.info("[poly_news] ingested {} new linked articles", new)
                    backoff.reset()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    d = backoff.next_delay()
                    # httpx ConnectError/ConnectTimeout stringify to '' —
                    # fall back to the class name so the log line isn't
                    # truncated after the colon.
                    detail = str(e) or type(e).__name__
                    logger.warning(
                        "[poly_news] error ({}), sleeping {:.1f}s: {}",
                        type(e).__name__, d, detail,
                    )
                    await self._sleep(d)
                    continue
                await self._sleep(poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _poll_cycle(
        self, client: httpx.AsyncClient, per_cycle: int, delay: float
    ) -> int:
        rows = await fetch_all(
            "SELECT market_id, question FROM markets WHERE active=1 ORDER BY liquidity DESC"
        )
        if not rows:
            return 0
        # Slide a window over the full active set so over many cycles
        # we cover everything, not just the top N by liquidity.
        if self._cursor >= len(rows):
            self._cursor = 0
        window = rows[self._cursor : self._cursor + per_cycle]
        if len(window) < per_cycle and self._cursor > 0:
            window = window + rows[: per_cycle - len(window)]
        self._cursor = (self._cursor + per_cycle) % max(len(rows), 1)

        new_total = 0
        for row in window:
            if self._stop.is_set():
                break
            mid_str = row["market_id"]
            question = row["question"] or ""
            try:
                payload = await self._fetch_market(client, mid_str)
            except httpx.HTTPStatusError as e:
                # 404s are normal (market id format mismatch between Gamma
                # /markets list and /markets/{id} lookup), don't shout.
                if e.response.status_code != 404:
                    logger.warning("[poly_news] {} HTTP {}", mid_str, e.response.status_code)
                payload = None
            if payload is not None:
                new_total += await self._ingest_market(mid_str, question, payload)
            if delay > 0:
                await self._sleep(delay)
        return new_total

    async def _fetch_market(
        self, client: httpx.AsyncClient, market_id: str
    ) -> dict[str, Any] | None:
        url = GAMMA_MARKET_URL.format(market_id=market_id)
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else None

    async def _ingest_market(
        self, market_id: str, question: str, payload: dict[str, Any]
    ) -> int:
        links = list(_extract_links(payload))
        new_count = 0
        for link in links:
            href = link["url"]
            h = url_hash(f"polynews:{market_id}:{href}")
            existing = await fetch_one(
                "SELECT id FROM feed_items WHERE url_hash=?", (h,)
            )
            if existing:
                continue
            title = link.get("title") or f"Linked news for: {question[:80]}"
            summary = link.get("summary") or ""
            meta = {
                "linked_market_id": market_id,
                "linked_question": question,
                "raw_source": link.get("source", ""),
            }
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at, meta)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    h,
                    "polymarket_news",
                    title.strip(),
                    summary.strip()[:1000],
                    href,
                    link.get("published_at") or now_ts(),
                    now_ts(),
                    json.dumps(meta),
                ),
            )
            new_count += 1
        return new_count


def _extract_links(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield {url, title, summary, source, published_at} dicts from
    whichever shape the Gamma response happens to use."""
    seen: set[str] = set()

    def emit(url: str, **fields: Any) -> Iterable[dict[str, Any]]:
        u = (url or "").strip()
        if not u or u in seen:
            return
        seen.add(u)
        yield {"url": u, **fields}

    # Structured field — prefer when present.
    related = payload.get("relatedNews") or payload.get("related_news") or []
    if isinstance(related, list):
        for item in related:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("link") or ""
            yield from emit(
                url,
                title=item.get("title") or item.get("headline") or "",
                summary=item.get("description") or item.get("summary") or "",
                source=item.get("source") or item.get("publisher") or "",
                published_at=item.get("publishedAt") or item.get("published_at"),
            )

    # Some markets put resource links inside a `resolutionSource` or
    # an array of objects under `events` / `attachments`.
    for key in ("resolutionSource", "resolution_source", "sourceUrl", "source_url"):
        val = payload.get(key)
        if isinstance(val, str):
            yield from emit(val, title=key)

    for key in ("events", "attachments", "resources"):
        val = payload.get(key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    yield from emit(
                        item.get("url") or item.get("link") or "",
                        title=item.get("title") or item.get("name") or "",
                        summary=item.get("description") or "",
                    )

    # Fallback: scrape URLs out of the description.
    description = payload.get("description") or ""
    if isinstance(description, str):
        for m in _URL_RE.finditer(description):
            yield from emit(m.group(0), title="", summary=description[:300], source="description")
