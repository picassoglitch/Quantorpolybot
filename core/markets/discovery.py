"""Polymarket Gamma API market discovery. Walks paginated /markets and
upserts active markets into the local SQLite cache.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, executemany
from core.utils.helpers import Backoff, now_ts, safe_float


class MarketDiscovery:
    component = "markets.discovery"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    async def run(self) -> None:
        cfg = get_config().get("markets") or {}
        poll = int(cfg.get("refresh_seconds", 300))
        backoff = Backoff(base=5, cap=300)
        logger.info("[markets] discovery loop started poll={}s", poll)
        while not self._stop.is_set():
            try:
                count = await self.refresh_once()
                logger.info("[markets] refreshed {} markets", count)
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = backoff.next_delay()
                logger.exception("[markets] refresh error, sleeping {:.1f}s: {}", delay, e)
                await self._sleep(delay)
                continue
            await self._sleep(poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def refresh_once(self) -> int:
        async with self._lock:
            cfg = get_config().get("markets") or {}
            base = cfg.get("gamma_url", "https://gamma-api.polymarket.com/markets")
            page_size = int(cfg.get("page_size", 100))
            max_pages = int(cfg.get("max_pages", 30))
            active_only = bool(cfg.get("active_only", True))

            total_upserted = 0
            offset = 0
            async with httpx.AsyncClient(timeout=20.0) as client:
                for _ in range(max_pages):
                    params: dict[str, Any] = {"limit": page_size, "offset": offset}
                    if active_only:
                        params["active"] = "true"
                        params["closed"] = "false"
                    r = await client.get(base, params=params)
                    r.raise_for_status()
                    payload = r.json()
                    markets = payload if isinstance(payload, list) else payload.get("data", [])
                    if not markets:
                        break
                    rows = [self._row_from_market(m) for m in markets]
                    rows = [r for r in rows if r is not None]
                    if rows:
                        await self._upsert(rows)
                        total_upserted += len(rows)
                    if len(markets) < page_size:
                        break
                    offset += page_size
            return total_upserted

    @staticmethod
    def _row_from_market(m: dict[str, Any]) -> tuple | None:
        market_id = m.get("id") or m.get("conditionId") or m.get("condition_id")
        if not market_id:
            return None
        question = (m.get("question") or m.get("title") or "").strip()
        slug = m.get("slug") or m.get("market_slug") or ""
        category = m.get("category") or (m.get("tags") or [{}])[0].get("label", "") if m.get("tags") else m.get("category", "")
        if isinstance(category, dict):
            category = category.get("label", "")
        active = 1 if m.get("active", True) and not m.get("closed", False) else 0
        close_time = m.get("end_date") or m.get("endDate") or m.get("close_time")
        token_ids = m.get("clobTokenIds") or m.get("tokenIds") or m.get("tokens")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                token_ids = [token_ids]
        if isinstance(token_ids, list):
            token_ids_json = json.dumps([str(t) for t in token_ids])
        else:
            token_ids_json = json.dumps([])
        liquidity = safe_float(m.get("liquidity") or m.get("liquidityNum"))
        last_price = safe_float(m.get("lastTradePrice") or m.get("last_price"))
        bid = safe_float(m.get("bestBid") or m.get("best_bid"))
        ask = safe_float(m.get("bestAsk") or m.get("best_ask"))
        return (
            str(market_id),
            question,
            slug,
            str(category or ""),
            active,
            str(close_time or ""),
            token_ids_json,
            bid,
            ask,
            last_price,
            liquidity,
            now_ts(),
        )

    @staticmethod
    async def _upsert(rows: list[tuple]) -> None:
        await executemany(
            """INSERT INTO markets
            (market_id, question, slug, category, active, close_time, token_ids,
             best_bid, best_ask, last_price, liquidity, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(market_id) DO UPDATE SET
              question=excluded.question,
              slug=excluded.slug,
              category=excluded.category,
              active=excluded.active,
              close_time=excluded.close_time,
              token_ids=excluded.token_ids,
              best_bid=excluded.best_bid,
              best_ask=excluded.best_ask,
              last_price=excluded.last_price,
              liquidity=excluded.liquidity,
              updated_at=excluded.updated_at
            """,
            rows,
        )

    async def deactivate_stale(self, max_age_seconds: float = 86400) -> int:
        """Mark markets inactive if we haven't seen them in a refresh cycle."""
        cutoff = now_ts() - max_age_seconds
        await execute("UPDATE markets SET active=0 WHERE updated_at < ?", (cutoff,))
        return 0
