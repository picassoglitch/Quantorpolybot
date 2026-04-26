"""Background REST-based price refresher.

The Polymarket WS pushes updates for subscribed tokens, but:
  * We subscribe to at most `subscribe_batch` tokens (50 by default), so
    any active market beyond that slice has a growing `updated_at` age.
  * After a WS dropout, all cached prices go stale until the reconnect
    and resubscribe finishes.

Either of those leaves the risk engine seeing `updated_at` older than
`stale_price_max_age_seconds` and rejecting the signal even though
Gamma would cheerfully hand us a fresh orderbook.

This task periodically finds stale active markets and does one Gamma
REST round-trip each to refresh `best_bid / best_ask / last_price /
updated_at`. The risk engine's stale-price guard stays on — so if both
the WS and Gamma REST are broken, signals still get rejected.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_all
from core.utils.helpers import Backoff, now_ts, safe_float

_GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets"


class PriceRefresher:
    component = "feeds.price_refresher"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "price_refresher") or {}
        if not cfg.get("enabled", True):
            logger.info("[price_refresher] disabled")
            return
        interval = float(cfg.get("poll_seconds", 60))
        backoff = Backoff(base=10, cap=300)
        logger.info("[price_refresher] started interval={}s", interval)
        while not self._stop.is_set():
            try:
                refreshed = await self.refresh_once()
                if refreshed:
                    logger.info("[price_refresher] refreshed {} stale markets", refreshed)
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = backoff.next_delay()
                logger.warning(
                    "[price_refresher] error ({}), sleeping {:.1f}s: {}",
                    type(e).__name__, delay, e,
                )
                await self._sleep(delay)
                continue
            await self._sleep(interval)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def refresh_once(self) -> int:
        cfg = get_config().get("feeds", "price_refresher") or {}
        risk_cfg = get_config().get("risk") or {}
        # Refetch anything older than half the risk engine's staleness
        # budget so the risk engine sees fresh data on the next pass.
        stale_budget = float(
            risk_cfg.get(
                "stale_price_max_age_seconds",
                risk_cfg.get("stale_price_seconds", 300),
            )
        )
        stale_after = float(cfg.get("stale_after_seconds", stale_budget / 2))
        max_per_cycle = int(cfg.get("max_markets_per_cycle", 40))
        request_delay = float(cfg.get("request_delay_seconds", 0.25))
        cutoff = now_ts() - stale_after

        rows = await fetch_all(
            """SELECT market_id, token_ids FROM markets
               WHERE active=1 AND (updated_at IS NULL OR updated_at < ?)
               ORDER BY updated_at ASC NULLS FIRST LIMIT ?""",
            (cutoff, max_per_cycle),
        )
        if not rows:
            return 0

        refreshed = 0
        async with httpx.AsyncClient(timeout=10.0) as client:
            for row in rows:
                if self._stop.is_set():
                    break
                market_id = row["market_id"]
                payload = await _gamma_fetch(client, market_id)
                if payload is None:
                    continue
                bid, ask, last = _extract_prices(payload, row["token_ids"])
                if bid is None and ask is None and last is None:
                    continue
                await execute(
                    """UPDATE markets
                       SET best_bid=COALESCE(?, best_bid),
                           best_ask=COALESCE(?, best_ask),
                           last_price=COALESCE(?, last_price),
                           updated_at=?
                       WHERE market_id=?""",
                    (bid, ask, last, now_ts(), market_id),
                )
                refreshed += 1
                if request_delay > 0:
                    await asyncio.sleep(request_delay)
        return refreshed


async def _gamma_fetch(client: httpx.AsyncClient, market_id: str) -> dict[str, Any] | None:
    try:
        r = await client.get(f"{_GAMMA_MARKET_URL}/{market_id}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.debug("[price_refresher] fetch failed for {}: {}", market_id, e)
        return None


def _extract_prices(
    payload: dict[str, Any], token_ids_json: str | None
) -> tuple[float | None, float | None, float | None]:
    bid_raw = payload.get("bestBid") or payload.get("best_bid")
    ask_raw = payload.get("bestAsk") or payload.get("best_ask")
    last_raw = payload.get("lastTradePrice") or payload.get("last_price")
    # markets.best_bid/ask are stored for the YES token (token_ids[0]).
    # Gamma returns them in that frame directly. Nothing to flip here.
    _ = token_ids_json
    bid = safe_float(bid_raw) if bid_raw is not None else None
    ask = safe_float(ask_raw) if ask_raw is not None else None
    last = safe_float(last_raw) if last_raw is not None else None
    return bid, ask, last
