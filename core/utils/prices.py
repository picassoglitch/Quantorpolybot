"""Shared price source for the shadow lanes.

Prefers the `price_ticks` table (WS-driven, fresh and cheap). Falls back
to a Gamma `/markets/{id}` fetch if the last tick is older than the
configured staleness window — WS drops happen and lanes must not make
exit decisions on a 10-minute-old price.

For spread checks at *exit* decision time the lanes call
`live_orderbook_snapshot`, which always hits Gamma so we're reading the
current orderbook, not whatever the WS last pushed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import fetch_one
from core.utils.helpers import now_ts, safe_float

# Fall back to Gamma when the last tick is older than this (seconds).
_STALE_TICK_SECONDS = 60.0
_GAMMA_MARKET_URL = "https://gamma-api.polymarket.com/markets"


@dataclass
class PriceSnapshot:
    token_id: str
    bid: float
    ask: float
    last: float
    ts: float
    source: str  # 'ticks' | 'gamma'

    @property
    def mid(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2.0
        return self.last or 0.0

    @property
    def spread(self) -> float:
        if self.bid and self.ask:
            return self.ask - self.bid
        return 1.0

    @property
    def spread_cents(self) -> float:
        return self.spread * 100.0


async def latest_tick(token_id: str) -> PriceSnapshot | None:
    row = await fetch_one(
        """SELECT token_id, bid, ask, last, ts
           FROM price_ticks
           WHERE token_id=?
           ORDER BY ts DESC LIMIT 1""",
        (token_id,),
    )
    if not row:
        return None
    return PriceSnapshot(
        token_id=row["token_id"],
        bid=safe_float(row["bid"]),
        ask=safe_float(row["ask"]),
        last=safe_float(row["last"]),
        ts=safe_float(row["ts"]),
        source="ticks",
    )


async def _gamma_market(market_id: str) -> dict[str, Any] | None:
    url = f"{_GAMMA_MARKET_URL}/{market_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.debug("[prices] gamma fetch failed for {}: {}", market_id, e)
        return None


def _token_ids_from_gamma(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("clobTokenIds") or payload.get("tokenIds") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [raw]
    return [str(t) for t in raw] if isinstance(raw, list) else []


async def current_price(market_id: str, token_id: str) -> PriceSnapshot | None:
    """Prefer fresh tick; fall back to Gamma if stale or missing."""
    tick = await latest_tick(token_id)
    if tick and (now_ts() - tick.ts) <= _STALE_TICK_SECONDS:
        return tick
    payload = await _gamma_market(market_id)
    if not payload:
        return tick  # stale is better than nothing
    bid = safe_float(payload.get("bestBid") or payload.get("best_bid"))
    ask = safe_float(payload.get("bestAsk") or payload.get("best_ask"))
    last = safe_float(payload.get("lastTradePrice") or payload.get("last_price"))
    tokens = _token_ids_from_gamma(payload)
    if tokens and token_id == tokens[0]:
        # yes token; bid/ask are for yes
        pass
    elif tokens and len(tokens) > 1 and token_id == tokens[1]:
        # flip for 'no' token
        bid, ask = (1.0 - ask) if ask else 0.0, (1.0 - bid) if bid else 0.0
        last = (1.0 - last) if last else 0.0
    return PriceSnapshot(
        token_id=token_id,
        bid=bid,
        ask=ask,
        last=last,
        ts=now_ts(),
        source="gamma",
    )


async def live_orderbook_snapshot(market_id: str, token_id: str) -> PriceSnapshot | None:
    """Always hit Gamma — used by scalping's liquidity exit where we
    need a fresh orderbook read, not a cached tick."""
    payload = await _gamma_market(market_id)
    if not payload:
        return None
    bid = safe_float(payload.get("bestBid") or payload.get("best_bid"))
    ask = safe_float(payload.get("bestAsk") or payload.get("best_ask"))
    last = safe_float(payload.get("lastTradePrice") or payload.get("last_price"))
    tokens = _token_ids_from_gamma(payload)
    if tokens and len(tokens) > 1 and token_id == tokens[1]:
        bid, ask = (1.0 - ask) if ask else 0.0, (1.0 - bid) if bid else 0.0
        last = (1.0 - last) if last else 0.0
    return PriceSnapshot(
        token_id=token_id,
        bid=bid,
        ask=ask,
        last=last,
        ts=now_ts(),
        source="gamma",
    )


async def volume_24h(market_id: str) -> float:
    """Best-effort 24h volume. Gamma exposes `volume24hr` on the market
    doc; fall back to total `volume` if that key is missing."""
    payload = await _gamma_market(market_id)
    if not payload:
        return 0.0
    for key in ("volume24hr", "volume_24hr", "volume24", "volume_24h"):
        v = payload.get(key)
        if v is not None:
            return safe_float(v)
    return safe_float(payload.get("volume"))


def parse_close_time(close_time: str) -> float | None:
    """Polymarket close_time comes as an ISO string. Returns unix seconds
    or None if unparseable."""
    if not close_time:
        return None
    s = close_time.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def days_until_resolve(close_time: str) -> float | None:
    ts = parse_close_time(close_time)
    if ts is None:
        return None
    return (ts - now_ts()) / 86400.0


async def latest_feed_item_ts(source: str | None = None) -> float:
    """Most recent ingest ts, globally or for one source. Used by the
    risk manager's feed-staleness check."""
    if source:
        row = await fetch_one(
            "SELECT MAX(ingested_at) AS m FROM feed_items WHERE source=?",
            (source,),
        )
    else:
        row = await fetch_one("SELECT MAX(ingested_at) AS m FROM feed_items")
    return safe_float(row["m"] if row else 0.0)
