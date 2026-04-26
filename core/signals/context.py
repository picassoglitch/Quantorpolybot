"""Market context enrichment for scoring prompts.

The bare prompt passes `(market.question, news_text, market.mid)`. That
leaves the model blind to: how the price has moved, how much fresh news
exists on this market, what prior model calls concluded. This module
assembles a compact JSON-able context dict from local DB tables only —
no network — so it's cheap enough to run on every scoring call.

Everything here is best-effort: a missing table, no recent ticks, or
zero peer signals all return gracefully empty fields rather than
raising. The prompt template degrades naturally.
"""

from __future__ import annotations

from typing import Any

from core.utils.db import fetch_all, fetch_one
from core.utils.helpers import now_ts, safe_float


async def _price_trajectory(market_id: str) -> dict[str, Any]:
    """Return current mid + delta vs 1h/6h/24h ago from price_ticks.

    Uses a single query window per lookback bucket so we pick the
    *closest* tick to the target timestamp rather than the newest one
    before it — cheap and good enough.
    """
    now = now_ts()
    windows = {"1h": 3600, "6h": 21600, "24h": 86400}
    out: dict[str, Any] = {}

    latest = await fetch_one(
        """SELECT bid, ask, last, ts
             FROM price_ticks
            WHERE market_id=?
            ORDER BY ts DESC LIMIT 1""",
        (market_id,),
    )
    if not latest:
        return {"note": "no recent ticks"}

    def _mid(row: Any) -> float:
        bid = safe_float(row["bid"])
        ask = safe_float(row["ask"])
        if bid and ask:
            return (bid + ask) / 2.0
        return safe_float(row["last"])

    cur = _mid(latest)
    out["current_mid"] = round(cur, 4)
    out["tick_age_s"] = int(now - safe_float(latest["ts"]))

    for label, span in windows.items():
        target = now - span
        row = await fetch_one(
            """SELECT bid, ask, last, ts
                 FROM price_ticks
                WHERE market_id=? AND ts <= ?
                ORDER BY ts DESC LIMIT 1""",
            (market_id, target),
        )
        if not row:
            continue
        prev = _mid(row)
        if prev > 0:
            out[f"delta_{label}"] = round(cur - prev, 4)
            out[f"pct_{label}"] = round((cur - prev) / prev * 100.0, 2)
    return out


async def _recent_news_count(market_id: str, hours: int = 6) -> int:
    """How many feed_items have been linked to this market in the last
    N hours. Proxy for 'this market is actively in the news'."""
    since = now_ts() - hours * 3600
    row = await fetch_one(
        """SELECT COUNT(*) AS n
             FROM feed_items
            WHERE meta LIKE ? AND ingested_at >= ?""",
        (f'%"linked_market_id":%"{market_id}"%', since),
    )
    return int(safe_float(row["n"]) if row else 0)


async def _peer_signals(market_id: str, limit: int = 3) -> list[dict[str, Any]]:
    """Last N prior scoring verdicts on this same market. Helps the
    model see if it's been hedging (all 0.5s) or previously took a
    strong stance that the current evidence should either confirm or
    override."""
    rows = await fetch_all(
        """SELECT implied_prob, confidence, side, created_at
             FROM signals
            WHERE market_id=?
            ORDER BY created_at DESC LIMIT ?""",
        (market_id, limit),
    )
    now = now_ts()
    return [
        {
            "prob": round(safe_float(r["implied_prob"]), 3),
            "conf": round(safe_float(r["confidence"]), 2),
            "side": r["side"] or "",
            "age_min": int((now - safe_float(r["created_at"])) / 60.0),
        }
        for r in rows
    ]


async def build_market_context(market_id: str) -> dict[str, Any]:
    """Assemble local-DB context for a scoring call. All fields
    degrade to empty/zero on missing data — never raises."""
    try:
        price = await _price_trajectory(market_id)
    except Exception:
        price = {}
    try:
        news_count = await _recent_news_count(market_id)
    except Exception:
        news_count = 0
    try:
        peers = await _peer_signals(market_id)
    except Exception:
        peers = []
    return {
        "price": price,
        "news_in_last_6h": news_count,
        "prior_model_calls": peers,
    }
