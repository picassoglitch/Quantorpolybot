"""Per-market trade cooldown.

Looks at the last order timestamp for a market and returns True if the
cooldown window hasn't elapsed yet.
"""

from __future__ import annotations

from core.utils.db import fetch_one
from core.utils.helpers import now_ts, safe_float


async def market_in_cooldown(market_id: str, cooldown_seconds: float) -> bool:
    if cooldown_seconds <= 0:
        return False
    row = await fetch_one(
        "SELECT MAX(created_at) AS last FROM orders WHERE market_id=?",
        (market_id,),
    )
    if row is None or row["last"] is None:
        return False
    last = safe_float(row["last"])
    return (now_ts() - last) < cooldown_seconds
