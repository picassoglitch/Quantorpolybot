"""Position lookup for dashboard widgets.

The legacy `positions` table is gone — all fills land in
`shadow_positions` (both shadow and real). This module exposes a thin
adapter so the dashboard and other read-side callers can keep using the
same `Position` dataclass shape they always used.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.utils.db import fetch_all, fetch_one
from core.utils.helpers import safe_float


@dataclass
class Position:
    id: int
    market_id: str
    token_id: str
    side: str
    size: float
    avg_price: float
    opened_at: float
    realized_pnl_usd: float
    status: str
    lane: str = ""
    is_real: bool = False


async def list_open(is_real: int | None = None) -> list[Position]:
    """Open + pending-fill positions across all lanes."""
    if is_real is None:
        rows = await fetch_all(
            """SELECT * FROM shadow_positions
               WHERE status IN ('OPEN','PENDING_FILL')
               ORDER BY entry_ts DESC"""
        )
    else:
        rows = await fetch_all(
            """SELECT * FROM shadow_positions
               WHERE status IN ('OPEN','PENDING_FILL')
                 AND COALESCE(is_real, 0)=?
               ORDER BY entry_ts DESC""",
            (is_real,),
        )
    out: list[Position] = []
    for r in rows:
        try:
            is_real_raw = r["is_real"]
        except (IndexError, KeyError):
            is_real_raw = 0
        out.append(
            Position(
                id=int(r["id"]),
                market_id=r["market_id"] or "",
                token_id=r["token_id"] or "",
                side=r["side"] or "",
                size=safe_float(r["size_shares"]),
                avg_price=safe_float(r["entry_price"]),
                opened_at=safe_float(r["entry_ts"]),
                realized_pnl_usd=safe_float(r["realized_pnl_usd"]),
                status=r["status"] or "",
                lane=r["strategy"] or "",
                is_real=bool(safe_float(is_real_raw)),
            )
        )
    return out


async def open_position_for(market_id: str) -> int:
    """Count of OPEN/PENDING_FILL positions on this market across all lanes."""
    row = await fetch_one(
        """SELECT COUNT(*) AS n FROM shadow_positions
           WHERE market_id=? AND status IN ('OPEN','PENDING_FILL')""",
        (market_id,),
    )
    return int(row["n"] if row else 0)
