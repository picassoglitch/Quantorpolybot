"""Bankroll, exposure, and PnL aggregations.

In dry-run we synthesise a virtual bankroll from `risk.max_total_exposure_usd
+ realized PnL`. In live mode the same number is exposed to the engine but
the user is expected to keep their wallet sufficiently funded — we do not
chain to on-chain balance lookups in this module.
"""

from __future__ import annotations

import time

from core.utils.config import get_config
from core.utils.db import fetch_one


async def total_open_exposure_usd() -> float:
    row = await fetch_one(
        """SELECT COALESCE(SUM(ABS(size * avg_price)), 0) AS exposure
           FROM positions WHERE status='OPEN'"""
    )
    return float(row["exposure"] if row else 0.0)


async def realized_pnl_usd() -> float:
    row = await fetch_one(
        "SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl FROM positions"
    )
    return float(row["pnl"] if row else 0.0)


async def daily_pnl_usd() -> float:
    cutoff = time.time() - 86400
    row = await fetch_one(
        """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
           FROM positions
           WHERE COALESCE(closed_at, opened_at) >= ?""",
        (cutoff,),
    )
    return float(row["pnl"] if row else 0.0)


async def current_bankroll_usd() -> float:
    cfg_max = float(get_config().get("risk", "max_total_exposure_usd", default=500))
    return cfg_max + await realized_pnl_usd()


async def equity_curve_points(limit: int = 100) -> list[tuple[float, float]]:
    """Return (ts, cumulative_pnl) pairs for the dashboard chart."""
    from core.utils.db import fetch_all

    rows = await fetch_all(
        """SELECT COALESCE(closed_at, opened_at) AS ts, realized_pnl_usd AS pnl
           FROM positions
           ORDER BY ts ASC"""
    )
    cum = 0.0
    points: list[tuple[float, float]] = []
    for r in rows:
        cum += float(r["pnl"] or 0.0)
        points.append((float(r["ts"] or 0.0), cum))
    return points[-limit:]
