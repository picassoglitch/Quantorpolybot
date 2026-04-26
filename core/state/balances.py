"""Bankroll, exposure, and PnL aggregations.

All three lanes (shadow + real) land in `shadow_positions`, so that's
the single source of truth. `is_real=0` is simulated money; `is_real=1`
is actual wallet funds. The dashboard overview can filter or split as
needed; the defaults here return the combined view.
"""

from __future__ import annotations

import time

from core.utils.config import get_config
from core.utils.db import fetch_all, fetch_one


async def total_open_exposure_usd(is_real: int | None = None) -> float:
    """USD tied up in open positions (including pending real fills)."""
    if is_real is None:
        row = await fetch_one(
            """SELECT COALESCE(SUM(size_usd), 0) AS exposure
               FROM shadow_positions
               WHERE status IN ('OPEN','PENDING_FILL')"""
        )
    else:
        row = await fetch_one(
            """SELECT COALESCE(SUM(size_usd), 0) AS exposure
               FROM shadow_positions
               WHERE status IN ('OPEN','PENDING_FILL')
                 AND COALESCE(is_real, 0)=?""",
            (is_real,),
        )
    return float(row["exposure"] if row else 0.0)


async def realized_pnl_usd(is_real: int | None = None) -> float:
    if is_real is None:
        row = await fetch_one(
            """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
               FROM shadow_positions WHERE status='CLOSED'"""
        )
    else:
        row = await fetch_one(
            """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
               FROM shadow_positions
               WHERE status='CLOSED' AND COALESCE(is_real, 0)=?""",
            (is_real,),
        )
    return float(row["pnl"] if row else 0.0)


async def daily_pnl_usd(is_real: int | None = None) -> float:
    cutoff = time.time() - 86400
    if is_real is None:
        row = await fetch_one(
            """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
               FROM shadow_positions
               WHERE status='CLOSED' AND close_ts >= ?""",
            (cutoff,),
        )
    else:
        row = await fetch_one(
            """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
               FROM shadow_positions
               WHERE status='CLOSED' AND close_ts >= ?
                 AND COALESCE(is_real, 0)=?""",
            (cutoff, is_real),
        )
    return float(row["pnl"] if row else 0.0)


async def current_bankroll_usd() -> float:
    """Bankroll available to Kelly sizing. Reads the *current mode's*
    capital pool (shadow_capital.total_usd or real_capital.total_usd)
    plus that mode's realized PnL — so at $15 shadow the risk engine
    sees $15 (+/- PnL), not the $12k exposure ceiling. Using the
    ceiling as bankroll was the bug that made the dashboard show a
    $35 Kelly-approved size on a $15 shadow budget.

    Falls back to ``risk.max_total_exposure_usd`` only if both capital
    pools are misconfigured (both zero), so a broken YAML still boots
    without crashing the risk engine."""
    cfg = get_config()
    mode = str(cfg.get("mode", default="shadow") or "shadow").strip().lower()
    mode = "real" if mode == "real" else "shadow"
    key = "real_capital" if mode == "real" else "shadow_capital"
    pool = cfg.get(key) or {}
    total = float(pool.get("total_usd") or 0.0)
    if total <= 0:
        # Graceful fallback — something in YAML is off, but risk still
        # needs a positive number. Use the exposure ceiling so Kelly
        # doesn't divide by zero.
        total = float(cfg.get("risk", "max_total_exposure_usd", default=500))
    is_real = 1 if mode == "real" else 0
    mode_pnl = await realized_pnl_usd(is_real=is_real)
    return max(0.0, total + mode_pnl)


async def equity_curve_points(
    limit: int = 100,
    is_real: int | None = None,
) -> list[tuple[float, float]]:
    """(ts, cumulative_pnl) pairs for the equity chart."""
    if is_real is None:
        rows = await fetch_all(
            """SELECT COALESCE(close_ts, entry_ts) AS ts, realized_pnl_usd AS pnl
               FROM shadow_positions
               WHERE status='CLOSED'
               ORDER BY ts ASC"""
        )
    else:
        rows = await fetch_all(
            """SELECT COALESCE(close_ts, entry_ts) AS ts, realized_pnl_usd AS pnl
               FROM shadow_positions
               WHERE status='CLOSED' AND COALESCE(is_real, 0)=?
               ORDER BY ts ASC""",
            (is_real,),
        )
    cum = 0.0
    points: list[tuple[float, float]] = []
    for r in rows:
        cum += float(r["pnl"] or 0.0)
        points.append((float(r["ts"] or 0.0), cum))
    return points[-limit:]
