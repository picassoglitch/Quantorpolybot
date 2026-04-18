"""Lightweight historical backtester.

Replays signals against historical price_ticks for a given parameter
set, producing a (sharpe, total_pnl, n_trades) tuple. Used by both the
optimization grid search and the prompt evaluator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.utils.db import fetch_all


@dataclass
class BacktestResult:
    sharpe: float
    total_pnl_usd: float
    n_trades: int
    win_rate: float


@dataclass
class BacktestParams:
    min_confidence: float
    min_edge: float
    kelly_fraction: float
    max_position_usd: float = 50.0
    edge_pricing_factor: float = 0.5


async def _signals_window(days: int) -> list[dict]:
    cutoff = time.time() - days * 86400
    rows = await fetch_all(
        """SELECT s.id, s.market_id, s.implied_prob, s.confidence, s.edge,
                  s.mid_price, s.side, s.created_at
           FROM signals s
           WHERE s.created_at >= ? AND s.implied_prob IS NOT NULL
           ORDER BY s.created_at ASC""",
        (cutoff,),
    )
    return [dict(r) for r in rows]


async def _resolution_price(market_id: str, after_ts: float) -> float | None:
    """Use the latest known tick after the signal timestamp as a stand-in
    for resolution. Good enough for relative parameter ranking.
    """
    rows = await fetch_all(
        """SELECT last, ts FROM price_ticks
           WHERE market_id=? AND ts >= ?
           ORDER BY ts DESC LIMIT 1""",
        (market_id, after_ts),
    )
    if not rows:
        return None
    return float(rows[0]["last"] or 0.0) or None


async def run_backtest(params: BacktestParams, days: int = 30) -> BacktestResult:
    signals = await _signals_window(days)
    pnls: list[float] = []
    wins = 0
    for s in signals:
        conf = float(s["confidence"] or 0)
        edge = float(s["edge"] or 0)
        if conf < params.min_confidence or abs(edge) < params.min_edge:
            continue
        mid = float(s["mid_price"] or 0)
        side = s["side"]
        if mid <= 0 or mid >= 1:
            continue
        target = mid + (edge * params.edge_pricing_factor) * (1 if side == "BUY" else -1)
        target = max(0.01, min(0.99, target))
        size_usd = min(
            params.max_position_usd,
            params.kelly_fraction * 100.0,  # synthesised stake
        )
        if size_usd <= 0:
            continue
        resolution = await _resolution_price(s["market_id"], float(s["created_at"]))
        if resolution is None:
            continue
        # crude PnL: long if BUY, short if SELL
        delta = (resolution - target) if side == "BUY" else (target - resolution)
        pnl = delta * (size_usd / target)
        pnls.append(pnl)
        if pnl > 0:
            wins += 1

    if not pnls:
        return BacktestResult(0.0, 0.0, 0, 0.0)
    arr = np.asarray(pnls, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = (mean / std) * math.sqrt(252) if std > 0 else 0.0
    return BacktestResult(
        sharpe=sharpe,
        total_pnl_usd=float(arr.sum()),
        n_trades=len(pnls),
        win_rate=wins / len(pnls),
    )


def iter_grid(grid: dict[str, list]) -> Iterable[dict]:
    keys = list(grid.keys())
    if not keys:
        yield {}
        return
    def _walk(idx: int, current: dict) -> Iterable[dict]:
        if idx == len(keys):
            yield dict(current)
            return
        k = keys[idx]
        for v in grid[k]:
            current[k] = v
            yield from _walk(idx + 1, current)
    yield from _walk(0, {})
