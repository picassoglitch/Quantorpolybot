"""Grid search over the configured parameter grid using the backtester."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from core.backtest.engine import BacktestParams, BacktestResult, iter_grid, run_backtest
from core.utils.config import get_config


@dataclass
class GridOutcome:
    best_params: dict
    best_result: BacktestResult
    baseline_result: BacktestResult


async def search() -> GridOutcome:
    cfg = get_config()
    grid = (cfg.get("optimization", "param_grid") or {})
    days = int(cfg.get("scheduler", "backtest_window_days", default=30))
    risk_cfg = cfg.get("risk") or {}
    kelly_cfg = cfg.get("kelly") or {}
    exec_cfg = cfg.get("execution") or {}

    baseline_params = BacktestParams(
        min_confidence=float(risk_cfg.get("min_confidence", 0.65)),
        min_edge=float(risk_cfg.get("min_edge", 0.04)),
        kelly_fraction=float(kelly_cfg.get("fraction", 0.25)),
        max_position_usd=float(risk_cfg.get("max_position_usd", 50)),
        edge_pricing_factor=float(exec_cfg.get("edge_pricing_factor", 0.5)),
    )
    baseline = await run_backtest(baseline_params, days=days)
    logger.info(
        "[opt] baseline sharpe={:.3f} pnl={:.2f} n={} win={:.1%}",
        baseline.sharpe, baseline.total_pnl_usd, baseline.n_trades, baseline.win_rate,
    )

    best_params = baseline_params
    best = baseline
    for combo in iter_grid(grid):
        params = BacktestParams(
            min_confidence=float(combo.get("min_confidence", baseline_params.min_confidence)),
            min_edge=float(combo.get("min_edge", baseline_params.min_edge)),
            kelly_fraction=float(combo.get("kelly_fraction", baseline_params.kelly_fraction)),
            max_position_usd=baseline_params.max_position_usd,
            edge_pricing_factor=baseline_params.edge_pricing_factor,
        )
        result = await run_backtest(params, days=days)
        if _is_better(result, best):
            best = result
            best_params = params
            logger.info(
                "[opt] new best {}: sharpe={:.3f} pnl={:.2f} n={}",
                combo, result.sharpe, result.total_pnl_usd, result.n_trades,
            )
    return GridOutcome(
        best_params=best_params.__dict__,
        best_result=best,
        baseline_result=baseline,
    )


def _is_better(candidate, baseline) -> bool:
    if candidate.n_trades < 5:
        return False
    if candidate.sharpe > baseline.sharpe + 0.01:
        return True
    if abs(candidate.sharpe - baseline.sharpe) < 0.01 and candidate.total_pnl_usd > baseline.total_pnl_usd:
        return True
    return False
