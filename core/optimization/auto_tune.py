"""Nightly auto-tuner. Runs the grid search, and if the best params beat
the baseline by more than `improvement_threshold` (Sharpe OR PnL), writes
them back to config.yaml and reloads the config in-process.
"""

from __future__ import annotations

import yaml
from loguru import logger

from core.optimization.grid_search import search
from core.utils.config import get_config
from core.utils.db import execute
from core.utils.helpers import now_ts
from core.utils.logging import audit


async def run() -> None:
    cfg_obj = get_config()
    if not cfg_obj.get("optimization", "enabled", default=True):
        logger.info("[auto_tune] disabled via config")
        return

    threshold = float(cfg_obj.get("optimization", "improvement_threshold", default=0.05))
    outcome = await search()
    base = outcome.baseline_result
    best = outcome.best_result

    sharpe_gain = (best.sharpe - base.sharpe) / max(abs(base.sharpe), 1e-6)
    pnl_gain = (best.total_pnl_usd - base.total_pnl_usd) / max(abs(base.total_pnl_usd), 1e-6)

    logger.info(
        "[auto_tune] sharpe_gain={:.2%} pnl_gain={:.2%} (threshold={:.0%})",
        sharpe_gain, pnl_gain, threshold,
    )

    if sharpe_gain < threshold and pnl_gain < threshold:
        logger.info("[auto_tune] no significant improvement; keeping config")
        return

    new_data = cfg_obj.as_dict()
    risk = dict(new_data.get("risk") or {})
    kelly = dict(new_data.get("kelly") or {})
    risk["min_confidence"] = float(outcome.best_params["min_confidence"])
    risk["min_edge"] = float(outcome.best_params["min_edge"])
    kelly["fraction"] = float(outcome.best_params["kelly_fraction"])
    new_data["risk"] = risk
    new_data["kelly"] = kelly

    old_yaml = yaml.safe_dump(cfg_obj.as_dict(), sort_keys=False)
    new_yaml = yaml.safe_dump(new_data, sort_keys=False)

    cfg_obj.save(new_data)
    cfg_obj.reload()

    await execute(
        """INSERT INTO config_overrides
           (ts, reason, old_yaml, new_yaml, sharpe_before, sharpe_after, pnl_before, pnl_after)
           VALUES (?,?,?,?,?,?,?,?)""",
        (now_ts(), "auto_tune", old_yaml, new_yaml, base.sharpe, best.sharpe, base.total_pnl_usd, best.total_pnl_usd),
    )
    audit(
        "config_auto_tuned",
        params=outcome.best_params,
        sharpe_before=base.sharpe,
        sharpe_after=best.sharpe,
        pnl_before=base.total_pnl_usd,
        pnl_after=best.total_pnl_usd,
    )
    logger.info("[auto_tune] config updated and reloaded in-process")
