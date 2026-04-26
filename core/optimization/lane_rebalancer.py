"""Nightly lane rebalancer — compound interest at the portfolio level.

Individual lanes already compound their own PnL: :func:`allocator.release`
adds realized PnL into ``total_budget`` so a winning lane grows its own
base over time. What we add here is *cross-lane* rebalancing — shifting
capital toward the lane that's earning the most right now.

Policy (deliberately conservative — this runs unsupervised at 4 AM):

  1. Reconcile `deployed` from actual open positions first, so a stale
     drift (e.g. from a crashed close path) doesn't poison the decision.
  2. Per-lane ROI over the lookback window
     (``realized_pnl_lookback / starting_budget``). Losing lanes get a
     score of 0 — the rebalancer only *adds* to winners; it doesn't
     take away from losers beyond the soft blend. A bad week
     shouldn't zero a lane out overnight.
  3. Softmax-ish normalization with hard floor + cap so no lane gets
     starved (floor ≥ ``min_budget_pct`` of portfolio) or runs away
     with everything (cap ≤ ``max_budget_pct``).
  4. Blend the target into the current weights at ``blend_rate`` — by
     default 20% per night, so it takes ~5 nights of consistent
     outperformance for a lane to fully claim its target share. Smooths
     out single-day noise.
  5. Re-apply the resulting weights to the *current portfolio total*
     (which already includes compounded PnL), so winners don't lose
     the gains they earned intrinsically — the rebalance is additive
     on top of per-lane compounding.

Safeguard: if total closes across the portfolio in the lookback are
below ``min_sample_size``, we skip entirely. First-day luck shouldn't
reshape the portfolio.

Config (all under ``lane_rebalancer:``):
  enabled: bool  (default True)
  lookback_days: int  (default 7)
  blend_rate: float  (default 0.2)
  min_budget_pct: float  (default 0.10) — floor as fraction of portfolio
  max_budget_pct: float  (default 0.60) — cap as fraction of portfolio
  min_sample_size: int  (default 10) — total closes required to act
  recompute_available: bool (default True)
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from core.execution import allocator
from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.helpers import now_ts


async def rebalance() -> dict[str, Any]:
    """Run one rebalance pass on the current mode's lanes. Returns a
    summary dict useful for logging + tests."""
    cfg = get_config().get("lane_rebalancer") or {}
    if not cfg.get("enabled", True):
        logger.info("[rebalancer] disabled via config")
        return {"skipped": "disabled"}

    lookback_days = int(cfg.get("lookback_days", 7))
    blend = float(cfg.get("blend_rate", 0.2))
    min_pct = float(cfg.get("min_budget_pct", 0.10))
    max_pct = float(cfg.get("max_budget_pct", 0.60))
    min_samples = int(cfg.get("min_sample_size", 10))
    # Per-lane minimum: a lane needs at least this many closes in the
    # window before its ROI counts. Under the threshold the lane is
    # "pinned" — target weight = current weight, no reshuffle from its
    # corner. Prevents a 2-close 100% WR lane from hoovering capital
    # on noise alone.
    min_closes_per_lane = int(cfg.get("min_closes_per_lane", 5))
    blend = max(0.0, min(1.0, blend))

    mode = allocator.current_mode()

    # 1. Reconcile deployed drift before we make decisions.
    drift = await allocator.reconcile_deployed(mode)
    if drift:
        logger.info("[rebalancer] reconciled drift: {}", drift)

    # 2. Gather state.
    states = [await allocator.get_state(l, mode) for l in allocator.LANES]
    states = [s for s in states if s is not None]
    if not states:
        return {"skipped": "no_lane_state"}

    total_portfolio = sum(s.total_budget for s in states)
    if total_portfolio <= 0:
        # Nothing to redistribute. A mode with zero capital (e.g. real
        # mode before the user funds it) sits quietly.
        return {"skipped": "zero_portfolio"}

    # 3. Per-lane realized PnL + close count over the lookback window.
    cutoff = time.time() - lookback_days * 86400
    lane_pnl: dict[str, float] = {}
    lane_closes: dict[str, int] = {}
    for s in states:
        pnl, closes = await _lane_window_stats(s.lane, mode, cutoff)
        lane_pnl[s.lane] = pnl
        lane_closes[s.lane] = closes
    total_closes = sum(lane_closes.values())
    if total_closes < min_samples:
        logger.info(
            "[rebalancer] only {} closes in last {}d (< {}); skipping",
            total_closes, lookback_days, min_samples,
        )
        return {
            "skipped": "insufficient_samples",
            "sample_size": total_closes,
        }

    # 4. Score each lane. Negative PnL → score 0 (no bonus, no penalty
    # beyond the blend shrinkage). Positive PnL → normalized by lane's
    # starting budget so a $5 win on a $10 lane outweighs a $5 win on a
    # $1000 lane — rewards efficient use of capital.
    #
    # Lanes with fewer than ``min_closes_per_lane`` closes are *pinned*:
    # we don't score them, and later we override their target weight to
    # equal their current weight so they stay put. That way a dry lane
    # or a lane with a thin 2-close sample doesn't either lose budget it
    # hasn't had a chance to justify, nor grab budget off a lucky streak.
    current = {s.lane: s.total_budget / total_portfolio for s in states}
    scores: dict[str, float] = {}
    pinned: set[str] = set()
    for s in states:
        if lane_closes[s.lane] < min_closes_per_lane:
            pinned.add(s.lane)
            continue
        pnl = lane_pnl[s.lane]
        base = max(s.total_budget, 1.0)  # avoid div-by-zero on tiny lanes
        scores[s.lane] = max(0.0, pnl / base)

    if not scores:
        # Every lane is under the sample threshold — nothing to judge.
        # Keep current weights, log, return.
        logger.info(
            "[rebalancer] all lanes under min_closes_per_lane={} "
            "(closes={}); skipping",
            min_closes_per_lane, lane_closes,
        )
        return {
            "skipped": "all_lanes_pinned",
            "lane_pnl": lane_pnl,
            "lane_closes": lane_closes,
        }

    total_score = sum(scores.values())
    if total_score <= 0:
        # All judged lanes flat or losing — don't rebalance. Let the
        # existing per-lane compounding continue unmolested. The
        # operator-facing signal here is "no lane is pulling the
        # portfolio up right now".
        logger.info(
            "[rebalancer] no positive-score lanes in last {}d; skipping",
            lookback_days,
        )
        return {
            "skipped": "no_winners",
            "lane_pnl": lane_pnl,
            "lane_closes": lane_closes,
            "pinned": sorted(pinned),
        }

    # 5. Target weights:
    #    - Pinned lanes keep their current share of the portfolio.
    #    - Judged lanes split the remaining share proportionally to score.
    #    Then apply floor + cap to the full weight vector.
    pinned_share = sum(current.get(l, 0.0) for l in pinned)
    free_share = max(0.0, 1.0 - pinned_share)
    raw_target: dict[str, float] = {l: current.get(l, 0.0) for l in pinned}
    for l, sc in scores.items():
        raw_target[l] = (sc / total_score) * free_share
    target = _apply_floor_and_cap(raw_target, min_pct, max_pct)

    # 6. Blend target into current weights (EMA-style).
    blended = {
        l: (1.0 - blend) * current.get(l, 0.0) + blend * target.get(l, 0.0)
        for l in target
    }
    # Re-normalize to kill rounding drift from the blend step.
    total_w = sum(blended.values())
    if total_w <= 0:
        return {"skipped": "degenerate_weights"}
    blended = {l: w / total_w for l, w in blended.items()}

    # 7. Apply — preserving total_portfolio exactly so compound gains
    # aren't lost. set_lane_budget recomputes available per lane, keeping
    # deployed untouched so open positions don't get constrained.
    summary: dict[str, Any] = {
        "mode": mode,
        "total_portfolio": round(total_portfolio, 2),
        "lane_pnl": lane_pnl,
        "lane_closes": lane_closes,
        "pinned": sorted(pinned),
        "before": {l: round(s.total_budget, 2) for l, s in zip(
            (s.lane for s in states), states,
        )},
        "after": {},
    }
    for lane, weight in blended.items():
        new_tb = round(total_portfolio * weight, 2)
        await allocator.set_lane_budget(
            lane, new_tb, mode=mode,
            reason=(
                f"rebalance win={lane_pnl[lane]:+.2f} "
                f"closes={lane_closes[lane]} w={weight:.2f}"
            ),
        )
        summary["after"][lane] = new_tb

    # Record the rebalance as a config_overrides row so the dashboard
    # history panel can surface "why did the lane budgets change?".
    await execute(
        """INSERT INTO config_overrides
           (ts, reason, old_yaml, new_yaml)
           VALUES (?,?,?,?)""",
        (
            now_ts(),
            f"lane_rebalance lookback={lookback_days}d "
            f"closes={total_closes}",
            _format_budgets(summary["before"]),
            _format_budgets(summary["after"]),
        ),
    )

    logger.info(
        "[rebalancer] applied new lane budgets (portfolio={:.2f}): {}",
        total_portfolio,
        {l: f"{summary['before'].get(l, 0.0):.2f}->{v:.2f}"
         for l, v in summary["after"].items()},
    )
    return summary


async def _lane_window_stats(
    lane: str, mode: str, cutoff: float,
) -> tuple[float, int]:
    """Realized PnL + close count for a lane over the cutoff window,
    summed across every strategy that funds from that lane bucket."""
    strategies = {
        s for s, l in allocator._STRATEGY_LANE_MAP.items() if l == lane
    }
    strategies.add(lane)  # the 1:1 default
    placeholders = ",".join("?" * len(strategies))
    is_real = 1 if mode == "real" else 0
    row = await fetch_one(
        f"""SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl,
                   COUNT(*) AS n
            FROM shadow_positions
            WHERE strategy IN ({placeholders})
              AND status='CLOSED'
              AND close_ts >= ?
              AND COALESCE(is_real, 0)=?""",
        (*strategies, cutoff, is_real),
    )
    if row is None:
        return 0.0, 0
    return float(row["pnl"] or 0.0), int(row["n"] or 0)


def _apply_floor_and_cap(
    weights: dict[str, float],
    min_pct: float,
    max_pct: float,
) -> dict[str, float]:
    """Clamp each weight into ``[min_pct, max_pct]``, then renormalize.
    Iterate twice to converge when both caps bite simultaneously —
    two passes is enough for the 3-lane setup we ship with; more
    lanes would call for a proper fixed-point loop.
    """
    if not weights:
        return weights
    n = len(weights)
    # min_pct * n might exceed 1 if caller misconfigures; clamp defensively.
    if min_pct * n > 1.0:
        min_pct = 1.0 / n / 2.0  # "give everyone a sliver" fallback
    if max_pct < min_pct:
        max_pct = min_pct
    w = dict(weights)
    for _ in range(2):
        w = {l: max(min_pct, min(max_pct, v)) for l, v in w.items()}
        total = sum(w.values())
        if total <= 0:
            # All-zero edge case: equal weights.
            return {l: 1.0 / n for l in w}
        w = {l: v / total for l, v in w.items()}
    return w


def _format_budgets(budgets: dict[str, float]) -> str:
    parts = [f"{l}={v:.2f}" for l, v in sorted(budgets.items())]
    return "; ".join(parts)
