"""Per-source trust calibration.

Nightly job that asks one question of every feed source we've ingested:
*when this source tagged a market that we later took a position on,
did those positions tend to make money?*

The output is a ``trust_weight`` per source that the hallucination
guard uses to weight "distinct sources" by quality rather than count.
Three Reddit posts from r/politics shouldn't unlock the same guard
threshold that one Reuters wire does — source_stats encodes that.

Algorithm (deliberately simple, so it stays interpretable):

  1. Pull every ``shadow_positions`` row closed in the last N days
     (default 30). Skip unresolved positions — they have no label yet.
  2. For each position, find feed_items whose ``meta.linked_market_id``
     matches the position's market_id AND whose ``ingested_at`` falls
     inside the 24h before ``entry_ts``. These are the sources the
     pipeline could plausibly have acted on.
  3. Attribute the position's realised PnL (or win/loss label) equally
     across those sources. One win split across 3 sources = 1/3 credit
     each. This is naive but avoids giving any single source full
     credit for a multi-source signal.
  4. Aggregate per source: samples, wins, losses, weighted_pnl.
  5. Convert to a trust_weight in roughly [0.3, 1.5] — we never zero
     out a source (a bad streak shouldn't permanently silence a
     feed), and we cap the upside so one lucky source can't
     single-handedly unlock the hallucination guard.

The table is rewritten atomically each run — source_stats is a
derived view, not a log.

Readers (``core.signals.pipeline._count_recent_sources``) lazily load
the weights into a module-level cache; ``invalidate_cache()`` is
called at the end of each calibrate() so the next pipeline read picks
up the fresh weights without a restart.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_all
from core.utils.helpers import now_ts

# ---- Weight shape --------------------------------------------------
# trust_weight = clip(BASE + SLOPE * weighted_pnl_per_sample,
#                     MIN_WEIGHT, MAX_WEIGHT)
#
# We center at 1.0 so an uncalibrated source (no samples) contributes
# exactly like it did pre-calibration. A source with strong positive
# PnL gets up to MAX_WEIGHT; a consistently bad source floors at
# MIN_WEIGHT. Tight bounds are intentional — this is a nudge, not a
# veto. The nightly cron also runs source_trust.calibrate() so a bad
# streak corrects back toward 1.0 within a few days.
_BASE_WEIGHT = 1.0
_SLOPE = 2.0
_MIN_WEIGHT = 0.3
_MAX_WEIGHT = 1.5
_MIN_SAMPLES_FOR_CALIBRATION = 3

# Module-level cache of source -> trust_weight. Populated lazily from
# the source_stats table on first read, invalidated after each
# calibrate() run. None sentinel means "not yet loaded"; an empty
# dict means "loaded, nothing stored yet" (first calibrate hasn't
# run or no positions have closed).
_WEIGHT_CACHE: dict[str, float] | None = None


async def get_source_weights() -> dict[str, float]:
    """Return the current source->trust_weight map.

    Loads lazily from ``source_stats`` on first call. Callers should
    use the returned dict with a fallback of 1.0 for missing sources —
    see ``trust_weight_for()``.
    """
    global _WEIGHT_CACHE
    if _WEIGHT_CACHE is not None:
        return _WEIGHT_CACHE
    rows = await fetch_all(
        "SELECT source, trust_weight FROM source_stats"
    )
    cache: dict[str, float] = {}
    for r in rows:
        src = (r["source"] or "").strip()
        if not src:
            continue
        w = float(r["trust_weight"] or _BASE_WEIGHT)
        cache[src] = max(_MIN_WEIGHT, min(_MAX_WEIGHT, w))
    _WEIGHT_CACHE = cache
    return _WEIGHT_CACHE


def invalidate_cache() -> None:
    """Drop the in-memory weights cache. Called after calibrate() so
    the pipeline reads fresh weights on its next evaluation."""
    global _WEIGHT_CACHE
    _WEIGHT_CACHE = None


def trust_weight_for(source: str, weights: dict[str, float]) -> float:
    """Fallback-safe lookup. Unknown source -> ``_BASE_WEIGHT`` so
    sources not yet seen by the calibrator contribute as before."""
    if not source:
        return _BASE_WEIGHT
    return weights.get(source, _BASE_WEIGHT)


async def calibrate() -> None:
    """Rebuild ``source_stats`` from recent closed positions.

    Safe to run repeatedly — each call REPLACES the rows. If
    ``learning.source_trust_enabled`` is false in config we skip,
    leaving whatever weights exist in place (explicit escape hatch
    for users who don't trust the calibration yet)."""
    cfg = get_config()
    if not cfg.get("learning", "source_trust_enabled", default=True):
        logger.info("[source_trust] disabled via config")
        return

    lookback_days = int(cfg.get("learning", "source_trust_lookback_days", default=30))
    cutoff = time.time() - lookback_days * 86400

    positions = await fetch_all(
        """SELECT id, market_id, entry_ts, realized_pnl_usd, size_usd, status
           FROM shadow_positions
           WHERE status='CLOSED' AND close_ts >= ? AND entry_ts IS NOT NULL""",
        (cutoff,),
    )
    if not positions:
        logger.info("[source_trust] no closed positions in lookback; skipping")
        return

    # source -> [samples, wins, losses, weighted_pnl]
    agg: dict[str, list[float]] = {}

    for pos in positions:
        market_id = str(pos["market_id"] or "")
        entry_ts = float(pos["entry_ts"] or 0.0)
        pnl = float(pos["realized_pnl_usd"] or 0.0)
        size = float(pos["size_usd"] or 0.0)
        if not market_id or entry_ts <= 0 or size <= 0:
            continue
        # 24h window before entry — if the bot took the trade on
        # evidence older than a day, it's a stretch to credit that
        # evidence for the outcome.
        window_start = entry_ts - 86400.0
        sources = await _sources_for_market_in_window(
            market_id, window_start, entry_ts,
        )
        if not sources:
            continue
        # Attribute equally across sources present in the window.
        # pnl_pct in [-1, +1]ish (clipped) is the normalized signal
        # the weight converter later consumes. Using absolute PnL
        # would let one jackpot dominate.
        pnl_pct = _clip(pnl / size, -1.0, 1.0)
        share = 1.0 / len(sources)
        for src in sources:
            bucket = agg.setdefault(src, [0.0, 0.0, 0.0, 0.0])
            bucket[0] += share                # samples (fractional)
            if pnl > 0:
                bucket[1] += share
            elif pnl < 0:
                bucket[2] += share
            bucket[3] += pnl_pct * share      # weighted_pnl

    if not agg:
        logger.info("[source_trust] no source<->position links found; skipping")
        return

    rows = []
    now = now_ts()
    for src, (samples, wins, losses, wpnl) in agg.items():
        if samples < _MIN_SAMPLES_FOR_CALIBRATION:
            # Not enough evidence — record the sample count but keep
            # weight at baseline so a single lucky (or unlucky) trade
            # doesn't swing the guard.
            weight = _BASE_WEIGHT
        else:
            per_sample = wpnl / samples if samples > 0 else 0.0
            weight = _BASE_WEIGHT + _SLOPE * per_sample
            weight = max(_MIN_WEIGHT, min(_MAX_WEIGHT, weight))
        win_rate = (wins / samples) if samples > 0 else 0.0
        rows.append(
            (src, int(round(samples)), int(round(wins)), int(round(losses)),
             round(win_rate, 4), round(wpnl, 4), round(weight, 4), now),
        )

    # Rewrite atomically: DELETE then INSERT in a single transaction.
    # REPLACE-per-row would leave deleted sources as stale rows.
    await execute("DELETE FROM source_stats")
    for row in rows:
        await execute(
            """INSERT INTO source_stats
               (source, samples, wins, losses, win_rate, weighted_pnl,
                trust_weight, last_computed_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            row,
        )

    invalidate_cache()
    logger.info(
        "[source_trust] recalibrated {} sources from {} positions "
        "(top={}, bottom={})",
        len(rows), len(positions),
        _extreme(rows, top=True),
        _extreme(rows, top=False),
    )


async def _sources_for_market_in_window(
    market_id: str, window_start: float, window_end: float,
) -> set[str]:
    """Distinct feed sources that tagged this market in the given
    pre-entry time window. Uses the same meta-LIKE predicate the
    pipeline's hallucination guard uses so the two stay in lockstep."""
    rows = await fetch_all(
        """SELECT DISTINCT source FROM feed_items
           WHERE meta LIKE ? AND ingested_at >= ? AND ingested_at <= ?""",
        (f'%"linked_market_id":%"{market_id}"%', window_start, window_end),
    )
    return {str(r["source"]) for r in rows if r["source"]}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _extreme(rows: list[tuple[Any, ...]], *, top: bool) -> str:
    """Return "source=weight" for the highest- or lowest-weight entry,
    used in the summary log line. Empty string when no rows."""
    if not rows:
        return ""
    # row layout: (source, samples, wins, losses, win_rate, weighted_pnl,
    #              trust_weight, last_computed_at)
    key = lambda r: r[6]  # trust_weight
    pick = max(rows, key=key) if top else min(rows, key=key)
    return f"{pick[0]}={pick[6]:.2f}"
