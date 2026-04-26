"""Aggregations over `signal_outcomes` for the CLI report.

Three views produced today:
  - ``source_performance``: one row per distinct source name
  - ``category_performance``: one row per event/lane category
  - ``market_mapper_performance``: one row per market_id

Each aggregator is a pure SQL query — read-only, no INSERTs. The
CLI prints them as tables; the dashboard (future) can read them
directly.

Counting rules (v1):
  - sample_size = number of signal_outcomes rows for the bucket
  - hit_rate = direction_correct=1 / direction_correct in (0,1)
    (rows with NULL direction_correct are EXCLUDED — they didn't
    have enough data to score)
  - avg_5m_move_abs = mean(|max_favorable_move_15m|) using rows
    with non-null price_after_5m as the proxy "did it move"
  - false_positive_rate = whether_market_moved=0 / whether_market_moved
    in (0,1). Rows with NULL are EXCLUDED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.utils.db import fetch_all


@dataclass(frozen=True)
class SourceMetrics:
    source: str
    sample_size: int                   # outcomes where this source contributed
    hit_rate: float | None             # direction_correct success rate
    false_positive_rate: float | None  # rate of "no movement"
    avg_move_5m_abs: float | None      # mean |price_after_5m - snapshot|
    avg_move_15m_abs: float | None
    avg_edge_captured: float | None    # mean realized PnL (USD) on accepted
    avg_edge_missed: float | None      # mean missed move on non-accepted
    accepted_count: int
    observed_count: int
    rejected_count: int


@dataclass(frozen=True)
class CategoryMetrics:
    category: str
    sample_size: int
    hit_rate: float | None
    avg_move_5m_abs: float | None
    avg_move_15m_abs: float | None
    accepted_count: int
    observed_count: int
    rejected_count: int


@dataclass(frozen=True)
class MarketMetrics:
    market_id: str
    sample_size: int
    hit_rate: float | None
    avg_move_5m_abs: float | None
    avg_move_15m_abs: float | None
    accepted_count: int
    rejected_count: int


# ============================================================
# Source performance
# ============================================================


_SOURCE_SQL = """
SELECT
  source,
  COUNT(*) AS sample_size,
  -- hit_rate: direction_correct=1 / direction_correct in (0,1)
  AVG(CASE WHEN direction_correct IS NULL THEN NULL
           ELSE direction_correct END) AS hit_rate,
  -- false_positive_rate: whether_market_moved=0 / not-NULL
  AVG(CASE WHEN whether_market_moved IS NULL THEN NULL
           ELSE 1 - whether_market_moved END) AS false_positive_rate,
  -- avg absolute 5m / 15m move from snapshot
  AVG(CASE
        WHEN price_after_5m IS NOT NULL AND snapshot_price IS NOT NULL
        THEN ABS(price_after_5m - snapshot_price)
        ELSE NULL
      END) AS avg_move_5m_abs,
  AVG(CASE
        WHEN price_after_15m IS NOT NULL AND snapshot_price IS NOT NULL
        THEN ABS(price_after_15m - snapshot_price)
        ELSE NULL
      END) AS avg_move_15m_abs,
  AVG(estimated_edge_captured) AS avg_edge_captured,
  AVG(estimated_edge_missed)   AS avg_edge_missed,
  SUM(CASE WHEN final_status='accepted' THEN 1 ELSE 0 END) AS accepted_count,
  SUM(CASE WHEN final_status='observed' THEN 1 ELSE 0 END) AS observed_count,
  SUM(CASE WHEN final_status='rejected' THEN 1 ELSE 0 END) AS rejected_count
FROM signal_outcomes
GROUP BY source
ORDER BY sample_size DESC
"""


async def source_performance() -> list[SourceMetrics]:
    rows = await fetch_all(_SOURCE_SQL)
    return [
        SourceMetrics(
            source=str(r["source"] or "unknown"),
            sample_size=int(r["sample_size"] or 0),
            hit_rate=_nullable_float(r["hit_rate"]),
            false_positive_rate=_nullable_float(r["false_positive_rate"]),
            avg_move_5m_abs=_nullable_float(r["avg_move_5m_abs"]),
            avg_move_15m_abs=_nullable_float(r["avg_move_15m_abs"]),
            avg_edge_captured=_nullable_float(r["avg_edge_captured"]),
            avg_edge_missed=_nullable_float(r["avg_edge_missed"]),
            accepted_count=int(r["accepted_count"] or 0),
            observed_count=int(r["observed_count"] or 0),
            rejected_count=int(r["rejected_count"] or 0),
        )
        for r in rows
    ]


# ============================================================
# Category performance
# ============================================================


_CATEGORY_SQL = """
SELECT
  category,
  COUNT(*) AS sample_size,
  AVG(CASE WHEN direction_correct IS NULL THEN NULL
           ELSE direction_correct END) AS hit_rate,
  AVG(CASE
        WHEN price_after_5m IS NOT NULL AND snapshot_price IS NOT NULL
        THEN ABS(price_after_5m - snapshot_price)
        ELSE NULL
      END) AS avg_move_5m_abs,
  AVG(CASE
        WHEN price_after_15m IS NOT NULL AND snapshot_price IS NOT NULL
        THEN ABS(price_after_15m - snapshot_price)
        ELSE NULL
      END) AS avg_move_15m_abs,
  SUM(CASE WHEN final_status='accepted' THEN 1 ELSE 0 END) AS accepted_count,
  SUM(CASE WHEN final_status='observed' THEN 1 ELSE 0 END) AS observed_count,
  SUM(CASE WHEN final_status='rejected' THEN 1 ELSE 0 END) AS rejected_count
FROM signal_outcomes
GROUP BY category
ORDER BY sample_size DESC
"""


async def category_performance() -> list[CategoryMetrics]:
    rows = await fetch_all(_CATEGORY_SQL)
    return [
        CategoryMetrics(
            category=str(r["category"] or "unknown"),
            sample_size=int(r["sample_size"] or 0),
            hit_rate=_nullable_float(r["hit_rate"]),
            avg_move_5m_abs=_nullable_float(r["avg_move_5m_abs"]),
            avg_move_15m_abs=_nullable_float(r["avg_move_15m_abs"]),
            accepted_count=int(r["accepted_count"] or 0),
            observed_count=int(r["observed_count"] or 0),
            rejected_count=int(r["rejected_count"] or 0),
        )
        for r in rows
    ]


# ============================================================
# Market-mapper performance
# ============================================================


_MARKET_SQL = """
SELECT
  market_id,
  COUNT(*) AS sample_size,
  AVG(CASE WHEN direction_correct IS NULL THEN NULL
           ELSE direction_correct END) AS hit_rate,
  AVG(CASE
        WHEN price_after_5m IS NOT NULL AND snapshot_price IS NOT NULL
        THEN ABS(price_after_5m - snapshot_price)
        ELSE NULL
      END) AS avg_move_5m_abs,
  AVG(CASE
        WHEN price_after_15m IS NOT NULL AND snapshot_price IS NOT NULL
        THEN ABS(price_after_15m - snapshot_price)
        ELSE NULL
      END) AS avg_move_15m_abs,
  SUM(CASE WHEN final_status='accepted' THEN 1 ELSE 0 END) AS accepted_count,
  SUM(CASE WHEN final_status='rejected' THEN 1 ELSE 0 END) AS rejected_count
FROM signal_outcomes
GROUP BY market_id
ORDER BY sample_size DESC
"""


async def market_mapper_performance(*, limit: int = 50) -> list[MarketMetrics]:
    rows = await fetch_all(_MARKET_SQL + f"\nLIMIT {int(limit)}")
    return [
        MarketMetrics(
            market_id=str(r["market_id"] or ""),
            sample_size=int(r["sample_size"] or 0),
            hit_rate=_nullable_float(r["hit_rate"]),
            avg_move_5m_abs=_nullable_float(r["avg_move_5m_abs"]),
            avg_move_15m_abs=_nullable_float(r["avg_move_15m_abs"]),
            accepted_count=int(r["accepted_count"] or 0),
            rejected_count=int(r["rejected_count"] or 0),
        )
        for r in rows
    ]


# ============================================================
# Targeted reports for the CLI
# ============================================================


async def missed_edge_candidates(*, limit: int = 20) -> list[dict]:
    """Top rejected/observed rows where the market DID move
    favorably afterwards. These are the highest-leverage signal
    sources to investigate — the bot saw the event, declined to
    trade, and the market then moved."""
    rows = await fetch_all(
        """SELECT source, category, market_id, detected_at,
                  snapshot_price, max_favorable_move_15m,
                  max_adverse_move_15m, final_status, reject_reason
           FROM signal_outcomes
           WHERE final_status IN ('rejected', 'observed')
             AND estimated_edge_missed IS NOT NULL
           ORDER BY estimated_edge_missed DESC
           LIMIT ?""",
        (int(limit),),
    )
    return [dict(r) for r in rows]


async def noisy_sources(
    *, min_sample_size: int = 20, min_false_positive_rate: float = 0.6,
) -> list[SourceMetrics]:
    """Sources with high false-positive rate AND enough samples to
    trust the rate. These are candidates for deprioritization."""
    metrics = await source_performance()
    return [
        m for m in metrics
        if m.sample_size >= min_sample_size
        and m.false_positive_rate is not None
        and m.false_positive_rate >= min_false_positive_rate
    ]


async def top_sources_by_5m_move(
    *, min_sample_size: int = 20, limit: int = 10,
) -> list[SourceMetrics]:
    """Sources with the largest average 5m move after they fire
    AND enough samples for the average to mean something."""
    metrics = await source_performance()
    qualifying = [
        m for m in metrics
        if m.sample_size >= min_sample_size
        and m.avg_move_5m_abs is not None
    ]
    qualifying.sort(key=lambda m: m.avg_move_5m_abs or 0.0, reverse=True)
    return qualifying[:limit]


async def top_categories_by_hit_rate(
    *, min_sample_size: int = 20, limit: int = 10,
) -> list[CategoryMetrics]:
    metrics = await category_performance()
    qualifying = [
        m for m in metrics
        if m.sample_size >= min_sample_size
        and m.hit_rate is not None
    ]
    qualifying.sort(key=lambda m: m.hit_rate or 0.0, reverse=True)
    return qualifying[:limit]


# ============================================================
# Helpers
# ============================================================


def _nullable_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
