"""Pattern Discovery Engine PR #1 tests.

Coverage:
  - signal_outcomes: extraction from each source table, source fan-out,
    price_after_N math (with and without ticks), max_favorable / adverse
    over a 15m window, missing-tick graceful handling.
  - aggregations: hit_rate / false_positive_rate / avg_move math.
  - trust_tiers: NEW / WATCH / TRUSTED / LATE_CONFIRMATION / NOISY /
    BLACKLIST classification, minimum sample size guard.
  - CLI smoke: imports + a no-op run on an empty DB.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from core.analytics import aggregations, signal_outcomes
from core.analytics.aggregations import SourceMetrics
from core.analytics.trust_tiers import (
    TIER_BLACKLIST,
    TIER_LATE_CONFIRMATION,
    TIER_NEW,
    TIER_NOISY,
    TIER_TRUSTED,
    TIER_WATCH,
    TrustTierConfig,
    classify_all,
    classify_tier,
)
from core.utils import db as db_module


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "analytics.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


async def _seed_event_market_candidate(
    *,
    event_id: str = "evt-1",
    market_id: str = "m-1",
    considered_at: float | None = None,
    status: str = "rejected",
    reject_reason: str = "polarity_unknown",
    side: str = "",
    sources: list[str] | None = None,
    category: str = "shooting",
    snapshot_price: float = 0.40,
    yes_token: str = "yes-tok-1",
) -> int:
    sources = sources or ["rss:bbc"]
    snap = {
        "event": {
            "category": category,
            "severity": 0.85,
            "confidence": 0.70,
            "source_count": len(sources),
            "sources": sources,
            "first_seen_timestamp": (considered_at or time.time()) - 30,
        },
        "market": {
            "mid": snapshot_price,
            "snapshot_price": snapshot_price,
            "snapshot_taken_at": considered_at or time.time(),
            "yes_token": yes_token,
        },
        "impact": {"polarity_source": "rules"},
    }
    return await db_module.execute(
        """INSERT INTO event_market_candidates
             (event_id, market_id, considered_at, status, reject_reason,
              side, true_prob, confidence, edge, market_mid,
              impact_snapshot, shadow_position_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id, market_id, considered_at or time.time(),
            status, reject_reason, side, 0.55, 0.50, 0.05,
            snapshot_price, json.dumps(snap), None,
        ),
    )


async def _seed_price_ticks(
    *,
    market_id: str = "m-1",
    base_ts: float | None = None,
    series: list[tuple[float, float]] | None = None,
) -> None:
    """`series` is a list of (delta_seconds_from_base, mid_price) pairs."""
    base_ts = base_ts or time.time()
    series = series or [(0.0, 0.40)]
    rows = []
    for dt, mid in series:
        rows.append((market_id, "tok", mid - 0.005, mid + 0.005, mid, base_ts + dt))
    await db_module.executemany(
        "INSERT INTO price_ticks (market_id, token_id, bid, ask, last, ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


# ============================================================
# 1. signal_outcomes — extraction + fan-out
# ============================================================


@pytest.mark.asyncio
async def test_rebuild_extracts_event_market_candidates(temp_db):
    await _seed_event_market_candidate(sources=["rss:bbc"])
    n = await signal_outcomes.rebuild()
    assert n == 1
    rows = await db_module.fetch_all("SELECT * FROM signal_outcomes")
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "rss:bbc"
    assert r["source_table"] == "event_market_candidates"
    assert r["category"] == "shooting"


@pytest.mark.asyncio
async def test_rebuild_fans_out_one_row_per_source(temp_db):
    """Two sources on one event → two rows in signal_outcomes."""
    await _seed_event_market_candidate(
        sources=["rss:bbc", "polymarket_news"],
    )
    n = await signal_outcomes.rebuild()
    assert n == 2
    rows = await db_module.fetch_all(
        "SELECT source FROM signal_outcomes ORDER BY source"
    )
    assert [r["source"] for r in rows] == ["polymarket_news", "rss:bbc"]


@pytest.mark.asyncio
async def test_rebuild_handles_missing_sources_gracefully(temp_db):
    """An event with no source list still produces ONE row (source='unknown')
    so we don't silently drop rejection data."""
    snap = {"event": {"category": "x", "sources": []}, "market": {"mid": 0.5}}
    await db_module.execute(
        """INSERT INTO event_market_candidates
             (event_id, market_id, considered_at, status, reject_reason,
              side, true_prob, confidence, edge, market_mid,
              impact_snapshot, shadow_position_id)
           VALUES ('e', 'm', ?, 'rejected', 'x', '', 0.5, 0.5, 0.0, 0.5, ?, NULL)""",
        (time.time(), json.dumps(snap)),
    )
    n = await signal_outcomes.rebuild()
    assert n == 1
    row = await db_module.fetch_one("SELECT source FROM signal_outcomes")
    assert row["source"] == "unknown"


@pytest.mark.asyncio
async def test_rebuild_idempotent_truncates_existing_rows(temp_db):
    """Calling rebuild twice should leave the table at the second
    rebuild's content, not double-INSERT."""
    await _seed_event_market_candidate(sources=["a"])
    await signal_outcomes.rebuild()
    await signal_outcomes.rebuild()
    rows = await db_module.fetch_all("SELECT id FROM signal_outcomes")
    assert len(rows) == 1


# ============================================================
# 2. price_after_N math + missing tick handling
# ============================================================


@pytest.mark.asyncio
async def test_price_after_N_picks_first_tick_after_horizon(temp_db):
    """Snapshot at t=0, tick stream at t=70s (between 1m and 5m).
    price_after_1m should pick the t=70s tick (first ge), price_after_5m
    should pick the t=310s tick."""
    base = time.time()
    await _seed_event_market_candidate(
        considered_at=base, market_id="m-1",
        snapshot_price=0.40, sources=["rss:bbc"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(70.0, 0.42), (310.0, 0.45), (1000.0, 0.50)],
    )
    await signal_outcomes.rebuild()
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert row["price_after_1m"] == 0.42
    assert row["price_after_5m"] == 0.45
    # 15m horizon = 900s; the 1000s tick is after, so price_after_15m=0.50
    assert row["price_after_15m"] == 0.50
    # 1h horizon = 3600s; no tick exists ≥ 3600s, so price_after_1h=NULL
    assert row["price_after_1h"] is None


@pytest.mark.asyncio
async def test_price_after_N_handles_no_ticks_gracefully(temp_db):
    """Candidate exists but no price_ticks at all → all outcome
    columns NULL, no exceptions."""
    await _seed_event_market_candidate(market_id="ghost-market")
    n = await signal_outcomes.rebuild()
    assert n == 1
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert row["price_after_1m"] is None
    assert row["price_after_5m"] is None
    assert row["max_favorable_move_15m"] is None
    assert row["whether_market_moved"] is None


@pytest.mark.asyncio
async def test_max_favorable_and_adverse_over_window(temp_db):
    """Snapshot at t=0, base 0.40. Ticks: +0.05 at t=200s, -0.03 at
    t=500s. Max favorable = +0.05; max adverse = -0.03."""
    base = time.time()
    await _seed_event_market_candidate(
        considered_at=base, market_id="m-1",
        snapshot_price=0.40, sources=["a"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(200.0, 0.45), (500.0, 0.37)],
    )
    await signal_outcomes.rebuild()
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert abs(row["max_favorable_move_15m"] - 0.05) < 1e-9
    assert abs(row["max_adverse_move_15m"] - (-0.03)) < 1e-9
    assert row["whether_market_moved"] == 1


@pytest.mark.asyncio
async def test_whether_market_moved_zero_under_noise_floor(temp_db):
    """Ticks within the noise floor (~0.5c) shouldn't count as
    movement — protects metrics from being noise-dominated."""
    base = time.time()
    await _seed_event_market_candidate(
        considered_at=base, market_id="m-1",
        snapshot_price=0.40, sources=["a"],
    )
    # Tiny moves: +0.001 / -0.002 over the window, both well under
    # the 0.005 noise floor.
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(120.0, 0.401), (480.0, 0.398)],
    )
    await signal_outcomes.rebuild()
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert row["whether_market_moved"] == 0


@pytest.mark.asyncio
async def test_direction_correct_buy_with_upward_move(temp_db):
    """status=accepted, side=BUY, market moves up → direction_correct=1."""
    base = time.time()
    await _seed_event_market_candidate(
        considered_at=base, market_id="m-1", status="accepted",
        side="BUY", snapshot_price=0.40, sources=["a"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(60.0, 0.42), (300.0, 0.45)],
    )
    await signal_outcomes.rebuild()
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert row["direction_correct"] == 1


@pytest.mark.asyncio
async def test_direction_correct_sell_with_upward_move(temp_db):
    """status=accepted, side=SELL, market moves up → direction_correct=0
    (we bet against the move)."""
    base = time.time()
    await _seed_event_market_candidate(
        considered_at=base, market_id="m-1", status="accepted",
        side="SELL", snapshot_price=0.40, sources=["a"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(60.0, 0.42), (300.0, 0.45)],
    )
    await signal_outcomes.rebuild()
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert row["direction_correct"] == 0


@pytest.mark.asyncio
async def test_direction_correct_null_when_no_side(temp_db):
    """Rejected/observed rows have no side → direction_correct=NULL."""
    base = time.time()
    await _seed_event_market_candidate(
        considered_at=base, market_id="m-1", status="rejected",
        side="", snapshot_price=0.40, sources=["a"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base, series=[(60.0, 0.45)],
    )
    await signal_outcomes.rebuild()
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert row["direction_correct"] is None


@pytest.mark.asyncio
async def test_estimated_edge_missed_populated_for_rejected(temp_db):
    """For non-accepted rows that move favorably, estimated_edge_missed
    is the magnitude of the move."""
    base = time.time()
    await _seed_event_market_candidate(
        considered_at=base, status="rejected",
        snapshot_price=0.40, sources=["a"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(60.0, 0.45), (300.0, 0.48)],  # +0.08 max
    )
    await signal_outcomes.rebuild()
    row = await db_module.fetch_one("SELECT * FROM signal_outcomes")
    assert row["estimated_edge_missed"] is not None
    assert row["estimated_edge_missed"] > 0.07


# ============================================================
# 3. Aggregations
# ============================================================


@pytest.mark.asyncio
async def test_source_performance_aggregates_by_distinct_source(temp_db):
    """Two events: one with source X (correct direction), one with
    source X (wrong direction). source_performance should show
    sample_size=2, hit_rate=0.5."""
    base = time.time()
    # Correct direction
    await _seed_event_market_candidate(
        event_id="e1", market_id="m-1", considered_at=base,
        status="accepted", side="BUY", snapshot_price=0.40,
        sources=["src-X"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(60.0, 0.42), (300.0, 0.45)],
    )
    # Wrong direction (BUY but price drops)
    await _seed_event_market_candidate(
        event_id="e2", market_id="m-2", considered_at=base + 1,
        status="accepted", side="BUY", snapshot_price=0.40,
        sources=["src-X"],
    )
    await _seed_price_ticks(
        market_id="m-2", base_ts=base + 1,
        series=[(60.0, 0.38), (300.0, 0.35)],
    )
    await signal_outcomes.rebuild()
    metrics = await aggregations.source_performance()
    src_x = next(m for m in metrics if m.source == "src-X")
    assert src_x.sample_size == 2
    assert src_x.hit_rate == 0.5
    assert src_x.accepted_count == 2
    assert src_x.rejected_count == 0


@pytest.mark.asyncio
async def test_missed_edge_candidates_lists_top_misses(temp_db):
    base = time.time()
    await _seed_event_market_candidate(
        event_id="e-miss", market_id="m-1", considered_at=base,
        status="rejected", reject_reason="corroboration",
        snapshot_price=0.40, sources=["src-Y"],
    )
    await _seed_price_ticks(
        market_id="m-1", base_ts=base,
        series=[(60.0, 0.50), (300.0, 0.55)],  # huge favorable move
    )
    await signal_outcomes.rebuild()
    rows = await aggregations.missed_edge_candidates(limit=10)
    assert len(rows) == 1
    assert rows[0]["source"] == "src-Y"


# ============================================================
# 4. Trust tiers
# ============================================================


def _metrics(
    *, source="src", sample_size=100, hit_rate=0.7, fp_rate=0.2,
    avg_5m=0.01, avg_15m=0.012, accepted=10, observed=20, rejected=70,
) -> SourceMetrics:
    return SourceMetrics(
        source=source, sample_size=sample_size,
        hit_rate=hit_rate, false_positive_rate=fp_rate,
        avg_move_5m_abs=avg_5m, avg_move_15m_abs=avg_15m,
        avg_edge_captured=None, avg_edge_missed=None,
        accepted_count=accepted, observed_count=observed,
        rejected_count=rejected,
    )


def test_tier_new_when_sample_too_small():
    m = _metrics(sample_size=5)
    a = classify_tier(m, TrustTierConfig(min_sample_new=20))
    assert a.tier == TIER_NEW


def test_tier_blacklist_when_low_hit_high_fp_large_sample():
    m = _metrics(sample_size=200, hit_rate=0.20, fp_rate=0.80)
    a = classify_tier(m)
    assert a.tier == TIER_BLACKLIST


def test_tier_noisy_when_high_fp_but_not_blacklist():
    m = _metrics(sample_size=80, hit_rate=0.55, fp_rate=0.65)
    a = classify_tier(m)
    assert a.tier == TIER_NOISY


def test_tier_late_confirmation_when_15m_swamps_5m():
    """avg_15m / avg_5m >> 1 → most of the move had happened by minute 5
    so the signal is trailing the move."""
    m = _metrics(
        sample_size=100, hit_rate=0.55, fp_rate=0.30,
        avg_5m=0.002, avg_15m=0.020,  # ratio = 10x
    )
    a = classify_tier(m)
    assert a.tier == TIER_LATE_CONFIRMATION


def test_tier_trusted_when_high_hit_real_move_large_sample():
    m = _metrics(
        sample_size=80, hit_rate=0.72, fp_rate=0.20,
        avg_5m=0.012, avg_15m=0.014,
    )
    a = classify_tier(m)
    assert a.tier == TIER_TRUSTED


def test_tier_watch_when_scoreable_but_unproven():
    m = _metrics(
        sample_size=30, hit_rate=0.55, fp_rate=0.35,
        avg_5m=0.006, avg_15m=0.008,
    )
    a = classify_tier(m)
    assert a.tier == TIER_WATCH


def test_classify_all_orders_by_severity():
    metrics = [
        _metrics(source="trusted", sample_size=80, hit_rate=0.72,
                 fp_rate=0.2, avg_5m=0.012, avg_15m=0.014),
        _metrics(source="bad", sample_size=200, hit_rate=0.20, fp_rate=0.80),
        _metrics(source="new", sample_size=5),
    ]
    out = classify_all(metrics)
    # BLACKLIST first, NEW last (severity ordering pinned).
    assert out[0].tier == TIER_BLACKLIST
    assert out[-1].tier == TIER_NEW


def test_min_sample_size_guard_in_blacklist():
    """Even with awful hit_rate / fp_rate, BLACKLIST shouldn't fire
    if sample_size is below the trusted-threshold floor — we don't
    blacklist on a handful of bad events."""
    m = _metrics(sample_size=25, hit_rate=0.20, fp_rate=0.80)
    cfg = TrustTierConfig(min_sample_trusted=50)
    a = classify_tier(m, cfg)
    assert a.tier != TIER_BLACKLIST


# ============================================================
# 5. CLI smoke
# ============================================================


def test_cli_module_imports_cleanly():
    """The CLI script must be importable without bot side effects."""
    import scripts.analyze_patterns  # noqa: F401


@pytest.mark.asyncio
async def test_rebuild_on_empty_db_returns_zero(temp_db):
    """Empty DB shouldn't error — just produce 0 rows."""
    n = await signal_outcomes.rebuild()
    assert n == 0
    rows = await db_module.fetch_all("SELECT * FROM signal_outcomes")
    assert rows == []


@pytest.mark.asyncio
async def test_aggregations_on_empty_table_return_empty_lists(temp_db):
    assert await aggregations.source_performance() == []
    assert await aggregations.category_performance() == []
    assert await aggregations.market_mapper_performance() == []
    assert await aggregations.missed_edge_candidates() == []
    assert await aggregations.noisy_sources() == []
