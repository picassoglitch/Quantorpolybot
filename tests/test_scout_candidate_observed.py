"""Candidate evaluator — observed-mode (PR #7).

Round-trips ImpactScore(observed=True) through `evaluate()` to verify:
  - status="observed" is persisted (not "rejected")
  - audit row has empty side / no shadow_position_id
  - decision.observed=True flag flows through
  - safety gates that fire BEFORE observed-mode (corroboration,
    age, spread) still reject correctly even with observed=True
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.markets.cache import Market
from core.scout import candidate as scout_candidate
from core.scout.event import Event, EventCategory
from core.scout.impact import ImpactScore
from core.scout.mapper import MarketMatch
from core.utils import db as db_module


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "scout_observed.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _market(question: str = "Will war end by Q3?", mid: float = 0.40) -> Market:
    return Market(
        market_id="m-x", question=question, slug="m",
        category="politics", active=True, close_time="",
        token_ids=["yes", "no"],
        best_bid=mid - 0.01, best_ask=mid + 0.01, last_price=mid,
        liquidity=25_000.0, updated_at=time.time(),
    )


def _event(
    category: EventCategory = EventCategory.CEASEFIRE,
    age_seconds: float = 60.0,
    sources: list[str] | None = None,
) -> Event:
    now = time.time()
    sources = sources or ["src-a", "src-b"]
    return Event(
        event_id="evt-obs", timestamp_detected=now - age_seconds,
        title="High-severity event", category=category,
        severity=0.85, confidence=0.70, location="",
        entities=["X"], source_count=len(sources), sources=sources,
        contradiction_score=0.0, raw_signal_ids=[1, 2],
        first_seen_at=now - age_seconds, last_seen_at=now,
    )


def _match(market: Market, score: float = 0.5) -> MarketMatch:
    return MarketMatch(
        market=market, score=score,
        entity_overlap=score, keyword_overlap=score,
        category_alignment=1.0, liquidity_quality=0.8,
        near_resolution_bonus=0.5,
    )


def _observed_impact(true_prob: float = 0.40) -> ImpactScore:
    """Mock the impact scorer's observed-mode output."""
    return ImpactScore(
        direction=0,
        true_prob=true_prob,
        confidence=0.20,
        expected_nudge=0.0,
        polarity_reasoning="no polarity rule matched",
        components={"category_nudge": 0.05, "match_score": 0.5},
        observed=True,
    )


# ============================================================
# Observed-mode persistence
# ============================================================


@pytest.mark.asyncio
async def test_observed_writes_audit_row_with_status_observed(temp_db):
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), _observed_impact(),
    )
    assert decision.observed is True
    assert decision.accepted is False  # NOT a trade
    assert decision.side == ""
    assert decision.edge == 0.0
    assert "observed" in decision.reason

    rows = await db_module.fetch_all(
        "SELECT * FROM event_market_candidates WHERE event_id=?",
        ("evt-obs",),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "observed"
    assert row["reject_reason"] == ""
    assert row["side"] == ""
    # No position attached.
    assert row["shadow_position_id"] is None


@pytest.mark.asyncio
async def test_observed_safety_gates_still_apply(temp_db):
    """An event TOO OLD to trade must reject as 'rejected' even
    when impact.observed=True. We don't accidentally bypass safety
    by going through the observed path."""
    old_event = _event(age_seconds=10_000.0)
    decision = await scout_candidate.evaluate(
        old_event, _match(_market()), _observed_impact(),
        cfg={"max_event_age_seconds": 1800.0},
    )
    assert decision.observed is False
    assert decision.accepted is False
    assert "too old" in decision.reason
    row = await db_module.fetch_one(
        "SELECT status FROM event_market_candidates WHERE event_id=?",
        ("evt-obs",),
    )
    assert row["status"] == "rejected"


@pytest.mark.asyncio
async def test_observed_with_no_corroboration_rejects(temp_db):
    """Single non-primary source → reject (corroboration check
    runs before observed-mode short-circuit)."""
    one_source_event = _event(sources=["someblog"])
    decision = await scout_candidate.evaluate(
        one_source_event, _match(_market()), _observed_impact(),
    )
    assert decision.observed is False
    assert decision.accepted is False
    assert "corroboration" in decision.reason


@pytest.mark.asyncio
async def test_observed_with_event_cooldown_rejects(temp_db):
    """If an accepted candidate already exists for this event_id,
    a follow-up observed candidate must respect the cooldown
    (same as a real trade would)."""
    market_a = _market(); market_a.market_id = "m-a"
    market_b = _market(); market_b.market_id = "m-b"

    # Manually pre-insert an accepted row to simulate a prior trade.
    from core.utils.helpers import now_ts
    await db_module.execute(
        """INSERT INTO event_market_candidates
             (event_id, market_id, considered_at, status, reject_reason,
              side, true_prob, confidence, edge, market_mid,
              impact_snapshot, shadow_position_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("evt-obs", "m-a", now_ts(), "accepted", "",
         "BUY", 0.55, 0.50, 0.05, 0.50, "{}", 1),
    )

    decision = await scout_candidate.evaluate(
        _event(), _match(market_b), _observed_impact(),
        cfg={"event_max_positions": 1, "max_event_age_seconds": 1800.0,
             "min_confidence": 0.40, "min_edge": 0.05,
             "require_primary_or_corroboration": False, "min_sources": 1},
    )
    assert decision.accepted is False
    assert decision.observed is False
    assert "cooldown" in decision.reason


# ============================================================
# Existing accepted/rejected paths still work (regression)
# ============================================================


@pytest.mark.asyncio
async def test_accepted_path_unchanged_by_observed_branch(temp_db):
    """A normal directional ImpactScore must still reach the
    accepted path. Observed-branch must NOT swallow it."""
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="rule matched",
        components={}, observed=False,
    )
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), impact,
    )
    assert decision.accepted is True
    assert decision.observed is False
    assert decision.side == "BUY"
    row = await db_module.fetch_one(
        "SELECT status FROM event_market_candidates WHERE event_id=?",
        ("evt-obs",),
    )
    assert row["status"] == "accepted"


@pytest.mark.asyncio
async def test_polarity_unknown_without_observed_still_rejects(temp_db):
    """direction=0 + observed=False + confidence=0 → reject as
    polarity_unknown (the v1 path, unchanged)."""
    impact = ImpactScore(
        direction=0, true_prob=0.50, confidence=0.0,
        expected_nudge=0.0, polarity_reasoning="no rule matched",
        components={}, observed=False,
    )
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), impact,
    )
    assert decision.accepted is False
    assert decision.observed is False
    assert "polarity_unknown" in decision.reason
    row = await db_module.fetch_one(
        "SELECT status FROM event_market_candidates WHERE event_id=?",
        ("evt-obs",),
    )
    assert row["status"] == "rejected"
