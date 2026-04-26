"""Candidate generator + safety gate + persistence tests.

Round-trips through a tmp DB so the audit-row contract is verified.
"""

from __future__ import annotations

import asyncio
import json
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
    db_path = tmp_path / "scout_candidate.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _market(question: str = "Will war ends by Q3?", mid: float = 0.40) -> Market:
    return Market(
        market_id="m-x",
        question=question,
        slug="m",
        category="politics",
        active=True,
        close_time="",
        token_ids=["yes", "no"],
        best_bid=mid - 0.01,
        best_ask=mid + 0.01,
        last_price=mid,
        liquidity=25_000.0,
        updated_at=time.time(),
    )


def _event(
    *,
    category: EventCategory = EventCategory.CEASEFIRE,
    age_seconds: float = 60.0,
    sources: list[str] | None = None,
    confidence: float = 0.7,
) -> Event:
    now = time.time()
    sources = sources or ["src-a", "src-b"]
    return Event(
        event_id="evt-1",
        timestamp_detected=now - age_seconds,
        title="Ceasefire signed",
        category=category,
        severity=0.85,
        confidence=confidence,
        location="",
        entities=["X"],
        source_count=len(sources),
        sources=sources,
        contradiction_score=0.0,
        raw_signal_ids=[1, 2],
        first_seen_at=now - age_seconds,
        last_seen_at=now,
    )


def _match(market: Market, score: float = 0.7) -> MarketMatch:
    return MarketMatch(
        market=market, score=score,
        entity_overlap=score, keyword_overlap=score,
        category_alignment=1.0, liquidity_quality=0.8,
        near_resolution_bonus=0.5,
    )


def _impact(direction: int = 1, confidence: float = 0.50, true_prob: float = 0.55) -> ImpactScore:
    return ImpactScore(
        direction=direction,
        true_prob=true_prob,
        confidence=confidence,
        expected_nudge=0.10,
        polarity_reasoning="test",
        components={"category_nudge": 0.15, "match_score": 0.7},
    )


# ---------------- Accept path ----------------


@pytest.mark.asyncio
async def test_accept_writes_audit_row_with_status_accepted(temp_db):
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), _impact(),
    )
    assert decision.accepted is True
    assert decision.side in ("BUY", "SELL")
    row = await db_module.fetch_one(
        "SELECT * FROM event_market_candidates WHERE event_id=?", ("evt-1",),
    )
    assert row["status"] == "accepted"
    assert row["reject_reason"] == ""
    assert row["side"] == decision.side
    assert row["impact_snapshot"]
    snap = json.loads(row["impact_snapshot"])
    # The snapshot must carry the three blocks the dashboard / Pattern
    # Discovery layer needs to reason about decisions.
    assert "event" in snap and "market_match" in snap and "impact" in snap


# ---------------- Reject paths ----------------


@pytest.mark.asyncio
async def test_reject_when_event_too_old(temp_db):
    decision = await scout_candidate.evaluate(
        _event(age_seconds=10_000.0), _match(_market()), _impact(),
        cfg={"max_event_age_seconds": 1800.0},
    )
    assert decision.accepted is False
    assert "too old" in decision.reason


@pytest.mark.asyncio
async def test_reject_when_insufficient_corroboration_and_no_primary(temp_db):
    """Single-source non-primary event with severity BELOW the v3
    high-sev-solo threshold (0.80) must still reject as insufficient
    corroboration. Above-threshold severity now passes through with
    a reduced size multiplier — see test_scout_v3.py for that path.
    """
    # Force severity 0.65 < 0.80 so the v3 high_sev_solo override
    # does NOT kick in; corroboration check still rejects.
    weak_event = _event()
    weak_event = type(weak_event)(  # rebuild dataclass with lower severity
        **{**weak_event.__dict__, "severity": 0.65,
           "sources": ["someblog"], "source_count": 1},
    )
    decision = await scout_candidate.evaluate(
        weak_event, _match(_market()), _impact(),
    )
    assert decision.accepted is False
    assert "corroboration" in decision.reason


@pytest.mark.asyncio
async def test_accept_when_single_primary_source(temp_db):
    """One trusted primary source satisfies the corroboration rule."""
    decision = await scout_candidate.evaluate(
        _event(sources=["reuters"]),
        _match(_market()), _impact(),
        cfg={
            "min_sources": 2,
            "primary_sources": ["reuters", "apnews", "bloomberg"],
            "min_edge": 0.05,
            "min_confidence": 0.40,
            "max_event_age_seconds": 1800.0,
        },
    )
    assert decision.accepted is True


@pytest.mark.asyncio
async def test_reject_when_polarity_unknown(temp_db):
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), _impact(direction=0),
    )
    assert decision.accepted is False
    assert "polarity" in decision.reason


@pytest.mark.asyncio
async def test_reject_when_edge_below_threshold(temp_db):
    decision = await scout_candidate.evaluate(
        _event(), _match(_market(mid=0.50)),
        _impact(direction=1, true_prob=0.51),  # edge 0.01
        cfg={"min_edge": 0.05, "min_confidence": 0.0,
             "max_event_age_seconds": 1800.0,
             "require_primary_or_corroboration": False},
    )
    assert decision.accepted is False
    assert "edge" in decision.reason


@pytest.mark.asyncio
async def test_reject_when_confidence_below_threshold(temp_db):
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), _impact(confidence=0.20),
        cfg={"min_confidence": 0.40,
             "min_edge": 0.0,
             "max_event_age_seconds": 1800.0,
             "require_primary_or_corroboration": False},
    )
    assert decision.accepted is False
    assert "confidence" in decision.reason


# ---------------- Cool-down ----------------


@pytest.mark.asyncio
async def test_cooldown_blocks_second_accept_for_same_event(temp_db):
    market_a = _market(); market_a.market_id = "m-a"
    market_b = _market(); market_b.market_id = "m-b"
    d1 = await scout_candidate.evaluate(
        _event(), _match(market_a), _impact(),
        cfg={"event_max_positions": 1,
             "max_event_age_seconds": 1800.0,
             "min_edge": 0.05, "min_confidence": 0.40,
             "min_sources": 1,  # single source ok for this test
             "require_primary_or_corroboration": False},
    )
    assert d1.accepted is True
    d2 = await scout_candidate.evaluate(
        _event(), _match(market_b), _impact(),
        cfg={"event_max_positions": 1,
             "max_event_age_seconds": 1800.0,
             "min_edge": 0.05, "min_confidence": 0.40,
             "min_sources": 1,
             "require_primary_or_corroboration": False},
    )
    assert d2.accepted is False
    assert "cooldown" in d2.reason


# ---------------- attach_position ----------------


@pytest.mark.asyncio
async def test_attach_position_writes_position_id(temp_db):
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), _impact(),
    )
    assert decision.audit_row_id > 0
    await scout_candidate.attach_position(decision.audit_row_id, 9999)
    row = await db_module.fetch_one(
        "SELECT shadow_position_id FROM event_market_candidates WHERE id=?",
        (decision.audit_row_id,),
    )
    assert row["shadow_position_id"] == 9999
