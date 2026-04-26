"""Heuristic impact scorer tests.

Pure function — no DB, no network.

Covers: polarity inference per category, neutral on unknown polarity,
nudge magnitude scales with severity × match score, confidence
hard-cap at 0.55.
"""

from __future__ import annotations

import time

from core.markets.cache import Market
from core.scout.event import Event, EventCategory
from core.scout.impact import score_impact
from core.scout.mapper import MarketMatch


def _market(question: str, mid: float = 0.50) -> Market:
    return Market(
        market_id="m-1",
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


def _event(category: EventCategory, severity: float = 0.9, confidence: float = 0.7) -> Event:
    return Event(
        event_id="e-1",
        timestamp_detected=time.time(),
        title="x",
        category=category,
        severity=severity,
        confidence=confidence,
        location="",
        entities=["X"],
        source_count=2,
        sources=["a", "b"],
        contradiction_score=0.0,
        raw_signal_ids=[1],
        first_seen_at=time.time(),
        last_seen_at=time.time(),
    )


def _match(market: Market, score: float = 0.7) -> MarketMatch:
    return MarketMatch(
        market=market,
        score=score,
        entity_overlap=score,
        keyword_overlap=score,
        category_alignment=1.0,
        liquidity_quality=0.8,
        near_resolution_bonus=0.5,
    )


# ---------------- Polarity inference ----------------


def test_ceasefire_buys_yes_on_war_ends_market():
    market = _market("Will the war ends by Q3?", mid=0.40)
    impact = score_impact(_event(EventCategory.CEASEFIRE), _match(market))
    assert impact.direction == 1
    assert impact.true_prob > 0.40


def test_war_escalation_sells_yes_on_ceasefire_market():
    market = _market("Will there be a ceasefire by Q3?", mid=0.50)
    impact = score_impact(_event(EventCategory.WAR_ESCALATION), _match(market))
    assert impact.direction == -1
    assert impact.true_prob < 0.50


def test_shooting_sells_yes_on_attendance_market():
    market = _market("Will the President attend WHCD?", mid=0.85)
    impact = score_impact(_event(EventCategory.SHOOTING), _match(market))
    assert impact.direction == -1
    assert impact.true_prob < 0.85


def test_resignation_buys_yes_on_resigns_market():
    market = _market("Will Director resigns by April?", mid=0.30)
    impact = score_impact(_event(EventCategory.RESIGNATION), _match(market))
    assert impact.direction == 1


def test_unknown_polarity_returns_zero_direction():
    market = _market("Some unrelated market question", mid=0.50)
    impact = score_impact(_event(EventCategory.MACRO_DATA_SURPRISE), _match(market))
    # MACRO_DATA_SURPRISE has no polarity rules in v1.
    assert impact.direction == 0
    assert impact.confidence == 0.0


# ---------------- Magnitude / severity ----------------


def test_higher_severity_yields_larger_nudge():
    market = _market("Will the war ends by Q3?", mid=0.50)
    weak_event = _event(EventCategory.CEASEFIRE, severity=0.4)
    strong_event = _event(EventCategory.CEASEFIRE, severity=1.0)
    weak = score_impact(weak_event, _match(market))
    strong = score_impact(strong_event, _match(market))
    assert abs(strong.true_prob - 0.50) > abs(weak.true_prob - 0.50)


def test_higher_match_score_yields_larger_nudge():
    market = _market("Will the war ends by Q3?", mid=0.50)
    event = _event(EventCategory.CEASEFIRE)
    weak_match = _match(market, score=0.30)
    strong_match = _match(market, score=0.90)
    weak = score_impact(event, weak_match)
    strong = score_impact(event, strong_match)
    assert abs(strong.true_prob - 0.50) > abs(weak.true_prob - 0.50)


# ---------------- Confidence cap ----------------


def test_confidence_capped_at_055_regardless_of_inputs():
    """Heuristic scorer must not return confidence > 0.55 even with
    a 1.0 event confidence and 1.0 match score. The lane's
    `min_confidence` gate (default 0.40) sits below this so the
    accepted band is well-defined."""
    market = _market("Will the war ends by Q3?", mid=0.50)
    event = _event(EventCategory.CEASEFIRE, severity=1.0, confidence=1.0)
    match = _match(market, score=1.0)
    impact = score_impact(event, match)
    assert impact.confidence <= 0.55


def test_explicit_confidence_cap_argument_is_respected():
    market = _market("Will the war ends by Q3?", mid=0.50)
    impact = score_impact(
        _event(EventCategory.CEASEFIRE), _match(market),
        confidence_cap=0.30,
    )
    assert impact.confidence <= 0.30


# ---------------- Mid clamping ----------------


def test_true_prob_clamped_to_open_interval():
    """Even a maximally bullish nudge on a near-1.0 mid stays < 1.0
    (and symmetric for near-0)."""
    market_high = _market("Will resigns by April?", mid=0.99)
    market_low = _market("Will the war ends by Q3?", mid=0.01)
    impact_high = score_impact(_event(EventCategory.RESIGNATION), _match(market_high))
    impact_low = score_impact(_event(EventCategory.WAR_ESCALATION), _match(market_low))
    assert 0.0 < impact_high.true_prob < 1.0
    assert 0.0 < impact_low.true_prob < 1.0
