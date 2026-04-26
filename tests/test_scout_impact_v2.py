"""Scout impact scoring v2 (PR #7) — broader polarity, better
confidence math, observed-mode.

Pure-function tests on `score_impact`. No DB, no async.
Round-trips for `candidate.evaluate(observed=True)` are in
`tests/test_scout_candidate_observed.py`.
"""

from __future__ import annotations

import time

import pytest

from core.markets.cache import Market
from core.scout.event import Event, EventCategory
from core.scout.impact import (
    _HEURISTIC_CONFIDENCE_CAP,
    _OBSERVED_CONFIDENCE,
    _OBSERVED_MIN_MATCH_SCORE,
    _OBSERVED_MIN_SEVERITY,
    _confidence,
    _infer_polarity,
    score_impact,
)
from core.scout.mapper import MarketMatch


def _market(question: str, mid: float = 0.50) -> Market:
    return Market(
        market_id="m-1", question=question, slug="m",
        category="politics", active=True, close_time="",
        token_ids=["yes", "no"],
        best_bid=mid - 0.01, best_ask=mid + 0.01, last_price=mid,
        liquidity=25_000.0, updated_at=time.time(),
    )


def _event(
    category: EventCategory,
    severity: float = 0.85,
    confidence: float = 0.70,
) -> Event:
    return Event(
        event_id="e-1", timestamp_detected=time.time(),
        title="x", category=category,
        severity=severity, confidence=confidence,
        location="", entities=["X"], source_count=2,
        sources=["a", "b"], contradiction_score=0.0,
        raw_signal_ids=[1], first_seen_at=time.time(),
        last_seen_at=time.time(),
    )


def _match(market: Market, score: float = 0.7) -> MarketMatch:
    return MarketMatch(
        market=market, score=score,
        entity_overlap=score, keyword_overlap=score,
        category_alignment=1.0, liquidity_quality=0.8,
        near_resolution_bonus=0.5,
    )


# ============================================================
# Broader polarity rules — sample coverage per category
# ============================================================


@pytest.mark.parametrize("question, expected_dir", [
    # CEASEFIRE bullish patterns
    ("Will the war end by Q3?", +1),
    ("Will a ceasefire be signed by April?", +1),
    ("Will Russia withdraw from territory by year end?", +1),
    # CEASEFIRE bearish (continuation) patterns
    ("Will hostilities escalate in Q4?", -1),
    ("Will the invasion expand to neighboring territory?", -1),
])
def test_ceasefire_polarity_covers_common_phrasings(question, expected_dir):
    d, _ = _infer_polarity(EventCategory.CEASEFIRE, question)
    assert d == expected_dir, f"failed for {question!r}"


@pytest.mark.parametrize("question, expected_dir", [
    ("Will Trump attend the rally on Saturday?", -1),
    ("Will the President appear at WHCD?", -1),
    ("Will the Secret Service director resign by April?", +1),
    ("Will the suspect be charged with attempted murder?", +1),
    ("Will Trump survive the next month?", +1),
])
def test_assassination_attempt_polarity(question, expected_dir):
    d, _ = _infer_polarity(EventCategory.ASSASSINATION_ATTEMPT, question)
    assert d == expected_dir, f"failed for {question!r}"


@pytest.mark.parametrize("question, expected_dir", [
    ("Will the rally happen as scheduled?", -1),
    ("Will the venue be evacuated by 5pm?", +1),
    ("Will the suspect be in custody by Friday?", +1),
    ("Will Senator X resign over the response?", +1),
    ("Will the event be cancelled?", +1),
])
def test_shooting_polarity(question, expected_dir):
    d, _ = _infer_polarity(EventCategory.SHOOTING, question)
    assert d == expected_dir, f"failed for {question!r}"


@pytest.mark.parametrize("question, expected_dir", [
    ("Will the minister resign by April?", +1),
    ("Will the CEO step down this quarter?", +1),
    ("Will the director be fired?", +1),
    ("Will the senator remain in office through 2027?", -1),
    ("Will the PM stay as leader by year end?", -1),
])
def test_resignation_polarity(question, expected_dir):
    d, _ = _infer_polarity(EventCategory.RESIGNATION, question)
    assert d == expected_dir, f"failed for {question!r}"


@pytest.mark.parametrize("question, expected_dir", [
    ("Will defendant be convicted by Q3?", +1),
    ("Will charges be filed by April?", +1),
    ("Will defendant be acquitted?", -1),
    ("Will the case be dismissed?", -1),
])
def test_arrest_polarity(question, expected_dir):
    d, _ = _infer_polarity(EventCategory.ARREST, question)
    assert d == expected_dir, f"failed for {question!r}"


@pytest.mark.parametrize("question, expected_dir", [
    ("Will the Supreme Court uphold the law?", +1),
    ("Will the ruling be reversed on appeal?", -1),
    ("Will the lower court ruling be struck down?", -1),
    ("Will the judge rule in favor of the defendant?", +1),
])
def test_court_ruling_polarity(question, expected_dir):
    d, _ = _infer_polarity(EventCategory.COURT_RULING, question)
    assert d == expected_dir, f"failed for {question!r}"


@pytest.mark.parametrize("question, expected_dir", [
    ("Will the QB play this Sunday?", -1),
    ("Will Player X start in the playoffs?", -1),
    ("Will the backup take over?", +1),
    ("Will the replacement be called up?", +1),
])
def test_sports_injury_polarity(question, expected_dir):
    d, _ = _infer_polarity(EventCategory.SPORTS_INJURY, question)
    assert d == expected_dir, f"failed for {question!r}"


def test_election_result_has_no_static_polarity():
    """Static rules can't infer who won — must defer to LLM tier
    or fall through to observed-mode."""
    d, _ = _infer_polarity(
        EventCategory.ELECTION_RESULT, "Will candidate X win the primary?",
    )
    assert d == 0


def test_macro_data_surprise_has_no_static_polarity():
    d, _ = _infer_polarity(
        EventCategory.MACRO_DATA_SURPRISE, "Will the Fed cut rates by July?",
    )
    assert d == 0


# ============================================================
# Confidence math v2 — weighted average × severity
# ============================================================


def test_confidence_lifts_off_with_strong_event_and_moderate_match():
    """v1 multiplicative formula returned 0.34 for these inputs
    (event_conf=0.85, match=0.4, sev=0.85). v2 should clear the
    0.40 lane gate."""
    c = _confidence(event_confidence=0.85, match_score=0.4, severity=0.85)
    assert c >= 0.40
    assert c <= _HEURISTIC_CONFIDENCE_CAP


def test_confidence_caps_at_055_even_on_perfect_inputs():
    c = _confidence(event_confidence=1.0, match_score=1.0, severity=1.0)
    assert c == _HEURISTIC_CONFIDENCE_CAP


def test_confidence_stays_low_on_weak_inputs():
    """Don't accidentally pass weak matches — operator depends on
    the floor for safety."""
    c = _confidence(event_confidence=0.6, match_score=0.25, severity=0.5)
    assert c < 0.40


def test_confidence_severity_floor_at_05():
    """A 0.0-severity event still gets sev_mult=0.5 (the lane's
    safety floor against pathological inputs). Matches the
    docstring contract."""
    c_zero = _confidence(event_confidence=0.85, match_score=0.4, severity=0.0)
    c_low = _confidence(event_confidence=0.85, match_score=0.4, severity=0.5)
    assert abs(c_zero - c_low) < 1e-9


# ============================================================
# Observed-mode — direction=0 + high severity + decent match
# ============================================================


def test_observed_mode_when_high_severity_and_match_but_no_polarity():
    """ELECTION_RESULT has no polarity rules. With severity=0.85
    and match_score=0.5, the scorer should flag observed=True
    instead of returning a hard zero."""
    market = _market("Will candidate X win the Q3 primary?")
    impact = score_impact(
        _event(EventCategory.ELECTION_RESULT, severity=0.85),
        _match(market, score=0.5),
    )
    assert impact.direction == 0
    assert impact.observed is True
    assert 0.0 < impact.confidence <= _OBSERVED_CONFIDENCE
    # true_prob falls back to the market mid (no nudge applied).
    assert abs(impact.true_prob - 0.50) < 1e-9


def test_not_observed_when_severity_below_threshold():
    market = _market("Will candidate X win the Q3 primary?")
    low_sev = _event(EventCategory.ELECTION_RESULT, severity=0.40)
    impact = score_impact(low_sev, _match(market, score=0.7))
    assert impact.direction == 0
    assert impact.observed is False
    assert impact.confidence == 0.0


def test_not_observed_when_match_score_below_threshold():
    market = _market("Will candidate X win the Q3 primary?")
    impact = score_impact(
        _event(EventCategory.ELECTION_RESULT, severity=0.85),
        _match(market, score=0.20),  # below the observed threshold
    )
    assert impact.direction == 0
    assert impact.observed is False
    assert impact.confidence == 0.0


def test_observed_thresholds_are_documented_constants():
    """Pin the constants in the test so a future loosening is
    reviewed deliberately."""
    assert _OBSERVED_MIN_SEVERITY == 0.60
    assert _OBSERVED_MIN_MATCH_SCORE == 0.30
    assert _OBSERVED_CONFIDENCE == 0.20


# ============================================================
# Existing happy-path still works (regression coverage)
# ============================================================


def test_ceasefire_buys_yes_on_war_ends_market_v2():
    market = _market("Will the war end by Q3?", mid=0.40)
    impact = score_impact(_event(EventCategory.CEASEFIRE), _match(market))
    assert impact.direction == +1
    assert impact.true_prob > 0.40
    assert impact.confidence >= 0.40
    assert impact.observed is False


def test_war_escalation_sells_yes_on_ceasefire_market_v2():
    market = _market("Will there be a ceasefire by Q3?", mid=0.50)
    impact = score_impact(_event(EventCategory.WAR_ESCALATION), _match(market))
    assert impact.direction == -1
    assert impact.true_prob < 0.50
    assert impact.observed is False
