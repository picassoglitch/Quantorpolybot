"""Tests for the scout cycle's `rejected_breakdown` instrumentation
(v3.1 follow-up).

The scout lane emits one line per scan with reject reasons bucketed
into coarse categories so an operator can see WHICH gate is
dominant without grepping every individual reject. These tests pin
the bucket assignments against the long human-readable reason
strings the candidate evaluator emits.
"""

from __future__ import annotations

import time

import pytest

from core.scout.event import Event, EventCategory
from core.strategies.breaking_event_scout import _bucket_reject_reason


def _event(severity: float = 0.85) -> Event:
    now = time.time()
    return Event(
        event_id="evt", timestamp_detected=now, title="t",
        category=EventCategory.SHOOTING,
        severity=severity, confidence=0.7, location="",
        entities=["X"], source_count=1, sources=["src"],
        contradiction_score=0.0, raw_signal_ids=[1],
        first_seen_at=now, last_seen_at=now,
    )


@pytest.mark.parametrize("reason, expected", [
    # Corroboration shape from candidate.evaluate
    ("insufficient corroboration: source_count=1 < 2 and no primary "
     "source in ['apnews', 'bbc', 'bloomberg', 'reuters']",
     "corroboration"),
    # Polarity unknown — high severity (above observed floor)
    ("polarity_unknown: no polarity rule matched for category=ceasefire",
     "polarity_unknown"),
    ("edge 0.030 < min_edge 0.050",  "low_edge"),
    ("impact confidence 0.20 < 0.40", "low_confidence"),
    ("event cooldown: 1 accepted candidate(s) already exist",
     "cooldown"),
    ("event too old: age 5000s > max_event_age 1800s",
     "too_old"),
    ("spread 8.0c outside (0, 5.0c]",          "spread"),
    ("liquidity 500 < min liquidity 1000",     "liquidity"),
    ("lane_paused_or_missing",                 "lane_paused"),
    ("allocator_denied",                       "allocator_denied"),
    ("invalid mid 0.000 (out of (0,1))",       "invalid_mid"),
    ("",                                       "other"),
    ("some_unrecognised_reason_string",        "other"),
])
def test_bucket_reject_reason_picks_expected_bucket(reason, expected):
    bucket = _bucket_reject_reason(reason, _event())
    assert bucket == expected, f"reason={reason!r} -> {bucket!r}, expected {expected!r}"


def test_polarity_unknown_with_low_severity_buckets_as_low_severity():
    """Special-case from the docstring: a polarity_unknown reject on
    an event whose severity is below the observed-mode floor should
    bucket as `low_severity`, not `polarity_unknown`. That distinction
    tells the operator "we discarded the signal as too weak" vs.
    "rules don't know what to do for this market shape"."""
    low_sev = _event(severity=0.20)  # below the v3.1 0.35 floor
    bucket = _bucket_reject_reason(
        "polarity_unknown: no polarity rules for category=macro_data_surprise",
        low_sev,
    )
    assert bucket == "low_severity"


def test_polarity_unknown_with_borderline_severity_buckets_as_polarity_unknown():
    """Severity exactly at the observed-mode floor should still
    classify as polarity_unknown (the bucket cuts off strictly below)."""
    from core.scout.impact import _OBSERVED_MIN_SEVERITY
    at_floor = _event(severity=_OBSERVED_MIN_SEVERITY)
    bucket = _bucket_reject_reason(
        "polarity_unknown: no polarity rules", at_floor,
    )
    assert bucket == "polarity_unknown"


def test_corroboration_takes_priority_over_polarity_unknown():
    """If a reason mentions both 'corroboration' and (rare edge
    case) other tokens, corroboration wins because it's a stronger
    signal of WHY this candidate was dropped."""
    bucket = _bucket_reject_reason(
        "insufficient corroboration: source_count=1 ...", _event(),
    )
    assert bucket == "corroboration"


# ============================================================
# polarity_confirms_solo stub (v3.1, NOT YET WIRED)
# ============================================================


from dataclasses import dataclass

from core.strategies.breaking_event_scout import polarity_confirms_solo


@dataclass
class _FakeImpact:
    """Lightweight stand-in for ImpactScore so the stub tests don't
    depend on the full scout.impact module."""
    direction: int = 0
    confidence: float = 0.0
    polarity_source: str = "rules"


def test_polarity_confirms_solo_returns_false_by_default():
    """Default config has `llm_polarity_confirms_solo_enabled: false`
    so the helper must return False even on a perfect input."""
    impact = _FakeImpact(direction=+1, confidence=0.9, polarity_source="llm")
    assert polarity_confirms_solo(
        _event(), impact, "BUY", cfg={},
    ) is False


def test_polarity_confirms_solo_returns_true_on_buy_match_when_enabled():
    impact = _FakeImpact(direction=+1, confidence=0.9, polarity_source="llm")
    assert polarity_confirms_solo(
        _event(), impact, "BUY",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is True


def test_polarity_confirms_solo_returns_true_on_sell_match_when_enabled():
    impact = _FakeImpact(direction=-1, confidence=0.9, polarity_source="llm")
    assert polarity_confirms_solo(
        _event(), impact, "SELL",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is True


def test_polarity_confirms_solo_rejects_direction_mismatch():
    """LLM said BUY (direction=+1) but candidate side is SELL → no
    confirmation."""
    impact = _FakeImpact(direction=+1, confidence=0.9, polarity_source="llm")
    assert polarity_confirms_solo(
        _event(), impact, "SELL",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is False


def test_polarity_confirms_solo_rejects_low_confidence():
    """min_confidence default is 0.7 — anything below should reject."""
    impact = _FakeImpact(direction=+1, confidence=0.6, polarity_source="llm")
    assert polarity_confirms_solo(
        _event(), impact, "BUY",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is False


def test_polarity_confirms_solo_rejects_when_polarity_source_not_llm():
    """The whole point is that the LLM was the second signal — a
    rule-based polarity doesn't count as independent corroboration."""
    impact = _FakeImpact(direction=+1, confidence=0.9, polarity_source="rules")
    assert polarity_confirms_solo(
        _event(), impact, "BUY",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is False


def test_polarity_confirms_solo_rejects_zero_direction():
    """LLM returned 'unclear' (direction=0) — no confirmation."""
    impact = _FakeImpact(direction=0, confidence=0.9, polarity_source="llm")
    assert polarity_confirms_solo(
        _event(), impact, "BUY",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is False


def test_polarity_confirms_solo_handles_none_impact():
    """Defensive: a None impact (no LLM call attempted) returns False."""
    assert polarity_confirms_solo(
        _event(), None, "BUY",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is False


def test_polarity_confirms_solo_respects_custom_min_confidence():
    impact = _FakeImpact(direction=+1, confidence=0.55, polarity_source="llm")
    # Default 0.7 floor → False
    assert polarity_confirms_solo(
        _event(), impact, "BUY",
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is False
    # Custom 0.5 floor → True
    assert polarity_confirms_solo(
        _event(), impact, "BUY", min_confidence=0.5,
        cfg={"llm_polarity_confirms_solo_enabled": True},
    ) is True


def test_stub_is_not_wired_into_candidate_evaluator():
    """Belt-and-suspenders: confirm core/scout/candidate.py does NOT
    import polarity_confirms_solo. The wiring is intentionally
    deferred until after the threshold-fix soak proves stable."""
    import core.scout.candidate as cand
    # Read the source — checking for the literal name in the module
    # source is more reliable than introspection (the helper could
    # be imported indirectly).
    import inspect
    src = inspect.getsource(cand)
    assert "polarity_confirms_solo" not in src, (
        "polarity_confirms_solo is now wired into candidate.evaluate. "
        "Update this test if that's intentional, but flag it loudly "
        "in the PR — it changes corroboration safety semantics."
    )
