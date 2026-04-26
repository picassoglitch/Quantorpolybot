"""Post-scoring guard tests (hallucination + long-horizon cap)."""

from __future__ import annotations

from core.signals.guards import apply_true_prob_cap, hallucination_reject


def test_hallucination_guard_rejects_spec_example() -> None:
    """Directive spec: true_prob=0.70 + 1 source + mid=0.02 must REJECT."""
    reason = hallucination_reject(
        true_prob=0.70, mid=0.02, num_sources=1,
    )
    assert reason is not None
    assert "hallucination_guard" in reason


def test_hallucination_guard_accepts_strong_corroboration() -> None:
    """Same high prior but with three independent sources passes."""
    reason = hallucination_reject(
        true_prob=0.70, mid=0.02, num_sources=3,
    )
    assert reason is None


def test_hallucination_guard_ignores_normal_mid() -> None:
    """Above the low-mid threshold, the guard is a no-op."""
    reason = hallucination_reject(
        true_prob=0.70, mid=0.20, num_sources=0,
    )
    assert reason is None


def test_hallucination_guard_allows_low_mid_low_true_prob() -> None:
    """Low mid + low LLM prob is fine — the guard only fires on confident
    upward bets that lack corroboration."""
    reason = hallucination_reject(
        true_prob=0.30, mid=0.02, num_sources=0,
    )
    assert reason is None


def test_long_horizon_cap_clamps_far_future() -> None:
    assert apply_true_prob_cap(0.97, days_to_resolve=45) == 0.90


def test_long_horizon_cap_noop_near_future() -> None:
    """Resolutions within the horizon window are not clamped."""
    assert apply_true_prob_cap(0.97, days_to_resolve=10) == 0.97


def test_long_horizon_cap_noop_when_unknown_horizon() -> None:
    """No close_time => no cap (safer than guessing)."""
    assert apply_true_prob_cap(0.97, days_to_resolve=None) == 0.97
