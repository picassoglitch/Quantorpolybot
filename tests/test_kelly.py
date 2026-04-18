"""Unit tests for Kelly sizing."""

from core.risk.kelly import kelly_fraction, kelly_size_usd


def test_kelly_fraction_zero_when_no_edge():
    assert kelly_fraction(0.5, 0.5) == 0.0


def test_kelly_fraction_positive_when_edge_exists():
    f = kelly_fraction(prob_win=0.6, price=0.5)
    assert f > 0.0
    assert f < 1.0


def test_kelly_fraction_clamps_negative_to_zero():
    assert kelly_fraction(prob_win=0.3, price=0.5) == 0.0


def test_kelly_size_respects_min():
    size = kelly_size_usd(0.6, 0.5, bankroll_usd=10, fraction=0.001, min_size_usd=1.0)
    assert size == 0.0  # below min


def test_kelly_size_respects_max():
    size = kelly_size_usd(0.95, 0.10, bankroll_usd=10_000, fraction=1.0, max_size_usd=50.0)
    assert size == 50.0


def test_kelly_size_zero_bankroll():
    assert kelly_size_usd(0.9, 0.5, bankroll_usd=0) == 0.0
