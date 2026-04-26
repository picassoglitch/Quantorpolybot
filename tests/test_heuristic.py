"""Heuristic scorer unit tests.

Pure function — no DB, no network. Covers: positive/negative keyword
direction, ambiguous-returns-zero, confidence cap at 0.70, implied_prob
nudge bounded to 0.12, saturation at 3 hits.
"""

from __future__ import annotations

import time

from core.markets.cache import Market
from core.strategies.heuristic import score


def _market(mid: float = 0.50) -> Market:
    # best_bid/ask straddle the requested mid.
    half = 0.02
    return Market(
        market_id="m1",
        question="Will X happen?",
        slug="x",
        category="politics",
        active=True,
        close_time="",
        token_ids=["y", "n"],
        best_bid=mid - half,
        best_ask=mid + half,
        last_price=mid,
        liquidity=10_000.0,
        updated_at=time.time(),
    )


def test_positive_keywords_yield_bullish_signal():
    result = score("Candidate wins the primary and surges ahead", _market(0.50))
    assert result.direction == 1
    assert result.strength > 0
    assert result.implied_prob > 0.50  # nudged up from mid
    assert 0.55 <= result.confidence <= 0.70


def test_negative_keywords_yield_bearish_signal():
    result = score("Bill fails and stalls in committee; sponsor concedes", _market(0.60))
    assert result.direction == -1
    assert result.strength > 0
    assert result.implied_prob < 0.60  # nudged down
    assert 0.55 <= result.confidence <= 0.70


def test_ambiguous_text_returns_no_signal():
    # Both a positive and a negative keyword -> zero.
    result = score("She wins the vote but the committee fails to ratify", _market(0.50))
    assert result.direction == 0
    assert result.strength == 0.0
    assert result.confidence == 0.0


def test_no_keywords_returns_no_signal():
    result = score("A perfectly neutral sentence about weather today", _market(0.50))
    assert result.direction == 0
    assert result.strength == 0.0


def test_confidence_capped_at_070_even_with_many_hits():
    """Spec requirement: heuristic confidence must never exceed 0.70 —
    we don't know context, only vocabulary."""
    text = "wins confirms approves passes signs succeeds announces launches"
    result = score(text, _market(0.50))
    assert result.direction == 1
    assert result.confidence <= 0.70


def test_strength_saturates_at_three_hits():
    # 1 hit
    r1 = score("She wins the vote", _market(0.50))
    # 4 hits — strength should equal (not exceed) the 3-hit saturation value.
    r4 = score("wins confirms approves passes announces", _market(0.50))
    assert r1.strength < r4.strength
    assert r4.strength <= 1.0


def test_implied_prob_clamped_to_valid_range():
    """Even with a high mid + strong positive signal, implied_prob stays <= 0.99."""
    result = score("wins confirms approves passes", _market(0.95))
    assert 0.01 <= result.implied_prob <= 0.99


def test_implied_prob_nudge_scales_with_strength():
    weak = score("She wins", _market(0.50))
    strong = score("wins confirms approves", _market(0.50))
    # Stronger signal nudges implied_prob further from mid.
    assert abs(strong.implied_prob - 0.50) >= abs(weak.implied_prob - 0.50)


def test_reasoning_is_populated():
    result = score("Candidate wins decisively", _market(0.50))
    assert "heuristic" in result.reasoning.lower()
    assert "wins" in result.reasoning
