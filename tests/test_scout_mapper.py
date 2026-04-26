"""Mapper tests — entity/keyword/category match scoring."""

from __future__ import annotations

import time

from core.markets.cache import Market
from core.scout.event import Event, EventCategory
from core.scout.mapper import map_event_to_markets


def _market(
    market_id: str,
    question: str,
    *,
    category: str = "politics",
    mid: float = 0.50,
    spread_cents: float = 1.0,
    liquidity: float = 25_000.0,
    close_time: str = "",
) -> Market:
    half = (spread_cents / 100.0) / 2.0
    return Market(
        market_id=market_id,
        question=question,
        slug=market_id,
        category=category,
        active=True,
        close_time=close_time,
        token_ids=["yes", "no"],
        best_bid=mid - half,
        best_ask=mid + half,
        last_price=mid,
        liquidity=liquidity,
        updated_at=time.time(),
    )


def _event(category: EventCategory, *, entities: list[str], title: str = "") -> Event:
    return Event(
        event_id="e-1",
        timestamp_detected=time.time(),
        title=title or " ".join(entities),
        category=category,
        severity=0.9,
        confidence=0.7,
        location="",
        entities=entities,
        source_count=2,
        sources=["src-a", "src-b"],
        contradiction_score=0.0,
        raw_signal_ids=[1, 2],
        first_seen_at=time.time(),
        last_seen_at=time.time(),
    )


def test_market_with_strong_entity_overlap_outranks_unrelated_market():
    event = _event(
        EventCategory.SHOOTING,
        entities=["Trump", "Pittsburgh"],
        title="Trump survives shooting in Pittsburgh",
    )
    markets = [
        _market("m-trump-attends", "Will Trump attend Pittsburgh rally?",
                category="politics"),
        _market("m-bitcoin", "Will Bitcoin reach $100k by year end?",
                category="crypto"),
    ]
    matches = map_event_to_markets(event, markets, top_k=5, min_score=0.0)
    assert matches
    # Trump market wins.
    assert matches[0].market.market_id == "m-trump-attends"
    # Bitcoin filtered or far below.
    if len(matches) > 1:
        assert matches[0].score > matches[1].score


def test_min_score_filters_out_irrelevant_markets():
    event = _event(EventCategory.SHOOTING, entities=["Bigfoot"])
    markets = [_market("m-1", "Will Trump win the 2024 election?")]
    matches = map_event_to_markets(event, markets, min_score=0.30)
    assert matches == []


def test_inactive_market_is_filtered():
    event = _event(EventCategory.SHOOTING, entities=["Trump"])
    market = _market("m-1", "Will Trump speak Pittsburgh?")
    market.active = False
    matches = map_event_to_markets(event, [market], min_score=0.0)
    assert matches == []


def test_low_liquidity_filtered_by_min_liquidity():
    event = _event(EventCategory.SHOOTING, entities=["Trump"])
    market = _market("m-1", "Will Trump speak?", liquidity=500.0)
    matches = map_event_to_markets(event, [market], min_liquidity=1000.0)
    assert matches == []


def test_wide_spread_filtered():
    event = _event(EventCategory.SHOOTING, entities=["Trump"])
    market = _market("m-1", "Will Trump speak?", spread_cents=10.0)
    matches = map_event_to_markets(event, [market], max_spread_cents=5.0)
    assert matches == []


def test_top_k_caps_returned_matches():
    event = _event(EventCategory.SHOOTING, entities=["Trump"])
    markets = [
        _market(f"m-{i}", f"Will Trump action {i}?", category="politics")
        for i in range(10)
    ]
    matches = map_event_to_markets(event, markets, top_k=3, min_score=0.0)
    assert len(matches) == 3


def test_category_alignment_boosts_score():
    """Same entity overlap, different categories → politics (aligned)
    should outrank crypto (unaligned)."""
    event = _event(
        EventCategory.ELECTION_RESULT, entities=["Smith"],
        title="Smith wins election",
    )
    politics_market = _market("m-pol", "Will Smith win election?", category="politics")
    crypto_market = _market("m-cry", "Will Smith win election?", category="crypto")
    matches = map_event_to_markets(
        event, [politics_market, crypto_market], top_k=5, min_score=0.0,
    )
    assert matches[0].market.market_id == "m-pol"
    assert matches[0].category_alignment > matches[1].category_alignment
