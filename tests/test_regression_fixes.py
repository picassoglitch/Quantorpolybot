"""Regression tests for the three post-incident fixes:

1. Ollama httpx singleton — the same client instance must be returned
   across calls so the connection pool and keep-alive state survive.
2. Market-universe filter + hard cap — 15k raw markets must be reduced
   to <= MAX_MARKETS, and the cap must be enforced by the
   volume/time-to-resolve composite rank.
3. PolymarketWS backoff jitter — retries at the same attempt count must
   produce different delays so two parallel failures don't lock-step.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from core.feeds.polymarket_ws import PolymarketWS
from core.markets.discovery import _apply_universe_filter
from core.signals.ollama_client import (
    _get_shared_client,
    reset_shared_client,
)


# -- 1. Singleton httpx client ------------------------------------------


@pytest.mark.asyncio
async def test_ollama_singleton_reuses_client():
    await reset_shared_client()
    try:
        a = await _get_shared_client()
        b = await _get_shared_client()
        assert a is b, "shared client must be reused across calls"
        assert not a.is_closed
    finally:
        await reset_shared_client()


@pytest.mark.asyncio
async def test_ollama_reset_rebuilds_client():
    await reset_shared_client()
    try:
        a = await _get_shared_client()
        await reset_shared_client()
        b = await _get_shared_client()
        assert a is not b, "reset must force a new client instance"
        assert a.is_closed
        assert not b.is_closed
    finally:
        await reset_shared_client()


# -- 2. Market universe filter cap --------------------------------------


def _synthetic_market(
    mid: int, *, volume: float, days_out: float, category: str = "politics",
    active: bool = True, closed: bool = False,
) -> dict:
    end = datetime.now(timezone.utc) + timedelta(days=days_out)
    return {
        "id": f"m-{mid}",
        "question": f"Question {mid}",
        "active": active,
        "closed": closed,
        "volume24hr": volume,
        "end_date": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "category": category,
    }


def test_universe_filter_caps_to_max():
    """15k raw markets (all nominally 'active') must drop to <= cap."""
    raw = []
    for i in range(15_000):
        raw.append(_synthetic_market(
            i,
            volume=20_000 + (i % 1000),
            days_out=1 + (i % 100),
        ))
    kept, dropped = _apply_universe_filter(
        raw, min_vol=10_000.0, max_days=120.0, cap=2000,
    )
    assert len(kept) == 2000
    # Nothing dropped for inactive / category since all synthetic markets
    # are active politics — dropped["volume"] and ["date"] can be 0,
    # ["inactive"]/["category"] must be 0.
    assert dropped["inactive"] == 0
    assert dropped["category"] == 0


def test_universe_filter_drops_low_volume_and_far_dated():
    raw = [
        _synthetic_market(1, volume=50_000, days_out=10),   # keep
        _synthetic_market(2, volume=500, days_out=10),      # drop: volume
        _synthetic_market(3, volume=50_000, days_out=400),  # drop: date
        _synthetic_market(4, volume=50_000, days_out=10, closed=True),  # drop: inactive
        _synthetic_market(5, volume=50_000, days_out=10, category="unknown"),  # drop: category
    ]
    kept, dropped = _apply_universe_filter(
        raw, min_vol=10_000.0, max_days=120.0, cap=100,
    )
    assert len(kept) == 1
    assert kept[0]["id"] == "m-1"
    assert dropped["volume"] == 1
    assert dropped["date"] == 1
    assert dropped["inactive"] == 1
    assert dropped["category"] == 1


def test_universe_filter_ranks_by_volume_over_days():
    """Within the cap, nearer-resolution + higher-volume wins."""
    raw = [
        _synthetic_market(1, volume=20_000, days_out=100),  # score 200
        _synthetic_market(2, volume=20_000, days_out=2),    # score 10_000
        _synthetic_market(3, volume=100_000, days_out=50),  # score 2_000
    ]
    kept, _ = _apply_universe_filter(
        raw, min_vol=10_000.0, max_days=120.0, cap=2,
    )
    kept_ids = [m["id"] for m in kept]
    assert kept_ids[0] == "m-2"  # highest score wins
    assert kept_ids[1] == "m-3"


# -- 3. WS backoff jitter ------------------------------------------------


def test_ws_backoff_produces_jittered_delays():
    """Backoff at the same attempt count must not be identical — jitter
    is what keeps two parallel failures from retrying in lock-step."""
    ws = PolymarketWS()
    ws._attempt = 3
    ws._backoff_cap = 60.0
    delays = {ws._backoff_delay() for _ in range(20)}
    # With 5s of uniform jitter over 20 samples, duplicates are astronomically
    # unlikely — the test fails deterministically if jitter is constant.
    assert len(delays) > 1
    # Base = min(60, 2*3) = 6; jitter 0..5 -> range [6, 11].
    for d in delays:
        assert 6.0 <= d <= 11.0


def test_ws_backoff_respects_cap():
    ws = PolymarketWS()
    ws._attempt = 1000
    ws._backoff_cap = 60.0
    # Base capped at 60, plus up to 5s jitter — never exceeds 65.
    for _ in range(10):
        d = ws._backoff_delay()
        assert 60.0 <= d <= 65.0
