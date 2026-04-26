"""Microstructure proxy scorer tests.

Seeds the price_ticks table with synthetic tick streams and asserts the
scorer produces the right direction/strength/confidence under each
regime: clean drift, no drift, stale book, crossed book, sparse ticks.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.markets.cache import Market
from core.strategies.microstructure import (
    _MAX_MICROSTRUCTURE_CONFIDENCE,
    score_microstructure,
)
from core.utils import db as db_module


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "micro.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _market(mid: float = 0.50, spread_cents: float = 1.0, liquidity: float = 25_000.0) -> Market:
    half = (spread_cents / 100.0) / 2.0
    return Market(
        market_id="mkt-1",
        question="Will X happen?",
        slug="x",
        category="politics",
        active=True,
        close_time="",
        token_ids=["yes", "no"],
        best_bid=mid - half,
        best_ask=mid + half,
        last_price=mid,
        liquidity=liquidity,
        updated_at=time.time(),
    )


async def _seed_ticks(market_id: str, mids: list[float], dt_seconds: float = 30.0) -> None:
    """Insert ticks for `market_id`, oldest first, 30s apart."""
    now = time.time()
    rows = []
    for i, mid in enumerate(mids):
        ts = now - (len(mids) - i) * dt_seconds
        rows.append((market_id, "yes", mid - 0.005, mid + 0.005, mid, ts))
    await db_module.executemany(
        "INSERT INTO price_ticks (market_id, token_id, bid, ask, last, ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


# ---------------- Hard-rejects (return None) ----------------


@pytest.mark.asyncio
async def test_returns_none_when_no_ticks_in_window(temp_db):
    sig = await score_microstructure(
        _market(),
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=50_000.0,
        window_seconds=600.0,
        min_ticks=6,
    )
    assert sig is None


@pytest.mark.asyncio
async def test_returns_none_below_min_ticks(temp_db):
    await _seed_ticks("mkt-1", [0.50, 0.51, 0.52])  # 3 ticks, need 6
    sig = await score_microstructure(
        _market(),
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=50_000.0,
        window_seconds=600.0,
        min_ticks=6,
    )
    assert sig is None


@pytest.mark.asyncio
async def test_returns_none_when_book_is_crossed(temp_db):
    """Spread <= 0 should hard-reject; we don't trade crossed books."""
    await _seed_ticks("mkt-1", [0.50] * 10)
    crossed = _market(mid=0.50, spread_cents=1.0)
    crossed.best_bid = 0.55  # > best_ask -> spread is negative
    crossed.best_ask = 0.45
    sig = await score_microstructure(
        crossed,
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=50_000.0,
    )
    assert sig is None


# ---------------- Direction = 0 (no drift) ----------------


@pytest.mark.asyncio
async def test_zero_direction_when_mids_are_flat(temp_db):
    await _seed_ticks("mkt-1", [0.50] * 12)
    sig = await score_microstructure(
        _market(),
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=50_000.0,
    )
    assert sig is not None
    assert sig.direction == 0
    assert sig.strength == 0.0
    assert sig.confidence == 0.0
    assert "no drift" in sig.reasoning


# ---------------- Bullish drift ----------------


@pytest.mark.asyncio
async def test_bullish_drift_returns_positive_direction(temp_db):
    # Strong upward drift: 50c -> 53c over 12 ticks.
    mids = [0.50 + 0.0025 * i for i in range(12)]
    await _seed_ticks("mkt-1", mids)
    sig = await score_microstructure(
        _market(mid=0.515),
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=50_000.0,
    )
    assert sig is not None
    assert sig.direction == 1
    assert sig.strength > 0.40
    assert sig.implied_prob > 0.515  # nudged above mid
    # Confidence is hard-capped at the module constant.
    assert sig.confidence <= _MAX_MICROSTRUCTURE_CONFIDENCE


@pytest.mark.asyncio
async def test_bearish_drift_returns_negative_direction(temp_db):
    mids = [0.50 - 0.0025 * i for i in range(12)]
    await _seed_ticks("mkt-1", mids)
    sig = await score_microstructure(
        _market(mid=0.485),
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=50_000.0,
    )
    assert sig is not None
    assert sig.direction == -1
    assert sig.implied_prob < 0.485


# ---------------- Confidence cap ----------------


@pytest.mark.asyncio
async def test_confidence_never_exceeds_cap_even_on_perfect_inputs(temp_db):
    """Even with maximal drift, max liquidity, max volume, min spread,
    confidence MUST be <= the module-level cap. This is the spec
    requirement: microstructure entries cannot size up like news
    entries."""
    # Strongest possible drift in 12 ticks within the saturation window.
    mids = [0.40 + 0.005 * i for i in range(12)]  # 6c drift
    await _seed_ticks("mkt-1", mids)
    sig = await score_microstructure(
        _market(mid=0.43, spread_cents=0.5, liquidity=200_000.0),
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=500_000.0,
    )
    assert sig is not None
    assert sig.confidence <= _MAX_MICROSTRUCTURE_CONFIDENCE


# ---------------- Components are exposed ----------------


@pytest.mark.asyncio
async def test_components_dict_is_populated(temp_db):
    mids = [0.50 + 0.002 * i for i in range(10)]
    await _seed_ticks("mkt-1", mids)
    sig = await score_microstructure(
        _market(),
        max_spread_cents=3.0,
        min_volume_24h=10_000.0,
        vol_24h=50_000.0,
    )
    assert sig is not None
    expected = {
        "spread_tightness",
        "tick_frequency",
        "volatility_score",
        "drift_magnitude",
        "drift_cents",
        "liquidity_score",
    }
    assert expected <= set(sig.components.keys())
    # Every score component is bounded [0,1] except drift_cents (raw).
    for key, val in sig.components.items():
        if key == "drift_cents":
            continue
        assert 0.0 <= val <= 1.0, f"{key}={val}"
