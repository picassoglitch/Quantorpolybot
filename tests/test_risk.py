"""Risk engine integration tests against an in-memory database."""

from __future__ import annotations

import asyncio
import json
import pathlib
import time

import pytest

from core.markets.cache import Market
from core.risk.rules import RiskEngine, RiskRejection
from core.utils import config as config_module
from core.utils import db as db_module


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the global DB at a temp file and reset the cached path."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _market(**overrides) -> Market:
    base = dict(
        market_id="m1",
        question="Will it rain?",
        slug="rain",
        category="weather",
        active=True,
        close_time="",
        token_ids=["tok1", "tok2"],
        best_bid=0.49,
        best_ask=0.51,
        last_price=0.50,
        liquidity=10_000.0,
        updated_at=time.time(),
    )
    base.update(overrides)
    return Market(**base)


@pytest.mark.asyncio
async def test_risk_blocks_stale_price():
    risk = RiskEngine()
    market = _market(updated_at=time.time() - 999)
    with pytest.raises(RiskRejection):
        await risk.evaluate(market, "BUY", implied_prob=0.7, confidence=0.9)


@pytest.mark.asyncio
async def test_risk_blocks_low_liquidity():
    risk = RiskEngine()
    market = _market(liquidity=10.0)
    with pytest.raises(RiskRejection):
        await risk.evaluate(market, "BUY", implied_prob=0.7, confidence=0.9)


@pytest.mark.asyncio
async def test_risk_blocks_wide_spread():
    risk = RiskEngine()
    market = _market(best_bid=0.10, best_ask=0.90)
    with pytest.raises(RiskRejection):
        await risk.evaluate(market, "BUY", implied_prob=0.7, confidence=0.9)


@pytest.mark.asyncio
async def test_risk_approves_clean_signal():
    risk = RiskEngine()
    market = _market()
    decision = await risk.evaluate(market, "BUY", implied_prob=0.65, confidence=0.9)
    assert decision.size_usd > 0
    assert 0.01 < decision.target_price < 0.99


@pytest.mark.asyncio
async def test_risk_passes_low_price_market_with_tight_absolute_spread():
    """A longshot-style market at mid=$0.0035 with a $0.003 absolute spread
    should PASS even though the relative spread is ~86%. Prior behaviour
    rejected all low-price markets because only the relative check existed.
    """
    risk = RiskEngine()
    market = _market(best_bid=0.002, best_ask=0.005, last_price=0.0035)
    # liquidity already 10_000 in the helper, comfortably above the floor.
    decision = await risk.evaluate(market, "BUY", implied_prob=0.20, confidence=0.9)
    assert decision.size_usd > 0


@pytest.mark.asyncio
async def test_risk_rejects_low_price_market_with_wide_absolute_spread():
    """Low-price markets still get blocked when the ABSOLUTE spread is
    wider than the configured ceiling (0.03 by default)."""
    risk = RiskEngine()
    market = _market(best_bid=0.01, best_ask=0.10, last_price=0.04)
    with pytest.raises(RiskRejection):
        await risk.evaluate(market, "BUY", implied_prob=0.5, confidence=0.9)
