"""Shadow execution engine unit tests.

Covers: open_position audit trail, global ceiling rejection, PnL math
(BUY and SELL), conviction trajectory append + max-points trim,
conviction_is_stable sliding-window logic, close_position capital release.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.execution import allocator, shadow
from core.markets.cache import Market
from core.utils import db as db_module
from core.utils.prices import PriceSnapshot


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shadow.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _market(best_bid: float = 0.48, best_ask: float = 0.52) -> Market:
    return Market(
        market_id="m-shadow-1",
        question="Will it happen?",
        slug="happen",
        category="politics",
        active=True,
        close_time="",
        token_ids=["yes-tok", "no-tok"],
        best_bid=best_bid,
        best_ask=best_ask,
        last_price=(best_bid + best_ask) / 2,
        liquidity=20_000.0,
        updated_at=time.time(),
    )


def _snapshot(bid: float = 0.48, ask: float = 0.52) -> PriceSnapshot:
    return PriceSnapshot(
        token_id="yes-tok",
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        ts=time.time(),
        source="ticks",
    )


@pytest.mark.asyncio
async def test_open_position_records_full_audit_trail():
    await allocator.init_lane_capital()
    await allocator.reserve("scalping", 75.0)

    pos_id = await shadow.open_position(
        strategy="scalping",
        market=_market(),
        side="BUY",
        snapshot=_snapshot(0.48, 0.52),
        size_usd=75.0,
        true_prob=0.62,
        confidence=0.78,
        entry_reason="test entry",
        evidence_ids=[1, 2, 3],
        evidence_snapshot={"note": "smoke test"},
        entry_latency_ms=45.0,
    )
    assert pos_id is not None
    positions = await shadow.open_positions_for("scalping")
    assert len(positions) == 1
    p = positions[0]
    assert p.strategy == "scalping"
    assert p.side == "BUY"
    assert p.entry_price == pytest.approx(0.52)  # filled at ask
    assert p.size_usd == pytest.approx(75.0)
    assert p.size_shares == pytest.approx(75.0 / 0.52, rel=1e-3)
    assert p.cited_evidence_ids == [1, 2, 3]
    assert p.true_prob_entry == pytest.approx(0.62)
    assert p.confidence_entry == pytest.approx(0.78)
    assert p.entry_latency_ms == pytest.approx(45.0)
    # Conviction trajectory seeded with the first point.
    assert len(p.conviction_trajectory) == 1


@pytest.mark.asyncio
async def test_open_position_sells_at_bid():
    """SELL (BUY-NO direction) should fill at the bid, not the ask."""
    await allocator.init_lane_capital()
    await allocator.reserve("scalping", 75.0)
    await shadow.open_position(
        strategy="scalping",
        market=_market(0.40, 0.44),
        side="SELL",
        snapshot=_snapshot(0.40, 0.44),
        size_usd=75.0,
        true_prob=0.30,
        confidence=0.80,
        entry_reason="sell test",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=10.0,
    )
    p = (await shadow.open_positions_for("scalping"))[0]
    assert p.side == "SELL"
    assert p.entry_price == pytest.approx(0.40)  # filled at bid


@pytest.mark.asyncio
async def test_open_position_rejected_above_global_per_trade_ceiling(monkeypatch):
    """Belt-and-suspenders: a $600 size must be rejected when the global
    risk.max_position_usd is $500, even if the lane approved it."""
    await allocator.init_lane_capital()
    # Allocator doesn't know about the global ceiling — reserve succeeds.
    # We bypass reserve and hand the ceiling an over-size bet directly.
    pos_id = await shadow.open_position(
        strategy="scalping",
        market=_market(),
        side="BUY",
        snapshot=_snapshot(),
        size_usd=600.0,  # > $500 per-trade cap
        true_prob=0.60,
        confidence=0.80,
        entry_reason="should be blocked",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=5.0,
    )
    assert pos_id is None
    assert await shadow.count_open("scalping") == 0


@pytest.mark.asyncio
async def test_open_position_rejected_with_bad_price():
    """entry_price <= 0 or >= 1 is nonsense — reject rather than record."""
    await allocator.init_lane_capital()
    # ask=0 -> reject
    pos_id = await shadow.open_position(
        strategy="scalping",
        market=_market(),
        side="BUY",
        snapshot=_snapshot(0.0, 0.0),
        size_usd=75.0,
        true_prob=0.60,
        confidence=0.80,
        entry_reason="bad price",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=5.0,
    )
    assert pos_id is None


def test_compute_unrealized_pnl_buy_side():
    # Bought at 0.50, price now 0.55 on 100 shares.
    assert shadow.compute_unrealized_pnl("BUY", 0.50, 0.55, 100.0) == pytest.approx(5.0)
    # Loss.
    assert shadow.compute_unrealized_pnl("BUY", 0.50, 0.45, 100.0) == pytest.approx(-5.0)


def test_compute_unrealized_pnl_sell_side():
    # SELL (short) wins when price drops.
    assert shadow.compute_unrealized_pnl("SELL", 0.50, 0.40, 100.0) == pytest.approx(10.0)
    assert shadow.compute_unrealized_pnl("SELL", 0.50, 0.60, 100.0) == pytest.approx(-10.0)


def test_compute_pnl_pct():
    assert shadow.compute_pnl_pct("BUY", 0.50, 0.55) == pytest.approx(10.0)
    assert shadow.compute_pnl_pct("SELL", 0.50, 0.45) == pytest.approx(10.0)
    assert shadow.compute_pnl_pct("BUY", 0.0, 0.10) == 0.0  # guard against /0


@pytest.mark.asyncio
async def test_update_price_records_unrealized_pnl():
    await allocator.init_lane_capital()
    await allocator.reserve("scalping", 100.0)
    await shadow.open_position(
        strategy="scalping",
        market=_market(0.48, 0.52),
        side="BUY",
        snapshot=_snapshot(0.48, 0.52),
        size_usd=100.0,
        true_prob=0.60,
        confidence=0.80,
        entry_reason="pnl test",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=5.0,
    )
    p = (await shadow.open_positions_for("scalping"))[0]
    # Price moves up: bid now 0.56 -> exit at 0.56.
    pct = await shadow.update_price(p, _snapshot(0.56, 0.60))
    # BUY at 0.52, exit at 0.56 -> (0.56-0.52)/0.52 * 100 ≈ 7.69%
    assert pct == pytest.approx(7.69, abs=0.1)
    p2 = (await shadow.open_positions_for("scalping"))[0]
    assert p2.unrealized_pnl_usd > 0


@pytest.mark.asyncio
async def test_append_conviction_trims_to_max_points():
    await allocator.init_lane_capital()
    await allocator.reserve("scalping", 75.0)
    await shadow.open_position(
        strategy="scalping",
        market=_market(),
        side="BUY",
        snapshot=_snapshot(),
        size_usd=75.0,
        true_prob=0.60,
        confidence=0.80,
        entry_reason="traj test",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=5.0,
    )
    p = (await shadow.open_positions_for("scalping"))[0]
    # Append beyond the cap.
    for i in range(10):
        p = (await shadow.open_positions_for("scalping"))[0]
        await shadow.append_conviction(p, 0.60 + i * 0.001, 0.50, max_points=5)
    p = (await shadow.open_positions_for("scalping"))[0]
    assert len(p.conviction_trajectory) == 5


def test_conviction_is_stable_detects_flat_tail():
    # Last three points within eps=0.02 of each other -> stable.
    traj = [[1, 0.60, 0.5], [2, 0.70, 0.5], [3, 0.71, 0.5], [4, 0.72, 0.5], [5, 0.715, 0.5]]
    assert shadow.conviction_is_stable(traj, eps=0.02) is True


def test_conviction_is_stable_false_when_moving():
    # Latest point jumped by 0.10 -> not stable.
    traj = [[1, 0.50, 0.5], [2, 0.50, 0.5], [3, 0.60, 0.5]]
    assert shadow.conviction_is_stable(traj, eps=0.02) is False


def test_conviction_is_stable_needs_three_points():
    """Fewer than 3 samples -> never stable (can't tell yet)."""
    assert shadow.conviction_is_stable([], eps=0.02) is False
    assert shadow.conviction_is_stable([[1, 0.6, 0.5]], eps=0.02) is False
    assert shadow.conviction_is_stable([[1, 0.6, 0.5], [2, 0.6, 0.5]], eps=0.02) is False


@pytest.mark.asyncio
async def test_close_position_releases_capital_and_records_pnl():
    await allocator.init_lane_capital()
    approved = await allocator.reserve("scalping", 100.0)
    assert approved == 100.0
    state_before = await allocator.get_state("scalping")
    deployed_before = state_before.deployed

    await shadow.open_position(
        strategy="scalping",
        market=_market(0.48, 0.52),
        side="BUY",
        snapshot=_snapshot(0.48, 0.52),
        size_usd=100.0,
        true_prob=0.60,
        confidence=0.80,
        entry_reason="close test",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=5.0,
    )
    p = (await shadow.open_positions_for("scalping"))[0]

    # Exit +10% — bid jumps to 0.572.
    realized = await shadow.close_position(p, _snapshot(0.572, 0.580), reason="take_profit")
    assert realized > 0

    state_after = await allocator.get_state("scalping")
    # Capital released; deployed back down.
    assert state_after.deployed == pytest.approx(deployed_before - 100.0)
    # Winner compounded into total_budget.
    assert state_after.total_budget > state_before.total_budget

    # Position flipped to CLOSED with the right reason.
    open_after = await shadow.open_positions_for("scalping")
    assert len(open_after) == 0


@pytest.mark.asyncio
async def test_lane_metrics_aggregates_closed_positions():
    await allocator.init_lane_capital()
    # Open + close a winner.
    await allocator.reserve("scalping", 100.0)
    await shadow.open_position(
        strategy="scalping",
        market=_market(0.48, 0.52),
        side="BUY",
        snapshot=_snapshot(0.48, 0.52),
        size_usd=100.0,
        true_prob=0.60,
        confidence=0.80,
        entry_reason="winner",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=5.0,
    )
    p = (await shadow.open_positions_for("scalping"))[0]
    await shadow.close_position(p, _snapshot(0.60, 0.62), reason="take_profit")

    m = await shadow.lane_metrics("scalping")
    assert m["closed"] == 1
    assert m["wins"] == 1
    assert m["win_rate"] == 1.0
    assert m["realized_pnl"] > 0
