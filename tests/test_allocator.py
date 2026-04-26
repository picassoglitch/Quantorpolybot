"""Allocator unit tests.

Covers: initial bootstrap + lane budgets, reserve/release accounting,
dynamic-cap clamping (spec requirement #1), pause/unpause semantics,
and the floor below which a slot is skipped entirely.
"""

from __future__ import annotations

import asyncio

import pytest

from core.execution import allocator
from core.utils import db as db_module


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "alloc.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


@pytest.mark.asyncio
async def test_init_creates_three_lanes_with_correct_budgets():
    await allocator.init_lane_capital()
    states = {s.lane: s for s in await allocator.all_states()}
    assert set(states.keys()) == {"scalping", "event_sniping", "longshot"}
    # Default config: $10k * (0.60 / 0.30 / 0.10)
    assert states["scalping"].total_budget == pytest.approx(6000.0)
    assert states["event_sniping"].total_budget == pytest.approx(3000.0)
    assert states["longshot"].total_budget == pytest.approx(1000.0)
    # All lanes start fully available.
    for s in states.values():
        assert s.deployed == 0
        assert s.available == s.total_budget
        assert not s.is_paused


@pytest.mark.asyncio
async def test_reserve_deducts_and_release_restores():
    await allocator.init_lane_capital()
    approved = await allocator.reserve("event_sniping", 100.0)
    assert approved == 100.0
    state = await allocator.get_state("event_sniping")
    assert state.deployed == pytest.approx(100.0)
    assert state.available == pytest.approx(2900.0)

    # Loss -> total_budget shrinks, deployed freed.
    await allocator.release("event_sniping", 100.0, realized_pnl_usd=-20.0)
    state = await allocator.get_state("event_sniping")
    assert state.deployed == pytest.approx(0.0)
    assert state.total_budget == pytest.approx(2980.0)
    assert state.available == pytest.approx(2980.0)


@pytest.mark.asyncio
async def test_reserve_clamps_to_available_not_skip():
    """Spec #1: 'cap dynamically, don't skip'. A $300 request must fill
    at $200 if only $200 is available."""
    await allocator.init_lane_capital()
    # Drain event lane down to $200.
    approved = await allocator.reserve("event_sniping", 2800.0)
    assert approved == 2800.0
    approved = await allocator.reserve("event_sniping", 300.0)
    assert approved == 200.0  # clamped, not skipped
    state = await allocator.get_state("event_sniping")
    assert state.available == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_reserve_returns_none_when_below_floor():
    """Below the $50 floor the slot isn't worth the overhead."""
    await allocator.init_lane_capital()
    await allocator.reserve("longshot", 975.0)  # leaves $25 available
    approved = await allocator.reserve("longshot", 100.0)
    assert approved is None


@pytest.mark.asyncio
async def test_pause_blocks_reserve():
    await allocator.init_lane_capital()
    import time
    await allocator.pause("scalping", until_ts=time.time() + 3600, reason="test")
    approved = await allocator.reserve("scalping", 75.0)
    assert approved is None
    await allocator.unpause("scalping")
    approved = await allocator.reserve("scalping", 75.0)
    assert approved == 75.0


@pytest.mark.asyncio
async def test_lanes_cannot_borrow_from_each_other():
    """Drain scalping to zero; event/longshot must still have full budget."""
    await allocator.init_lane_capital()
    # Drain scalping.
    await allocator.reserve("scalping", 6000.0)
    # Event and longshot untouched.
    e = await allocator.get_state("event_sniping")
    l = await allocator.get_state("longshot")
    assert e.available == pytest.approx(3000.0)
    assert l.available == pytest.approx(1000.0)
    # Next scalping request returns None.
    assert await allocator.reserve("scalping", 75.0) is None


@pytest.mark.asyncio
async def test_pause_all_affects_every_lane():
    await allocator.init_lane_capital()
    await allocator.pause_all(3600, "test")
    for lane in allocator.LANES:
        state = await allocator.get_state(lane)
        assert state.is_paused


def test_clamp_position_size_pure(monkeypatch):
    """The pure math helper — no DB, for fast regression coverage.

    Returns ``(size, skip_reason)`` so callers can distinguish the
    two skip causes (below-floor vs genuine exhaustion) for logging.
    Monkeypatches the floor so the test is independent of whatever
    YAML the repo ships with (user configs freely change it).
    """
    monkeypatch.setattr(
        allocator, "_capital_cfg_for",
        lambda mode: {"min_lane_available_usd": 50.0},
    )
    # Below floor with headroom -> skip with below_min_available.
    size, reason = allocator.clamp_position_size("event_sniping", 100.0, 49.0)
    assert size == 0.0
    assert reason == "below_min_available"
    # Above floor, under available -> requested, no reason.
    size, reason = allocator.clamp_position_size("event_sniping", 100.0, 500.0)
    assert size == 100.0
    assert reason == ""
    # Above available -> clamped to what's left.
    size, reason = allocator.clamp_position_size("event_sniping", 300.0, 200.0)
    assert size == 200.0
    assert reason == ""
    # Exactly floor edge -> dispatches (< is strict).
    size, reason = allocator.clamp_position_size("event_sniping", 100.0, 50.0)
    assert size == 50.0
    assert reason == ""
    # Zero availability -> exhausted, no matter the floor.
    size, reason = allocator.clamp_position_size("event_sniping", 100.0, 0.0)
    assert size == 0.0
    assert reason == "exhausted"
