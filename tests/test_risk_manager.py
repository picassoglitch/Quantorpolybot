"""Shadow risk manager trigger tests.

Exercises each of the five circuit breakers via ``check_once``:
  1. Lane daily drawdown -> pause that lane
  2. Portfolio daily drawdown -> pause all lanes
  3. Rolling win-rate alert (no pause, just alert flag)
  4. Concentration (>3 open on one market) -> block set
  5. Feed staleness -> pause event lane
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.execution import allocator, risk_manager
from core.execution.risk_manager import ShadowRiskManager
from core.utils import db as db_module
from core.utils.db import execute


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "risk.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    # Reset the module-level blocked-markets set between tests.
    risk_manager._BLOCKED_MARKETS.clear()
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


async def _insert_closed_position(
    *,
    strategy: str,
    market_id: str,
    realized_pnl: float,
    close_ts: float | None = None,
) -> int:
    return await execute(
        """INSERT INTO shadow_positions
           (strategy, market_id, token_id, side, entry_price, size_usd,
            size_shares, entry_ts, status, close_ts, close_reason,
            realized_pnl_usd, unrealized_pnl_usd)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            strategy,
            market_id,
            "tok",
            "BUY",
            0.50,
            100.0,
            200.0,
            (close_ts or time.time()) - 60,
            "CLOSED",
            close_ts or time.time(),
            "test",
            realized_pnl,
            0.0,
        ),
    )


async def _insert_open_position(*, strategy: str, market_id: str) -> int:
    return await execute(
        """INSERT INTO shadow_positions
           (strategy, market_id, token_id, side, entry_price, size_usd,
            size_shares, entry_ts, status, unrealized_pnl_usd)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (strategy, market_id, "tok", "BUY", 0.50, 50.0, 100.0, time.time(), "OPEN", 0.0),
    )


async def _fresh_feed() -> None:
    """Insert a recent feed item so the staleness trigger stays silent
    while we exercise unrelated breakers."""
    await execute(
        "INSERT INTO feed_items (url_hash, source, title, ingested_at) VALUES (?,?,?,?)",
        (f"fresh-{time.time()}", "rss", "fresh", time.time()),
    )


@pytest.mark.asyncio
async def test_lane_daily_drawdown_pauses_that_lane():
    """Default config: lane_dd 5%. Event lane budget $3000 -> $150 loss triggers."""
    await allocator.init_lane_capital()
    await _fresh_feed()
    # -$200 on event_sniping > 5% of $3000.
    await _insert_closed_position(
        strategy="event_sniping", market_id="m1", realized_pnl=-200.0,
    )
    mgr = ShadowRiskManager()
    result = await mgr.check_once()
    assert "event_sniping" in result["lane_drawdown_pauses"]
    state = await allocator.get_state("event_sniping")
    assert state.is_paused
    # Other lanes untouched.
    scalp = await allocator.get_state("scalping")
    assert not scalp.is_paused


@pytest.mark.asyncio
async def test_portfolio_drawdown_pauses_all_lanes():
    """Default config: portfolio_dd 3%. Total shadow $10k -> $300 loss triggers."""
    await allocator.init_lane_capital()
    # Spread -$400 across lanes so no individual lane triggers at 5%.
    # scalping: 5% of $6000 = $300 -> -$250 safe
    # event:    5% of $3000 = $150 -> -$100 safe
    # longshot: 5% of $1000 = $50  -> -$50 right at edge; use -$40
    await _insert_closed_position(strategy="scalping", market_id="a", realized_pnl=-250.0)
    await _insert_closed_position(strategy="event_sniping", market_id="b", realized_pnl=-100.0)
    await _insert_closed_position(strategy="longshot", market_id="c", realized_pnl=-40.0)
    # Total -$390 > 3% of $10k
    mgr = ShadowRiskManager()
    result = await mgr.check_once()
    assert result["portfolio_pause"] is True
    for lane in allocator.LANES:
        state = await allocator.get_state(lane)
        assert state.is_paused


@pytest.mark.asyncio
async def test_low_rolling_win_rate_raises_alert_without_pausing():
    """Under 40% win rate fires an alert but doesn't pause the lane —
    this is signal, not an automatic stop."""
    await allocator.init_lane_capital()
    # Isolate: keep feeds fresh so we only exercise the win-rate trigger.
    await execute(
        "INSERT INTO feed_items (url_hash, source, title, ingested_at) VALUES (?,?,?,?)",
        ("wr-fresh", "rss", "keep feed fresh", time.time()),
    )
    # 10 bets, 2 wins, 8 losses (all tiny so we don't trigger drawdown).
    for i in range(2):
        await _insert_closed_position(
            strategy="scalping", market_id=f"w{i}", realized_pnl=1.0,
        )
    for i in range(8):
        await _insert_closed_position(
            strategy="scalping", market_id=f"l{i}", realized_pnl=-1.0,
        )
    mgr = ShadowRiskManager()
    result = await mgr.check_once()
    assert result["win_rate_alert"] is True
    # No lane paused by this trigger alone.
    for lane in allocator.LANES:
        assert not (await allocator.get_state(lane)).is_paused


@pytest.mark.asyncio
async def test_concentration_blocks_market_with_more_than_three_open():
    """Four open positions on one market across lanes -> blocked."""
    await allocator.init_lane_capital()
    for lane in ("scalping", "event_sniping", "longshot", "scalping"):
        await _insert_open_position(strategy=lane, market_id="hot-market")
    # Another market, only 2 positions -> not blocked.
    await _insert_open_position(strategy="scalping", market_id="ok-market")
    await _insert_open_position(strategy="longshot", market_id="ok-market")

    mgr = ShadowRiskManager()
    await mgr.check_once()
    blocked = risk_manager.concentration_blocked()
    assert "hot-market" in blocked
    assert "ok-market" not in blocked


@pytest.mark.asyncio
async def test_feed_staleness_pauses_event_lane():
    """No feed items at all -> age is infinite -> event lane paused."""
    await allocator.init_lane_capital()
    mgr = ShadowRiskManager()
    result = await mgr.check_once()
    assert result["feed_stale"] is True
    event_state = await allocator.get_state("event_sniping")
    assert event_state.is_paused
    # Scalping + longshot not punished by feed staleness.
    assert not (await allocator.get_state("scalping")).is_paused
    assert not (await allocator.get_state("longshot")).is_paused


@pytest.mark.asyncio
async def test_fresh_feed_does_not_pause_event_lane():
    """A recent feed item keeps the event lane running."""
    await allocator.init_lane_capital()
    await execute(
        """INSERT INTO feed_items (url_hash, source, title, ingested_at)
           VALUES (?,?,?,?)""",
        ("h1", "rss", "breaking news", time.time() - 30),
    )
    mgr = ShadowRiskManager()
    result = await mgr.check_once()
    assert result["feed_stale"] is False
    assert not (await allocator.get_state("event_sniping")).is_paused


@pytest.mark.asyncio
async def test_check_once_clears_stale_blocks_when_concentration_drops():
    """If a market was blocked and positions close, the next tick unblocks it."""
    await allocator.init_lane_capital()
    for lane in ("scalping", "event_sniping", "longshot", "scalping"):
        await _insert_open_position(strategy=lane, market_id="hot-market")
    mgr = ShadowRiskManager()
    await mgr.check_once()
    assert "hot-market" in risk_manager.concentration_blocked()
    # Close three of the four so only 1 remains open.
    await execute(
        "UPDATE shadow_positions SET status='CLOSED', close_ts=? WHERE market_id=? AND id <= ?",
        (time.time(), "hot-market", 3),
    )
    await mgr.check_once()
    assert "hot-market" not in risk_manager.concentration_blocked()
