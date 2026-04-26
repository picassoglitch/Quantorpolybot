"""Regression tests for the April 2026 shadow-run hardening commits.

Covers, in the priority order requested by the operator:

1. (must-have) ``clob_client.ensure_ready()`` dispatches the sync build
   to an executor so the event loop doesn't stall on
   ``create_or_derive_api_creds()`` — this was the root cause of the
   100-150 s "Ollama silent" watchdog triggers in the April 2026 soak.
2. (must-have) ``event_sniper`` rejects markets with mid outside the
   configured ``price_range`` (default [0.02, 0.98]) BEFORE spending
   any Ollama cycles or querying volume_24h. The lane previously let
   mid=0.001 trades through via the legacy scoring path that bypasses
   the signal pipeline's plausibility gate.
3. (nice-to-have) ``microscalp`` skips markets whose resolution is
   further out than ``max_horizon_hours`` (default 72 h). This lane's
   thesis is 5-15 min mean reversion; multi-month election contracts
   have no business here.
4. (nice-to-have) Real-mode daily-loss cap: when realized P&L on
   ``is_real=1`` positions breaches ``-limits.daily_loss_cap_usd``,
   every real lane pauses for 24 h. Shadow lanes stay untouched so the
   bot keeps learning while the real book cools off.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.execution import allocator, clob_client, risk_manager
from core.execution.risk_manager import ShadowRiskManager
from core.markets.cache import Market
from core.strategies import event_sniper, microscalp
from core.utils import db as db_module
from core.utils.db import execute


# ---------------------------------------------------------------------------
# Shared fixture — fresh temp DB + clean module-level risk_manager state.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shadow_fixes.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    risk_manager._BLOCKED_MARKETS.clear()
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _mk_market(mid: float, *, market_id: str = "m-t") -> Market:
    half_spread = 0.001
    return Market(
        market_id=market_id,
        question="Will Panama win the World Cup?",
        slug="panama-wc",
        category="sports",
        active=True,
        close_time="",
        token_ids=["t-yes", "t-no"],
        best_bid=max(0.0, mid - half_spread),
        best_ask=mid + half_spread,
        last_price=mid,
        liquidity=50_000.0,
        updated_at=time.time(),
    )


def _wrap_async(value):
    """Adapter so ``monkeypatch.setattr(lane, '_match_markets', ...)``
    can replace an ``async def`` method with a lambda returning a
    coroutine that yields ``value``."""
    async def _coro(*_a, **_kw):
        return value
    return _coro


# ===========================================================================
# 1. clob_client.ensure_ready() — async-safe; must not block event loop.
# ===========================================================================
@pytest.mark.asyncio
async def test_ensure_ready_does_not_block_event_loop(monkeypatch):
    """Simulate a 500 ms blocking build. While ``ensure_ready`` is
    pending, a companion heartbeat coroutine must keep ticking. If the
    build ran inline on the event loop (pre-fix behavior) the heartbeat
    would stall for the full 500 ms; dispatched off-loop, it keeps
    ticking at ~10 ms granularity."""
    # Force a fresh build path — reset caches.
    monkeypatch.setattr(clob_client, "_client", None)
    monkeypatch.setattr(clob_client, "_last_fail_ts", 0.0)
    monkeypatch.setattr(clob_client, "_last_fail_reason", "")

    def fake_build():
        # Synchronous sleep simulates the HTTP round-trip to
        # clob.polymarket.com inside create_or_derive_api_creds.
        time.sleep(0.5)
        return None  # "creds missing" branch; _client stays None

    monkeypatch.setattr(clob_client, "_build_client", fake_build)

    ticks = 0
    start = time.monotonic()

    async def heartbeat():
        nonlocal ticks
        deadline = start + 0.6
        while time.monotonic() < deadline:
            ticks += 1
            await asyncio.sleep(0.01)

    ensure_task = asyncio.create_task(clob_client.ensure_ready())
    beat_task = asyncio.create_task(heartbeat())
    await asyncio.gather(ensure_task, beat_task)

    # Under off-loop dispatch, the heartbeat ticks ~50-60 times in
    # 600 ms. Under the pre-fix inline build, it would tick <5 times.
    # 25 is a comfortable middle threshold that tolerates slow CI.
    assert ticks >= 25, (
        f"event loop was starved during ensure_ready (ticks={ticks}); "
        "build did not dispatch to executor"
    )
    assert ensure_task.result() is False  # fake_build returned None


# ===========================================================================
# 2. event_sniper tail-price gate rejects mid=0.001 before volume_24h.
# ===========================================================================
@pytest.mark.asyncio
async def test_event_sniper_tail_price_gate_rejects_extreme_mid(monkeypatch):
    """At mid=0.001 the lane must bail BEFORE calling ``volume_24h`` or
    the Ollama scorer. We assert via a tripwire on volume_24h — it
    lives right after the tail gate in the scan path."""
    await allocator.init_lane_capital()

    lane = event_sniper.EventSniperLane()
    monkeypatch.setattr(
        lane, "_match_markets",
        _wrap_async([_mk_market(0.001, market_id="m-tail")]),
    )

    volume_calls: list[str] = []

    async def tripwire_volume(mid: str) -> float:
        volume_calls.append(mid)
        return 10_000.0

    monkeypatch.setattr(event_sniper, "volume_24h", tripwire_volume)

    item = {
        "id": 1,
        "title": "Panama stun Brazil 3-0",
        "summary": "Shock result",
        "ingested_at": time.time() - 30,  # fresh enough
    }
    await lane._process_item(item)
    assert volume_calls == [], (
        f"tail-price gate did not fire: volume_24h called with "
        f"{volume_calls} for a mid=0.001 market"
    )


@pytest.mark.asyncio
async def test_event_sniper_tail_gate_lets_mid_market_through(monkeypatch):
    """Flip: at mid=0.50 the gate must NOT fire — volume_24h should be
    reached (after which the test can stop caring; we just need proof
    the gate is specific, not blanket)."""
    await allocator.init_lane_capital()

    lane = event_sniper.EventSniperLane()
    monkeypatch.setattr(
        lane, "_match_markets",
        _wrap_async([_mk_market(0.50, market_id="m-mid")]),
    )

    volume_calls: list[str] = []

    async def tripwire_volume(mid: str) -> float:
        volume_calls.append(mid)
        # Return 0 to short-circuit the rest of the pipeline (fails
        # min_volume check) — we only care that we *got* this far.
        return 0.0

    monkeypatch.setattr(event_sniper, "volume_24h", tripwire_volume)

    item = {
        "id": 2,
        "title": "Panama stun Brazil 3-0",
        "summary": "Shock result",
        "ingested_at": time.time() - 30,
    }
    await lane._process_item(item)
    assert volume_calls == ["m-mid"], (
        "gate over-rejected: mid=0.50 market didn't reach volume_24h"
    )


# ===========================================================================
# 3. microscalp horizon filter rejects long-horizon (e.g. 2027 elections).
# ===========================================================================
@pytest.mark.asyncio
async def test_microscalp_horizon_filter_rejects_long_horizon(monkeypatch):
    """A market resolving in 200 days must be skipped before volume_24h
    or any entry logic runs — microscalp's thesis is 5-15 min mean
    reversion, not multi-month capital lockup."""
    await allocator.init_lane_capital()

    long_horizon_market = _mk_market(mid=0.50, market_id="m-election-2027")

    async def fake_list_active(limit: int = 500):
        return [long_horizon_market]

    monkeypatch.setattr(
        microscalp.market_cache, "list_active", fake_list_active,
    )
    # 200 days ≫ 72 h / 24 ≈ 3 days → must reject.
    monkeypatch.setattr(microscalp, "days_until_resolve", lambda _ct: 200.0)

    volume_calls: list[str] = []

    async def tripwire_volume(mid: str) -> float:
        volume_calls.append(mid)
        return 10_000.0

    monkeypatch.setattr(microscalp, "volume_24h", tripwire_volume)

    lane = microscalp.MicroscalpLane()
    # Prime the rolling history with two samples spanning the move
    # window so the scan would otherwise have a prior_mid to compare
    # against — this ensures we're specifically testing the horizon
    # gate, not tripping on "no history yet".
    now = time.time()
    from collections import deque
    lane._history[long_horizon_market.market_id] = deque(
        [(now - 400, 0.45), (now - 10, 0.50)], maxlen=128,
    )

    entered = await lane.scan_once()
    assert entered == 0
    assert volume_calls == [], (
        f"horizon filter did not reject 200-day market: volume_24h was "
        f"reached with {volume_calls}"
    )


@pytest.mark.asyncio
async def test_microscalp_horizon_filter_rejects_unknown_close_time(monkeypatch):
    """``days_until_resolve`` returns None on unparseable close_time.
    The gate must treat that as 'too-risky, skip' rather than
    'unlimited horizon, enter'."""
    await allocator.init_lane_capital()
    mkt = _mk_market(mid=0.50, market_id="m-no-close")

    async def fake_list_active(limit: int = 500):
        return [mkt]

    monkeypatch.setattr(
        microscalp.market_cache, "list_active", fake_list_active,
    )
    monkeypatch.setattr(microscalp, "days_until_resolve", lambda _ct: None)

    volume_calls: list[str] = []

    async def tripwire_volume(mid: str) -> float:
        volume_calls.append(mid)
        return 10_000.0

    monkeypatch.setattr(microscalp, "volume_24h", tripwire_volume)

    lane = microscalp.MicroscalpLane()
    await lane.scan_once()
    assert volume_calls == []


# ===========================================================================
# 4. Real-mode daily-loss cap — pauses all real lanes on breach.
# ===========================================================================
async def _insert_real_closed_loss(
    *, strategy: str, market_id: str, pnl: float,
) -> int:
    return await execute(
        """INSERT INTO shadow_positions
           (strategy, market_id, token_id, side, entry_price, size_usd,
            size_shares, entry_ts, status, close_ts, close_reason,
            realized_pnl_usd, unrealized_pnl_usd, is_real)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            strategy, market_id, "tok-real", "BUY", 0.50, 10.0, 20.0,
            time.time() - 60, "CLOSED", time.time() - 30, "test",
            pnl, 0.0, 1,
        ),
    )


@pytest.mark.asyncio
async def test_real_daily_loss_cap_pauses_all_real_lanes():
    """Default ``limits.daily_loss_cap_usd`` is 1.50 (config.yaml). Two
    real closed losses totalling -2.10 must trip the cap and pause
    every real lane for 24 h.

    NOTE: we don't assert the shadow side stays untouched here. The
    shadow circuit breakers (``lane_daily_pnl``, ``portfolio_daily_pnl``)
    predate the is_real split and sum across both modes — so real
    losses also register as shadow losses. That's a separate, pre-
    existing leak; cleaning it up is out of scope for the April-2026
    hardening pass. The cap's own behavior (pause real lanes, emit
    ``real_loss_cap_pause``) is what this test guards.
    """
    await allocator.init_lane_capital()
    # Arm the real scalping lane so the "any_real_active" check sees a
    # budgeted, non-paused target. Real budgets ship at 0; test-only.
    await allocator.set_lane_budget("scalping", 20.0, mode="real")

    # Fresh feed so feed-staleness doesn't add noise to the logs.
    await execute(
        "INSERT INTO feed_items (url_hash, source, title, ingested_at) "
        "VALUES (?,?,?,?)",
        ("cap-fresh", "rss", "fresh", time.time()),
    )

    # Use a bespoke strategy tag so these positions don't accidentally
    # end up attributed to a real lane bucket — the cap's own path
    # computes PnL via ``real_portfolio_daily_pnl`` (sums across all
    # is_real=1 regardless of strategy), so the tag doesn't affect the
    # breach condition.
    await _insert_real_closed_loss(strategy="real_test", market_id="a", pnl=-1.20)
    await _insert_real_closed_loss(strategy="real_test", market_id="b", pnl=-0.90)

    mgr = ShadowRiskManager()
    result = await mgr.check_once()
    assert result["real_loss_cap_pause"] is True

    # Every real lane paused for ~24 h.
    for lane in allocator.LANES:
        st = await allocator.get_state(lane, mode="real")
        assert st is not None, f"real lane {lane} missing"
        assert st.is_paused, (
            f"real lane {lane} not paused after loss-cap breach"
        )


@pytest.mark.asyncio
async def test_real_daily_loss_cap_noop_when_no_real_budget():
    """With all real budgets at 0 (ship config), a real loss must NOT
    pause the real lanes — the "any active real lane" check requires
    ``total_budget > 0`` so zero-budget lanes don't count as pause-
    able. Forward-looking guardrail: stays a no-op in pure shadow
    mode until the operator flips budgets on."""
    await allocator.init_lane_capital()
    # Don't arm any real lane — all real budgets stay at 0.

    await execute(
        "INSERT INTO feed_items (url_hash, source, title, ingested_at) "
        "VALUES (?,?,?,?)",
        ("cap-quiet", "rss", "fresh", time.time()),
    )
    await _insert_real_closed_loss(strategy="real_test", market_id="a", pnl=-5.00)

    mgr = ShadowRiskManager()
    result = await mgr.check_once()
    assert result["real_loss_cap_pause"] is False
    # All real lanes stay unpaused (budget 0 → nothing to pause).
    for lane in allocator.LANES:
        st = await allocator.get_state(lane, mode="real")
        assert st is not None
        assert not st.is_paused, (
            f"real lane {lane} paused with zero budget — should be a "
            "no-op until operator flips budgets on"
        )
