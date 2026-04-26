"""Lane 5: microscalp — high-frequency price-based mean reversion.

The LLM-driven lanes (scalping / event_sniping / resolution_day / longshot)
all share one weakness: they need a fresh evidence signal before they'll
consider an entry. That's fine on a $1000+ budget where a handful of
conviction bets cover the rent, but on a small budget the user notices
the silence: markets move all day, trades barely trickle in.

This lane fills that gap. It ignores the LLM entirely and trades a
simple pure-price hypothesis: when a Polymarket contract mid has moved
≥ ``min_move_pct`` over the last few minutes *with no linked news*,
that move is likely order-flow noise and will mean-revert. We enter
against the move, size small, hold at most 15 minutes, and exit on
either a tight TP/SL, peak-drawdown trailing, or a hard max-hold timer.

Why it's safe on small budgets:
  * Tiny per-trade size (4-7% of lane) so 12 concurrent entries only
    eat the lane's headroom, not all of it.
  * Zero LLM calls — adds no load to the GPU queue and never blocks on
    model-swap latency.
  * No-news guard: we skip any market that had a linked feed_item in
    the last ``no_news_window_seconds``, so we're not fading informed
    moves we didn't have time to ingest.
  * Hard 15-minute max-hold means capital cycles fast even when every
    exit path fails — no position can tie up lane budget for long.

Budget: reuses the ``scalping`` lane bucket (same trick as
resolution_day), so enabling/disabling doesn't require touching the
60/30/10 allocation split. Concentration checks still apply per market.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from loguru import logger

from core.execution import allocator, shadow
from core.execution.risk_manager import concentration_blocked
from core.markets import cache as market_cache
from core.utils.config import get_config
from core.utils.db import fetch_all
from core.utils.helpers import now_ts, safe_float
from core.utils.prices import (
    current_price,
    days_until_resolve,
    live_orderbook_snapshot,
    volume_24h,
)

# Reuses scalping's budget bucket — see module docstring.
LANE = "scalping"
STRATEGY_TAG = "microscalp"

# Register the shared bucket so allocator.release() credits the scalping
# lane when a microscalp position closes. Without this, every close
# logged "release on unknown lane 'shadow:microscalp'" and the scalping
# bucket's deployed counter drifted up until lane_budget_exhausted.
allocator.register_strategy_lane(STRATEGY_TAG, LANE)

# Per-market rolling price history (ts, mid) for the reversion signal.
# Keyed on market_id. Bounded by the deque maxlen so memory stays flat
# even if the active market set rotates heavily.
_HISTORY_MAXLEN = 128


def _cfg() -> dict[str, Any]:
    return get_config().get("microscalp") or {}


def _directional_move(pos: Any, current_mid: float) -> float:
    """Signed move in our favor, in absolute probability units.

    BUY position wants price up from entry; SELL wants price down.
    Returns a positive number when the trade has moved our way and
    negative when against us — the caller then thresholds on
    ``signal_decay_exit_max_move``. Used by the signal-decay exit
    so we can detect "position is stagnant" without confusing it
    with a small adverse move (which the stop_loss handles).
    """
    entry = float(getattr(pos, "entry_price", 0.0) or 0.0)
    if entry <= 0:
        return 0.0
    delta = current_mid - entry
    return delta if getattr(pos, "side", "BUY") == "BUY" else -delta


class MicroscalpLane:
    component = "strategies.microscalp"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        # market_id -> deque[(ts, mid)]
        self._history: dict[str, deque[tuple[float, float]]] = {}
        # Per-market "don't re-enter until" timestamp. Populated by the
        # monitor loop after each close; consulted by the entry scan.
        # Without this, a stop_loss on market X would immediately re-arm
        # the same fade signal (YES still sits at the post-move mid),
        # and we'd lose on the same market 8-10 times in 5 minutes —
        # exactly the pattern the first run surfaced in the log.
        self._cooldown_until: dict[str, float] = {}

    async def run(self) -> None:
        if not _cfg().get("enabled", False):
            logger.info("[microscalp] lane disabled")
            return
        logger.info("[microscalp] lane started")
        try:
            await asyncio.gather(
                self._entry_loop(),
                self._monitor_loop(),
            )
        except asyncio.CancelledError:
            raise

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    # ---- Entry scan --------------------------------------------------

    async def _entry_loop(self) -> None:
        interval = safe_float(_cfg().get("entry_scan_seconds", 15))
        while not self._stop.is_set():
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[microscalp] entry scan error: {}", e)
            await self._sleep(interval)

    async def scan_once(self) -> int:
        cfg = _cfg()
        now = now_ts()
        window = safe_float(cfg.get("move_window_seconds", 300))
        min_move = safe_float(cfg.get("min_move_pct", 3.0))
        # Absolute-move gate (new). A 3% relative move at mid=0.20 is
        # only 0.6¢ in absolute probability units — below the 2¢ tick
        # that makes the fade round-trip profitable after spread. We
        # require BOTH the relative AND absolute conditions so cheap
        # low-mid markets don't slip through with tiny moves. Set to 0
        # to disable the absolute gate and restore pre-Apr-2026 behavior.
        min_abs_move = safe_float(cfg.get("min_absolute_move_cents", 2.0))
        no_news_window = safe_float(cfg.get("no_news_window_seconds", 600))
        min_mid = safe_float(cfg.get("min_mid", 0.20))
        max_mid = safe_float(cfg.get("max_mid", 0.80))
        max_spread_cents = safe_float(cfg.get("max_spread_cents", 2))
        min_volume = safe_float(cfg.get("min_volume_24h", 500))
        max_concurrent = int(cfg.get("max_concurrent", 12))
        cooldown = safe_float(cfg.get("same_market_cooldown_seconds", 300))
        # Horizon filter (new). Political election markets months out
        # should never be microscalp candidates — we saw multi-week-
        # horizon 2026 Colombian/Peruvian election trades slip into
        # this lane. Days_until_resolve is a float; we convert to hours.
        # Set to 0 to disable.
        max_horizon_hours = safe_float(cfg.get("max_horizon_hours", 72.0))
        # Stale-mid skip (new). If no fresh mid update was appended to
        # the rolling history for this long, the market is dark and
        # whatever move we "see" is an artifact of the scanner re-
        # appending the same value, not a real dislocation. Set to 0
        # to disable.
        max_signal_age_s = safe_float(cfg.get("max_signal_age_seconds", 600.0))

        open_count = await shadow.count_open(STRATEGY_TAG)
        if open_count >= max_concurrent:
            return 0

        state = await allocator.get_state(LANE)
        if state is None or state.is_paused:
            return 0

        candidates = await market_cache.list_active(limit=500)
        blocked = concentration_blocked()
        active_ids: set[str] = set()
        entered = 0

        for market in candidates:
            active_ids.add(market.market_id)
            mid = market.mid
            if mid <= 0:
                continue
            # Update per-market history first so the next scan has one
            # more data point regardless of whether we enter this round.
            hist = self._history.setdefault(
                market.market_id, deque(maxlen=_HISTORY_MAXLEN),
            )
            hist.append((now, mid))
            # Trim anything older than 2x window — keeps the deque
            # bounded even when markets go quiet for long stretches.
            while hist and (now - hist[0][0]) > window * 2:
                hist.popleft()

            if open_count + entered >= max_concurrent:
                continue
            if market.market_id in blocked:
                continue
            # Per-market cooldown after the previous close on this market,
            # regardless of outcome. A winning fade is also suspect for
            # immediate re-entry — if the mid moved back to fair, there's
            # no dislocation left to fade.
            cool_until = self._cooldown_until.get(market.market_id, 0.0)
            if cool_until > now:
                continue
            if await shadow.count_open_for_market_in_lane(
                market.market_id, STRATEGY_TAG,
            ) > 0:
                continue
            if mid < min_mid or mid > max_mid:
                continue
            spread_cents = (market.best_ask - market.best_bid) * 100.0
            if spread_cents <= 0 or spread_cents > max_spread_cents:
                continue

            # Horizon filter: microscalp's thesis is mean-reversion on a
            # 5-15 minute timescale, which has no business holding
            # political election contracts resolving in months. Skip any
            # market whose resolution is further out than the configured
            # horizon. days_until_resolve -> None when close_time is
            # missing/unparseable; treat "unknown" as too-risky for this
            # lane and skip.
            if max_horizon_hours > 0:
                days_left = days_until_resolve(market.close_time)
                if days_left is None or (days_left * 24.0) > max_horizon_hours:
                    continue

            # Stale-mid skip: if the most recent history sample on this
            # market is older than max_signal_age_seconds, the market's
            # quote stream has gone dark. Any "move" we detect against a
            # stale prior is an artifact, not a real dislocation.
            if max_signal_age_s > 0 and hist:
                newest_ts = hist[-1][0]
                if (now - newest_ts) > max_signal_age_s:
                    continue

            # Need at least one history point older than ``window`` seconds
            # to compute the recent move. On fresh markets we'll simply
            # skip until the deque fills.
            prior_mid = None
            for t, m in hist:
                if (now - t) >= window and m > 0:
                    prior_mid = m
                    break
            if prior_mid is None:
                continue
            move_pct = (mid - prior_mid) / prior_mid * 100.0
            if abs(move_pct) < min_move:
                continue
            # Absolute-move gate: relative-% can be misleading at low
            # mid (3% of 0.20 = 0.6¢ which doesn't clear the spread).
            # Require an absolute delta of at least min_abs_move cents.
            move_cents_abs = abs(mid - prior_mid) * 100.0
            if min_abs_move > 0 and move_cents_abs < min_abs_move:
                continue

            # No-news guard: skip any market that had a linked feed item
            # in the last no_news_window_seconds. If there's fresh news,
            # the other lanes should handle it with their LLM context —
            # we don't fade informed moves.
            recent = await fetch_all(
                """SELECT 1 FROM feed_items
                   WHERE meta LIKE ? AND ingested_at >= ?
                   LIMIT 1""",
                (
                    f'%"linked_market_id":%"{market.market_id}"%',
                    now - no_news_window,
                ),
            )
            if recent:
                continue

            vol = await volume_24h(market.market_id)
            if vol < min_volume:
                continue

            # Fade the move: move up -> SELL, move down -> BUY.
            side = "SELL" if move_pct > 0 else "BUY"
            # Scale pseudo-confidence with move magnitude so bigger
            # dislocations get a bigger slice of the sizing ramp inside
            # allocator.compute_position_size. Capped at 0.80 — this
            # isn't a high-conviction lane by design.
            confidence = min(0.80, 0.55 + abs(move_pct) / 25.0)

            wanted = await allocator.compute_position_size(LANE, confidence, cfg)
            approved = await allocator.reserve(LANE, wanted)
            if approved is None:
                continue

            snap = await current_price(
                market.market_id, market.yes_token() or "",
            )
            if snap is None or snap.mid <= 0:
                await allocator.release(LANE, approved, 0.0)
                continue

            pos_id = await shadow.open_position(
                strategy=STRATEGY_TAG,
                market=market,
                side=side,
                snapshot=snap,
                size_usd=approved,
                # Our "model" says the pre-move mid was correct —
                # record that as true_prob for post-hoc learning.
                true_prob=prior_mid,
                confidence=confidence,
                entry_reason=(
                    f"microscalp: move={move_pct:+.1f}% over {window:.0f}s "
                    f"no_news spread={spread_cents:.1f}c"
                ),
                evidence_ids=[],
                evidence_snapshot={
                    "move_pct": round(move_pct, 2),
                    "window_seconds": int(window),
                    "prior_mid": round(prior_mid, 4),
                    "current_mid": round(mid, 4),
                },
                entry_latency_ms=0.0,
                # No LLM client by design — this lane never validates.
            )
            if pos_id is None:
                await allocator.release(LANE, approved, 0.0)
                continue
            entered += 1

        # Garbage-collect history entries for markets that fell out of
        # the active set. Otherwise the dict grows unboundedly over the
        # lifetime of the process. Do the same sweep on cooldown entries
        # whose timestamp has already elapsed.
        for mid_id in list(self._history.keys()):
            if mid_id not in active_ids:
                del self._history[mid_id]
        for mid_id in list(self._cooldown_until.keys()):
            if self._cooldown_until[mid_id] <= now:
                del self._cooldown_until[mid_id]

        if entered:
            logger.info("[microscalp] entered {} positions", entered)
        return entered

    # ---- Monitor / exit ----------------------------------------------

    async def _monitor_loop(self) -> None:
        interval = safe_float(_cfg().get("monitor_seconds", 10))
        while not self._stop.is_set():
            try:
                await self.monitor_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[microscalp] monitor error: {}", e)
            await self._sleep(interval)

    async def monitor_once(self) -> int:
        cfg = _cfg()
        tp_pct = safe_float(cfg.get("take_profit_pct", 5))
        sl_pct = safe_float(cfg.get("stop_loss_pct", 5))
        trail_arm = safe_float(cfg.get("trailing_activate_pct", 0))
        trail_dd = safe_float(cfg.get("trailing_drawdown_pct", 0))
        max_hold = safe_float(cfg.get("max_hold_seconds", 900))
        liquidity_cents = safe_float(cfg.get("liquidity_exit_spread_cents", 4))
        cooldown = safe_float(cfg.get("same_market_cooldown_seconds", 300))
        # Signal-decay early exit (new). If a position has aged past
        # ``decay_age_s`` without the price moving at least
        # ``decay_max_move`` in OUR direction, the fade thesis has
        # failed quietly — close at current price and report
        # ``signal_decay`` so analysis can separate these from real
        # ``max_hold`` expiries.  Note: with default ``max_hold=900s``
        # and ``decay_age_s=1800s``, max_hold fires first. Either lower
        # ``signal_decay_exit_age_seconds`` below ``max_hold_seconds``
        # or raise ``max_hold_seconds`` for this exit to ever actually
        # trigger. Set ``decay_age_s=0`` to disable.
        decay_age_s = safe_float(cfg.get("signal_decay_exit_age_seconds", 1800))
        decay_max_move = safe_float(cfg.get("signal_decay_exit_max_move", 0.005))

        positions = await shadow.open_positions_for(STRATEGY_TAG)
        closed = 0
        for pos in positions:
            snap = await current_price(pos.market_id, pos.token_id)
            if snap is None or snap.mid <= 0:
                continue
            pnl_pct = await shadow.update_price(pos, snap)
            reason: str | None = None

            if pnl_pct >= tp_pct:
                reason = f"take_profit {pnl_pct:+.1f}%"
            elif pnl_pct <= -sl_pct:
                reason = f"stop_loss {pnl_pct:+.1f}%"
            elif (
                trail_arm > 0 and trail_dd > 0
                and pos.peak_pnl_pct >= trail_arm
                and (pos.peak_pnl_pct - pnl_pct) >= trail_dd
            ):
                reason = (
                    f"trailing_exit peak={pos.peak_pnl_pct:+.1f}% "
                    f"now={pnl_pct:+.1f}%"
                )
            elif (
                decay_age_s > 0
                and pos.age_seconds >= decay_age_s
                and _directional_move(pos, snap.mid) < decay_max_move
            ):
                # Note the placement: before ``max_hold`` so that when
                # both triggers qualify, we emit the more-informative
                # ``signal_decay`` reason (operator wants to measure
                # these separately vs. raw time expiries).
                reason = (
                    f"signal_decay age={pos.age_seconds:.0f}s "
                    f"move={_directional_move(pos, snap.mid):+.4f}"
                )
            elif pos.age_seconds > max_hold:
                reason = f"max_hold age={pos.age_seconds:.0f}s"
            else:
                live = await live_orderbook_snapshot(
                    pos.market_id, pos.token_id,
                )
                if live and live.spread_cents > liquidity_cents:
                    reason = f"liquidity_exit spread={live.spread_cents:.1f}c"

            if reason is not None:
                await shadow.close_position(pos, snap, reason)
                # Arm the per-market cooldown so the next scan doesn't
                # immediately re-enter the same fade signal.
                self._cooldown_until[pos.market_id] = now_ts() + cooldown
                closed += 1
        return closed
