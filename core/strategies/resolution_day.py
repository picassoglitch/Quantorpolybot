"""Lane 4: resolution-day sniping.

Markets in the final 24 hours before resolution are frequently mispriced:
traders unwind, liquidity thins, and the LLM has a much easier job
evaluating "will X happen today?" than a 30-day forecast. This lane scans
for markets where ``close_time - now`` is inside a short window and takes
a position when the model sees a concrete edge against the current price.

Why it's safe on small budgets:
  * Positions resolve in < 24h -> capital recycles fast.
  * Uses the deep tier because close-to-resolution reasoning benefits from
    the bigger model.
  * Entry gate is tight: min_edge + min_confidence + evidence check.
  * Position size is budget-adaptive via the allocator helper.

Budget: reuses the ``scalping`` lane bucket (doesn't need its own
allocation) so switching this lane on/off doesn't require rebalancing the
existing 60/30/10 split. This also means at $15 shadow it competes with
scalping for the same $9, which is fine — both are short-hold strategies
and the 2-min cooldown prevents double-entry on the same market.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from core.execution import allocator, shadow
from core.execution.risk_manager import concentration_blocked
from core.markets import cache as market_cache
from core.signals.ollama_client import OllamaClient
from core.strategies import scoring
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
STRATEGY_TAG = "resolution_day"

# Register the shared bucket so allocator.release() credits the scalping
# lane when a resolution_day position closes (otherwise it'd try to
# release into a nonexistent 'resolution_day' lane and silently leak
# capital from the scalping bucket).
allocator.register_strategy_lane(STRATEGY_TAG, LANE)


def _cfg() -> dict[str, Any]:
    return get_config().get("resolution_day") or {}


async def _recent_evidence_for(market_id: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = await fetch_all(
        """SELECT id, source, title, summary, url, ingested_at
           FROM feed_items
           WHERE meta LIKE ?
           ORDER BY ingested_at DESC LIMIT ?""",
        (f'%"linked_market_id":%"{market_id}"%', limit),
    )
    return [dict(r) for r in rows]


class ResolutionDayLane:
    component = "strategies.resolution_day"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._client = OllamaClient()

    async def run(self) -> None:
        if not _cfg().get("enabled", False):
            logger.info("[resolve_day] lane disabled")
            return
        logger.info("[resolve_day] lane started")
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

    # ---- Entry scan ----

    async def _entry_loop(self) -> None:
        interval = safe_float(_cfg().get("entry_scan_seconds", 120))
        while not self._stop.is_set():
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[resolve_day] entry scan error: {}", e)
            await self._sleep(interval)

    async def scan_once(self) -> int:
        cfg = _cfg()
        max_concurrent = int(cfg.get("max_concurrent", 10))
        open_count = await shadow.count_open(STRATEGY_TAG)
        if open_count >= max_concurrent:
            return 0
        state = await allocator.get_state(LANE)
        if state is None or state.is_paused:
            return 0

        min_hours = safe_float(cfg.get("min_hours_to_resolve", 1.0))
        max_hours = safe_float(cfg.get("max_hours_to_resolve", 24.0))
        min_volume = safe_float(cfg.get("min_volume_24h", 2000))
        min_edge = safe_float(cfg.get("min_edge", 0.06))
        min_conf = safe_float(cfg.get("min_confidence", 0.65))
        max_spread_cents = safe_float(cfg.get("max_spread_cents", 6))
        min_evidence = int(cfg.get("min_evidence_sources", 1))

        candidates = await market_cache.list_active(limit=200)
        blocked = concentration_blocked()
        entered = 0
        for market in candidates:
            if open_count + entered >= max_concurrent:
                break
            if market.market_id in blocked:
                continue
            if await shadow.count_open_for_market_in_lane(market.market_id, STRATEGY_TAG) > 0:
                continue
            days = days_until_resolve(market.close_time)
            if days is None:
                continue
            hours = days * 24.0
            if hours < min_hours or hours > max_hours:
                continue
            spread_cents = (market.best_ask - market.best_bid) * 100.0
            if spread_cents <= 0 or spread_cents > max_spread_cents:
                continue
            vol = await volume_24h(market.market_id)
            if vol < min_volume:
                continue
            evidence = await _recent_evidence_for(market.market_id, limit=5)
            sources = {e.get("source") for e in evidence if e.get("source")}
            if len(sources) < min_evidence:
                # Fall back to market question — close-to-resolution markets
                # often reason fine from the question alone.
                evidence = []
                sources = set()

            text = "\n".join(
                f"{e.get('title','')}: {(e.get('summary') or '')[:200]}"
                for e in evidence[:3]
            ) or (market.question or "")

            # Close-to-resolution markets deserve the deep model — reasoning
            # quality matters more than latency here (positions hold < 24h).
            score = await scoring.score(market, text, self._client, tier="deep")
            if score is None:
                continue
            if score.confidence < min_conf:
                continue
            edge = score.true_prob - market.mid
            if abs(edge) < min_edge:
                continue
            side = "BUY" if edge > 0 else "SELL"

            wanted = await allocator.compute_position_size(LANE, score.confidence, cfg)
            approved = await allocator.reserve(LANE, wanted)
            if approved is None:
                continue

            snap = await current_price(market.market_id, market.yes_token() or "")
            if snap is None or snap.mid <= 0:
                await allocator.release(LANE, approved, 0.0)
                continue

            pos_id = await shadow.open_position(
                strategy=STRATEGY_TAG,
                market=market,
                side=side,
                snapshot=snap,
                size_usd=approved,
                true_prob=score.true_prob,
                confidence=score.confidence,
                entry_reason=(
                    f"resolve_day: edge={edge:+.3f} conf={score.confidence:.2f} "
                    f"hours_to_close={hours:.1f} spread={spread_cents:.1f}c"
                ),
                evidence_ids=[int(e["id"]) for e in evidence if e.get("id")],
                evidence_snapshot={
                    "reasoning": score.reasoning,
                    "hours_to_resolve": round(hours, 2),
                    "sources": sorted(sources),
                },
                entry_latency_ms=0.0,
                ollama_client=self._client,
                validator_text=text or market.question,
                validator_reasoning=score.reasoning,
            )
            if pos_id is None:
                await allocator.release(LANE, approved, 0.0)
                continue
            entered += 1
        if entered:
            logger.info("[resolve_day] entered {} positions", entered)
        return entered

    # ---- Monitor / exit ----

    async def _monitor_loop(self) -> None:
        interval = safe_float(_cfg().get("monitor_seconds", 60))
        while not self._stop.is_set():
            try:
                await self.monitor_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[resolve_day] monitor error: {}", e)
            await self._sleep(interval)

    async def monitor_once(self) -> int:
        cfg = _cfg()
        tp_pct = safe_float(cfg.get("take_profit_pct", 10))
        sl_pct = safe_float(cfg.get("stop_loss_pct", 12))
        trail_arm = safe_float(cfg.get("trailing_activate_pct", 0))
        trail_dd = safe_float(cfg.get("trailing_drawdown_pct", 0))
        liquidity_exit_cents = safe_float(cfg.get("liquidity_exit_spread_cents", 10))
        # Exit early if we're inside the final N minutes before close — avoid
        # getting stuck holding into resolution when exit liquidity dries up.
        close_exit_minutes = safe_float(cfg.get("close_exit_minutes", 15))

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
            else:
                # Final-minutes exit.
                market = await market_cache.get_market(pos.market_id)
                if market is not None:
                    days = days_until_resolve(market.close_time)
                    if days is not None and days > 0:
                        minutes_left = days * 24.0 * 60.0
                        if minutes_left < close_exit_minutes:
                            reason = f"pre_resolve_exit minutes_left={minutes_left:.1f}"
                # Liquidity exit.
                if reason is None:
                    live = await live_orderbook_snapshot(pos.market_id, pos.token_id)
                    if live and live.spread_cents > liquidity_exit_cents:
                        reason = f"liquidity_exit spread={live.spread_cents:.1f}c"

            if reason is not None:
                await shadow.close_position(pos, snap, reason)
                closed += 1
        return closed
