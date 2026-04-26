"""Lane 2: event sniping.

React to breaking news faster than the market reprices. Entry is
cursor-driven over feed_items rather than market-driven — every fresh
item within the freshness window is evaluated for trade-worthiness.
Uses the timeout-bounded scoring path so Ollama stalls can't eat our
edge; falls back to a keyword heuristic with a smaller size when the
LLM misses the window.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from core.execution import allocator, shadow
from core.execution.risk_manager import concentration_blocked
from core.markets import cache as market_cache
from core.markets.cache import Market
from core.signals.candidates import scored_candidates_for
from core.signals.ollama_client import OllamaClient
from core.strategies import heuristic, scoring
from core.strategies.scoring import Score
from core.utils.config import get_config
from core.utils.db import fetch_all, fetch_one
from core.utils.helpers import clamp, now_ts, safe_float
from core.utils.prices import current_price, days_until_resolve, volume_24h

LANE = "event_sniping"


def _cfg() -> dict[str, Any]:
    return get_config().get("event_sniping") or {}


async def _position_size(confidence: float) -> float:
    """Delegate to the budget-adaptive allocator helper. See
    :func:`core.execution.allocator.compute_position_size`."""
    return await allocator.compute_position_size(LANE, confidence, _cfg())


async def _fallback_size(lane_cfg: dict) -> float:
    """Heuristic-fallback size is configured as a fraction of lane so it
    stays sane across budget sizes too. Falls back to the legacy absolute
    value when the pct key is missing."""
    state = await allocator.get_state(LANE)
    total = safe_float(state.total_budget) if state else 0.0
    pct = safe_float(lane_cfg.get("heuristic_fallback_pct", 0.0))
    if pct > 0 and total > 0:
        return round(total * pct, 2)
    return safe_float(lane_cfg.get("heuristic_fallback_size_usd", 1.5))


def _market_moved_recently(market: Market, pct: float, window_s: float = 900) -> bool:
    """Heuristic: compare current mid to last_price. `price_ticks` would
    give a better history, but for the entry gate we're just checking
    'has the market already reacted?' — a basic cache compare is fine."""
    if market.last_price <= 0:
        return False
    delta = abs(market.mid - market.last_price) / max(market.last_price, 1e-6)
    return delta * 100.0 > pct


class EventSniperLane:
    component = "strategies.event_sniper"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._client = OllamaClient()
        self._cursor_id = 0

    async def run(self) -> None:
        if not _cfg().get("enabled", True):
            logger.info("[event] lane disabled")
            return
        await self._init_cursor()
        logger.info("[event] lane started cursor={}", self._cursor_id)
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

    async def _init_cursor(self) -> None:
        row = await fetch_one("SELECT MAX(id) AS m FROM feed_items")
        self._cursor_id = int((row["m"] or 0) if row else 0)

    # ---- Entry (feed-driven) ----

    async def _entry_loop(self) -> None:
        """Poll feed_items rapidly. Event lane needs to react in <60s of
        ingest so we don't batch or sleep long between polls."""
        while not self._stop.is_set():
            try:
                processed = await self._drain_new_items()
                if processed == 0:
                    await self._sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[event] entry loop error: {}", e)
                await self._sleep(5)

    async def _drain_new_items(self) -> int:
        rows = await fetch_all(
            "SELECT * FROM feed_items WHERE id > ? ORDER BY id ASC LIMIT 25",
            (self._cursor_id,),
        )
        if not rows:
            return 0
        for row in rows:
            try:
                await self._process_item(dict(row))
            except Exception as e:
                logger.exception("[event] item {} failed: {}", row["id"], e)
            self._cursor_id = max(self._cursor_id, int(row["id"]))
        return len(rows)

    async def _process_item(self, item: dict[str, Any]) -> None:
        cfg = _cfg()
        fresh_window = safe_float(cfg.get("feed_freshness_minutes", 5)) * 60.0
        ingested_at = safe_float(item.get("ingested_at"))
        age = now_ts() - ingested_at if ingested_at else 9999
        if age > fresh_window:
            return  # old news; not event-lane material

        # Triggered rescore on open positions for matched markets (fast
        # path — don't wait for the 2m cron if a breaking story lands).
        matched_markets = await self._match_markets(item)
        if not matched_markets:
            return

        # Capacity check before spending Ollama cycles.
        state = await allocator.get_state(LANE)
        if state is None or state.is_paused:
            return
        max_concurrent = int(cfg.get("max_concurrent", 15))
        if await shadow.count_open(LANE) >= max_concurrent:
            await self._trigger_rescore_for_held(matched_markets, item)
            return

        min_volume = safe_float(cfg.get("min_volume_24h", 5000))
        min_edge = safe_float(cfg.get("min_edge", 0.10))
        min_conf = safe_float(cfg.get("min_confidence", 0.70))
        unmoved_pct = safe_float(cfg.get("market_unmoved_pct", 3))
        timeout_s = safe_float(cfg.get("heuristic_fallback_timeout_seconds", 10))
        fallback_size = await _fallback_size(cfg)
        max_latency_s = safe_float(cfg.get("max_entry_latency_seconds", 60))
        # Duration filter: event_sniping is meant for catalysts that
        # resolve within days, not long-duration futures. If the market's
        # resolution is further out than this, skip — Ollama routinely
        # hallucinates high edge on FIFA-World-Cup-style dark-horse
        # markets that trade at implied<5% and lock capital for months.
        # Set to 0 or negative to disable.
        max_days_to_resolve = safe_float(cfg.get("max_days_to_resolution", 14))
        # Half-width of the "model hedged" band around 0.5. true_prob
        # inside [0.5 - band, 0.5 + band] is treated as "no opinion" —
        # see config.yaml:event_sniping.model_no_opinion_band for why.
        no_op_band = safe_float(cfg.get("model_no_opinion_band", 0.04))

        text = f"{item.get('title','')}\n{item.get('summary','')}".strip()
        blocked = concentration_blocked()

        # Absolute tail-price gate (added 2026-04-23). The existing
        # plausibility gate below only triggers on |edge|>0.25 in the
        # extreme tails — a model saying "true_p=0.20 vs market 0.001"
        # produces edge=+0.199 which slipped through. An absolute
        # price-range check is independent of what the model says: at
        # mid<=0.02 or mid>=0.98 the round-trip spread eats any
        # plausible edge and the lane has no business taking the trade.
        # The April 2026 soak saw 2 event trades enter at mid=0.001
        # that the manual cleanup script had to sweep — this gate
        # closes that hole at the lane level (separate from the
        # signal-pipeline plausibility guard, because event_sniper has
        # its own scoring path that doesn't traverse the signal
        # pipeline). Config: event_sniping.price_range: [lo, hi].
        price_range = cfg.get("price_range") or [0.02, 0.98]
        try:
            price_lo = safe_float(price_range[0])
            price_hi = safe_float(price_range[1])
        except (TypeError, IndexError, ValueError):
            price_lo, price_hi = 0.02, 0.98

        for market in matched_markets:
            if market.market_id in blocked:
                continue
            if await shadow.count_open_for_market_in_lane(market.market_id, LANE) > 0:
                continue
            # Tail-price gate: reject markets whose CURRENT mid is at
            # either extreme. Lives BEFORE the volume/score calls so we
            # fail fast and don't spend Ollama cycles on dart-throw
            # markets. Note: this checks market.mid (what we'd pay), not
            # the model's true_prob (which the plausibility gate below
            # handles as a secondary check).
            if market.mid < price_lo or market.mid > price_hi:
                logger.debug(
                    "[event] skip {} mid={:.4f} outside price_range=[{:.2f},{:.2f}]",
                    market.market_id, market.mid, price_lo, price_hi,
                )
                continue
            # Unmoved check (market hasn't digested news yet).
            if _market_moved_recently(market, unmoved_pct):
                continue
            # Fast volume gate.
            vol = await volume_24h(market.market_id)
            if vol < min_volume:
                continue
            # Duration gate: skip markets that resolve beyond the
            # lane's horizon. days_until_resolve returns None when
            # close_time is missing/unparseable — treat that as
            # "unknown, don't commit capital here".
            if max_days_to_resolve > 0:
                days_left = days_until_resolve(market.close_time)
                if days_left is None or days_left > max_days_to_resolve:
                    logger.debug(
                        "[event] skip {} resolution={} > max={}d",
                        market.market_id,
                        f"{int(days_left)}d" if days_left is not None else "unknown",
                        int(max_days_to_resolve),
                    )
                    continue

            # Score with bounded timeout -> heuristic fallback. Fast tier
            # is the right target here (small model, <5s). If the fast
            # queue is already saturated we skip Ollama entirely rather
            # than stalling — event lane edge decays fast.
            if OllamaClient.fast_queue_saturated():
                logger.warning(
                    "[event] fast queue saturated (>={}), using heuristic for {}",
                    OllamaClient.pending_fast, market.market_id,
                )
                h = heuristic.score(text, market)
                score = Score(
                    true_prob=h.implied_prob,
                    confidence=h.confidence,
                    reasoning=h.reasoning,
                    source="heuristic",
                )
            else:
                score = await scoring.score_with_timeout(
                    market, text, timeout_seconds=timeout_s,
                    client=self._client, tier="fast",
                )
            if score.confidence < min_conf and score.source == "ollama":
                continue
            # Model hedged near 0.5 = no opinion, not a signal. On an
            # extreme-price market this otherwise manufactures a fake
            # ~0.5-wide "edge" and the lane auto-shorts the extreme.
            # Applies to both Ollama and heuristic sources (the heuristic
            # also returns ~0.5 when keywords are ambiguous).
            if abs(score.true_prob - 0.5) < no_op_band:
                logger.info(
                    "[event] skip {} true_p={:.3f} ≈ 0.5 (model hedged, no opinion)",
                    market.market_id, score.true_prob,
                )
                continue
            edge = score.true_prob - market.mid
            if abs(edge) < min_edge:
                continue
            # Plausibility gate — mirror of the one in signals/pipeline.py.
            # Event_sniper has its own scoring path that doesn't traverse
            # the signal pipeline, so the gate has to live here too or
            # implausible setups like "Panama wins World Cup at true_p=0.6
            # while market says 0.001" leak through (incident 2026-04-22).
            # Symmetric: catches both tails.
            risk_cfg = get_config().get("risk") or {}
            plaus_max_edge = safe_float(risk_cfg.get("plausibility_max_edge", 0.25))
            plaus_min_impl = safe_float(risk_cfg.get("plausibility_min_implied", 0.05))
            if plaus_max_edge > 0 and abs(edge) > plaus_max_edge:
                in_low_tail = market.mid < plaus_min_impl
                in_high_tail = market.mid > (1.0 - plaus_min_impl)
                if in_low_tail or in_high_tail:
                    logger.info(
                        "[event] skip {} implausible edge={:+.3f} mid={:.3f}",
                        market.market_id, edge, market.mid,
                    )
                    continue
            side = "BUY" if edge > 0 else "SELL"

            # Heuristic fallback forces a smaller size.
            if score.source == "heuristic":
                if score.confidence <= 0:
                    continue
                wanted = fallback_size
            else:
                wanted = await _position_size(score.confidence)

            approved = await allocator.reserve(LANE, wanted)
            if approved is None:
                continue

            snap = await current_price(market.market_id, market.yes_token() or "")
            if snap is None or snap.mid <= 0:
                await allocator.release(LANE, approved, 0.0)
                continue

            latency_ms = max(0.0, (now_ts() - ingested_at) * 1000.0) if ingested_at else 0.0
            if latency_ms / 1000.0 > max_latency_s:
                logger.warning(
                    "[event] latency {:.1f}s > max {:.0f}s — logging but still entering",
                    latency_ms / 1000.0, max_latency_s,
                )

            pos_id = await shadow.open_position(
                strategy=LANE,
                market=market,
                side=side,
                snapshot=snap,
                size_usd=approved,
                true_prob=score.true_prob,
                confidence=score.confidence,
                entry_reason=(
                    f"event[{score.source}]: edge={edge:+.3f} conf={score.confidence:.2f} "
                    f"latency={latency_ms:.0f}ms"
                ),
                evidence_ids=[int(item["id"])],
                evidence_snapshot={
                    "feed_item": {
                        "id": int(item["id"]),
                        "source": item.get("source"),
                        "title": item.get("title"),
                        "url": item.get("url"),
                    },
                    "reasoning": score.reasoning,
                    "score_source": score.source,
                },
                entry_latency_ms=latency_ms,
                ollama_client=self._client,
                validator_text=text,
                validator_reasoning=score.reasoning,
            )
            if pos_id is None:
                await allocator.release(LANE, approved, 0.0)
                continue

    async def _match_markets(self, item: dict[str, Any]) -> list[Market]:
        """Use the existing candidate scorer (+ linked_market_id hints) to
        find markets for this feed item."""
        meta_raw = item.get("meta")
        meta = {}
        if isinstance(meta_raw, str):
            try:
                meta = json.loads(meta_raw)
            except (TypeError, ValueError):
                meta = {}
        elif isinstance(meta_raw, dict):
            meta = meta_raw
        linked = (meta.get("linked_market_id") or "").strip() if isinstance(meta, dict) else ""
        if linked:
            m = await market_cache.get_market(str(linked))
            return [m] if m else []
        text = f"{item.get('title','')}\n{item.get('summary','')}".strip()
        if not text:
            return []
        scored = await scored_candidates_for(text)
        return [m for _, m in scored[:2]] if scored else []

    async def _trigger_rescore_for_held(
        self, markets: list[Market], item: dict[str, Any],
    ) -> None:
        """Capacity is full, but a new feed item for a market we already
        hold is the whole point of the 'fast-path re-score' — force a
        rescore on that position immediately rather than waiting for the
        cron tick."""
        text = f"{item.get('title','')}\n{item.get('summary','')}".strip()
        for m in markets:
            pos_rows = await fetch_all(
                """SELECT * FROM shadow_positions
                   WHERE strategy=? AND market_id=? AND status='OPEN'""",
                (LANE, m.market_id),
            )
            for row in pos_rows:
                pos = shadow._row_to_position(row)
                snap = await current_price(pos.market_id, pos.token_id)
                if snap is None:
                    continue
                score = await scoring.score_with_timeout(
                    m, text,
                    timeout_seconds=safe_float(_cfg().get("heuristic_fallback_timeout_seconds", 10)),
                    client=self._client, tier="fast",
                )
                await shadow.append_conviction(pos, score.true_prob, snap.mid)
                # Contradicting news check: if re-score flips + strong conf -> close.
                new_edge = score.true_prob - snap.mid
                contradicts = (
                    (pos.side == "BUY" and new_edge <= -0.02)
                    or (pos.side == "SELL" and new_edge >= 0.02)
                )
                if contradicts and score.confidence >= 0.70:
                    await shadow.close_position(
                        pos, snap, f"flip_exit new_news true_p={score.true_prob:.2f}",
                    )

    # ---- Monitor / exit ----

    async def _monitor_loop(self) -> None:
        interval = safe_float(_cfg().get("monitor_seconds", 30))
        while not self._stop.is_set():
            try:
                await self.monitor_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[event] monitor error: {}", e)
            await self._sleep(interval)

    async def monitor_once(self) -> int:
        cfg = _cfg()
        tp_pct = safe_float(cfg.get("take_profit_pct", 20))
        sl_pct = safe_float(cfg.get("stop_loss_pct", 15))
        trail_arm = safe_float(cfg.get("trailing_activate_pct", 0))
        trail_dd = safe_float(cfg.get("trailing_drawdown_pct", 0))
        max_age_hours = safe_float(cfg.get("max_age_hours", 6))
        rescore_seconds = safe_float(cfg.get("rescore_seconds", 120))

        positions = await shadow.open_positions_for(LANE)
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
            # Trailing TP: arms on peak, exits on drawdown from peak.
            elif (
                trail_arm > 0 and trail_dd > 0
                and pos.peak_pnl_pct >= trail_arm
                and (pos.peak_pnl_pct - pnl_pct) >= trail_dd
            ):
                reason = (
                    f"trailing_exit peak={pos.peak_pnl_pct:+.1f}% "
                    f"now={pnl_pct:+.1f}%"
                )
            elif pos.age_hours > max_age_hours:
                reason = f"time_exit age={pos.age_hours:.1f}h"

            if reason is None and (now_ts() - pos.last_rescored_ts) > rescore_seconds:
                reason = await self._rescore_and_maybe_flip(pos, snap)

            if reason is not None:
                await shadow.close_position(pos, snap, reason)
                closed += 1
        return closed

    async def _rescore_and_maybe_flip(
        self, pos: shadow.ShadowPosition, snap: Any,
    ) -> str | None:
        if shadow.conviction_is_stable(pos.conviction_trajectory):
            await shadow.append_conviction(pos, pos.true_prob_entry, snap.mid)
            return None
        market = await market_cache.get_market(pos.market_id)
        if market is None:
            return None
        timeout_s = safe_float(_cfg().get("heuristic_fallback_timeout_seconds", 10))
        score = await scoring.score_with_timeout(
            market, market.question, timeout_seconds=timeout_s,
            client=self._client, tier="fast",
        )
        await shadow.append_conviction(pos, score.true_prob, snap.mid)
        new_edge = score.true_prob - snap.mid
        if pos.side == "BUY" and new_edge <= -0.02:
            return f"flip_exit rescored true_p={score.true_prob:.2f}"
        if pos.side == "SELL" and new_edge >= 0.02:
            return f"flip_exit rescored true_p={score.true_prob:.2f}"
        return None
