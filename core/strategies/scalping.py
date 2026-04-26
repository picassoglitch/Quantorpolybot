"""Lane 1: scalping.

Catch small mispricings on liquid markets, exit fast. Liquidity is the
core filter here — no illiquid markets, no wide spreads, no stale
prices. Exits trigger on PnL thresholds, spread widening, position age,
or a conviction flip from the Ollama re-score.
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
from core.utils.db import fetch_all, fetch_one
from core.utils.helpers import now_ts, safe_float
from core.utils.prices import (
    current_price,
    days_until_resolve,
    live_orderbook_snapshot,
    volume_24h,
)

LANE = "scalping"


def _cfg() -> dict[str, Any]:
    return get_config().get("scalping") or {}


async def _position_size(confidence: float) -> float:
    """Delegate to the budget-adaptive allocator helper so this sizing
    auto-scales when total_usd changes (winners compound, dashboard
    edits, etc.) without re-tuning the lane config."""
    return await allocator.compute_position_size(LANE, confidence, _cfg())


async def _recent_evidence_for(market_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Pull the latest feed items tagged to this market (polymarket_news
    and google_news per-market feeds set meta.linked_market_id)."""
    rows = await fetch_all(
        """SELECT id, source, title, summary, url, ingested_at
           FROM feed_items
           WHERE meta LIKE ?
           ORDER BY ingested_at DESC LIMIT ?""",
        (f'%"linked_market_id":%"{market_id}"%', limit),
    )
    return [dict(r) for r in rows]


class ScalpingLane:
    component = "strategies.scalping"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._client = OllamaClient()

    async def run(self) -> None:
        if not _cfg().get("enabled", True):
            logger.info("[scalping] lane disabled")
            return
        logger.info("[scalping] lane started")
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
        interval = safe_float(_cfg().get("entry_scan_seconds", 60))
        while not self._stop.is_set():
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[scalping] entry scan error: {}", e)
            await self._sleep(interval)

    async def scan_once(self) -> int:
        cfg = _cfg()
        max_concurrent = int(cfg.get("max_concurrent", 40))
        open_count = await shadow.count_open(LANE)
        if open_count >= max_concurrent:
            return 0

        state = await allocator.get_state(LANE)
        if state is None or state.is_paused:
            return 0

        min_volume = safe_float(cfg.get("min_volume_24h", 10000))
        max_spread_cents = safe_float(cfg.get("max_spread_cents", 3))
        min_edge = safe_float(cfg.get("min_edge", 0.04))
        min_conf = safe_float(cfg.get("min_confidence", 0.60))
        min_days = safe_float(cfg.get("min_resolve_days", 2))
        max_days = safe_float(cfg.get("max_resolve_days", 14))
        min_evidence = int(cfg.get("min_evidence_sources", 2))

        # Liquid markets first. liquidity is our pre-proxy for volume; we
        # confirm 24h volume via gamma before entry.
        #
        # Pull a wider pool (300) and pre-filter by the lane's date window
        # BEFORE we slice to 40 workable candidates. Why: the top-40 by
        # liquidity on Polymarket is dominated by long-horizon futures
        # (elections, World Cup, crypto H2) which all sit outside
        # scalping's 2-14d window. A naive `list_active(limit=40)`
        # returned 37/40 mismatches and the lane never reached Ollama —
        # that was the April 2026 idle-scalping bug. The 300-wide pull
        # with in-window filter exposes the liquid-short-dated subset.
        candidate_pool = int(cfg.get("candidate_pool_size", 300))
        scan_cap = int(cfg.get("scan_cap", 40))
        pool = await market_cache.list_active(limit=candidate_pool)
        windowed: list[Any] = []
        for m in pool:
            d = days_until_resolve(m.close_time)
            if d is not None and min_days <= d <= max_days:
                windowed.append(m)
            if len(windowed) >= scan_cap:
                break
        candidates = windowed
        blocked = concentration_blocked()
        # Funnel counters: each bucket is "skipped because of X" so we
        # can tell whether the lane is idle because nothing qualified,
        # Ollama rejected everything, or something else. Without this,
        # a silent lane is indistinguishable from a dead lane.
        total = len(candidates)
        skipped: dict[str, int] = {}
        scored_n = 0
        ollama_n = 0      # of scored, came from a real Ollama response
        heuristic_n = 0   # of scored, came from the keyword fallback
        low_conf = 0
        low_edge = 0
        entered = 0
        for market in candidates:
            if open_count + entered >= max_concurrent:
                break
            if market.market_id in blocked:
                skipped["concentration"] = skipped.get("concentration", 0) + 1
                continue
            if await shadow.count_open_for_market_in_lane(market.market_id, LANE) > 0:
                skipped["already_open"] = skipped.get("already_open", 0) + 1
                continue
            # Date window already enforced in the pool pre-filter above;
            # no need to re-check here.
            # Spread gate (from cache first; cheap).
            spread_cents = (market.best_ask - market.best_bid) * 100.0
            if spread_cents <= 0 or spread_cents > max_spread_cents:
                skipped["spread"] = skipped.get("spread", 0) + 1
                continue
            # Evidence requirement.
            evidence = await _recent_evidence_for(market.market_id, limit=5)
            sources = {e.get("source") for e in evidence if e.get("source")}
            if len(sources) < min_evidence:
                skipped["evidence"] = skipped.get("evidence", 0) + 1
                continue
            # Volume check (fresh from gamma; avoid scalp on volume traps).
            vol = await volume_24h(market.market_id)
            if vol < min_volume:
                skipped["volume"] = skipped.get("volume", 0) + 1
                continue
            # Score via Ollama (or keyword heuristic on saturation /
            # timeout — see core.strategies.scoring.score_with_fallback).
            # Lanes used to call ``scoring.score`` here, which returned
            # ``None`` whenever the fast queue was saturated or Ollama
            # stalled; that produced "scan total=40 ... scored=0" for
            # entire 10-minute windows. ``score_with_fallback`` always
            # returns a Score so a degraded GPU never zeros out entries.
            text = "\n".join(
                f"{e.get('title','')}: {(e.get('summary') or '')[:200]}"
                for e in evidence[:3]
            )
            # Scalping entry uses the fast tier: ~40 candidates per cycle
            # means we can't afford the 15-30s deep model here.
            score = await scoring.score_with_fallback(
                market, text, client=self._client, tier="fast",
            )
            scored_n += 1
            if score.source == "ollama":
                ollama_n += 1
            else:
                heuristic_n += 1
            # Heuristic confidence is capped at 0.70 by design (see
            # core.strategies.heuristic) — the existing min_conf gate
            # already filters thin keyword signals, so fallback Scores
            # only enter on a clear directional headline.
            if score.confidence < min_conf:
                low_conf += 1
                continue
            edge = score.true_prob - market.mid
            if abs(edge) < min_edge:
                low_edge += 1
                continue
            side = "BUY" if edge > 0 else "SELL"

            # Capital reservation + dynamic cap.
            wanted = await _position_size(score.confidence)
            approved = await allocator.reserve(LANE, wanted)
            if approved is None:
                skipped["budget"] = skipped.get("budget", 0) + 1
                continue

            # Fresh price snapshot at entry.
            snap = await current_price(market.market_id, market.yes_token() or "")
            if snap is None or snap.mid <= 0:
                await allocator.release(LANE, approved, 0.0)
                continue

            pos_id = await shadow.open_position(
                strategy=LANE,
                market=market,
                side=side,
                snapshot=snap,
                size_usd=approved,
                true_prob=score.true_prob,
                confidence=score.confidence,
                entry_reason=(
                    f"scalp: edge={edge:+.3f} conf={score.confidence:.2f} "
                    f"spread={spread_cents:.1f}c vol={vol:.0f}"
                ),
                evidence_ids=[int(e["id"]) for e in evidence if e.get("id")],
                evidence_snapshot={
                    "reasoning": score.reasoning,
                    "evidence_count": len(evidence),
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
        # Scan summary: always log so the operator can tell at a glance
        # whether the lane is actively looking (and where candidates
        # die). `scored` is the count that reached Ollama — the pricey
        # bucket; the rest are cheap gate skips.
        if total > 0:
            skip_summary = ", ".join(
                f"{k}={v}" for k, v in sorted(skipped.items())
            ) or "none"
            logger.info(
                "[scalping] scan total={} skip=[{}] scored={} "
                "(ollama={} heuristic={}) low_conf={} low_edge={} entered={}",
                total, skip_summary, scored_n, ollama_n, heuristic_n,
                low_conf, low_edge, entered,
            )
        return entered

    # ---- Monitor / exit ----

    async def _monitor_loop(self) -> None:
        interval = safe_float(_cfg().get("monitor_seconds", 30))
        while not self._stop.is_set():
            try:
                await self.monitor_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[scalping] monitor error: {}", e)
            await self._sleep(interval)

    async def monitor_once(self) -> int:
        cfg = _cfg()
        tp_pct = safe_float(cfg.get("take_profit_pct", 8))
        sl_pct = safe_float(cfg.get("stop_loss_pct", 15))
        trail_arm = safe_float(cfg.get("trailing_activate_pct", 0))
        trail_dd = safe_float(cfg.get("trailing_drawdown_pct", 0))
        max_age_hours = safe_float(cfg.get("max_age_hours", 24))
        liquidity_exit_cents = safe_float(cfg.get("liquidity_exit_spread_cents", 5))
        rescore_seconds = safe_float(cfg.get("rescore_seconds", 300))

        positions = await shadow.open_positions_for(LANE)
        closed = 0
        for pos in positions:
            snap = await current_price(pos.market_id, pos.token_id)
            if snap is None or snap.mid <= 0:
                continue
            pnl_pct = await shadow.update_price(pos, snap)
            reason: str | None = None

            # ---- Price-based exits ----
            if pnl_pct >= tp_pct:
                reason = f"take_profit {pnl_pct:+.1f}%"
            elif pnl_pct <= -sl_pct:
                reason = f"stop_loss {pnl_pct:+.1f}%"
            # ---- Trailing TP: armed once peak crosses the activate line,
            # triggers when pnl falls more than trail_dd below peak. ----
            elif (
                trail_arm > 0 and trail_dd > 0
                and pos.peak_pnl_pct >= trail_arm
                and (pos.peak_pnl_pct - pnl_pct) >= trail_dd
            ):
                reason = (
                    f"trailing_exit peak={pos.peak_pnl_pct:+.1f}% "
                    f"now={pnl_pct:+.1f}%"
                )
            # ---- Age exit ----
            elif pos.age_hours > max_age_hours:
                reason = f"time_exit age={pos.age_hours:.1f}h"
            # ---- Liquidity exit — re-check spread from live orderbook ----
            else:
                live = await live_orderbook_snapshot(pos.market_id, pos.token_id)
                if live and live.spread_cents > liquidity_exit_cents:
                    reason = f"liquidity_exit spread={live.spread_cents:.1f}c"

            # ---- Conviction re-score (flip exit) ----
            if reason is None and (now_ts() - pos.last_rescored_ts) > rescore_seconds:
                reason = await self._rescore_and_maybe_flip(pos, snap)

            if reason is not None:
                await shadow.close_position(pos, snap, reason)
                closed += 1
        return closed

    async def _rescore_and_maybe_flip(self, pos: shadow.ShadowPosition, snap: Any) -> str | None:
        market = await market_cache.get_market(pos.market_id)
        if market is None:
            return None
        # Stable-position skip: on hot GPU, don't waste cycles on a
        # flat conviction trajectory.
        if shadow.conviction_is_stable(pos.conviction_trajectory):
            await shadow.append_conviction(pos, pos.true_prob_entry, snap.mid)
            return None
        evidence = await _recent_evidence_for(pos.market_id, limit=3)
        text = "\n".join(
            f"{e.get('title','')}: {(e.get('summary') or '')[:150]}" for e in evidence
        ) or market.question
        # Re-scores are the hottest path — always fast tier. Use the
        # fallback-aware scorer so a saturated GPU doesn't quietly skip
        # the conviction update; the heuristic at least gives us a
        # directional read on a fresh headline.
        score = await scoring.score_with_fallback(
            market, text, client=self._client, tier="fast",
        )
        await shadow.append_conviction(pos, score.true_prob, snap.mid)
        # Flip: edge crosses mid relative to original side.
        new_edge = score.true_prob - snap.mid
        if pos.side == "BUY" and new_edge <= -0.02:
            return f"flip_exit rescored true_p={score.true_prob:.2f} mid={snap.mid:.2f}"
        if pos.side == "SELL" and new_edge >= 0.02:
            return f"flip_exit rescored true_p={score.true_prob:.2f} mid={snap.mid:.2f}"
        return None
