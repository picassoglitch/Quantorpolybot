"""Lane 3: longshot.

Bet small on mispriced low-probability outcomes and hold to resolution.
Fixed size ($25) forces diversification. Early exits only fire on
extreme repricing, a dying market, or strongly contradicting evidence.

Hardened after a first live run where the loose gates let the lane
enter a $0.002 sports market with no real thesis. The current rules
require entry price ≥ `min_entry_price`, two independent evidence
sources, a non-sports category, a 30-180-day resolution window, and a
named catalyst (date or concrete event) in the LLM reasoning — weasel
words like "could", "might" without a date are rejected.
"""

from __future__ import annotations

import asyncio
import re
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
from core.utils.prices import current_price, days_until_resolve, volume_24h

LANE = "longshot"

# A named catalyst is a date fragment, month name, or a concrete
# event keyword. We require at least one of these patterns alongside
# any weasel words so the bot doesn't buy on pure hand-waving.
_MONTHS = (
    "january february march april may june july august september "
    "october november december jan feb mar apr jun jul aug sep oct nov dec"
).split()
_CATALYST_PATTERNS: list[re.Pattern[str]] = [
    # Dates: 2025-03-10, 3/10/2025, March 10, 10 March 2025
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(rf"\b(?:{'|'.join(_MONTHS)})\s+\d{{1,2}}\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}\s+(?:{'|'.join(_MONTHS)})\b", re.IGNORECASE),
    # Resolution/event keywords (must co-occur with a date/month, OR alone
    # be enough on their own — they are all concrete event references).
    re.compile(r"\b(?:election day|inauguration|earnings|fomc|debate|verdict|"
               r"ruling|hearing|summit|announcement|deadline|primary day|"
               r"convention|launch|mainnet|halving|resolve[sd]?|expires?)\b",
               re.IGNORECASE),
]


def _cfg() -> dict[str, Any]:
    return get_config().get("longshot") or {}


def _has_named_catalyst(text: str) -> bool:
    if not text:
        return False
    for pat in _CATALYST_PATTERNS:
        if pat.search(text):
            return True
    return False


def _has_weasel_word(text: str, weasel_words: list[str]) -> bool:
    if not text or not weasel_words:
        return False
    lower = text.lower()
    for w in weasel_words:
        if re.search(rf"\b{re.escape(w.lower())}\b", lower):
            return True
    return False


async def _recent_evidence_for(market_id: str, limit: int = 5) -> list[dict[str, Any]]:
    rows = await fetch_all(
        """SELECT id, source, title, summary, url, ingested_at
           FROM feed_items
           WHERE meta LIKE ?
           ORDER BY ingested_at DESC LIMIT ?""",
        (f'%"linked_market_id":%"{market_id}"%', limit),
    )
    return [dict(r) for r in rows]


class LongshotLane:
    component = "strategies.longshot"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._client = OllamaClient()
        # pos_id -> first_seen_below_floor_ts (monotonic-ish unix seconds).
        # In-memory is fine: worst case we lose the grace timer on
        # restart and wait another 6h before closing a dead position.
        self._dead_floor_since: dict[int, float] = {}

    async def run(self) -> None:
        if not _cfg().get("enabled", True):
            logger.info("[longshot] lane disabled")
            return
        logger.info("[longshot] lane started")
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

    # ---- Entry ----

    async def _entry_loop(self) -> None:
        interval = safe_float(_cfg().get("entry_scan_seconds", 600))
        while not self._stop.is_set():
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[longshot] entry scan error: {}", e)
            await self._sleep(interval)

    async def scan_once(self) -> int:
        cfg = _cfg()
        max_concurrent = int(cfg.get("max_concurrent", 40))
        if await shadow.count_open(LANE) >= max_concurrent:
            return 0
        state = await allocator.get_state(LANE)
        if state is None or state.is_paused:
            return 0

        min_entry_price = safe_float(cfg.get("min_entry_price", 0.04))
        max_entry_price = safe_float(cfg.get("max_entry_price", 0.15))
        min_edge_mult = safe_float(cfg.get("min_edge_multiple", 2.0))
        min_conf = safe_float(cfg.get("min_confidence", 0.60))
        min_days = safe_float(cfg.get("min_resolve_days", 30))
        max_days = safe_float(cfg.get("max_resolve_days", 180))
        max_per_market = int(cfg.get("max_per_market", 1))
        # Budget-adaptive: use the allocator helper which reads
        # position_pct (if configured) or falls back to fixed_position.
        # Longshot uses a single sizing tier (no confidence scale) so we
        # pass a neutral confidence to hit the 'base' arm of the helper.
        fixed = await allocator.compute_position_size(LANE, 0.50, cfg)
        if fixed <= 0:
            fixed = safe_float(cfg.get("fixed_position", 25))
        min_evidence_sources = int(cfg.get("min_evidence_sources", 2))
        allowed = {c.lower() for c in (cfg.get("allowed_categories") or [])}
        disallowed = {c.lower() for c in (cfg.get("disallowed_categories") or [])}
        weasel = [w for w in (cfg.get("weasel_words") or []) if isinstance(w, str)]

        candidates = await market_cache.list_active(limit=200)
        blocked = concentration_blocked()
        entered = 0
        for market in candidates:
            if await shadow.count_open(LANE) >= max_concurrent:
                break
            if market.market_id in blocked:
                continue
            mid = market.mid
            if mid < min_entry_price or mid > max_entry_price:
                continue
            # Category gate — reject disallowed first (sports), then
            # require the remaining market to be on the allow-list if
            # one is configured.
            category = (market.category or "").strip().lower()
            if category and category in disallowed:
                continue
            if allowed and category not in allowed:
                continue
            days = days_until_resolve(market.close_time)
            if days is None or days < min_days or days > max_days:
                continue
            if await shadow.count_open_for_market_in_lane(market.market_id, LANE) >= max_per_market:
                continue
            evidence = await _recent_evidence_for(market.market_id, limit=10)
            sources = {e.get("source") for e in evidence if e.get("source")}
            if len(sources) < min_evidence_sources:
                continue
            text = "\n".join(
                f"{e.get('title','')}: {(e.get('summary') or '')[:200]}"
                for e in evidence[:5]
            ) or market.question
            # Longshot bets depend on deep analysis of thin evidence —
            # always deep tier, even though the lane is paused by default.
            score = await scoring.score(market, text, self._client, tier="deep")
            if score is None:
                continue
            if score.confidence < min_conf:
                continue
            if score.true_prob < min_edge_mult * mid:
                continue
            # Named-catalyst / weasel-word guard. If the LLM reasoning
            # contains hedging words ("likely", "could", ...) without a
            # concrete date or resolution event, the thesis is too soft
            # to hold for weeks. Reject.
            has_catalyst = _has_named_catalyst(score.reasoning)
            if _has_weasel_word(score.reasoning, weasel) and not has_catalyst:
                logger.debug(
                    "[longshot] reject {} — weasel words without catalyst: {!r}",
                    market.market_id, score.reasoning[:120],
                )
                continue
            if not has_catalyst:
                # Even without weasel words we want a concrete event /
                # date referenced somewhere in the reasoning. Skip if
                # not present.
                logger.debug(
                    "[longshot] reject {} — no named catalyst in reasoning",
                    market.market_id,
                )
                continue
            vol = await volume_24h(market.market_id)
            if vol <= 0:
                continue

            side = "BUY"  # longshots are always YES bets on mispriced lows
            approved = await allocator.reserve(LANE, fixed)
            if approved is None:
                continue
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
                    f"longshot: mid={mid:.3f} true_p={score.true_prob:.3f} "
                    f"multiple={score.true_prob/mid:.2f}x days={days:.0f} "
                    f"sources={len(sources)} category={category or 'unknown'}"
                ),
                evidence_ids=[int(e["id"]) for e in evidence if e.get("id")],
                evidence_snapshot={
                    "reasoning": score.reasoning,
                    "evidence_count": len(evidence),
                    "sources": sorted(sources),
                    "catalyst_present": has_catalyst,
                },
                entry_latency_ms=0.0,
            )
            if pos_id is None:
                await allocator.release(LANE, approved, 0.0)
                continue
            entered += 1
        if entered:
            logger.info("[longshot] entered {} positions", entered)
        return entered

    # ---- Monitor / exit ----

    async def _monitor_loop(self) -> None:
        interval = safe_float(_cfg().get("monitor_seconds", 120))
        while not self._stop.is_set():
            try:
                await self.monitor_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[longshot] monitor error: {}", e)
            await self._sleep(interval)

    async def monitor_once(self) -> int:
        cfg = _cfg()
        tp_mult = safe_float(cfg.get("take_profit_multiple", 2.0))
        dead_floor = safe_float(cfg.get("dead_floor_price", 0.015))
        dead_floor_grace = safe_float(cfg.get("dead_floor_hours", 6.0)) * 3600.0
        contradicting_conf = safe_float(cfg.get("contradicting_confidence", 0.80))
        rescore_seconds = safe_float(cfg.get("rescore_seconds", 3600))

        positions = await shadow.open_positions_for(LANE)
        open_ids = {pos.id for pos in positions}
        # Clean up grace timers for positions that have closed.
        for pid in list(self._dead_floor_since):
            if pid not in open_ids:
                self._dead_floor_since.pop(pid, None)

        closed = 0
        for pos in positions:
            snap = await current_price(pos.market_id, pos.token_id)
            if snap is None or snap.mid <= 0:
                continue
            await shadow.update_price(pos, snap)
            reason: str | None = None

            if pos.entry_price > 0 and snap.mid >= tp_mult * pos.entry_price:
                reason = (
                    f"take_profit price surged "
                    f"{snap.mid/pos.entry_price:.2f}x from {pos.entry_price:.3f}"
                )
            elif snap.mid < dead_floor:
                first_seen = self._dead_floor_since.get(pos.id)
                if first_seen is None:
                    self._dead_floor_since[pos.id] = now_ts()
                elif (now_ts() - first_seen) >= dead_floor_grace:
                    reason = (
                        f"dead_floor mid={snap.mid:.3f} held≥{dead_floor_grace/3600:.1f}h"
                    )
            else:
                # Price recovered above floor — reset the grace timer.
                self._dead_floor_since.pop(pos.id, None)

            if reason is None and (now_ts() - pos.last_rescored_ts) > rescore_seconds:
                reason = await self._rescore_and_maybe_close(pos, snap, contradicting_conf)

            if reason is not None:
                await shadow.close_position(pos, snap, reason)
                self._dead_floor_since.pop(pos.id, None)
                closed += 1
        return closed

    async def _rescore_and_maybe_close(
        self, pos: shadow.ShadowPosition, snap: Any, min_contradict_conf: float,
    ) -> str | None:
        if shadow.conviction_is_stable(pos.conviction_trajectory):
            await shadow.append_conviction(pos, pos.true_prob_entry, snap.mid)
            return None
        market = await market_cache.get_market(pos.market_id)
        if market is None:
            return None
        evidence = await _recent_evidence_for(pos.market_id, limit=3)
        text = "\n".join(
            f"{e.get('title','')}: {(e.get('summary') or '')[:200]}" for e in evidence
        ) or market.question
        # Conviction re-scoring on longshots: deep tier — these holds
        # span weeks and we rescore rarely, so latency is cheap.
        score = await scoring.score(market, text, self._client, tier="deep")
        if score is None:
            return None
        await shadow.append_conviction(pos, score.true_prob, snap.mid)
        if score.true_prob < snap.mid and score.confidence >= min_contradict_conf:
            return (
                f"contradicting_evidence true_p={score.true_prob:.2f} "
                f"mid={snap.mid:.3f} conf={score.confidence:.2f}"
            )
        return None
