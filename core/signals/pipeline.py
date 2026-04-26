"""End-to-end signal pipeline:

   feed_items (unprocessed) -> candidate markets -> Ollama JSON ->
   edge calc -> persisted signal -> handed to execution

Runs as a single async loop polling for new feed_items rows.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from core.markets.cache import get_market
from core.risk.rules import RiskEngine, RiskRejection
from core.signals.candidates import (
    scored_candidates_for,
    serialize_candidates,
)
from core.signals.context import build_market_context
from core.learning.source_trust import get_source_weights, trust_weight_for
from core.signals.guards import apply_true_prob_cap, guard_config, hallucination_reject
from core.signals.ollama_client import OllamaClient, deep_realtime_enabled
from core.strategies import heuristic
from core.utils.config import get_config, get_prompts
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import clamp, now_ts, safe_float
from core.utils.prices import days_until_resolve


class SignalPipeline:
    component = "signals.pipeline"

    def __init__(self, risk: RiskEngine) -> None:
        self._stop = asyncio.Event()
        self._ollama = OllamaClient()
        self._risk = risk
        self._cursor_id = 0  # last processed feed_items.id
        # Throttle the "processed N items" log: during a backlog catch-up
        # the pipeline chews through thousands of pre-filter skips per
        # second, which drowns the log. Summarise to one line every 5s.
        self._processed_since_log = 0
        self._last_log_ts = 0.0
        # Per-market "last Ollama-scored at" timestamp. Multiple feeds
        # (google_news, polymarket_news, RSS cross-posts) often tag the
        # same market in rapid succession — the dashboard showed the
        # same Croatia WC market being deep-scored 7 times back-to-back
        # for 0 benefit. Throttle re-scoring per market so we spend GPU
        # time on genuinely new candidates instead. Runtime-mutable so
        # ``signals.market_rescore_cooldown_seconds`` takes effect on
        # config reload without a restart.
        self._market_last_scored: dict[str, float] = {}

    async def run(self) -> None:
        await self._init_cursor()
        logger.info("[signals] pipeline started; cursor={}", self._cursor_id)
        while not self._stop.is_set():
            try:
                processed = await self._process_batch()
                if processed == 0:
                    if self._processed_since_log:
                        logger.info(
                            "[signals] processed {} items (backlog drained)",
                            self._processed_since_log,
                        )
                        self._processed_since_log = 0
                        self._last_log_ts = now_ts()
                    await self._sleep(2)
                else:
                    self._processed_since_log += processed
                    ts = now_ts()
                    if ts - self._last_log_ts >= 5.0:
                        logger.info(
                            "[signals] processed {} items (last 5s)",
                            self._processed_since_log,
                        )
                        self._processed_since_log = 0
                        self._last_log_ts = ts
                    # Yield briefly on a full batch so other coroutines
                    # (WS ingest, order monitor) aren't starved during
                    # catch-up. 10ms is below the batch runtime floor.
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[signals] loop error: {}", e)
                await self._sleep(5)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _init_cursor(self) -> None:
        # Resume from the latest signal we have (so a restart doesn't
        # re-process the entire history).
        row = await fetch_one(
            "SELECT MAX(feed_item_id) AS m FROM signals"
        )
        self._cursor_id = int((row["m"] or 0) if row else 0)

    async def _process_batch(self) -> int:
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
                logger.exception("[signals] item {} failed: {}", row["id"], e)
            self._cursor_id = max(self._cursor_id, int(row["id"]))
        return len(rows)

    async def _process_item(self, item: dict[str, Any]) -> None:
        text = f"{item.get('title', '')}\n{item.get('summary', '')}".strip()
        if not text:
            return
        scored = await scored_candidates_for(text)
        if not scored:
            return
        markets = [m for _, m in scored]

        # Per-market rescore cooldown. If the top candidate was Ollama-
        # scored within the cooldown window, skip the LLM call and log
        # a cheap throttled signal row so the audit trail shows why the
        # feed item didn't fire. Without this guard the pipeline
        # happily burns ~5s of deep-tier time on duplicate signals for
        # the same market every time a new news item tags it.
        cooldown = float(
            get_config().get(
                "signals", "market_rescore_cooldown_seconds", default=180,
            )
        )
        if cooldown > 0 and markets:
            top_mid = str(markets[0].market_id)
            last_ts = self._market_last_scored.get(top_mid, 0.0)
            age = now_ts() - last_ts
            if last_ts > 0 and age < cooldown:
                await execute(
                    """INSERT INTO signals
                    (feed_item_id, market_id, implied_prob, confidence, edge,
                     mid_price, side, size_usd, reasoning, prompt_version,
                     created_at, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        int(item["id"]),
                        top_mid,
                        0.0, 0.0, 0.0, 0.0,
                        "NONE",
                        0.0,
                        (
                            f"market_rescore_throttle age={age:.0f}s "
                            f"cooldown={cooldown:.0f}s"
                        ),
                        "prefilter",
                        now_ts(),
                        "market_throttled",
                    ),
                )
                return
        # Pre-filter: feeds that already nominate a specific market
        # (polymarket_news, predictit_xref, google_news per-market query)
        # bypass the keyword check; everything else must clear the
        # configured min_keyword_overlap or we save a 'keyword_mismatch'
        # row and skip Ollama. Saves big on local LLM time for the
        # broad-topic feeds (Reuters, BBC, Wikipedia).
        meta = _decode_meta(item.get("meta"))
        linked_id = (meta.get("linked_market_id") or "").strip()
        if not linked_id:
            min_overlap = float(
                get_config().get("signals", "min_keyword_overlap", default=0.05)
            )
            best_score = scored[0][0]
            if best_score < min_overlap:
                await execute(
                    """INSERT INTO signals
                    (feed_item_id, market_id, implied_prob, confidence, edge,
                     mid_price, side, size_usd, reasoning, prompt_version,
                     created_at, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        int(item["id"]),
                        None,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        "NONE",
                        0.0,
                        f"keyword_mismatch best_score={best_score:.3f} "
                        f"min={min_overlap:.3f}",
                        "prefilter",
                        now_ts(),
                        "keyword_mismatch",
                    ),
                )
                return
        # Long-horizon candidate pre-filter. We don't run any strategy
        # that benefits from >12-month markets (longshot's widest
        # window is 180d; event_sniping's is 14d; scalping wants
        # near-term) and historically these markets are thin-liquidity
        # junk — 2028 elections, far-future crypto, World Cup round-
        # of-16 prop bets. Drop them BEFORE the Ollama call. If every
        # candidate is too far out, log one `long_horizon_skip` row
        # for the audit trail and skip the item.
        sig_cfg = get_config().get("signals") or {}
        max_candidate_days = safe_float(
            sig_cfg.get("max_candidate_days", 365)
        )
        if max_candidate_days > 0:
            filtered_markets: list[Any] = []
            for m in markets:
                d = days_until_resolve(m.close_time)
                if d is None or d <= max_candidate_days:
                    filtered_markets.append(m)
            if not filtered_markets:
                # Represent the skip so the signals page / reviewer
                # can see what happened — attributed to the top
                # heuristic candidate so the market_id column stays
                # populated.
                top_mid = str(markets[0].market_id)
                await execute(
                    """INSERT INTO signals
                    (feed_item_id, market_id, implied_prob, confidence, edge,
                     mid_price, side, size_usd, reasoning, prompt_version,
                     created_at, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        int(item["id"]),
                        top_mid,
                        0.0, 0.0, 0.0, 0.0,
                        "NONE",
                        0.0,
                        f"long_horizon_skip all candidates resolve "
                        f"> {max_candidate_days:.0f}d out",
                        "prefilter",
                        now_ts(),
                        "long_horizon",
                    ),
                )
                return
            markets = filtered_markets

        prompt_version, template = get_prompts().active()
        news_item_json = json.dumps(
            {
                "source": item.get("source"),
                "title": item.get("title"),
                "summary": (item.get("summary") or "")[:1000],
                "url": item.get("url"),
            },
            ensure_ascii=False,
        )
        # ---- Branch: deep tier off-realtime mode --------------------
        # When ``ollama.deep_realtime_enabled`` is false (typical for
        # CPU-only Ollama deployments), the signal pipeline can't
        # afford a 20-45s deep call per feed item. Skip the LLM and
        # use the keyword heuristic instead — same downstream guards,
        # same risk gates, same signal-row schema, just a faster
        # source. The bot stays useful while waiting for GPU/Ollama
        # to come back or for the operator to re-enable the flag.
        if not deep_realtime_enabled():
            await self._process_with_heuristic(
                item=item,
                markets=markets,
                prompt_version="heuristic",
                text=text,
            )
            return

        candidates_json = json.dumps(serialize_candidates(markets), ensure_ascii=False)
        # Build per-candidate context concurrently so the prompt can reason
        # about price trajectory / peer signals / news frequency per market
        # — not just the news text in isolation. Best-effort: any failure
        # gives an empty dict for that market so the prompt still renders.
        #
        # Cap the parallel fan-out: each ``build_market_context`` opens 3
        # aiosqlite connections (price trajectory + recent news count +
        # peer signals). With 25 candidates per pipeline call, that's 75
        # concurrent connections fighting WAL serialization, which was
        # one of the contributors to the watchdog-observed loop stalls.
        # Top-K is configurable so this can be tuned without a code
        # change; 5 covers >95% of feed items in practice (the candidate
        # scorer only rarely produces more than a handful of strong matches).
        ctx_top_k = int(
            get_config().get("signals", "context_top_k", default=5)
        )
        if ctx_top_k > 0 and len(markets) > ctx_top_k:
            head, tail = markets[:ctx_top_k], markets[ctx_top_k:]
        else:
            head, tail = list(markets), []
        contexts = await asyncio.gather(
            *(build_market_context(m.market_id) for m in head),
            return_exceptions=True,
        )
        ctx_map: dict[str, dict[str, Any]] = {}
        for m, c in zip(head, contexts):
            ctx_map[m.market_id] = c if isinstance(c, dict) else {}
        for m in tail:
            # Tail markets get an empty context dict so the prompt still
            # serializes them — the LLM just won't have price/news/peer
            # context for those candidates. They're the lowest-scoring
            # heuristic matches anyway.
            ctx_map[m.market_id] = {}
        context_json = json.dumps(ctx_map, ensure_ascii=False)
        # Use .replace() not .format() — the prompt template contains a literal
        # JSON schema example with {} braces that would otherwise blow up
        # str.format with KeyError on the inner JSON keys.
        prompt = (
            template
            .replace("{news_item}", news_item_json)
            .replace("{candidates}", candidates_json)
            .replace("{context}", context_json)
        )
        result = await self._ollama.generate_json(
            prompt, tag=f"pipeline:{int(item['id'])}",
        )
        if not result:
            # Ollama unavailable (cooldown / timeout / unparseable). The
            # pipeline used to silently return here, which made a
            # degraded LLM look identical to "no candidate matched" in
            # the signals table. Persist a row so the dashboard /
            # operator triage can tell the difference and so feed items
            # aren't re-processed forever (the cursor only advances when
            # _process_item returns).
            top_mid = str(markets[0].market_id) if markets else None
            await execute(
                """INSERT INTO signals
                (feed_item_id, market_id, implied_prob, confidence, edge,
                 mid_price, side, size_usd, reasoning, prompt_version,
                 created_at, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    int(item["id"]),
                    top_mid,
                    0.0, 0.0, 0.0, 0.0,
                    "NONE",
                    0.0,
                    "ollama_unavailable: deep call returned no parseable JSON",
                    prompt_version,
                    now_ts(),
                    "ollama_unavailable",
                ),
            )
            return
        market_id = result.get("market_id")
        if not market_id or market_id == "null":
            return
        market = await get_market(str(market_id))
        if not market:
            return

        # Record that we just scored this market so the next feed item
        # tagging it hits the cooldown above. We key on the market the
        # LLM actually picked (not the top heuristic candidate), since
        # the prompt can and does rebalance across the candidate list.
        self._market_last_scored[str(market.market_id)] = now_ts()
        # GC the dict when it grows past ~2k entries — pathological feeds
        # would otherwise accumulate one entry per ever-seen market.
        if len(self._market_last_scored) > 2000:
            cutoff = now_ts() - max(
                float(get_config().get(
                    "signals", "market_rescore_cooldown_seconds", default=180,
                )) * 4,
                900.0,
            )
            self._market_last_scored = {
                mid: ts for mid, ts in self._market_last_scored.items()
                if ts >= cutoff
            }

        implied = clamp(safe_float(result.get("implied_prob")), 0.0, 1.0)
        confidence = clamp(safe_float(result.get("confidence")), 0.0, 1.0)
        reasoning = (result.get("reasoning") or "")[:500]
        mid = market.mid

        # ---- Sanity guards on the LLM output ----
        # Long-horizon cap: resolutions > N days out shouldn't get
        # 99%-confidence priors. Clamp true_prob so the edge calc
        # doesn't explode.
        guards = guard_config(get_config().get("risk"))
        days = days_until_resolve(market.close_time)
        capped_implied = apply_true_prob_cap(
            implied,
            days,
            long_horizon_days=guards["long_horizon_days"],
            cap=guards["long_horizon_cap"],
        )
        if capped_implied != implied:
            reasoning = (
                reasoning + f" [long_horizon_cap {implied:.2f}->{capped_implied:.2f}]"
            )[:500]
            implied = capped_implied

        # Hallucination guard: on markets where the current mid is
        # very low, high LLM priors are the #1 cause of bad trades.
        # Count unique evidence sources linked to this market in
        # recent feed items and reject when corroboration is thin.
        # Weighted count reflects per-source trust calibration (see
        # core.learning.source_trust); unknown sources fall back to
        # 1.0 so pre-calibration behaviour is preserved.
        num_sources, weighted_sources = await self._count_recent_sources(
            str(market.market_id),
        )
        reject_reason = hallucination_reject(
            true_prob=implied,
            mid=mid,
            num_sources=num_sources,
            weighted_sources=weighted_sources,
            low_mid_threshold=guards["low_mid_threshold"],
            high_true_prob_threshold=guards["high_true_prob_threshold"],
            min_sources=guards["min_sources"],
        )

        edge = implied - mid
        side = "BUY" if edge > 0 else "SELL"

        signal_id = await execute(
            """INSERT INTO signals
            (feed_item_id, market_id, implied_prob, confidence, edge, mid_price,
             side, size_usd, reasoning, prompt_version, created_at, status,
             category)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                int(item["id"]),
                market.market_id,
                implied,
                confidence,
                edge,
                mid,
                side,
                0.0,
                reasoning,
                prompt_version,
                now_ts(),
                "PENDING",
                (market.category or "").strip(),
            ),
        )

        if reject_reason:
            await self._reject(signal_id, reject_reason)
            return

        cfg = get_config().get("risk") or {}
        if confidence < float(cfg.get("min_confidence", 0.65)):
            await self._reject(signal_id, "low confidence")
            return
        if abs(edge) < float(cfg.get("min_edge", 0.04)):
            await self._reject(signal_id, "edge below threshold")
            return
        # Plausibility gate: when the market trades at a tiny implied
        # probability and the LLM is claiming a huge edge, we're almost
        # always looking at a hallucination — e.g. "Panama wins 2026
        # World Cup" priced at 0.002 where Ollama quotes true_prob=0.30
        # from a vague news headline. The hallucination_reject guard
        # above catches the *extreme* version (low_mid + high_true_prob
        # with thin sources); this is the softer cousin that triggers
        # whenever the edge itself is implausibly large relative to
        # the market price, even with sources. Keeps Ollama honest on
        # long-tail futures without interfering with genuine event
        # repricings (which sit closer to 0.5).
        plausibility_max_edge = float(
            cfg.get("plausibility_max_edge", 0.25)
        )
        plausibility_min_implied = float(
            cfg.get("plausibility_min_implied", 0.05)
        )
        # Symmetric: catch both tails. mid<0.05 with edge>+0.25 is the
        # "dark horse to win" hallucination; mid>0.95 with edge<-0.25
        # is the mirror ("favorite will actually lose") case. Either
        # way the market is already pricing an extreme — outsized edge
        # against that pricing is almost always model error.
        in_low_tail = mid < plausibility_min_implied
        in_high_tail = mid > (1.0 - plausibility_min_implied)
        if (
            plausibility_max_edge > 0
            and abs(edge) > plausibility_max_edge
            and (in_low_tail or in_high_tail)
        ):
            await self._reject(
                signal_id,
                f"implausible_edge edge={edge:+.2f} mid={mid:.3f}",
            )
            return

        try:
            decision = await self._risk.evaluate(market, side, implied, confidence)
        except RiskRejection as rej:
            await self._reject(signal_id, str(rej))
            return

        # Signals surface candidates; the three lanes (core.strategies.*)
        # own entry decisions and size via the allocator. This pipeline
        # persists the risk-checked size on the signal row for audit but
        # does not submit any orders — lanes pull their own opportunities
        # from feed_items / market state directly.
        await execute(
            "UPDATE signals SET status=?, size_usd=? WHERE id=?",
            ("APPROVED", decision.size_usd, signal_id),
        )

    async def _process_with_heuristic(
        self,
        *,
        item: dict[str, Any],
        markets: list[Any],
        prompt_version: str,
        text: str,
    ) -> None:
        """Heuristic-only path used when ``deep_realtime_enabled`` is
        False. Picks a single market — the linked one if the feed item
        has ``meta.linked_market_id``, otherwise the top heuristic
        candidate by keyword overlap — runs ``heuristic.score`` against
        it, and feeds the result through the same guards + risk
        evaluator as the deep path. The signal row's
        ``prompt_version`` is set to ``heuristic`` so the dashboard /
        operator can tell the source apart.
        """
        if not markets:
            return

        # Prefer the explicitly-linked market when the feed already
        # tagged one (polymarket_news, predictit_xref, per-market
        # google_news). Fall back to the top heuristic candidate.
        meta = _decode_meta(item.get("meta"))
        linked_id = (meta.get("linked_market_id") or "").strip()
        chosen_market: Any | None = None
        if linked_id:
            chosen_market = await get_market(linked_id)
        if chosen_market is None:
            chosen_market = markets[0]

        # Record so the rescore-cooldown guard above counts this market
        # as recently scored.
        self._market_last_scored[str(chosen_market.market_id)] = now_ts()

        h = heuristic.score(text, chosen_market)
        if h.direction == 0:
            # No directional signal from keywords. Persist a clearly-
            # labelled row so the operator can see the pipeline
            # processed the item but had nothing to act on.
            await execute(
                """INSERT INTO signals
                (feed_item_id, market_id, implied_prob, confidence, edge,
                 mid_price, side, size_usd, reasoning, prompt_version,
                 created_at, status, category)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    int(item["id"]),
                    chosen_market.market_id,
                    chosen_market.mid, 0.0, 0.0, chosen_market.mid,
                    "NONE", 0.0,
                    "heuristic: ambiguous (no directional keywords)",
                    prompt_version, now_ts(),
                    "heuristic_neutral",
                    (chosen_market.category or "").strip(),
                ),
            )
            return

        await self._finalize_signal(
            item=item,
            market=chosen_market,
            implied=h.implied_prob,
            confidence=h.confidence,
            reasoning=f"heuristic: {h.reasoning}"[:500],
            prompt_version=prompt_version,
        )

    async def _finalize_signal(
        self,
        *,
        item: dict[str, Any],
        market: Any,
        implied: float,
        confidence: float,
        reasoning: str,
        prompt_version: str,
    ) -> None:
        """Apply guards + insert + risk eval — shared between the deep
        and heuristic paths so they can't drift. The deep path inlines
        the same logic for historical reasons; this helper is the
        canonical implementation for the heuristic path and any
        future scoring source. It does NOT submit any orders.
        """
        mid = market.mid

        # Long-horizon cap on the implied prob — same rule as the deep path.
        guards = guard_config(get_config().get("risk"))
        days = days_until_resolve(market.close_time)
        capped_implied = apply_true_prob_cap(
            implied,
            days,
            long_horizon_days=guards["long_horizon_days"],
            cap=guards["long_horizon_cap"],
        )
        if capped_implied != implied:
            reasoning = (
                reasoning + f" [long_horizon_cap {implied:.2f}->{capped_implied:.2f}]"
            )[:500]
            implied = capped_implied

        # Hallucination guard. Heuristic confidence is already capped
        # at 0.70 by core.strategies.heuristic, but this guard is
        # symmetric across sources — a heuristic that cooks up a high
        # true_prob on a thin-evidence low-mid market should still be
        # rejected.
        num_sources, weighted_sources = await self._count_recent_sources(
            str(market.market_id),
        )
        reject_reason = hallucination_reject(
            true_prob=implied,
            mid=mid,
            num_sources=num_sources,
            weighted_sources=weighted_sources,
            low_mid_threshold=guards["low_mid_threshold"],
            high_true_prob_threshold=guards["high_true_prob_threshold"],
            min_sources=guards["min_sources"],
        )

        edge = implied - mid
        side = "BUY" if edge > 0 else "SELL"

        signal_id = await execute(
            """INSERT INTO signals
            (feed_item_id, market_id, implied_prob, confidence, edge, mid_price,
             side, size_usd, reasoning, prompt_version, created_at, status,
             category)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                int(item["id"]),
                market.market_id,
                implied, confidence, edge, mid, side, 0.0,
                reasoning, prompt_version, now_ts(),
                "PENDING",
                (market.category or "").strip(),
            ),
        )

        if reject_reason:
            await self._reject(signal_id, reject_reason)
            return

        cfg = get_config().get("risk") or {}
        if confidence < float(cfg.get("min_confidence", 0.65)):
            await self._reject(signal_id, "low confidence")
            return
        if abs(edge) < float(cfg.get("min_edge", 0.04)):
            await self._reject(signal_id, "edge below threshold")
            return

        plausibility_max_edge = float(cfg.get("plausibility_max_edge", 0.25))
        plausibility_min_implied = float(cfg.get("plausibility_min_implied", 0.05))
        in_low_tail = mid < plausibility_min_implied
        in_high_tail = mid > (1.0 - plausibility_min_implied)
        if (
            plausibility_max_edge > 0
            and abs(edge) > plausibility_max_edge
            and (in_low_tail or in_high_tail)
        ):
            await self._reject(
                signal_id,
                f"implausible_edge edge={edge:+.2f} mid={mid:.3f}",
            )
            return

        try:
            decision = await self._risk.evaluate(market, side, implied, confidence)
        except RiskRejection as rej:
            await self._reject(signal_id, str(rej))
            return

        await execute(
            "UPDATE signals SET status=?, size_usd=? WHERE id=?",
            ("APPROVED", decision.size_usd, signal_id),
        )

    @staticmethod
    async def _reject(signal_id: int, reason: str) -> None:
        await execute(
            "UPDATE signals SET status=?, reasoning=COALESCE(reasoning,'') || ?  WHERE id=?",
            ("REJECTED", f" | rejected: {reason}", signal_id),
        )

    @staticmethod
    async def _count_recent_sources(
        market_id: str, limit: int = 25,
    ) -> tuple[int, float]:
        """Distinct feed sources that recently tagged this market,
        returned as ``(raw_count, weighted_count)``. The weighted
        count sums per-source ``trust_weight`` from the nightly
        source_trust calibration — a well-calibrated source counts
        for more than a noisy one. Missing sources fall back to 1.0
        so pre-calibration behaviour is unchanged.

        Used by the hallucination guard on low-mid markets."""
        rows = await fetch_all(
            """SELECT DISTINCT source FROM feed_items
               WHERE meta LIKE ?
               ORDER BY ingested_at DESC LIMIT ?""",
            (f'%"linked_market_id":%"{market_id}"%', limit),
        )
        sources = {r["source"] for r in rows if r["source"]}
        raw = len(sources)
        weights = await get_source_weights()
        weighted = sum(trust_weight_for(s, weights) for s in sources)
        return raw, weighted


def _decode_meta(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}
    except (TypeError, ValueError):
        return {}
