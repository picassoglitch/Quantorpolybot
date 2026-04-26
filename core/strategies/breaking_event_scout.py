"""Lane: Breaking Event Scout (Step #3 PR #1).

Orchestrates the scout pipeline:

  1. Pull recent unprocessed `scout_signals` (since last_scan_ts).
  2. Normalize them into `Event` clusters (`core.scout.normalizer`).
  3. UPSERT into `breaking_events` so duplicate clusters across scan
     ticks collapse on `event_id`.
  4. For each NEW event (first-seen in this scan):
       a. Map to top-K markets (`core.scout.mapper`).
       b. For each mapped market: heuristic ImpactScore
          (`core.scout.impact`).
       c. Apply safety gate (`core.scout.candidate.evaluate`) — every
          decision (accept/reject) is persisted to
          `event_market_candidates` with a reason.
       d. For accepted candidates: allocator.reserve →
          shadow.open_position → attach the resulting position id back
          to the candidate row.

SHADOW only: positions are opened via `core.execution.shadow.open_position`,
which routes by `current_mode()`. With `mode: shadow` (and real lane
budgets at 0) no real CLOB orders can be placed even by accident.

The lane runs at a slow cadence (default 60s) — GDELT updates every
15 min and an event that's older than `max_event_age_seconds` (default
30 min) is rejected by the safety gate anyway, so faster polling
doesn't buy anything.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from core.execution import allocator, shadow
from core.markets import cache as market_cache
from core.scout import candidate as scout_candidate
from core.scout import impact as scout_impact
from core.scout import mapper as scout_mapper
from core.scout import normalizer as scout_normalizer
from core.scout.event import Event, EventCategory, Signal
from core.signals.ollama_client import OllamaClient
from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import now_ts, safe_float
from core.utils.prices import current_price


LANE = "breaking_event_scout"
allocator.register_strategy_lane(LANE, LANE)


def _cfg() -> dict[str, Any]:
    return get_config().get("breaking_event_scout") or {}


def _signal_from_row(row: Any) -> Signal:
    """Hydrate a scout_signals row into a Signal dataclass."""
    try:
        entities = json.loads(row["entities"] or "[]")
    except (TypeError, ValueError):
        entities = []
    try:
        raw = json.loads(row["raw_payload"] or "{}")
    except (TypeError, ValueError):
        raw = {}
    return Signal(
        source=row["source"] or "",
        source_type=row["source_type"] or "",
        timestamp=safe_float(row["published_at"]) or safe_float(row["ingested_at"]),
        title=row["title"] or "",
        body=row["body"] or "",
        entities=entities,
        category_hint=row["category_hint"] or "",
        url=row["url"] or "",
        confidence=safe_float(row["confidence"]),
        raw_payload=raw,
    )


class BreakingEventScoutLane:
    component = "strategies.breaking_event_scout"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        # Cursor: highest scout_signals.id we've already normalized.
        # Persisted nowhere — re-resumes from `MAX(last_seen_at)` on
        # restart via `_init_cursor`. A small replay window is
        # acceptable; the breaking_events UPSERT will collapse
        # duplicates by event_id.
        self._cursor_id: int = 0
        # v3: shared Ollama client for the LLM polarity inference
        # path. Lazily-instantiated process-wide in OllamaClient
        # itself, so this is essentially free.
        self._ollama = OllamaClient()

    async def run(self) -> None:
        cfg = _cfg()
        if not cfg.get("enabled", True):
            logger.info("[scout] lane disabled")
            return
        await self._init_cursor()
        interval = safe_float(cfg.get("scan_seconds", 60.0))
        logger.info("[scout] lane started (scan every {}s)", interval)
        try:
            while not self._stop.is_set():
                try:
                    await self.scan_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception("[scout] scan error: {}", e)
                await self._sleep(interval)
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
        """Resume cursor: events whose `last_seen_at` is recent come
        from signals we've already processed. Pick MAX(scout_signals.id)
        whose published_at <= MAX(last_seen_at) so we don't miss the
        ones still pending."""
        row = await fetch_one(
            "SELECT MAX(id) AS m FROM scout_signals "
            "WHERE ingested_at <= COALESCE("
            "  (SELECT MAX(last_seen_at) FROM breaking_events), 0)"
        )
        self._cursor_id = int((row["m"] or 0) if row else 0)

    async def scan_once(self) -> dict[str, int]:
        """One scan cycle.

        Returns a counter dict (consumed by tests + the scan summary
        log) so the operator and the test harness can assert what
        happened: `signals`, `events_seen`, `events_new`, `mapped`,
        `accepted`, `rejected`.
        """
        cfg = _cfg()
        max_signals_per_scan = int(cfg.get("max_signals_per_scan", 200))
        max_markets_per_event = int(cfg.get("max_markets_per_event", 5))
        cluster_window_seconds = safe_float(
            cfg.get("cluster_window_seconds", 1800.0)
        )
        # Mapper-side filters: keep aligned with the candidate gate so
        # markets that would fail the safety check don't even get
        # scored.
        mapper_min_score = safe_float(cfg.get("mapper_min_score", 0.20))
        mapper_min_liquidity = safe_float(cfg.get("min_liquidity", 1000.0))
        mapper_max_spread_cents = safe_float(cfg.get("max_spread_cents", 5.0))
        market_pool = int(cfg.get("market_pool_size", 500))

        counts = {
            "signals": 0, "events_seen": 0, "events_new": 0,
            "mapped": 0, "accepted": 0, "rejected": 0,
            # PR #7: scout noticed the event but no polarity rule
            # fired. Persisted with status="observed", no trade.
            "observed": 0,
            "no_mapping": 0,
        }

        # ---- 1. Pull new scout signals ----
        rows = await fetch_all(
            "SELECT * FROM scout_signals WHERE id > ? ORDER BY id ASC LIMIT ?",
            (self._cursor_id, max_signals_per_scan),
        )
        if not rows:
            return counts
        counts["signals"] = len(rows)

        signals: list[tuple[int, Signal]] = [
            (int(r["id"]), _signal_from_row(r)) for r in rows
        ]

        # ---- 2. Cluster into Events ----
        events = scout_normalizer.normalize(
            signals,
            cluster_window_seconds=cluster_window_seconds,
        )
        counts["events_seen"] = len(events)

        # ---- 3. Persist breaking_events (UPSERT-like by event_id) ----
        new_events = await self._upsert_events(events)
        counts["events_new"] = len(new_events)

        if not new_events:
            # Advance cursor; nothing else to do this tick.
            self._cursor_id = max(int(r["id"]) for r in rows)
            return counts

        # ---- 4. Map each new Event -> markets and evaluate ----
        markets = await market_cache.list_active(limit=market_pool)
        for event in new_events:
            matches = scout_mapper.map_event_to_markets(
                event,
                markets,
                top_k=max_markets_per_event,
                min_score=mapper_min_score,
                min_liquidity=mapper_min_liquidity,
                max_spread_cents=mapper_max_spread_cents,
            )
            if not matches:
                counts["no_mapping"] += 1
                logger.info(
                    "[scout] event={} cat={} sev={:.2f} -> no mapped markets",
                    event.event_id, event.category.value, event.severity,
                )
                continue
            counts["mapped"] += len(matches)
            for match in matches:
                outcome = await self._evaluate_and_maybe_open(event, match)
                if outcome == "accepted":
                    counts["accepted"] += 1
                elif outcome == "observed":
                    counts["observed"] += 1
                else:
                    counts["rejected"] += 1

        # Advance cursor only after a successful pass — partial
        # failures keep the unprocessed signals in scope for next
        # tick.
        self._cursor_id = max(int(r["id"]) for r in rows)

        if counts["events_new"] > 0:
            logger.info(
                "[scout] scan signals={} events_new={} mapped={} "
                "accepted={} observed={} rejected={} no_mapping={}",
                counts["signals"], counts["events_new"], counts["mapped"],
                counts["accepted"], counts["observed"], counts["rejected"],
                counts["no_mapping"],
            )
        return counts

    async def _upsert_events(self, events: list[Event]) -> list[Event]:
        """INSERT new events; UPDATE last_seen_at on existing.
        Returns the subset that were freshly inserted (i.e. new to
        the DB). The lane only runs the mapper/scorer/candidate
        pipeline for those — re-running them on a re-seen cluster
        would create duplicate audit rows."""
        new_events: list[Event] = []
        now_t = now_ts()
        for ev in events:
            existing = await fetch_one(
                "SELECT first_seen_at FROM breaking_events WHERE event_id=?",
                (ev.event_id,),
            )
            if existing is not None:
                # Touch last_seen_at + source_count so a long-running
                # cluster's metadata stays fresh.
                await execute(
                    """UPDATE breaking_events
                       SET last_seen_at=?, source_count=?, sources=?,
                           confidence=?, severity=?
                       WHERE event_id=?""",
                    (
                        now_t,
                        ev.source_count,
                        json.dumps(ev.sources),
                        ev.confidence,
                        ev.severity,
                        ev.event_id,
                    ),
                )
                continue
            try:
                await execute(
                    """INSERT INTO breaking_events
                         (event_id, timestamp_detected, title, category,
                          severity, confidence, location, entities,
                          source_count, sources, contradiction_score,
                          raw_signal_ids, first_seen_at, last_seen_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ev.event_id,
                        ev.timestamp_detected,
                        ev.title,
                        ev.category.value,
                        ev.severity,
                        ev.confidence,
                        ev.location,
                        json.dumps(ev.entities),
                        ev.source_count,
                        json.dumps(ev.sources),
                        ev.contradiction_score,
                        json.dumps(ev.raw_signal_ids),
                        ev.first_seen_at,
                        ev.last_seen_at,
                    ),
                )
                new_events.append(ev)
            except Exception as e:
                logger.warning(
                    "[scout] persist event {} failed: {}", ev.event_id, e,
                )
        return new_events

    async def _evaluate_and_maybe_open(
        self, event: Event, match: scout_mapper.MarketMatch,
    ) -> str:
        """Score impact, run the safety gate, and open a SHADOW
        position if accepted.

        Returns one of:
          - ``"accepted"`` — SHADOW position was opened
          - ``"observed"`` — scout noticed the event but no polarity
            rule fired; row persisted with ``status="observed"``,
            no trade
          - ``"rejected"`` — failed a safety/edge/confidence gate;
            row persisted with ``status="rejected"`` and a reason
        """
        # Get a fresh price snapshot — the cached market.mid may be
        # stale by minutes; for breaking-news entries that's the
        # difference between hitting an exit-able price and not.
        snap = await current_price(
            match.market.market_id, match.market.yes_token() or ""
        )
        market_mid = (
            snap.mid if snap is not None and snap.mid > 0 else match.market.mid
        )

        # v3: route through the async wrapper so the LLM-polarity
        # path can fire on high-severity events the rule table can't
        # resolve. The wrapper falls through to the rule-based score
        # on any LLM failure, so this never blocks the lane on
        # Ollama latency beyond the configured timeout.
        cfg_for_eval = _cfg()
        impact = await scout_impact.score_impact_async(
            event, match,
            market_mid=market_mid,
            ollama_client=self._ollama,
            llm_enabled=bool(cfg_for_eval.get("llm_polarity_enabled", True)),
            llm_severity_floor=safe_float(
                cfg_for_eval.get("llm_polarity_severity_threshold", 0.70),
            ),
            llm_timeout_seconds=safe_float(
                cfg_for_eval.get("llm_polarity_timeout_seconds", 5.0),
            ),
        )
        decision = await scout_candidate.evaluate(
            event, match, impact, market_mid=market_mid, cfg=cfg_for_eval,
        )
        if decision.observed:
            # Visible-not-traded path. Logged at INFO so the operator
            # sees that the scout DID notice the event (separate from
            # rejected, which means a safety rule blocked it).
            logger.info(
                "[scout] OBSERVED event={} market={} sev={:.2f} "
                "match_score={:.2f} (no polarity rule, no trade)",
                event.event_id, match.market.market_id,
                event.severity, match.score,
            )
            return "observed"
        if not decision.accepted:
            logger.info(
                "[scout] reject event={} market={} side={} edge={:.3f} "
                "reason={}",
                event.event_id, match.market.market_id,
                decision.side or "-", decision.edge, decision.reason,
            )
            return "rejected"

        # Lane state checks before reserving capital.
        state = await allocator.get_state(LANE)
        if state is None or state.is_paused:
            logger.info(
                "[scout] event={} market={} accepted but lane paused/missing — "
                "skipping open",
                event.event_id, match.market.market_id,
            )
            return "rejected"

        size_usd = await _position_size(decision.confidence, _cfg())
        # v3: apply the high-sev-solo size multiplier (0.30x by
        # default) when the candidate passed corroboration via the
        # single-source override. Cooldown / max-exposure / edge
        # gates already ran inside scout_candidate.evaluate — this
        # is sizing-only.
        if decision.size_multiplier != 1.0:
            size_usd = max(0.0, size_usd * decision.size_multiplier)
            logger.info(
                "[scout] event={} market={} sized down to {:.2f} "
                "(multiplier={:.2f}, high_sev_solo={})",
                event.event_id, match.market.market_id,
                size_usd, decision.size_multiplier,
                decision.high_sev_solo,
            )
        approved = await allocator.reserve(LANE, size_usd)
        if approved is None:
            logger.info(
                "[scout] event={} market={} accepted but allocator denied "
                "(wanted={:.2f})",
                event.event_id, match.market.market_id, size_usd,
            )
            return "rejected"

        if snap is None or snap.mid <= 0:
            await allocator.release(LANE, approved, 0.0)
            logger.info(
                "[scout] event={} market={} no live snapshot at open time",
                event.event_id, match.market.market_id,
            )
            return "rejected"

        pos_id = await shadow.open_position(
            strategy=LANE,
            market=match.market,
            side=decision.side,
            snapshot=snap,
            size_usd=approved,
            true_prob=decision.true_prob,
            confidence=decision.confidence,
            entry_reason=(
                f"scout[{event.category.value}] event={event.event_id} "
                f"edge={decision.edge:+.3f} sev={event.severity:.2f} "
                f"sources={event.source_count}"
            ),
            evidence_ids=event.raw_signal_ids,
            evidence_snapshot={
                "lane": LANE,
                "event_id": event.event_id,
                "event_title": event.title,
                "event_category": event.category.value,
                "event_severity": event.severity,
                "event_confidence": event.confidence,
                "event_sources": event.sources,
                "event_entities": event.entities,
                "match_score": match.score,
                "impact_components": impact.components,
                "polarity_reasoning": impact.polarity_reasoning,
            },
            entry_latency_ms=max(0.0, (now_ts() - event.timestamp_detected) * 1000.0),
        )
        if pos_id is None:
            await allocator.release(LANE, approved, 0.0)
            logger.warning(
                "[scout] open_position returned None for event={} market={}",
                event.event_id, match.market.market_id,
            )
            return "rejected"

        await scout_candidate.attach_position(decision.audit_row_id, pos_id)
        logger.info(
            "[scout] OPENED event={} market={} side={} size={:.2f} "
            "edge={:.3f} conf={:.2f}",
            event.event_id, match.market.market_id, decision.side,
            approved, decision.edge, decision.confidence,
        )
        return "accepted"


async def _position_size(confidence: float, cfg: dict[str, Any]) -> float:
    """Lane-side sizing — independent of the LLM-Kelly path the main
    pipeline uses. Linear blend between `base_position` and
    `max_position` driven by confidence.
    """
    base = safe_float(cfg.get("base_position_usd", 1.0))
    cap = safe_float(cfg.get("max_position_usd", 3.0))
    # Confidence cap is 0.55 from the heuristic scorer; treat 0.40
    # as the lower anchor (matches the impact scorer's confidence
    # floor).
    if cap <= base:
        return base
    span = max(0.0, min(1.0, (confidence - 0.40) / (0.55 - 0.40)))
    return base + (cap - base) * span
