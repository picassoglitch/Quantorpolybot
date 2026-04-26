"""SHADOW candidate generator + safety gate + persistence.

Sits between the impact scorer and the lane. For each (Event, Market,
ImpactScore) triple, applies the spec'd safety rules and returns
either an `AcceptedCandidate` (the lane should open a SHADOW
position) or a `RejectedCandidate` (the lane logs and skips).

EVERY decision (accept or reject) is persisted to
`event_market_candidates` so the dashboard / pattern-discovery layer
can answer "why did the scout reject Event X for Market Y" without
log spelunking.

Safety rules (from the user spec):

  1. Event has 2+ credible sources OR a single primary source
  2. Market liquidity is sufficient (gated upstream by mapper, but
     re-checked here as a belt-and-suspenders)
  3. Spread is acceptable (also re-checked)
  4. Expected edge exceeds threshold
  5. No contradicting higher-quality source (v1: stub — needs
     Step #3 PR 1.5+ to materialise)
  6. Event is fresh enough to matter
  7. Per-event max exposure (cool-down): only ONE accepted candidate
     per `event_id` per `event_max_positions` window
  8. Per-event max total size across all mapped markets

This module never opens positions itself — the lane does that. This
keeps the `accept` decision pure-ish (one DB write for the candidate
audit row, no allocator touch) and easy to test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from core.markets.cache import Market
from core.scout.event import Event
from core.scout.impact import ImpactScore
from core.scout.mapper import MarketMatch
from core.utils.db import execute, fetch_one
from core.utils.helpers import now_ts, safe_float


@dataclass
class CandidateDecision:
    """One side of the accept/reject/observed union. Lane checks
    `accepted` for the trade decision; `observed` for the
    "noticed-but-not-traded" path.

    `audit_row_id` is the rowid of the persisted row in
    `event_market_candidates`; the lane updates it with the
    `shadow_position_id` once the position opens (so the audit trail
    is complete). For observed candidates, no position is opened and
    the row's status remains "observed".
    """

    accepted: bool
    event_id: str
    market_id: str
    side: str               # "BUY" / "SELL" (= BUY NO) / "" if rejected/observed
    true_prob: float
    confidence: float
    edge: float
    market_mid: float
    reason: str
    impact_snapshot: dict[str, Any] = field(default_factory=dict)
    audit_row_id: int = 0
    # PR #7: marks an "observed" decision — the scout noticed the
    # event but no polarity rule fired. Persisted with
    # status="observed", no shadow trade opened. Lane uses this to
    # increment the observed-counter in scan summaries.
    observed: bool = False


async def evaluate(
    event: Event,
    match: MarketMatch,
    impact: ImpactScore,
    *,
    market_mid: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> CandidateDecision:
    """Apply safety rules → CandidateDecision → persist audit row.

    Pure rules + one DB INSERT. Does NOT call the allocator and does
    NOT open shadow positions. The lane is responsible for both.

    `cfg` is the scout's lane config block (defaults supplied).
    """
    cfg = cfg or {}
    market = match.market
    mid = market_mid if market_mid is not None else market.mid

    min_sources = int(cfg.get("min_sources", 2))
    primary_sources = set(cfg.get("primary_sources") or [
        "reuters", "apnews", "bloomberg",
        # GDELT-side: domain-tier 0.85 articles count as primary
        # via the "tier1_domain" check below. The connector tags
        # those in raw_payload but the simpler check is on
        # event.confidence (mean of signal confidences) — primary
        # sources push event.confidence high.
    ])
    min_edge = safe_float(cfg.get("min_edge", 0.05))
    min_confidence = safe_float(cfg.get("min_confidence", 0.40))
    max_event_age_seconds = safe_float(cfg.get("max_event_age_seconds", 1800.0))
    min_liquidity = safe_float(cfg.get("min_liquidity", 1000.0))
    max_spread_cents = safe_float(cfg.get("max_spread_cents", 5.0))
    event_max_positions = int(cfg.get("event_max_positions", 1))
    require_primary_or_corroboration = bool(
        cfg.get("require_primary_or_corroboration", True)
    )

    # ---- Build the audit snapshot up front (used for both branches) ----
    audit_snapshot = {
        "event": {
            "category": event.category.value,
            "severity": event.severity,
            "confidence": event.confidence,
            "source_count": event.source_count,
            "sources": event.sources,
            "entities": event.entities,
            "title": event.title,
        },
        "market_match": {
            "score": match.score,
            "entity_overlap": match.entity_overlap,
            "keyword_overlap": match.keyword_overlap,
            "category_alignment": match.category_alignment,
            "liquidity_quality": match.liquidity_quality,
            "near_resolution_bonus": match.near_resolution_bonus,
        },
        "impact": {
            "direction": impact.direction,
            "true_prob": impact.true_prob,
            "confidence": impact.confidence,
            "expected_nudge": impact.expected_nudge,
            "polarity_reasoning": impact.polarity_reasoning,
            "components": impact.components,
        },
        "market": {
            "mid": mid,
            "best_bid": market.best_bid,
            "best_ask": market.best_ask,
            "spread_cents": (market.best_ask - market.best_bid) * 100.0,
            "liquidity": market.liquidity,
            "yes_token": market.yes_token() or "",
        },
    }

    # ---- Run rules in cheap-first order ----
    reject = _check_rules(
        event=event,
        market=market,
        impact=impact,
        mid=mid,
        min_sources=min_sources,
        primary_sources=primary_sources,
        min_edge=min_edge,
        min_confidence=min_confidence,
        max_event_age_seconds=max_event_age_seconds,
        min_liquidity=min_liquidity,
        max_spread_cents=max_spread_cents,
        require_primary_or_corroboration=require_primary_or_corroboration,
    )

    # ---- Per-event cool-down: only N accepted candidates per event_id ----
    if reject is None:
        accepted_for_event = await _count_accepted_for_event(event.event_id)
        if accepted_for_event >= event_max_positions:
            reject = (
                f"event cooldown: {accepted_for_event} accepted candidate(s) "
                f"already exist for this event_id (cap {event_max_positions})"
            )

    # ---- Observed-mode short-circuit (PR #7) ----
    # When the impact scorer flagged this as a watchlist signal
    # (high severity + strong match but no polarity rule), persist
    # as status="observed" — visible in audit, NOT traded. We do
    # this AFTER the safety/cooldown checks (a bad-corroboration
    # event still records as "rejected") but BEFORE the
    # direction/edge gates (which would otherwise reject it as
    # polarity_unknown).
    if reject is None and impact.observed:
        row_id = await _persist(
            event_id=event.event_id,
            market_id=market.market_id,
            status="observed",
            reject_reason="",
            side="",
            true_prob=impact.true_prob,
            confidence=impact.confidence,
            edge=0.0,
            market_mid=mid,
            impact_snapshot=audit_snapshot,
        )
        return CandidateDecision(
            accepted=False,
            event_id=event.event_id,
            market_id=market.market_id,
            side="",
            true_prob=impact.true_prob,
            confidence=impact.confidence,
            edge=0.0,
            market_mid=mid,
            reason=(
                f"observed: severity={event.severity:.2f} "
                f"match_score={match.score:.2f} "
                f"polarity_unknown=true (no trade)"
            ),
            impact_snapshot=audit_snapshot,
            audit_row_id=row_id,
            observed=True,
        )

    # ---- Direction must be non-zero (after observed-mode check) ----
    if reject is None and impact.direction == 0:
        reject = f"polarity_unknown: {impact.polarity_reasoning}"

    edge = impact.true_prob - mid
    if reject is None and abs(edge) < min_edge:
        reject = f"edge {abs(edge):.3f} < min_edge {min_edge:.3f}"

    side = "BUY" if impact.direction > 0 else "SELL"

    if reject is not None:
        row_id = await _persist(
            event_id=event.event_id,
            market_id=market.market_id,
            status="rejected",
            reject_reason=reject,
            side="",
            true_prob=impact.true_prob,
            confidence=impact.confidence,
            edge=edge,
            market_mid=mid,
            impact_snapshot=audit_snapshot,
        )
        return CandidateDecision(
            accepted=False,
            event_id=event.event_id,
            market_id=market.market_id,
            side="",
            true_prob=impact.true_prob,
            confidence=impact.confidence,
            edge=edge,
            market_mid=mid,
            reason=reject,
            impact_snapshot=audit_snapshot,
            audit_row_id=row_id,
        )

    row_id = await _persist(
        event_id=event.event_id,
        market_id=market.market_id,
        status="accepted",
        reject_reason="",
        side=side,
        true_prob=impact.true_prob,
        confidence=impact.confidence,
        edge=edge,
        market_mid=mid,
        impact_snapshot=audit_snapshot,
    )
    return CandidateDecision(
        accepted=True,
        event_id=event.event_id,
        market_id=market.market_id,
        side=side,
        true_prob=impact.true_prob,
        confidence=impact.confidence,
        edge=edge,
        market_mid=mid,
        reason=(
            f"accepted: edge={edge:+.3f} conf={impact.confidence:.2f} "
            f"sources={event.source_count} sev={event.severity:.2f}"
        ),
        impact_snapshot=audit_snapshot,
        audit_row_id=row_id,
    )


def _check_rules(
    *,
    event: Event,
    market: Market,
    impact: ImpactScore,
    mid: float,
    min_sources: int,
    primary_sources: set[str],
    min_edge: float,
    min_confidence: float,
    max_event_age_seconds: float,
    min_liquidity: float,
    max_spread_cents: float,
    require_primary_or_corroboration: bool,
) -> str | None:
    """Returns first failing rule's reason string, or None on pass."""
    age = now_ts() - event.timestamp_detected
    if age > max_event_age_seconds:
        return (
            f"event too old: age {age:.0f}s > max_event_age "
            f"{max_event_age_seconds:.0f}s"
        )
    if mid <= 0.0 or mid >= 1.0:
        return f"invalid mid {mid:.3f} (out of (0,1))"
    if market.liquidity < min_liquidity:
        return f"liquidity {market.liquidity:.0f} < min {min_liquidity:.0f}"
    spread_cents = (market.best_ask - market.best_bid) * 100.0
    if spread_cents <= 0 or spread_cents > max_spread_cents:
        return (
            f"spread {spread_cents:.1f}c outside (0, {max_spread_cents:.1f}c]"
        )
    # Skip the impact.confidence gate when direction == 0. Both
    # downstream paths (observed-mode and polarity_unknown) handle
    # the no-direction case explicitly with their own logic, and
    # observed-mode is BELOW this gate by design (~0.20). Without
    # this, observed candidates would always reject as
    # "impact confidence 0.20 < 0.40" and never reach the observed
    # branch.
    if impact.direction != 0 and impact.confidence < min_confidence:
        return (
            f"impact confidence {impact.confidence:.2f} < "
            f"{min_confidence:.2f}"
        )
    if require_primary_or_corroboration:
        # Two paths to satisfaction:
        #   a) >= min_sources distinct sources (corroboration)
        #   b) at least one source name in primary_sources
        has_corroboration = event.source_count >= min_sources
        has_primary = bool(set(s.lower() for s in event.sources) & primary_sources)
        if not (has_corroboration or has_primary):
            return (
                f"insufficient corroboration: source_count={event.source_count} "
                f"< {min_sources} and no primary source in {sorted(primary_sources)}"
            )
    return None


async def _count_accepted_for_event(event_id: str) -> int:
    row = await fetch_one(
        """SELECT COUNT(*) AS n
           FROM event_market_candidates
           WHERE event_id=? AND status='accepted'""",
        (event_id,),
    )
    return int(row["n"] if row else 0)


async def _persist(
    *,
    event_id: str,
    market_id: str,
    status: str,
    reject_reason: str,
    side: str,
    true_prob: float,
    confidence: float,
    edge: float,
    market_mid: float,
    impact_snapshot: dict[str, Any],
) -> int:
    """Single INSERT; returns rowid. No-throw — a persistence failure
    must NOT prevent the lane from making a decision (the rule
    evaluation already produced an in-memory result that the lane will
    log). Returns 0 on failure so the lane can detect a missing audit
    row if it cares."""
    try:
        return await execute(
            """INSERT INTO event_market_candidates
                 (event_id, market_id, considered_at, status, reject_reason,
                  side, true_prob, confidence, edge, market_mid,
                  impact_snapshot, shadow_position_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                market_id,
                now_ts(),
                status,
                reject_reason,
                side,
                true_prob,
                confidence,
                edge,
                market_mid,
                json.dumps(impact_snapshot, default=str)[:8000],
                None,
            ),
        )
    except Exception as e:
        logger.warning(
            "[scout] persist candidate failed (event={}, market={}): {}",
            event_id, market_id, e,
        )
        return 0


async def attach_position(audit_row_id: int, shadow_position_id: int) -> None:
    """Link an accepted candidate audit row to the shadow_positions row
    that the lane just opened. Called by the lane after `open_position`
    succeeds. No-throw."""
    if audit_row_id <= 0 or shadow_position_id <= 0:
        return
    try:
        await execute(
            "UPDATE event_market_candidates SET shadow_position_id=? WHERE id=?",
            (shadow_position_id, audit_row_id),
        )
    except Exception as e:
        logger.debug(
            "[scout] attach_position failed (audit_id={}, position={}): {}",
            audit_row_id, shadow_position_id, e,
        )
