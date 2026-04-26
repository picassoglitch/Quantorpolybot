"""Heuristic impact scorer for the scout lane (PR #1, no LLM).

Given an Event + a mapped Market + the market's mid price snapshot,
estimate the probability impact of the event:

  - direction (BUY YES / SELL YES / no signal)
  - magnitude (how much should the price move)
  - confidence (how sure are we in this estimate)

Direction is the hard part. v1 uses a small lookup table of category
priors crossed with simple polarity inference from the event title:

  - "ceasefire" + market mentions "war ends by ..." -> BUY YES
  - "shooting" + market mentions "X attends Y" -> SELL YES (reduces
    attendance probability)
  - "indictment" + market mentions "X wins primary" -> direction
    depends on the historical base rate (which we don't know in v1);
    the scorer returns NEUTRAL and the lane skips.

The scorer is conservative: when polarity is ambiguous, it returns
direction=0 and the lane records a `polarity_unknown` reject reason
rather than guess.

Magnitude (v1):
  Per-category nudge size (in [0,1] probability units), scaled by
  event severity and the (Event, Market) match score.

Confidence (v1):
  ``min(0.55, event.confidence * match.score)``. Hard-capped at 0.55
  — a heuristic scorer cannot match the conviction of an LLM-backed
  scorer and the lane's gates should treat it accordingly. The LLM
  variant (roadmap, future PR) can lift the cap.

TODO(scout-llm): Replace polarity inference with an LLM call against
the deep tier (or a fine-tuned smaller model) once the basic loop is
proven and the budget for the call is justified by realized PnL.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.scout.event import Event, EventCategory
from core.scout.mapper import MarketMatch
from core.utils.helpers import clamp


@dataclass
class ImpactScore:
    """Result of `score_impact`. `direction` is +1 (BUY YES), -1
    (SELL YES = BUY NO), or 0 (no actionable signal).

    `true_prob` is the scorer's estimate of the YES probability after
    the event. The lane computes `edge = true_prob - market_mid` and
    gates accordingly.
    """

    direction: int
    true_prob: float
    confidence: float
    expected_nudge: float          # absolute probability move scorer expects
    polarity_reasoning: str
    components: dict[str, float]


# Per-category baseline nudge magnitudes (probability units). Conservative;
# multiplied by severity * match_score before being applied.
_CATEGORY_NUDGE: dict[EventCategory, float] = {
    EventCategory.ASSASSINATION_ATTEMPT: 0.15,
    EventCategory.SHOOTING: 0.08,
    EventCategory.WAR_ESCALATION: 0.12,
    EventCategory.CEASEFIRE: 0.15,
    EventCategory.ELECTION_RESULT: 0.20,
    EventCategory.COURT_RULING: 0.12,
    EventCategory.INDICTMENT: 0.10,
    EventCategory.RESIGNATION: 0.10,
    EventCategory.ARREST: 0.08,
    EventCategory.DEATH_INJURY: 0.10,
    EventCategory.EVACUATION: 0.06,
    EventCategory.MACRO_DATA_SURPRISE: 0.05,
    EventCategory.SPORTS_INJURY: 0.08,
    EventCategory.OTHER: 0.0,
}


# Polarity inference: per-category list of (market-question keywords,
# direction) tuples. The first match wins. If no entry matches the
# market's question, `direction=0` is returned (the lane logs
# `polarity_unknown` and skips — the spec is "Do not trade blindly
# from one headline").
#
# Keywords are checked as substrings against the lowered market
# question. Order matters within a category: more specific keywords
# should come first.
#
# Keyword stems are picked to substring-match common conjugations
# without false positives — "attend" matches "attend"/"attends"/
# "attending"; "resign" matches "resign"/"resigns"/"resigned"/
# "resignation". Order matters within a category: more specific
# keywords should come first.
_POLARITY_RULES: dict[EventCategory, list[tuple[str, int]]] = {
    EventCategory.CEASEFIRE: [
        ("war end", +1), ("ceasefire", +1), ("peace deal", +1),
        ("conflict end", +1), ("escalat", -1),
    ],
    EventCategory.WAR_ESCALATION: [
        ("ceasefire", -1), ("peace deal", -1), ("war end", -1),
        ("escalat", +1), ("invasion", +1),
    ],
    EventCategory.ASSASSINATION_ATTEMPT: [
        ("attend", -1), ("appear", -1), ("speak", -1),
        ("survive", +1),
    ],
    EventCategory.SHOOTING: [
        ("attend", -1), ("appear", -1), ("happen", -1),
        ("evacuat", +1),
    ],
    EventCategory.RESIGNATION: [
        ("remain", -1), ("stay as", -1), ("resign", +1),
        ("step down", +1),
    ],
    EventCategory.ARREST: [
        ("convict", +1), ("arrest", +1), ("charge", +1),
        ("walks free", -1), ("acquit", -1),
    ],
    EventCategory.INDICTMENT: [
        ("indict", +1), ("charge", +1), ("convict", +1),
        ("acquit", -1), ("dropped", -1),
    ],
    EventCategory.COURT_RULING: [
        ("ruled in favor", +1), ("ruled against", -1),
        ("uphold", +1), ("strike down", -1),
    ],
    EventCategory.ELECTION_RESULT: [
        # Hard to infer direction without knowing WHO won. Left
        # empty; lane will log polarity_unknown unless a future
        # version learns the winner from the event title.
    ],
    EventCategory.EVACUATION: [
        ("happen", -1), ("attend", -1), ("on schedule", -1),
        ("postpon", +1), ("cancel", +1),
    ],
    EventCategory.DEATH_INJURY: [
        ("attend", -1), ("appear", -1), ("recover", +1),
    ],
    EventCategory.SPORTS_INJURY: [
        ("play", -1), ("start", -1), ("appear", -1),
        ("score", -1), ("win", -1),
    ],
    EventCategory.MACRO_DATA_SURPRISE: [
        # Direction depends on the surprise sign and which market;
        # too noisy to encode as static rules. v1 returns
        # polarity_unknown.
    ],
}


def _infer_polarity(category: EventCategory, market_question: str) -> tuple[int, str]:
    """Return (direction, reasoning_string)."""
    rules = _POLARITY_RULES.get(category, [])
    if not rules or not market_question:
        return 0, f"no polarity rules for category={category.value}"
    low = market_question.lower()
    for keyword, direction in rules:
        if keyword in low:
            return direction, (
                f"polarity rule matched: '{keyword}' in market -> "
                f"{'BUY YES' if direction > 0 else 'SELL YES'}"
            )
    return 0, (
        f"no polarity rule matched for category={category.value} "
        f"in market question"
    )


def score_impact(
    event: Event,
    match: MarketMatch,
    *,
    market_mid: float | None = None,
    confidence_cap: float = 0.55,
) -> ImpactScore:
    """Compute the heuristic ImpactScore for one (Event, Market) pair.

    `market_mid` defaults to `match.market.mid` — pass an explicit
    value when the lane has a fresher snapshot than the cached
    market record.
    """
    market = match.market
    mid = market_mid if market_mid is not None else market.mid
    mid = clamp(mid, 0.01, 0.99)

    direction, polarity_reason = _infer_polarity(event.category, market.question)
    base_nudge = _CATEGORY_NUDGE.get(event.category, 0.0)
    # Effective nudge: shrink by severity × match score so weak
    # matches don't claim category-baseline edge.
    effective_nudge = base_nudge * event.severity * max(0.3, match.score)

    if direction == 0 or effective_nudge <= 0.0:
        return ImpactScore(
            direction=0,
            true_prob=mid,
            confidence=0.0,
            expected_nudge=0.0,
            polarity_reasoning=polarity_reason,
            components={
                "category_nudge": base_nudge,
                "event_severity": event.severity,
                "match_score": match.score,
                "effective_nudge": effective_nudge,
                "market_mid": mid,
            },
        )

    true_prob = clamp(mid + direction * effective_nudge, 0.01, 0.99)
    confidence = min(
        confidence_cap,
        event.confidence * max(0.3, match.score),
    )
    return ImpactScore(
        direction=direction,
        true_prob=true_prob,
        confidence=confidence,
        expected_nudge=effective_nudge,
        polarity_reasoning=polarity_reason,
        components={
            "category_nudge": base_nudge,
            "event_severity": event.severity,
            "match_score": match.score,
            "effective_nudge": effective_nudge,
            "market_mid": mid,
            "event_confidence": event.confidence,
        },
    )
