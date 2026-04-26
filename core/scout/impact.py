"""Heuristic impact scorer for the scout lane (PR #1, no LLM in path).

Given an Event + a mapped Market + the market's mid price snapshot,
estimate the probability impact of the event:

  - direction (BUY YES / SELL YES / no signal)
  - magnitude (how much should the price move)
  - confidence (how sure are we in this estimate)
  - observed flag (signal we noticed but can't trade — see below)

Direction comes from a per-category polarity rule table. Each category
has a list of ``(market_question_substring, direction)`` rules. The
first match wins. Compared to v1, the rules are now broad enough to
cover the common Polymarket question shapes a scouted event maps to:

  - "Will X happen by [date]?"    — outcome markets
  - "Will X attend / appear / speak at Y?"  — appearance markets
  - "Will X resign / step down by [date]?"  — resignation markets
  - "Will X be charged / convicted / acquitted?" — legal markets
  - "Will X win [election/race/primary]?"   — political markets

Observed-mode (PR #7):

  When polarity inference fails (no rule matches the market question)
  but the event is high-severity AND the market match is strong, the
  scorer returns ``direction=0, observed=True, confidence>0``. The
  candidate evaluator persists this with ``status="observed"`` — the
  scout NOTICED the event but doesn't trade it. That keeps the
  Pattern Discovery layer's audit trail complete (we know the event
  reached us, and we know we couldn't price it) without firing a
  reckless trade on a no-polarity signal.

Magnitude (v1, unchanged):

  Per-category nudge size (in [0,1] probability units), scaled by
  event severity and the (Event, Market) match score.

Confidence (v2, this PR):

  Old formula was multiplicative — ``event.confidence * max(0.3,
  match.score)`` — which punished perfectly-scored events with a
  middling match (0.85 × 0.4 = 0.34, below the 0.40 gate). New
  formula is a weighted average × severity multiplier:

      base       = 0.6 * event.confidence + 0.4 * max(0.3, match.score)
      sev_mult   = max(0.5, event.severity)
      confidence = min(0.55, base * sev_mult)

  Same 0.55 cap. Strong events with moderate matches now clear the
  gate; weak matches still get filtered (same outputs as v1 in the
  weak-match regime).

TODO(scout-llm): Replace polarity inference with an LLM call against
the deep tier (or a fine-tuned smaller model) once realized PnL
justifies the call cost. The seam is ``_infer_polarity()``.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.scout.event import Event, EventCategory
from core.scout.mapper import MarketMatch
from core.utils.helpers import clamp


# Confidence cap for the heuristic scorer. Even maximally-strong
# inputs cannot lift confidence above this — we're keyword-matching,
# not actually understanding context. The lane's per-tier
# `min_confidence` (default 0.40) sits below this cap so accepted
# candidates have a defined band.
_HEURISTIC_CONFIDENCE_CAP = 0.55

# Observed-mode thresholds. When polarity fails but BOTH the event
# severity AND the (event, market) match score clear these floors,
# the scorer returns a low-confidence ImpactScore with ``observed=True``
# so the lane records a watchlist entry instead of silently rejecting.
_OBSERVED_MIN_SEVERITY = 0.60
_OBSERVED_MIN_MATCH_SCORE = 0.30
_OBSERVED_CONFIDENCE = 0.20  # Below the 0.40 trade gate, above 0.0


@dataclass
class ImpactScore:
    """Result of `score_impact`. ``direction`` is +1 (BUY YES), -1
    (SELL YES = BUY NO), or 0 (no actionable directional signal).

    ``true_prob`` is the scorer's estimate of the YES probability
    after the event. The lane computes ``edge = true_prob - market_mid``
    and gates accordingly.

    ``observed`` (PR #7) marks an ImpactScore that the lane should
    persist for visibility but NOT trade on. Direction is 0 in this
    case; confidence is ~0.20 (below the trade gate by design).
    """

    direction: int
    true_prob: float
    confidence: float
    expected_nudge: float          # absolute probability move scorer expects
    polarity_reasoning: str
    components: dict[str, float]
    observed: bool = False         # PR #7: watchlist signal, no trade


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


# Polarity inference rules. Per-category list of
# ``(market_question_substring, direction)`` tuples — substring-matched
# against the lowered market question. First match wins. Order within
# a category matters: more specific phrases come before more general
# ones.
#
# Direction conventions:
#   +1 → BUY YES (event makes the YES outcome MORE likely)
#   -1 → SELL YES = BUY NO (event makes the YES outcome LESS likely)
#
# Keyword stems use root forms so they substring-match common
# conjugations: "attend" matches "attend"/"attends"/"attending";
# "resign" matches "resign"/"resigns"/"resigned"/"resignation".
#
# Coverage philosophy (v2): broad enough to catch the realistic
# Polymarket question shapes for each category. Roughly 8-15 patterns
# per category. Empty categories (ELECTION_RESULT, MACRO_DATA_SURPRISE)
# have no static polarity — those need the LLM-backed scorer to be
# useful at all (TODO marker above).
_POLARITY_RULES: dict[EventCategory, list[tuple[str, int]]] = {
    EventCategory.CEASEFIRE: [
        # YES side benefits from a ceasefire announcement.
        ("war end", +1), ("war ends", +1),
        ("ceasefire", +1), ("cease-fire", +1),
        ("peace deal", +1), ("peace agreement", +1),
        ("conflict end", +1), ("hostilities end", +1),
        ("agreement reached", +1),
        ("withdraw", +1), ("withdrawal", +1),
        # YES side hurt by a ceasefire (continued-fighting markets).
        ("escalat", -1), ("invasion", -1),
        ("territory captured", -1),
        ("war continues", -1),
    ],
    EventCategory.WAR_ESCALATION: [
        # YES side hurt by escalation.
        ("ceasefire", -1), ("cease-fire", -1),
        ("peace deal", -1), ("peace agreement", -1),
        ("war end", -1), ("withdraw", -1),
        # YES side benefits from escalation.
        ("escalat", +1), ("invasion", +1),
        ("airstrike", +1), ("missile strike", +1),
        ("ground offensive", +1),
        ("territory captured", +1),
        ("nuclear", +1),  # markets framed as "will Russia use a nuke"
    ],
    EventCategory.ASSASSINATION_ATTEMPT: [
        # Target's appearance / event attendance: attempt → bearish
        # (target may pull out, especially for short-horizon events).
        ("attend", -1), ("appear", -1),
        ("speak at", -1), ("present at", -1),
        ("rally", -1), ("debate", -1),
        ("public event", -1),
        # Target survival markets: attempt + survives → bullish.
        ("survive", +1), ("survives", +1),
        # Aftermath markets — Secret Service / law enforcement.
        ("director resign", +1), ("director steps down", +1),
        ("charged with", +1), ("indicted", +1),
        # Election odds historically RISE after assassination
        # attempts (sympathy effect). Conservative: don't encode
        # this — leave as no rule and let observed-mode flag it.
    ],
    EventCategory.SHOOTING: [
        # Attendance / event-happening markets.
        ("attend", -1), ("appear", -1),
        ("happen", -1), ("take place", -1),
        ("rally", -1), ("event continues", -1),
        ("speak at", -1), ("present at", -1),
        # Aftermath / response markets.
        ("evacuat", +1), ("postpon", +1),
        ("cancel", +1), ("delay", +1),
        ("charged with", +1), ("indicted", +1),
        ("arrest", +1), ("in custody", +1),
        ("resign", +1), ("step down", +1),
    ],
    EventCategory.RESIGNATION: [
        # Markets framed as "will X resign / step down".
        ("resign", +1), ("step down", +1),
        ("steps down", +1), ("removed from", +1),
        ("ousted", +1), ("fired", +1),
        ("leaves office", +1),
        # Markets framed as "will X stay / remain in office".
        ("remain in", -1), ("stay as", -1),
        ("stays in", -1), ("keep position", -1),
        ("survives", -1),  # "survives no-confidence vote" etc.
        ("re-elected", -1), ("reelected", -1),
    ],
    EventCategory.ARREST: [
        # Charge / conviction markets benefit from an arrest.
        # Use root stems so "charge"/"charged"/"charges"/"charging"
        # all match.
        ("arrest", +1), ("in custody", +1), ("detain", +1),
        ("charge", +1), ("convict", +1),
        ("indict", +1),
        ("found guilty", +1),
        # Acquittal / freedom markets hurt by an arrest.
        ("acquit", -1), ("walks free", -1),
        ("dropped", -1), ("dismissed", -1),
        ("not guilty", -1),
        # Re-election / candidacy markets — arrest of a candidate
        # may help or hurt; don't encode either way (let observed
        # mode catch it).
    ],
    EventCategory.INDICTMENT: [
        # Same shape as ARREST, slightly stronger because indictment
        # is closer to conviction.
        ("indict", +1), ("charged", +1), ("convict", +1),
        ("found guilty", +1), ("guilty plea", +1),
        ("plea deal", +1),
        ("acquit", -1), ("dismissed", -1), ("dropped", -1),
        ("not guilty", -1), ("walks free", -1),
        ("cleared", -1),
    ],
    EventCategory.COURT_RULING: [
        # Direct legal-outcome language. Stems chosen so they match
        # both "rule"/"ruled"/"ruling" — substring matching means
        # "rule" matches "ruled" but not the reverse.
        ("rule in favor", +1), ("upheld", +1), ("uphold", +1),
        ("affirmed", +1),
        ("rule against", -1), ("struck down", -1),
        ("strike down", -1), ("overturned", -1),
        ("reversed", -1),
        # Aftermath markets.
        ("appeal", -1),  # likely to appeal a positive ruling
    ],
    EventCategory.DEATH_INJURY: [
        # Appearance / continuation markets.
        ("attend", -1), ("appear", -1),
        ("continue", -1), ("complete", -1), ("finish", -1),
        ("play", -1), ("compete", -1),  # athlete edge
        # Recovery / replacement markets.
        ("recover", +1), ("recovers", +1),
        ("replace", +1), ("successor", +1),
        ("step in", +1),
    ],
    EventCategory.EVACUATION: [
        # Event-happening markets.
        ("happen", -1), ("take place", -1), ("attend", -1),
        ("on schedule", -1), ("as planned", -1),
        ("complete", -1), ("finish", -1),
        # Postponement / cancellation markets.
        ("postpon", +1), ("cancel", +1), ("delay", +1),
        ("rescheduled", +1),
        ("evacuat", +1), ("emergency", +1),
    ],
    EventCategory.SPORTS_INJURY: [
        # Player participation / performance markets.
        ("play", -1), ("plays", -1), ("start", -1),
        ("appear", -1), ("compete", -1),
        ("score", -1), ("scores", -1),
        ("win", -1), ("wins", -1),
        ("mvp", -1), ("all-star", -1),
        # Backup / replacement markets.
        ("backup", +1), ("replacement", +1),
        ("called up", +1),
    ],
    EventCategory.ELECTION_RESULT: [
        # Hard to infer direction without knowing WHO won. Static
        # rules can't capture that — observed-mode catches these.
        # The LLM-backed scorer (TODO) is the right place to handle.
    ],
    EventCategory.MACRO_DATA_SURPRISE: [
        # Direction depends on the surprise sign AND which market
        # — too noisy for static rules. Observed-mode catches these.
    ],
}


def _infer_polarity(category: EventCategory, market_question: str) -> tuple[int, str]:
    """Return ``(direction, reasoning_string)``."""
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


def _confidence(
    *,
    event_confidence: float,
    match_score: float,
    severity: float,
    cap: float = _HEURISTIC_CONFIDENCE_CAP,
) -> float:
    """v2 confidence math: weighted average × severity multiplier.

    Old (v1):  event_confidence * max(0.3, match_score)
              — multiplicative; punished strong events with mid matches.

    New (v2):  base = 0.6*event_conf + 0.4*max(0.3, match_score)
              sev  = max(0.5, severity)
              out  = min(cap, base * sev)
              — gives partial credit for either side being strong.

    Same cap as v1 (0.55). Same lower-tail behaviour: weak matches
    still don't clear the lane's 0.40 gate.
    """
    base = 0.6 * event_confidence + 0.4 * max(0.3, match_score)
    sev_mult = max(0.5, severity)
    return clamp(base * sev_mult, 0.0, cap)


def score_impact(
    event: Event,
    match: MarketMatch,
    *,
    market_mid: float | None = None,
    confidence_cap: float = _HEURISTIC_CONFIDENCE_CAP,
) -> ImpactScore:
    """Compute the heuristic ImpactScore for one (Event, Market) pair.

    ``market_mid`` defaults to ``match.market.mid`` — pass an explicit
    value when the lane has a fresher snapshot than the cached
    market record.

    Three return shapes:

      1. direction != 0 (polarity rule matched):
           Full ImpactScore with confidence in (0, cap].
           Lane evaluates against min_confidence + edge gates.

      2. direction == 0, observed == True (no polarity but
         severity + match_score clear the watchlist floors):
           ImpactScore with confidence ~0.20 — below the trade gate
           by design. Candidate evaluator persists with
           status="observed", no trade opened.

      3. direction == 0, observed == False (true no-signal):
           ImpactScore with confidence == 0.0. Candidate evaluator
           rejects with reason "polarity_unknown".
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
        # Observed-mode candidacy. High-severity events with strong
        # mappings still deserve to be SEEN by the operator and the
        # Pattern Discovery layer, even when polarity inference can't
        # produce a directional read. The lane persists these as
        # status="observed" — no trade, but the audit trail is
        # complete.
        observed = (
            event.severity >= _OBSERVED_MIN_SEVERITY
            and match.score >= _OBSERVED_MIN_MATCH_SCORE
        )
        watch_conf = _OBSERVED_CONFIDENCE if observed else 0.0
        return ImpactScore(
            direction=0,
            true_prob=mid,
            confidence=watch_conf,
            expected_nudge=0.0,
            polarity_reasoning=polarity_reason,
            components={
                "category_nudge": base_nudge,
                "event_severity": event.severity,
                "match_score": match.score,
                "effective_nudge": effective_nudge,
                "market_mid": mid,
            },
            observed=observed,
        )

    true_prob = clamp(mid + direction * effective_nudge, 0.01, 0.99)
    confidence = _confidence(
        event_confidence=event.confidence,
        match_score=match.score,
        severity=event.severity,
        cap=confidence_cap,
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
        observed=False,
    )
