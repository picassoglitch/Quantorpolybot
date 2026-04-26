"""Heuristic impact scorer for the scout lane.

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
#
# v3 lowered both floors so more events surface as observed candidates
# — the previous 0.60/0.30 was too strict against the corpus we're
# actually seeing from GDELT. Lowering doesn't loosen safety; observed
# candidates STILL don't trade (confidence stays below the trade gate).
_OBSERVED_MIN_SEVERITY = 0.50
_OBSERVED_MIN_MATCH_SCORE = 0.25
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

    ``polarity_source`` (v3) records how `direction` was determined:
    "rules" / "llm" / "none". The candidate audit snapshot logs this
    so the dashboard can distinguish rule-based from LLM-inferred
    decisions for post-hoc analysis.
    """

    direction: int
    true_prob: float
    confidence: float
    expected_nudge: float          # absolute probability move scorer expects
    polarity_reasoning: str
    components: dict[str, float]
    observed: bool = False         # PR #7: watchlist signal, no trade
    polarity_source: str = "rules"  # v3: "rules" | "llm" | "none"


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
    direction_override: int | None = None,
    polarity_source_override: str = "",
    polarity_reason_override: str = "",
) -> ImpactScore:
    """Compute the heuristic ImpactScore for one (Event, Market) pair.

    ``market_mid`` defaults to ``match.market.mid`` — pass an explicit
    value when the lane has a fresher snapshot than the cached
    market record.

    ``direction_override`` (v3): when supplied (e.g. by an LLM
    polarity call), bypasses ``_infer_polarity`` and uses the given
    direction. ``polarity_source_override`` should be set to "llm"
    in that case so the audit snapshot records it. Used by
    ``score_impact_async`` below — most callers leave this None.

    Three return shapes:

      1. direction != 0 (polarity rule matched OR override supplied):
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

    if direction_override is not None and direction_override != 0:
        direction = direction_override
        polarity_reason = (
            polarity_reason_override
            or f"direction_override (source={polarity_source_override or 'unknown'})"
        )
        polarity_source = polarity_source_override or "override"
    else:
        direction, polarity_reason = _infer_polarity(event.category, market.question)
        polarity_source = "rules" if direction != 0 else "none"
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
        # DEBUG-level decision trace so an operator can SEE exactly
        # why an event was promoted to observed-mode (or not). Only
        # surfaces under LOGURU_LEVEL=DEBUG; INFO logs stay clean.
        from loguru import logger as _logger
        _logger.debug(
            "[scout] observed_decision event={} cat={} severity={:.2f}"
            "(>={:.2f}={}) match_score={:.2f}(>={:.2f}={}) -> observed={}",
            event.event_id, event.category.value,
            event.severity, _OBSERVED_MIN_SEVERITY,
            event.severity >= _OBSERVED_MIN_SEVERITY,
            match.score, _OBSERVED_MIN_MATCH_SCORE,
            match.score >= _OBSERVED_MIN_MATCH_SCORE,
            observed,
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
            polarity_source=polarity_source,
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
        polarity_source=polarity_source,
    )


# ============================================================
# LLM polarity inference (v3) — async wrapper around score_impact
# ============================================================


# When the rule table can't resolve a direction AND the event is
# severe enough to be worth the LLM cost, we ask the model. The
# call is wrapped in a hard timeout; on failure / unparseable
# response we fall through to the rule-based score (which then
# routes to observed-mode if thresholds are met).
_LLM_POLARITY_DEFAULT_TIMEOUT_SECONDS = 5.0
_LLM_POLARITY_DEFAULT_SEVERITY_FLOOR = 0.70


def _build_polarity_prompt(event: "Event", market: "Any", mid: float) -> str:
    """Tight prompt asking ONLY for a direction. Kept short so the
    fast tier can return in 1-3 seconds."""
    sources = ", ".join((event.sources or [])[:5]) or "unknown"
    return (
        "You are scoring a breaking-news event against one Polymarket\n"
        "prediction market. Decide whether the event makes the YES\n"
        "outcome MORE LIKELY (BUY YES), LESS LIKELY (SELL YES = BUY NO),\n"
        "or has NO clear effect on the YES probability.\n"
        "\n"
        f"EVENT:\n"
        f"  title: {event.title[:200]}\n"
        f"  category: {event.category.value}\n"
        f"  severity: {event.severity:.2f}\n"
        f"  sources: {sources}\n"
        "\n"
        f"MARKET:\n"
        f"  question: {market.question[:200]}\n"
        f"  current YES probability: {mid:.2f}\n"
        "\n"
        "Respond with JSON ONLY (no commentary, no markdown):\n"
        '  {"direction": "buy" | "sell" | "unclear", '
        '"reason": "<one short sentence>"}'
    )


def _parse_polarity_response(payload: dict | None) -> tuple[int, str]:
    """Map the JSON response to (direction, reasoning). Tolerant of
    common synonyms ("yes"/"bullish" → buy, "no"/"bearish" → sell)
    so we don't reject correct answers on cosmetic differences."""
    if not isinstance(payload, dict):
        return 0, "llm: empty/unparseable response"
    raw = payload.get("direction") or payload.get("polarity") or ""
    reason = (payload.get("reason") or payload.get("reasoning") or "")[:200]
    if not isinstance(raw, str):
        return 0, f"llm: non-string direction={raw!r}"
    d = raw.strip().lower()
    if d in ("buy", "yes", "bullish", "positive", "+1", "buy_yes"):
        return +1, f"llm: {d} ({reason})" if reason else f"llm: {d}"
    if d in ("sell", "no", "bearish", "negative", "-1", "sell_yes", "buy_no"):
        return -1, f"llm: {d} ({reason})" if reason else f"llm: {d}"
    return 0, f"llm: unclear ({d!r}) {reason}".strip()


async def _llm_infer_polarity(
    ollama_client: "Any",
    event: "Event",
    market: "Any",
    mid: float,
    *,
    timeout_seconds: float = _LLM_POLARITY_DEFAULT_TIMEOUT_SECONDS,
) -> tuple[int, str]:
    """Single-shot LLM polarity inference. Returns
    ``(direction, reason)`` where direction ∈ {-1, 0, +1}.

    Failures (timeout, exception, empty response, unparseable
    direction) all return (0, reason). Caller treats 0 as "no LLM
    signal" and falls through to the rule-based / observed-mode
    path. NEVER raises.
    """
    import asyncio
    if ollama_client is None:
        return 0, "llm: no client configured"
    prompt = _build_polarity_prompt(event, market, mid)
    try:
        result = await asyncio.wait_for(
            ollama_client.fast_score(
                prompt, tag=f"scout:polarity:{event.event_id}",
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return 0, f"llm: timeout (>{timeout_seconds:.1f}s)"
    except Exception as e:
        return 0, f"llm: error {type(e).__name__}: {str(e)[:120]}"
    return _parse_polarity_response(result)


async def score_impact_async(
    event: "Event",
    match: "MarketMatch",
    *,
    market_mid: float | None = None,
    confidence_cap: float = _HEURISTIC_CONFIDENCE_CAP,
    ollama_client: "Any" = None,
    llm_severity_floor: float = _LLM_POLARITY_DEFAULT_SEVERITY_FLOOR,
    llm_timeout_seconds: float = _LLM_POLARITY_DEFAULT_TIMEOUT_SECONDS,
    llm_enabled: bool = True,
) -> ImpactScore:
    """Async variant of ``score_impact`` that consults the LLM for
    polarity ONLY when:

      - ``llm_enabled`` is True (config opt-out)
      - ``ollama_client`` was provided
      - The rule-based ``_infer_polarity`` returned 0 (rules
        couldn't resolve direction)
      - ``event.severity >= llm_severity_floor`` (default 0.70 —
        only worth the LLM cost on high-severity events)

    On any LLM failure (timeout, exception, unclear response), falls
    back to ``score_impact`` (which then routes to observed-mode if
    severity + match thresholds are met). The LLM call NEVER raises
    to the caller.
    """
    market = match.market
    mid_value = market_mid if market_mid is not None else market.mid
    mid_value = clamp(mid_value, 0.01, 0.99)

    # Fast path: rules already produced a direction, OR the LLM is
    # disabled / unavailable. Either way, score synchronously.
    rule_dir, _ = _infer_polarity(event.category, market.question)
    should_call_llm = (
        llm_enabled
        and ollama_client is not None
        and rule_dir == 0
        and event.severity >= llm_severity_floor
    )
    if not should_call_llm:
        return score_impact(
            event, match,
            market_mid=mid_value, confidence_cap=confidence_cap,
        )

    # LLM polarity attempt. Failures fall through to score_impact
    # without override → observed-mode if thresholds are met.
    llm_dir, llm_reason = await _llm_infer_polarity(
        ollama_client, event, market, mid_value,
        timeout_seconds=llm_timeout_seconds,
    )
    from loguru import logger as _logger
    _logger.debug(
        "[scout] llm_polarity event={} cat={} severity={:.2f} -> "
        "direction={} reason={!r}",
        event.event_id, event.category.value, event.severity,
        llm_dir, llm_reason,
    )
    if llm_dir == 0:
        # LLM couldn't resolve. Fall through to score_impact —
        # observed-mode handles the "noticed but can't price" case.
        return score_impact(
            event, match,
            market_mid=mid_value, confidence_cap=confidence_cap,
        )

    return score_impact(
        event, match,
        market_mid=mid_value, confidence_cap=confidence_cap,
        direction_override=llm_dir,
        polarity_source_override="llm",
        polarity_reason_override=llm_reason,
    )
