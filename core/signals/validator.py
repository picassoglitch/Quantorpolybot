"""High-stakes cross-validation.

For any proposed entry at or above ``validator_high_stakes_usd``, run a
second LLM pass with a different model (``validator_model``, low temp)
against the same evidence. The validator is a pure sanity check — it
doesn't generate new edge, it just catches cases where the primary
scorer is overconfident or on the wrong side of the trade.

Decision rules (all configurable from the ollama.* block):

  - |validator.true_prob - original.true_prob| > max_prob_drift
        -> shrink size by 50% ("halved").
  - validator flips direction relative to mid (e.g. original wanted BUY
    because true_prob > mid, validator says true_prob < mid)
        -> skip the entry entirely ("direction_skip").
  - otherwise accept as-is ("ok").

The caller writes the returned snapshot into
``shadow_positions.validator_snapshot`` so we can audit how often the
validator disagreed and whether its disagreements correlated with
bad outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.markets.cache import Market
from core.signals.ollama_client import OllamaClient
from core.utils.config import get_config
from core.utils.helpers import clamp, safe_float


@dataclass
class ValidationResult:
    decision: str            # 'ok' | 'halved' | 'direction_skip' | 'unavailable'
    adjusted_size: float     # final size to use (0 on direction_skip)
    validator_true_prob: float
    validator_confidence: float
    validator_reasoning: str
    original_true_prob: float
    drift: float
    notes: str

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "adjusted_size": round(self.adjusted_size, 2),
            "validator_true_prob": round(self.validator_true_prob, 4),
            "validator_confidence": round(self.validator_confidence, 4),
            "validator_reasoning": self.validator_reasoning[:400],
            "original_true_prob": round(self.original_true_prob, 4),
            "drift": round(self.drift, 4),
            "notes": self.notes,
        }


def _build_prompt(
    market: Market,
    side: str,
    original_true_prob: float,
    original_reasoning: str,
    evidence_text: str,
) -> str:
    """Deliberately plain-language prompt. The validator is a sceptic —
    tell it the proposed trade and ask it to agree or disagree. We reuse
    the same JSON schema the primary scorer returns so `_parse_response`
    can share logic."""
    mid = market.mid
    return (
        "You are a cautious validator reviewing a proposed trade on a "
        "prediction market. Respond with strict JSON only, keys: "
        '"implied_prob" (float 0-1), "confidence" (float 0-1), '
        '"reasoning" (string, <200 chars).\n\n'
        "MARKET:\n"
        f"  question: {market.question}\n"
        f"  current_mid: {mid:.3f}\n"
        f"  category: {market.category or 'unknown'}\n\n"
        "PROPOSED_TRADE:\n"
        f"  side: {side}\n"
        f"  scorer_true_prob: {original_true_prob:.3f}\n"
        f"  scorer_reasoning: {original_reasoning[:400]}\n\n"
        "EVIDENCE:\n"
        f"{evidence_text[:1500]}\n\n"
        "Return your OWN estimate of the true probability for the YES "
        "outcome given this evidence. Be strict — if the evidence does "
        "not clearly justify the scorer's estimate, say so in your "
        "reasoning and give a more conservative implied_prob."
    )


def _parse_response(result: dict[str, Any] | None) -> tuple[float, float, str] | None:
    if not result:
        return None
    implied = result.get("implied_prob")
    if implied is None:
        return None
    try:
        tp = clamp(safe_float(implied), 0.0, 1.0)
        conf = clamp(safe_float(result.get("confidence")), 0.0, 1.0)
    except (TypeError, ValueError):
        return None
    reasoning = (result.get("reasoning") or "")[:400]
    return tp, conf, reasoning


def _direction_from_edge(true_prob: float, mid: float) -> str:
    return "BUY" if true_prob >= mid else "SELL"


async def cross_validate(
    *,
    client: OllamaClient,
    market: Market,
    side: str,
    original_true_prob: float,
    original_reasoning: str,
    evidence_text: str,
    size_usd: float,
) -> ValidationResult:
    """Run the validator model and return the adjusted size + audit snapshot.

    Assumes the caller has already decided this entry is high-stakes
    (size >= validator_high_stakes_usd). On any failure (cooldown, bad
    JSON, timeout) the result is ``unavailable`` and size is unchanged —
    we do not reject entries just because the validator is down.
    """
    cfg = get_config().get("ollama") or {}
    max_drift = safe_float(cfg.get("validator_max_prob_drift", 0.15))

    prompt = _build_prompt(
        market, side, original_true_prob, original_reasoning, evidence_text,
    )
    try:
        raw = await client.validate(prompt, context={"market_id": market.market_id})
    except Exception as e:
        logger.warning("[validator] call failed for {}: {}", market.market_id, e)
        raw = None
    parsed = _parse_response(raw)
    if parsed is None:
        return ValidationResult(
            decision="unavailable",
            adjusted_size=size_usd,
            validator_true_prob=0.0,
            validator_confidence=0.0,
            validator_reasoning="",
            original_true_prob=original_true_prob,
            drift=0.0,
            notes="validator unavailable or unparseable",
        )

    v_tp, v_conf, v_reasoning = parsed
    drift = abs(v_tp - original_true_prob)
    validator_dir = _direction_from_edge(v_tp, market.mid)

    if validator_dir != side:
        logger.warning(
            "[validator] direction mismatch on {} (side={} validator_dir={} "
            "scorer_tp={:.2f} validator_tp={:.2f}) — skipping entry",
            market.market_id, side, validator_dir, original_true_prob, v_tp,
        )
        return ValidationResult(
            decision="direction_skip",
            adjusted_size=0.0,
            validator_true_prob=v_tp,
            validator_confidence=v_conf,
            validator_reasoning=v_reasoning,
            original_true_prob=original_true_prob,
            drift=drift,
            notes=f"validator said {validator_dir}",
        )

    if drift > max_drift:
        halved = round(size_usd * 0.5, 2)
        logger.warning(
            "[validator] drift {:.2f} > {:.2f} on {} — halving size {:.0f} -> {:.0f}",
            drift, max_drift, market.market_id, size_usd, halved,
        )
        return ValidationResult(
            decision="halved",
            adjusted_size=halved,
            validator_true_prob=v_tp,
            validator_confidence=v_conf,
            validator_reasoning=v_reasoning,
            original_true_prob=original_true_prob,
            drift=drift,
            notes=f"drift {drift:.2f} > {max_drift:.2f}",
        )

    return ValidationResult(
        decision="ok",
        adjusted_size=size_usd,
        validator_true_prob=v_tp,
        validator_confidence=v_conf,
        validator_reasoning=v_reasoning,
        original_true_prob=original_true_prob,
        drift=drift,
        notes="within tolerance",
    )
