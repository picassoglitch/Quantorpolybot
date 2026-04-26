"""Post-scoring sanity guards.

The LLM will confidently hand back `implied_prob=0.70` on a market
that's trading at $0.02 mid with no supporting evidence. Those are
exactly the trades that lose money. These guards cap / reject the
LLM's output before it gets sized.
"""

from __future__ import annotations

from typing import Any

# Defaults mirror what `apply_true_prob_cap` / `hallucination_reject`
# read from config — change config.yaml rather than these.
_DEFAULT_LOW_MID = 0.05
_DEFAULT_HIGH_TRUE_PROB = 0.60
_DEFAULT_MIN_SOURCES_FOR_LOW_MID = 3
_DEFAULT_LONG_HORIZON_DAYS = 30.0
_DEFAULT_MAX_TRUE_PROB_LONG = 0.90


def apply_true_prob_cap(
    true_prob: float,
    days_to_resolve: float | None,
    *,
    long_horizon_days: float = _DEFAULT_LONG_HORIZON_DAYS,
    cap: float = _DEFAULT_MAX_TRUE_PROB_LONG,
) -> float:
    """Cap `true_prob` at `cap` when the market is ≥ `long_horizon_days`
    from resolution. The LLM has no business being 99% confident a
    political outcome six months out will resolve YES."""
    if days_to_resolve is None:
        return true_prob
    if days_to_resolve >= long_horizon_days and true_prob > cap:
        return cap
    return true_prob


def hallucination_reject(
    *,
    true_prob: float,
    mid: float,
    num_sources: int,
    weighted_sources: float | None = None,
    low_mid_threshold: float = _DEFAULT_LOW_MID,
    high_true_prob_threshold: float = _DEFAULT_HIGH_TRUE_PROB,
    min_sources: int = _DEFAULT_MIN_SOURCES_FOR_LOW_MID,
) -> str | None:
    """If the LLM says true_prob > high_threshold on a market with mid
    below low_threshold, require strong corroboration (≥ min_sources
    independent evidence sources). Otherwise we treat it as a
    hallucination and return a reject reason.

    ``weighted_sources`` is the trust-weighted source count produced
    by ``core.learning.source_trust`` — when provided it's compared
    against ``min_sources`` instead of the raw count. Defaults to
    ``num_sources`` when omitted, preserving pre-calibration behaviour.

    Returns None when the signal passes the guard, or a human-readable
    reason string when it should be rejected.
    """
    if mid >= low_mid_threshold:
        return None
    if true_prob <= high_true_prob_threshold:
        return None
    effective = weighted_sources if weighted_sources is not None else float(num_sources)
    if effective >= float(min_sources):
        return None
    weighted_note = (
        f" (weighted {effective:.2f})"
        if weighted_sources is not None and abs(effective - num_sources) > 0.01
        else ""
    )
    return (
        f"hallucination_guard: true_prob={true_prob:.2f} at mid={mid:.3f} "
        f"requires ≥{min_sources} sources, got {num_sources}{weighted_note}"
    )


def guard_config(risk_cfg: dict[str, Any] | None) -> dict[str, float]:
    """Pull guard thresholds from a `risk` config dict, falling back to
    module defaults. Small helper so callers don't duplicate the
    default values."""
    cfg = risk_cfg or {}
    return {
        "low_mid_threshold": float(cfg.get("hallucination_low_mid", _DEFAULT_LOW_MID)),
        "high_true_prob_threshold": float(
            cfg.get("hallucination_high_true_prob", _DEFAULT_HIGH_TRUE_PROB)
        ),
        "min_sources": int(cfg.get("hallucination_min_sources", _DEFAULT_MIN_SOURCES_FOR_LOW_MID)),
        "long_horizon_days": float(cfg.get("long_horizon_cap_days", _DEFAULT_LONG_HORIZON_DAYS)),
        "long_horizon_cap": float(cfg.get("long_horizon_cap_true_prob", _DEFAULT_MAX_TRUE_PROB_LONG)),
    }
