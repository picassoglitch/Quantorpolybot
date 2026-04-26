"""Microstructure-only scoring for the scalping lane.

When a market has no usable news evidence, we can still extract a weak
directional signal from the orderbook + tick stream alone:

  - **Spread tightness** — a tight spread relative to the lane's max
    means the orderbook is healthy and slippage is bounded.
  - **Tick frequency** — markets that tick often are being actively
    quoted; markets that haven't ticked in 10 minutes are dead and
    not worth the slippage risk.
  - **Recent volatility** — meaningful (not noise-floor) mid moves
    indicate price-discovery is happening; a flat market gives no
    edge.
  - **Drift** — the direction of the recent mid move IS the
    microstructure signal. We bet with the drift, not against it
    (counter-trend on no-evidence is reckless).
  - **Volume / liquidity** — the lane already gates on these, but we
    fold them into the strength score so a barely-liquid market gets
    a lower-confidence Score even when the drift is clean.

This is **not** a real microstructure model. We don't have orderbook
depth (Polymarket Gamma + WS only expose top-of-book), so we can't
compute true bid-side / ask-side imbalance. The proxy bundle above is
what we *can* compute from current data; it's intentionally
conservative — confidence is capped well below the news-driven heuristic
so the lane can't size up on a price-history-only view.

TODO(orderbook-depth): once we have a real orderbook stream (e.g. via
the L2 endpoint or a market-maker dump), replace the volatility/drift
proxy with a true bid_size/(bid_size+ask_size) imbalance over a fixed
depth window. The dataclass is already shaped to carry it as another
component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.markets.cache import Market
from core.utils.db import fetch_all
from core.utils.helpers import clamp, now_ts, safe_float


# Hard cap on the confidence a microstructure-only Score can carry.
# Even a textbook-clean drift on a deep orderbook should not let the
# lane size up the same way a corroborated news signal would. The
# scalping lane's `min_confidence` (default 0.60) gates against this:
# microstructure entries fire only when the proxy score is well above
# noise.
_MAX_MICROSTRUCTURE_CONFIDENCE = 0.55

# Below this many ticks in the lookback window the market is stale; we
# don't compute volatility from <3 samples.
_MIN_TICKS_FOR_VOLATILITY = 3


@dataclass
class MicrostructureSignal:
    """Output of `score_microstructure`.

    `direction` is +1 (bullish on YES) / -1 (bearish on YES) / 0 (no
    signal). `strength` is the raw proxy score in [0,1]; `confidence`
    is the capped scalar that the lane compares against `min_confidence`.
    `implied_prob` is a small nudge off the current mid in the
    direction of the drift, scaled by strength — symmetric with how the
    keyword heuristic in `core.strategies.heuristic` produces its
    implied prob, so callers can treat the two outputs interchangeably
    when sizing.
    """

    direction: int
    strength: float
    confidence: float
    implied_prob: float
    reasoning: str
    components: dict[str, float] = field(default_factory=dict)


async def _recent_ticks(
    market_id: str, window_seconds: float, limit: int = 200
) -> list[dict[str, float]]:
    """Pull the most recent N ticks for `market_id` within the window.

    `price_ticks` is indexed by `(market_id, ts)`; even at 200 ticks
    this is a single B-tree lookup. Returned in chronological order
    so callers can compute drift = last - first without re-sorting.
    """
    cutoff = now_ts() - window_seconds
    rows = await fetch_all(
        """SELECT bid, ask, last, ts
           FROM price_ticks
           WHERE market_id=? AND ts >= ?
           ORDER BY ts ASC
           LIMIT ?""",
        (market_id, cutoff, limit),
    )
    out: list[dict[str, float]] = []
    for r in rows:
        bid = safe_float(r["bid"])
        ask = safe_float(r["ask"])
        last = safe_float(r["last"])
        ts = safe_float(r["ts"])
        mid = (bid + ask) / 2.0 if (bid and ask) else last
        if mid <= 0:
            continue
        out.append({"bid": bid, "ask": ask, "last": last, "ts": ts, "mid": mid})
    return out


def _spread_tightness(market: Market, max_spread_cents: float) -> float:
    """1.0 = pinned at zero spread, 0.0 = at the max acceptable.

    Negative spreads (crossed book) and zero-bid/ask are clamped to
    zero — they'd otherwise score as "tight" which is the opposite
    of safe. The lane already filters wide spreads upstream; this is
    the differentiator within the accepted range.
    """
    if max_spread_cents <= 0:
        return 0.0
    spread_cents = (market.best_ask - market.best_bid) * 100.0
    if spread_cents <= 0:
        return 0.0
    return clamp(1.0 - (spread_cents / max_spread_cents), 0.0, 1.0)


def _tick_frequency(ticks: list[dict[str, float]], window_seconds: float) -> float:
    """1.0 ~ a tick every 5s on average over the window.

    The 5s/tick anchor is a soft saturation point — anything faster
    is hyper-active and we don't want strength to keep climbing.
    """
    if not ticks or window_seconds <= 0:
        return 0.0
    expected_at_anchor = max(1.0, window_seconds / 5.0)
    return clamp(len(ticks) / expected_at_anchor, 0.0, 1.0)


def _volatility(ticks: list[dict[str, float]]) -> float:
    """Sample standard deviation of consecutive mid moves, in cents.

    Returns 0 when there aren't enough samples. The score (next fn)
    maps this raw cents value into [0,1]: noise floor is ~0.5c, an
    actively repricing market sits at 1-3c.
    """
    if len(ticks) < _MIN_TICKS_FOR_VOLATILITY:
        return 0.0
    moves_cents = [
        (ticks[i]["mid"] - ticks[i - 1]["mid"]) * 100.0
        for i in range(1, len(ticks))
    ]
    if not moves_cents:
        return 0.0
    n = len(moves_cents)
    mean = sum(moves_cents) / n
    var = sum((m - mean) ** 2 for m in moves_cents) / n
    return math.sqrt(var)


def _volatility_score(vol_cents: float) -> float:
    """Map raw volatility (cents) -> [0,1] strength contribution.

    Below 0.3c is noise; above 3c we cap (a market thrashing wider
    than that is a hazard, not an opportunity).
    """
    if vol_cents <= 0.3:
        return 0.0
    if vol_cents >= 3.0:
        return 1.0
    # Linear ramp between the two anchors.
    return (vol_cents - 0.3) / (3.0 - 0.3)


def _drift_score(ticks: list[dict[str, float]]) -> tuple[int, float, float]:
    """Compute drift direction, normalized magnitude, and raw mid delta.

    Returns ``(direction, magnitude_score, raw_delta_cents)`` where
    ``direction`` is +1 / -1 / 0 and ``magnitude_score`` is in [0,1].
    The 1.5c anchor (saturation) is intentionally tight — without
    news, we don't want to chase a 5-cent run that's likely already
    priced in by the time we'd see it.
    """
    if len(ticks) < 2:
        return 0, 0.0, 0.0
    delta_cents = (ticks[-1]["mid"] - ticks[0]["mid"]) * 100.0
    if abs(delta_cents) < 0.4:
        return 0, 0.0, delta_cents
    direction = 1 if delta_cents > 0 else -1
    magnitude = clamp(abs(delta_cents) / 1.5, 0.0, 1.0)
    return direction, magnitude, delta_cents


def _liquidity_score(market: Market, vol_24h: float, min_volume: float) -> float:
    """Combine the cache-side liquidity figure with 24h volume.

    Both are floors-not-curves: the lane's existing min_volume_24h
    gate already rejects below-floor markets, so this score lives
    in (0,1] for everything that reaches us. Anchor 5x the floor as
    "fully liquid for scalp purposes."
    """
    if min_volume <= 0:
        return 0.0
    vol_component = clamp(vol_24h / (5.0 * min_volume), 0.0, 1.0)
    liq_component = clamp(market.liquidity / 50_000.0, 0.0, 1.0)
    # Average the two; either alone is a partial story.
    return (vol_component + liq_component) / 2.0


async def score_microstructure(
    market: Market,
    *,
    max_spread_cents: float,
    min_volume_24h: float,
    vol_24h: float,
    window_seconds: float = 600.0,
    min_ticks: int = 6,
) -> MicrostructureSignal | None:
    """Compute a microstructure-only directional signal for `market`.

    Returns ``None`` when the input data is too thin to support any
    score (no recent ticks, no spread, no liquidity) — the lane treats
    that as `tier=NONE` and skips the market with a clear reason.

    Returns a `MicrostructureSignal` otherwise. Note the confidence is
    capped at `_MAX_MICROSTRUCTURE_CONFIDENCE` (0.55) regardless of how
    strong the components are, so a scalping config with
    `min_confidence: 0.60` (the default) will reject the signal — that's
    intentional. Configs that opt into microstructure entries must
    lower `min_confidence` for the microstructure tier explicitly.
    """
    ticks = await _recent_ticks(market.market_id, window_seconds)

    # Hard reject: not enough ticks to compute anything meaningful.
    if len(ticks) < min_ticks:
        return None

    spread = _spread_tightness(market, max_spread_cents)
    if spread <= 0.0:
        # Crossed or zero book — we're not going to touch it.
        return None

    freq = _tick_frequency(ticks, window_seconds)
    vol_cents = _volatility(ticks)
    vol_score = _volatility_score(vol_cents)
    direction, drift_mag, drift_cents = _drift_score(ticks)
    liq = _liquidity_score(market, vol_24h, min_volume_24h)

    # No directional bias = no trade. The other components only matter
    # if drift gives us a side to lean on.
    if direction == 0:
        return MicrostructureSignal(
            direction=0,
            strength=0.0,
            confidence=0.0,
            implied_prob=market.mid if 0 < market.mid < 1 else 0.5,
            reasoning=(
                "microstructure: no drift "
                f"(delta={drift_cents:+.2f}c spread_tight={spread:.2f} "
                f"freq={freq:.2f} vol={vol_cents:.2f}c liq={liq:.2f})"
            ),
            components={
                "spread_tightness": spread,
                "tick_frequency": freq,
                "volatility_score": vol_score,
                "drift_magnitude": 0.0,
                "drift_cents": drift_cents,
                "liquidity_score": liq,
            },
        )

    # Strength: weighted blend. Drift and liquidity matter most — drift
    # IS the signal, liquidity gates whether we can act on it without
    # eating spread. Tick frequency and volatility are
    # quality-of-microstructure modifiers; spread tightness is a small
    # bonus on top.
    #
    # Weights chosen so:
    #   - Max possible strength = 1.0 (when every component is 1.0).
    #   - A clean directional drift on a stale book (low freq, low vol)
    #     scores ~0.45-0.50, comfortably below the conservative gate.
    #   - A clean drift on an active book (freq, vol > 0.5) scores
    #     0.65-0.80, which combined with the 0.55 confidence cap and
    #     a tier-specific min_confidence still requires explicit opt-in.
    strength = (
        0.40 * drift_mag
        + 0.20 * liq
        + 0.15 * freq
        + 0.15 * vol_score
        + 0.10 * spread
    )
    strength = clamp(strength, 0.0, 1.0)

    # Confidence floor 0.40 once we have a directional signal at all,
    # ramped up by strength but capped at the module-wide cap.
    confidence = clamp(0.40 + (strength * 0.30), 0.40, _MAX_MICROSTRUCTURE_CONFIDENCE)

    # Implied prob: nudge off the current mid in the direction of the
    # drift. Max nudge 0.08 (smaller than the keyword heuristic's 0.12)
    # since we have no narrative reason for the move.
    mid = market.mid if 0 < market.mid < 1 else 0.5
    nudge = 0.08 * strength * direction
    implied = clamp(mid + nudge, 0.01, 0.99)

    reasoning = (
        f"microstructure: {'bullish' if direction == 1 else 'bearish'} drift "
        f"delta={drift_cents:+.2f}c strength={strength:.2f} "
        f"spread_tight={spread:.2f} freq={freq:.2f} vol={vol_cents:.2f}c "
        f"liq={liq:.2f} implied={implied:.2f} (mid was {mid:.2f})"
    )
    return MicrostructureSignal(
        direction=direction,
        strength=strength,
        confidence=confidence,
        implied_prob=implied,
        reasoning=reasoning,
        components={
            "spread_tightness": spread,
            "tick_frequency": freq,
            "volatility_score": vol_score,
            "drift_magnitude": drift_mag,
            "drift_cents": drift_cents,
            "liquidity_score": liq,
        },
    )
