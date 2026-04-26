"""Heuristic fallback scorer for the event lane.

When Ollama times out (>10s) or returns malformed JSON on a fresh feed
item, the event lane falls back to this simple keyword + market-price
heuristic so we don't miss the trade entirely. Conservative by design:
- confidence is capped below Ollama's typical range
- size is forced to the configured fallback_size_usd (smaller than a
  normal event entry)
- only fires when we have a clear directional signal
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.markets.cache import Market

_POSITIVE = {
    "wins", "won", "confirms", "confirmed", "approves", "approved",
    "passes", "passed", "agrees", "agreed", "signs", "signed",
    "surges", "soars", "rallies", "ahead", "leads", "leading",
    "succeeds", "succeeded", "triumphs", "beats", "delivers",
    "announces", "announced", "launches", "launched",
}

_NEGATIVE = {
    "loses", "lost", "denies", "denied", "rejects", "rejected",
    "fails", "failed", "collapses", "crashes", "plunges",
    "delays", "delayed", "postpones", "postponed", "cancels", "cancelled",
    "concedes", "conceded", "withdraws", "withdrew", "behind", "trailing",
    "blocks", "blocked", "vetoed", "vetoes",
}

_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class HeuristicResult:
    direction: int  # +1 bullish on YES, -1 bearish (BUY NO), 0 no signal
    strength: float  # 0..1 hit ratio
    implied_prob: float
    confidence: float
    reasoning: str


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def score(text: str, market: Market) -> HeuristicResult:
    """Return a directional heuristic score, or zero-strength if
    ambiguous. Caller decides whether strength is enough to act on."""
    toks = _tokens(text)
    pos_hits = toks & _POSITIVE
    neg_hits = toks & _NEGATIVE
    mid = market.mid if 0 < market.mid < 1 else 0.5

    # Ambiguous: both sides or neither -> no signal.
    if (pos_hits and neg_hits) or (not pos_hits and not neg_hits):
        return HeuristicResult(
            direction=0,
            strength=0.0,
            implied_prob=mid,
            confidence=0.0,
            reasoning="heuristic: ambiguous or no directional keywords",
        )

    direction = 1 if pos_hits else -1
    hits = pos_hits if direction == 1 else neg_hits
    strength = min(1.0, len(hits) / 3.0)  # saturate at 3 hits
    # Nudge implied_prob away from the market mid in the direction of
    # the signal. Max nudge 0.12; scaled by strength.
    nudge = 0.12 * strength * direction
    implied = max(0.01, min(0.99, mid + nudge))
    # Heuristic confidence: capped at 0.70 no matter how strong the
    # keyword hits — we don't know context, only vocabulary.
    confidence = 0.55 + 0.15 * strength
    confidence = max(0.0, min(0.70, confidence))
    hit_str = ", ".join(sorted(hits)[:5])
    reasoning = (
        f"heuristic fallback: {'bullish' if direction == 1 else 'bearish'} "
        f"hits=[{hit_str}] strength={strength:.2f} "
        f"implied_prob={implied:.2f} (mid was {mid:.2f})"
    )
    return HeuristicResult(
        direction=direction,
        strength=strength,
        implied_prob=implied,
        confidence=confidence,
        reasoning=reasoning,
    )
