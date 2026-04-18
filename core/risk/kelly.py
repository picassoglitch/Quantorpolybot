"""Fractional Kelly sizing for binary outcomes.

For a YES bet at price p with our estimated win prob q:
    payoff if win  = (1 - p) / p   (per $1 staked, not counting stake)
    full Kelly f*  = q - (1 - q) / b   where b = (1 - p) / p
We then scale by `fraction` (default 0.25) and clamp to USD limits.
"""

from __future__ import annotations

from core.utils.helpers import clamp


def kelly_fraction(prob_win: float, price: float) -> float:
    """Return the Kelly stake fraction in [0, 1]. Negative -> 0."""
    p = clamp(price, 1e-4, 1 - 1e-4)
    q = clamp(prob_win, 0.0, 1.0)
    b = (1 - p) / p
    f = q - (1 - q) / b
    return max(0.0, f)


def kelly_size_usd(
    prob_win: float,
    price: float,
    bankroll_usd: float,
    fraction: float = 0.25,
    min_size_usd: float = 0.0,
    max_size_usd: float = 1e9,
) -> float:
    """Convert Kelly fraction to a USD stake."""
    if bankroll_usd <= 0:
        return 0.0
    f = kelly_fraction(prob_win, price) * max(0.0, fraction)
    raw = bankroll_usd * f
    if raw < min_size_usd:
        return 0.0
    return clamp(raw, min_size_usd, max_size_usd)
