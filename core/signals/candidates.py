"""Candidate market generation for an incoming feed item."""

from __future__ import annotations

from typing import Any

from core.markets.cache import Market, candidate_markets
from core.utils.config import get_config


async def candidates_for(text: str) -> list[Market]:
    n = int(get_config().get("signals", "candidates_per_item", default=5))
    return await candidate_markets(text, top_n=n)


def serialize_candidates(markets: list[Market]) -> list[dict[str, Any]]:
    return [
        {
            "market_id": m.market_id,
            "question": m.question,
            "category": m.category,
            "mid_price": round(m.mid, 4),
            "best_bid": m.best_bid,
            "best_ask": m.best_ask,
            "liquidity": m.liquidity,
        }
        for m in markets
    ]
