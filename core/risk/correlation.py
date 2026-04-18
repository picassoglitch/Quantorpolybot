"""Category-level correlation guard.

Treats all open positions in the same `category` as correlated and
caps total exposure per bucket at `max_correlated_exposure_usd`.
"""

from __future__ import annotations

from core.utils.db import fetch_all


async def category_exposure_usd() -> dict[str, float]:
    rows = await fetch_all(
        """SELECT m.category AS category, SUM(ABS(p.size * p.avg_price)) AS exposure
           FROM positions p
           JOIN markets m ON m.market_id = p.market_id
           WHERE p.status='OPEN'
           GROUP BY m.category"""
    )
    return {(r["category"] or "uncategorised"): float(r["exposure"] or 0.0) for r in rows}


async def exposure_for_category(category: str) -> float:
    bucket = await category_exposure_usd()
    return bucket.get(category or "uncategorised", 0.0)
