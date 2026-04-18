"""Read-side helpers for the markets table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.utils.db import fetch_all, fetch_one
from core.utils.helpers import keywords, jaccard, safe_float


@dataclass
class Market:
    market_id: str
    question: str
    slug: str
    category: str
    active: bool
    close_time: str
    token_ids: list[str]
    best_bid: float
    best_ask: float
    last_price: float
    liquidity: float
    updated_at: float

    @property
    def mid(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.last_price or 0.0

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 1.0  # treat as wide

    def yes_token(self) -> str | None:
        return self.token_ids[0] if self.token_ids else None

    def no_token(self) -> str | None:
        return self.token_ids[1] if len(self.token_ids) > 1 else None


def _row_to_market(row: Any) -> Market:
    raw_tokens = row["token_ids"] or "[]"
    try:
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else list(raw_tokens)
    except json.JSONDecodeError:
        tokens = []
    return Market(
        market_id=row["market_id"],
        question=row["question"] or "",
        slug=row["slug"] or "",
        category=row["category"] or "",
        active=bool(row["active"]),
        close_time=row["close_time"] or "",
        token_ids=[str(t) for t in tokens],
        best_bid=safe_float(row["best_bid"]),
        best_ask=safe_float(row["best_ask"]),
        last_price=safe_float(row["last_price"]),
        liquidity=safe_float(row["liquidity"]),
        updated_at=safe_float(row["updated_at"]),
    )


async def get_market(market_id: str) -> Market | None:
    row = await fetch_one("SELECT * FROM markets WHERE market_id=?", (market_id,))
    return _row_to_market(row) if row else None


async def list_active(limit: int = 500) -> list[Market]:
    rows = await fetch_all(
        "SELECT * FROM markets WHERE active=1 ORDER BY liquidity DESC LIMIT ?",
        (limit,),
    )
    return [_row_to_market(r) for r in rows]


async def candidate_markets(text: str, top_n: int = 5) -> list[Market]:
    """Cheap keyword/Jaccard candidate match against active markets."""
    target = keywords(text)
    if not target:
        return []
    rows = await fetch_all(
        "SELECT * FROM markets WHERE active=1 ORDER BY liquidity DESC LIMIT 1000"
    )
    scored: list[tuple[float, Market]] = []
    for r in rows:
        m = _row_to_market(r)
        terms = keywords(f"{m.question} {m.category} {m.slug}")
        score = jaccard(target, terms)
        if score <= 0:
            continue
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_n]]
