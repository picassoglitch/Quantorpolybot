"""Event -> Polymarket market mapper.

For each Event, find the most relevant active markets. v1 is purely
keyword/category-based — no embeddings, no NER lib dep. The mapper
is intentionally split out from the scorer so future PRs can swap in
a stronger matcher (e.g. an embedding-based recall stage) without
touching the impact scorer.

Scoring per (Event, Market):

  match_score =
      0.50 * entity_overlap_jaccard(event.entities, market.question)
    + 0.20 * keyword_overlap_jaccard(event.title, market.question)
    + 0.15 * category_alignment(event.category, market.category)
    + 0.10 * liquidity_quality(market)
    + 0.05 * near_resolution_bonus(market)

Top-K markets per event are returned (default 5).

Hard filters applied BEFORE scoring (cheap):
  - market.active == 1
  - non-zero liquidity
  - non-crossed book

Hard filters applied AFTER ranking (the user-spec'd liquidity/spread
filter mention):
  - liquidity >= `min_liquidity`
  - spread_cents <= `max_spread_cents`

A future PR will replace the keyword overlap with embedding similarity
once we have a vector store; the function signature won't change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from core.markets.cache import Market
from core.scout.event import Event, EventCategory


# How well an Event category aligns with a Polymarket market category.
# Higher = stronger fit. Asymmetric — an "election_result" event hits
# politics markets hard but barely registers for crypto. Default 0.2
# for any combination not listed (a small non-zero floor so we don't
# completely zero out a high-entity-overlap match in an unmapped
# category bucket).
_CATEGORY_ALIGNMENT: dict[EventCategory, dict[str, float]] = {
    EventCategory.ASSASSINATION_ATTEMPT: {"politics": 1.0, "us-current-affairs": 0.9, "world": 0.7},
    EventCategory.SHOOTING: {"politics": 0.8, "us-current-affairs": 0.9, "world": 0.7},
    EventCategory.WAR_ESCALATION: {"world": 1.0, "geopolitics": 1.0, "politics": 0.7, "macro": 0.5},
    EventCategory.CEASEFIRE: {"world": 1.0, "geopolitics": 1.0, "politics": 0.6},
    EventCategory.ELECTION_RESULT: {"politics": 1.0, "elections": 1.0, "us-current-affairs": 0.8},
    EventCategory.COURT_RULING: {"politics": 0.9, "legal": 1.0, "us-current-affairs": 0.8},
    EventCategory.INDICTMENT: {"politics": 1.0, "legal": 0.9, "crypto": 0.4},
    EventCategory.RESIGNATION: {"politics": 0.9, "business": 0.9, "crypto": 0.7},
    EventCategory.ARREST: {"politics": 0.7, "crypto": 0.8, "legal": 0.9},
    EventCategory.DEATH_INJURY: {"politics": 0.7, "world": 0.8},
    EventCategory.EVACUATION: {"world": 0.8, "geopolitics": 0.8},
    EventCategory.MACRO_DATA_SURPRISE: {"macro": 1.0, "crypto": 0.7, "stocks": 0.8},
    EventCategory.SPORTS_INJURY: {"sports": 1.0},
    EventCategory.OTHER: {},
}


@dataclass
class MarketMatch:
    """One scored (Event, Market) candidate. Consumed by the impact
    scorer to compute the final ImpactScore + decide direction."""
    market: Market
    score: float
    entity_overlap: float
    keyword_overlap: float
    category_alignment: float
    liquidity_quality: float
    near_resolution_bonus: float


def map_event_to_markets(
    event: Event,
    markets: Iterable[Market],
    *,
    top_k: int = 5,
    min_score: float = 0.20,
    min_liquidity: float = 1000.0,
    max_spread_cents: float = 5.0,
    now_ts_value: float | None = None,
) -> list[MarketMatch]:
    """Pure function: returns up to `top_k` `MarketMatch` ranked by
    score, after applying hard filters.

    `min_score` is a floor — anything below it is dropped before the
    top-K cut so we don't return obviously-irrelevant matches just to
    fill the slot count. The lane treats an empty list as "no
    mapping" and logs accordingly.
    """
    event_entities = {e.lower() for e in event.entities if e}
    event_words = _significant_words(event.title)

    candidates: list[MarketMatch] = []
    for market in markets:
        # ---- Hard pre-filters ----
        if not market.active:
            continue
        if market.liquidity < min_liquidity:
            continue
        if market.best_bid <= 0 or market.best_ask <= 0:
            continue
        spread_cents = (market.best_ask - market.best_bid) * 100.0
        if spread_cents <= 0 or spread_cents > max_spread_cents:
            continue

        market_words = _significant_words(market.question)
        # ---- Score components ----
        ent_overlap = _jaccard(event_entities, market_words)
        kw_overlap = _jaccard(event_words, market_words)
        align = _category_alignment(event.category, market.category)
        liq_q = _liquidity_quality(market.liquidity)
        near_res = _near_resolution_bonus(market.close_time, now_ts_value)

        score = (
            0.50 * ent_overlap
            + 0.20 * kw_overlap
            + 0.15 * align
            + 0.10 * liq_q
            + 0.05 * near_res
        )
        if score < min_score:
            continue
        candidates.append(MarketMatch(
            market=market,
            score=score,
            entity_overlap=ent_overlap,
            keyword_overlap=kw_overlap,
            category_alignment=align,
            liquidity_quality=liq_q,
            near_resolution_bonus=near_res,
        ))

    candidates.sort(key=lambda m: m.score, reverse=True)
    return candidates[:top_k]


# ---------------- internal helpers ----------------


_WORD_RE = re.compile(r"[a-z]{4,}")
_STOPWORDS: frozenset[str] = frozenset({
    "will", "with", "from", "this", "that", "have", "into", "than",
    "then", "when", "what", "which", "their", "there", "would", "could",
    "should", "after", "before", "about", "between", "without",
})


def _significant_words(text: str) -> set[str]:
    if not text:
        return set()
    raw = _WORD_RE.findall(text.lower())
    return {w for w in raw if w not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _category_alignment(event_category: EventCategory, market_category: str) -> float:
    """Look up the per-category alignment weight; default 0.2."""
    if not market_category:
        return 0.2
    table = _CATEGORY_ALIGNMENT.get(event_category, {})
    # Polymarket categories arrive lower-case dash-separated; the
    # alignment table is keyed the same way.
    return table.get(market_category.strip().lower(), 0.2)


def _liquidity_quality(liquidity: float) -> float:
    """0 -> 1 sigmoid-ish ramp anchored at $50k."""
    if liquidity <= 0:
        return 0.0
    # Linear up to 50k, saturate beyond.
    return min(1.0, liquidity / 50_000.0)


def _near_resolution_bonus(close_time: str, now_ts_value: float | None) -> float:
    """Markets resolving in the next 30 days get a small bonus; the
    scout's edge is largest when the price has time to react but the
    event still matters. Markets a year out are essentially noise.
    """
    from core.utils.helpers import now_ts
    from core.utils.prices import parse_close_time

    ts = parse_close_time(close_time)
    if ts is None:
        return 0.0
    now_t = now_ts_value if now_ts_value is not None else now_ts()
    days = (ts - now_t) / 86400.0
    if days < 0 or days > 90:
        return 0.0
    if days <= 7:
        return 1.0
    if days <= 30:
        return 0.7
    return 0.3
