"""Google News RSS — dynamic per-market queries.

Builds a query string from each active market title (drops stop words
and the boilerplate "Will/Did" prefix), dedupes, and rotates a window
of N queries per cycle so we cover the whole market universe over time
without thrashing Google.

Each item is stored with ``meta.linked_market_id`` so the signal
pipeline knows it's tied to a specific market and can bypass the
generic keyword pre-filter.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Iterable
from urllib.parse import quote_plus

import feedparser
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts
from core.utils.watchdog import is_degraded

DEGRADED_POLL_MULTIPLIER = 2

GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

# Words that pollute every prediction-market title and add zero search
# signal — strip them before constructing the Google query.
_STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "by", "to",
    "for", "with", "is", "are", "be", "been", "will", "would", "could",
    "should", "did", "does", "do", "have", "has", "had", "this", "that",
    "these", "those", "than", "then", "from", "up", "down", "out", "over",
    "under", "as", "if", "it", "its", "into", "before", "after", "by",
    "vs", "vs.", "between",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")

# Defensive ceiling for what we'll accept as a publisher suffix. Real
# publishers are short ("BBC", "Reuters", "The New York Times", "Jamaica
# Observer"); a long tail almost always means the " - " was inside the
# headline body, not a separator. Sampling 12 live titles showed all
# publishers <= 30 chars; 60 leaves headroom without letting a
# half-headline through.
_PUBLISHER_MAX_LEN = 60


def _split_title_publisher(title: str) -> tuple[str, str | None]:
    """Split a Google News title on the LAST ``" - "`` separator.

    Google News appends the source name to every entry title with a
    literal ``" - "`` separator (ASCII space-hyphen-space, never an
    em-dash, verified across a 12-title sample). The headline itself
    can contain hyphens or even ``" - "`` substrings, so we use
    :func:`str.rpartition` to peel the final segment only.

    Defensive cases — return ``(title, None)`` and leave the title
    unchanged when:

    - the title is empty / falsy
    - no ``" - "`` separator is present
    - the suffix is empty (e.g. trailing separator)
    - the suffix is longer than :data:`_PUBLISHER_MAX_LEN` (almost
      certainly a sentence fragment, not a publisher name)
    - the prefix is empty (don't strip the title to nothing)

    Returning ``None`` for the publisher signals to the caller "do not
    set ``meta.publisher``"; the caller keeps the original title intact.
    """
    if not title:
        return title, None
    head, sep, tail = title.rpartition(" - ")
    if not sep:
        return title, None
    publisher = tail.strip()
    cleaned = head.strip()
    if not publisher or not cleaned:
        return title, None
    if len(publisher) > _PUBLISHER_MAX_LEN:
        return title, None
    return cleaned, publisher


class GoogleNewsFeed:
    component = "feed.google_news"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._cursor: int = 0
        # query-string -> market_id, used so we can stamp meta.linked_market_id
        # back onto each ingested article without an extra DB hop.
        self._query_to_market: dict[str, str] = {}

    async def run(self) -> None:
        cfg = get_config().get("feeds", "google_news") or {}
        if not cfg.get("enabled", True):
            logger.info("[gnews] disabled")
            return
        poll = int(cfg.get("poll_seconds", 300))
        per_cycle = int(cfg.get("queries_per_cycle", 10))
        delay = float(cfg.get("request_delay_seconds", 1.0))
        hl = cfg.get("hl", "en-US")
        gl = cfg.get("gl", "US")
        ceid = cfg.get("ceid", "US:en")
        backoff = Backoff(base=5, cap=300)

        logger.info("[gnews] starting; poll={}s per_cycle={}", poll, per_cycle)
        while not self._stop.is_set():
            try:
                queries = await self._build_query_window(per_cycle)
                new = 0
                for q in queries:
                    if self._stop.is_set():
                        break
                    new += await self._poll_query(q, hl, gl, ceid)
                    if delay > 0:
                        await self._sleep(delay)
                if new:
                    logger.info("[gnews] ingested {} new items across {} queries",
                                new, len(queries))
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                d = backoff.next_delay()
                logger.warning(
                    "[gnews] error ({}), sleeping {:.1f}s: {}",
                    type(e).__name__, d, e,
                )
                await self._sleep(d)
                continue
            await self._sleep(poll * DEGRADED_POLL_MULTIPLIER if is_degraded() else poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _build_query_window(self, per_cycle: int) -> list[str]:
        """Pull active markets, derive deduped queries, return the next
        `per_cycle` chunk in cursor order."""
        rows = await fetch_all(
            "SELECT market_id, question FROM markets WHERE active=1 "
            "ORDER BY liquidity DESC LIMIT 1000"
        )
        all_queries: list[str] = []
        seen: set[str] = set()
        self._query_to_market.clear()
        for r in rows:
            q = build_query(r["question"] or "")
            if not q or q in seen:
                continue
            seen.add(q)
            all_queries.append(q)
            self._query_to_market[q] = str(r["market_id"])
        if not all_queries:
            return []
        if self._cursor >= len(all_queries):
            self._cursor = 0
        window = all_queries[self._cursor : self._cursor + per_cycle]
        if len(window) < per_cycle and self._cursor > 0:
            window = window + all_queries[: per_cycle - len(window)]
        self._cursor = (self._cursor + per_cycle) % max(len(all_queries), 1)
        return window

    async def _poll_query(self, query: str, hl: str, gl: str, ceid: str) -> int:
        url = GOOGLE_NEWS_URL.format(
            q=quote_plus(query), hl=hl, gl=gl, ceid=quote_plus(ceid)
        )
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, url)
        new = 0
        market_id = self._query_to_market.get(query, "")
        for entry in parsed.entries or []:
            link = (entry.get("link") or "").strip()
            if not link:
                continue
            # Use (query, link) as the dedup key so the same article
            # surfacing for two different market queries isn't dropped.
            h = url_hash(f"gnews:{query}:{link}")
            existing = await fetch_one(
                "SELECT id FROM feed_items WHERE url_hash=?", (h,)
            )
            if existing:
                continue
            raw_title = (entry.get("title") or "").strip()
            title, publisher = _split_title_publisher(raw_title)
            summary = (entry.get("summary") or entry.get("description") or "").strip()
            published = self._published(entry)
            meta: dict[str, Any] = {
                "query": query,
                "linked_market_id": market_id,
            }
            if publisher:
                meta["publisher"] = publisher
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at, meta)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    h,
                    "google_news",
                    title,
                    summary[:1000],
                    link,
                    published,
                    now_ts(),
                    json.dumps(meta),
                ),
            )
            new += 1
        return new

    @staticmethod
    def _published(entry: Any) -> float:
        for key in ("published_parsed", "updated_parsed"):
            value = entry.get(key)
            if value:
                try:
                    import calendar
                    return float(calendar.timegm(value))
                except Exception:
                    continue
        return now_ts()


def build_query(question: str) -> str:
    """Turn a market title into a Google News search string."""
    if not question:
        return ""
    tokens = _TOKEN_RE.findall(question)
    cleaned: list[str] = []
    for t in tokens:
        low = t.lower()
        if low in _STOPWORDS:
            continue
        cleaned.append(t)
    # Cap to the first ~6 keywords; longer queries get throttled by Google.
    return " ".join(cleaned[:6]).strip()


def _iter_unique_queries(rows: Iterable[Any]) -> Iterable[tuple[str, str]]:
    seen: set[str] = set()
    for r in rows:
        q = build_query(r["question"] or "")
        if not q or q in seen:
            continue
        seen.add(q)
        yield q, str(r["market_id"])
