"""Async RSS poller.

Two input shapes are supported for backward compatibility:

  1. Bare URL strings (legacy) — source stored as the URL.
  2. Dicts ``{"name": "...", "url": "...", "weight": 0.85}`` — source
     stored as ``name`` and ``source_weight`` propagated to the enricher.

feedparser is synchronous; we run it via ``run_in_executor`` so one
slow feed doesn't block the loop. One failing feed never kills the
others — failures are logged and the next feed runs.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import feedparser
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts
from core.utils.watchdog import is_degraded

# Multiplier applied to the configured poll interval when the watchdog
# reports DEGRADED state. Keeps low-value ingest work out of the way of
# order/signal traffic.
DEGRADED_POLL_MULTIPLIER = 3

# Default seed used when config/config.yaml doesn't supply one. Fully
# self-hosted — no API keys, no vendor lock-in. Trust weight is a
# subjective 0-1 score used by the enricher to bias
# market_relevance. Tier-1 wires (Reuters) get the highest; topical
# aggregators + community feeds get lower.
#
# Spanish-language sources commented out intentionally — will enable
# after verifying Ollama enrichment handles ES correctly.
DEFAULT_FEEDS: list[dict[str, Any]] = [
    # Crypto / prediction-market adjacent
    {"name": "coindesk",         "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",      "weight": 0.75},
    {"name": "cointelegraph",    "url": "https://cointelegraph.com/rss",                        "weight": 0.65},
    {"name": "theblock",         "url": "https://www.theblock.co/rss.xml",                      "weight": 0.80},
    {"name": "decrypt",          "url": "https://decrypt.co/feed",                              "weight": 0.65},
    {"name": "bitcoinmagazine",  "url": "https://bitcoinmagazine.com/.rss/full/",               "weight": 0.55},
    {"name": "cryptoslate",      "url": "https://cryptoslate.com/feed/",                        "weight": 0.55},
    # General / macro / politics
    {"name": "reuters_markets",  "url": "https://feeds.reuters.com/reuters/marketsNews",        "weight": 0.95},
    {"name": "axios",            "url": "https://api.axios.com/feed/",                          "weight": 0.80},
    {"name": "politico",         "url": "https://rss.politico.com/politics-news.xml",           "weight": 0.80},
    # Community — low trust weight but real signal on velocity shifts
    {"name": "reddit_crypto",    "url": "https://www.reddit.com/r/CryptoCurrency/.rss",         "weight": 0.30},
    {"name": "reddit_polymarket","url": "https://www.reddit.com/r/Polymarket/.rss",             "weight": 0.40},
    {"name": "hn_crypto",        "url": "https://hnrss.org/newest?q=crypto+OR+polymarket+OR+bitcoin", "weight": 0.45},
    # TODO(es): enable Spanish sources after Ollama ES enrichment is verified.
    # {"name": "criptonoticias",   "url": "https://www.criptonoticias.com/feed/",                "weight": 0.55},
    # {"name": "cointelegraph_es", "url": "https://es.cointelegraph.com/rss",                    "weight": 0.55},
]


@dataclass
class FeedSpec:
    name: str
    url: str
    weight: float


def _normalise(entry: Any) -> FeedSpec:
    if isinstance(entry, str):
        # Legacy: bare URL. Derive a name from the host.
        name = entry
        try:
            from urllib.parse import urlparse
            host = urlparse(entry).hostname or entry
            name = host.replace("www.", "")
        except Exception:
            pass
        return FeedSpec(name=name, url=entry, weight=0.5)
    if isinstance(entry, dict):
        url = entry.get("url") or ""
        name = (entry.get("name") or url).strip() or url
        weight = float(entry.get("weight") or 0.5)
        return FeedSpec(name=name, url=url, weight=max(0.0, min(1.0, weight)))
    raise ValueError(f"unrecognised RSS feed spec: {entry!r}")


def configured_feeds() -> list[FeedSpec]:
    """Read feeds from config.yaml; fall back to DEFAULT_FEEDS."""
    cfg = get_config().get("feeds", "rss") or {}
    raw = cfg.get("sources")
    if not raw:
        return [_normalise(e) for e in DEFAULT_FEEDS]
    out: list[FeedSpec] = []
    for entry in raw:
        try:
            out.append(_normalise(entry))
        except ValueError as e:
            logger.warning("[rss] skipping bad feed spec ({}): {}", e, entry)
    return out


class RSSFeed:
    component = "feed.rss"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._feeds: list[FeedSpec] = []
        # Per-feed stats so the dashboard can show health without a DB roundtrip.
        self._last_ok_ts: dict[str, float] = {}
        self._last_err: dict[str, str] = {}
        self._last_new: dict[str, int] = {}

    async def run(self) -> None:
        cfg = get_config().get("feeds", "rss") or {}
        if not cfg.get("enabled", True):
            logger.info("[rss] disabled")
            return
        self._feeds = configured_feeds()
        poll = int(cfg.get("poll_seconds", 60))
        backoff = Backoff(base=2, cap=60)

        logger.info("[rss] starting; {} sources poll={}s", len(self._feeds), poll)
        while not self._stop.is_set():
            try:
                total_new = 0
                ok = 0
                for spec in self._feeds:
                    try:
                        new = await self._poll_one(spec)
                        total_new += new
                        self._last_ok_ts[spec.name] = now_ts()
                        self._last_new[spec.name] = new
                        self._last_err.pop(spec.name, None)
                        ok += 1
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self._last_err[spec.name] = f"{type(e).__name__}: {e}"
                        logger.warning(
                            "[rss] {} failed ({}): {}", spec.name, type(e).__name__, e,
                        )
                if total_new:
                    logger.info(
                        "[rss] cycle feeds_ok={}/{} new={}",
                        ok, len(self._feeds), total_new,
                    )
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = backoff.next_delay()
                logger.warning(
                    "[rss] cycle error ({}), sleeping {:.1f}s: {}",
                    type(e).__name__, delay, e,
                )
                await self._sleep(delay)
                continue
            await self._sleep(poll * DEGRADED_POLL_MULTIPLIER if is_degraded() else poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _poll_one(self, spec: FeedSpec) -> int:
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, spec.url)
        new_items = 0
        for entry in parsed.entries or []:
            link = (entry.get("link") or "").strip()
            if not link:
                continue
            h = url_hash(link)
            existing = await fetch_one(
                "SELECT id FROM feed_items WHERE url_hash=?", (h,)
            )
            if existing:
                continue
            title = (entry.get("title") or "").strip()
            summary = (entry.get("summary") or entry.get("description") or "").strip()
            if len(summary) > 500:
                summary = summary[:497] + "…"
            published = self._published(entry)
            meta: dict[str, Any] = {
                "tags": [t.get("term") for t in (entry.get("tags") or []) if t.get("term")],
                "author": entry.get("author"),
                "feed_url": spec.url,
            }
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at,
                 meta, source_weight)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    h, spec.name, title, summary, link, published,
                    now_ts(), json.dumps(meta), float(spec.weight),
                ),
            )
            new_items += 1
        return new_items

    # ---------- Public introspection for /api/news/sources ----------

    def snapshot(self) -> list[dict[str, Any]]:
        out = []
        for spec in (self._feeds or configured_feeds()):
            out.append({
                "name": spec.name,
                "url": spec.url,
                "weight": spec.weight,
                "last_ok_ts": self._last_ok_ts.get(spec.name),
                "last_new": self._last_new.get(spec.name, 0),
                "last_error": self._last_err.get(spec.name),
            })
        return out

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
