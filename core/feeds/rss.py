"""Async RSS poller. feedparser is sync — we run it in a thread."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import feedparser
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts


class RSSFeed:
    component = "feed.rss"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "rss") or {}
        if not cfg.get("enabled", True):
            logger.info("[rss] disabled")
            return
        sources: list[str] = list(cfg.get("sources") or [])
        poll = int(cfg.get("poll_seconds", 60))
        backoff = Backoff(base=2, cap=60)

        logger.info("[rss] starting; {} sources poll={}s", len(sources), poll)
        while not self._stop.is_set():
            try:
                count = 0
                for url in sources:
                    count += await self._poll_one(url)
                if count:
                    logger.info("[rss] ingested {} new items", count)
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = backoff.next_delay()
                logger.exception("[rss] error, sleeping {:.1f}s: {}", delay, e)
                await self._sleep(delay)
                continue
            await self._sleep(poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _poll_one(self, url: str) -> int:
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, feedparser.parse, url)
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
            published = self._published(entry)
            meta: dict[str, Any] = {
                "tags": [t.get("term") for t in (entry.get("tags") or []) if t.get("term")],
                "author": entry.get("author"),
            }
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at, meta)
                VALUES (?,?,?,?,?,?,?,?)""",
                (h, url, title, summary, link, published, now_ts(), json.dumps(meta)),
            )
            new_items += 1
        return new_items

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
