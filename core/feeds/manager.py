"""Spawns and supervises all feed coroutines."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from core.feeds.enrich import NewsEnricher
from core.feeds.fred import FredFeed
from core.feeds.google_news import GoogleNewsFeed
from core.feeds.metaculus import MetaculusFeed
from core.feeds.polymarket_news import PolymarketNewsFeed
from core.feeds.polymarket_ws import PolymarketWS
from core.feeds.predictit import PredictItFeed
from core.feeds.rss import RSSFeed
from core.feeds.wikipedia import WikipediaFeed


class FeedManager:
    def __init__(self) -> None:
        self.poly_ws = PolymarketWS()
        self.rss = RSSFeed()
        self.news_enricher = NewsEnricher()
        self.feeds: list[Any] = [
            self.rss,
            FredFeed(),
            MetaculusFeed(),
            self.poly_ws,
            PolymarketNewsFeed(),
            GoogleNewsFeed(),
            WikipediaFeed(),
            self.news_enricher,
            PredictItFeed(),
        ]
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    async def _supervise(self, feed: Any) -> None:
        """Restart a feed if it crashes. Each feed already has its own
        retry loop for expected errors (network, 5xx, etc.) — this is a
        last-resort supervisor for bugs that leak past ``feed.run()``
        entirely, so one broken feed can't silently stay dead for the
        rest of the process's lifetime."""
        backoff = 5.0
        while not self._stopping:
            try:
                await feed.run()
                return  # clean exit (e.g. feed disabled in config)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(
                    "[feeds] {} crashed, restarting in {:.0f}s: {}",
                    feed.component, backoff, e,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, 120.0)

    async def start(self) -> None:
        for feed in self.feeds:
            task = asyncio.create_task(
                self._supervise(feed), name=f"feed:{feed.component}",
            )
            self._tasks.append(task)
        logger.info("[feeds] started {} feeds", len(self._tasks))

    async def stop(self) -> None:
        self._stopping = True
        for feed in self.feeds:
            try:
                await feed.stop()
            except Exception as e:
                logger.warning("[feeds] stop error on {}: {}", feed.component, e)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("[feeds] stopped")
