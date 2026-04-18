"""Spawns and supervises all feed coroutines."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from core.feeds.fred import FredFeed
from core.feeds.metaculus import MetaculusFeed
from core.feeds.polymarket_ws import PolymarketWS
from core.feeds.rss import RSSFeed


class FeedManager:
    def __init__(self) -> None:
        self.feeds: list[Any] = [
            RSSFeed(),
            FredFeed(),
            MetaculusFeed(),
            PolymarketWS(),
        ]
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        for feed in self.feeds:
            task = asyncio.create_task(feed.run(), name=f"feed:{feed.component}")
            self._tasks.append(task)
        logger.info("[feeds] started {} feeds", len(self._tasks))

    async def stop(self) -> None:
        for feed in self.feeds:
            try:
                await feed.stop()
            except Exception as e:
                logger.warning("[feeds] stop error on {}: {}", feed.component, e)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("[feeds] stopped")
