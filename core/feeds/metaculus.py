"""Metaculus async poller. Pulls recent active questions."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts

METACULUS_URL = "https://www.metaculus.com/api2/questions/"
_HEADERS = {
    "User-Agent": "Quantorpolybot/0.1 (+https://github.com/local)",
    "Accept": "application/json",
}


class MetaculusFeed:
    component = "feed.metaculus"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "metaculus") or {}
        if not cfg.get("enabled", True):
            logger.info("[metaculus] disabled")
            return
        poll = int(cfg.get("poll_seconds", 1800))
        backoff = Backoff(base=5, cap=300)
        logger.info("[metaculus] starting poll={}s", poll)
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
            while not self._stop.is_set():
                try:
                    new = await self._poll(client)
                    if new:
                        logger.info("[metaculus] ingested {} new questions", new)
                    backoff.reset()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    delay = backoff.next_delay()
                    logger.exception("[metaculus] error, sleeping {:.1f}s: {}", delay, e)
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

    async def _poll(self, client: httpx.AsyncClient) -> int:
        r = await client.get(
            METACULUS_URL, params={"order_by": "-last_activity_time", "limit": 50}
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        new = 0
        for q in results:
            qid = q.get("id")
            if qid is None:
                continue
            uniq = f"metaculus:{qid}"
            h = url_hash(uniq)
            existing = await fetch_one(
                "SELECT id FROM feed_items WHERE url_hash=?", (h,)
            )
            if existing:
                continue
            title = (q.get("title") or "").strip()
            summary = (q.get("description") or "")[:500]
            community_pred = (q.get("community_prediction") or {}).get("full", {}) or {}
            meta: dict[str, Any] = {
                "metaculus_id": qid,
                "community_prediction": community_pred.get("q2"),
                "resolve_time": q.get("resolve_time"),
            }
            url = f"https://www.metaculus.com/questions/{qid}/"
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at, meta)
                VALUES (?,?,?,?,?,?,?,?)""",
                (h, "metaculus", title, summary, url, now_ts(), now_ts(), json.dumps(meta)),
            )
            new += 1
        return new
