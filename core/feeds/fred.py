"""FRED (Federal Reserve Economic Data) async poller."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from core.utils.config import env, get_config
from core.utils.db import execute, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


class FredFeed:
    component = "feed.fred"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "fred") or {}
        if not cfg.get("enabled", True):
            logger.info("[fred] disabled")
            return
        api_key = env("FRED_API_KEY")
        if not api_key:
            logger.warning("[fred] FRED_API_KEY missing; feed disabled at runtime")
            return
        series: list[str] = list(cfg.get("series") or [])
        poll = int(cfg.get("poll_seconds", 3600))
        backoff = Backoff(base=5, cap=300)

        logger.info("[fred] starting; {} series poll={}s", len(series), poll)
        async with httpx.AsyncClient(timeout=20.0) as client:
            while not self._stop.is_set():
                try:
                    new = 0
                    for sid in series:
                        new += await self._poll_series(client, api_key, sid)
                    if new:
                        logger.info("[fred] ingested {} new observations", new)
                    backoff.reset()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    delay = backoff.next_delay()
                    logger.exception("[fred] error, sleeping {:.1f}s: {}", delay, e)
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

    async def _poll_series(
        self, client: httpx.AsyncClient, api_key: str, series_id: str
    ) -> int:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }
        r = await client.get(FRED_BASE, params=params)
        r.raise_for_status()
        data = r.json()
        observations = data.get("observations") or []
        new = 0
        for obs in observations:
            date = obs.get("date")
            value = obs.get("value")
            if not date or value in (None, "."):
                continue
            uniq = f"fred:{series_id}:{date}"
            h = url_hash(uniq)
            existing = await fetch_one(
                "SELECT id FROM feed_items WHERE url_hash=?", (h,)
            )
            if existing:
                continue
            title = f"{series_id} {date}: {value}"
            summary = f"Federal Reserve series {series_id} new observation on {date} = {value}."
            meta: dict[str, Any] = {"series_id": series_id, "value": value, "date": date}
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at, meta)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    h,
                    f"fred:{series_id}",
                    title,
                    summary,
                    uniq,
                    now_ts(),
                    now_ts(),
                    json.dumps(meta),
                ),
            )
            new += 1
        return new
