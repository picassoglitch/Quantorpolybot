"""CryptoPanic free-tier news poller.

The public endpoint accepts an empty auth_token but returns much
better rate limits when CRYPTOPANIC_API_TOKEN is set (free key from
https://cryptopanic.com/developers/api/). Polls every couple of
minutes by default.

Items are filtered by the configured currency list (BTC/ETH/SOL by
default). Each ingested item carries the matched currencies in meta
so the signal pipeline can route it appropriately.
"""

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

API_URL = "https://cryptopanic.com/api/free/v1/posts/"
_HEADERS = {
    "User-Agent": "Quantorpolybot/0.1 (+https://github.com/local)",
    "Accept": "application/json",
}


class CryptoPanicFeed:
    component = "feed.cryptopanic"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._auth_warned = False

    async def run(self) -> None:
        cfg = get_config().get("feeds", "cryptopanic") or {}
        if not cfg.get("enabled", True):
            logger.info("[cryptopanic] disabled")
            return
        currencies: list[str] = [c.upper() for c in (cfg.get("currencies") or [])]
        poll = int(cfg.get("poll_seconds", 120))
        backoff = Backoff(base=10, cap=600)

        logger.info("[cryptopanic] starting; currencies={} poll={}s",
                    ",".join(currencies) or "ALL", poll)
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
            while not self._stop.is_set():
                try:
                    new = await self._poll(client, currencies)
                    if new:
                        logger.info("[cryptopanic] ingested {} new items", new)
                    backoff.reset()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    d = backoff.next_delay()
                    logger.exception("[cryptopanic] error, sleeping {:.1f}s: {}", d, e)
                    await self._sleep(d)
                    continue
                await self._sleep(poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _poll(self, client: httpx.AsyncClient, currencies: list[str]) -> int:
        token = env("CRYPTOPANIC_API_TOKEN", "")
        if not token and not self._auth_warned:
            logger.warning(
                "[cryptopanic] no CRYPTOPANIC_API_TOKEN set — public-only mode "
                "is heavily rate-limited. Add the token in the dashboard's "
                "Settings page."
            )
            self._auth_warned = True
        params: dict[str, Any] = {
            "auth_token": token,
            "public": "true",
        }
        if currencies:
            params["currencies"] = ",".join(currencies)
        r = await client.get(API_URL, params=params)
        # CryptoPanic answers 200 even on auth issues but inserts an
        # `info`/`error` field — surface that and stop instead of looping.
        if r.status_code == 401 or r.status_code == 403:
            if not self._auth_warned:
                logger.warning("[cryptopanic] auth rejected ({})", r.status_code)
                self._auth_warned = True
            return 0
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        new = 0
        for item in results:
            cid = item.get("id")
            url = (item.get("url") or "").strip()
            if not cid and not url:
                continue
            uniq = url or f"cryptopanic:{cid}"
            h = url_hash(uniq)
            existing = await fetch_one(
                "SELECT id FROM feed_items WHERE url_hash=?", (h,)
            )
            if existing:
                continue
            title = (item.get("title") or "").strip()
            currencies_meta = [
                c.get("code") for c in (item.get("currencies") or []) if c.get("code")
            ]
            source_meta = (item.get("source") or {})
            published = _parse_iso(item.get("published_at"))
            meta: dict[str, Any] = {
                "currencies": currencies_meta,
                "kind": item.get("kind"),
                "domain": source_meta.get("domain"),
                "votes": item.get("votes") or {},
            }
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at, meta)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    h,
                    "cryptopanic",
                    title,
                    title[:1000],
                    url or uniq,
                    published or now_ts(),
                    now_ts(),
                    json.dumps(meta),
                ),
            )
            new += 1
        return new


def _parse_iso(value: Any) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        # CryptoPanic uses RFC3339 with trailing Z.
        from datetime import datetime, timezone
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None
