"""Wikipedia Current Events Portal poller.

Fetches the rendered HTML of the Current Events Portal (or any other
configured HTML URL) and extracts every top-level bulleted news item
as its own feed_item. Dedup is per-URL + trimmed bullet text so the
same bullet across days isn't re-stored.

The previous implementation fetched the page as raw wikitext with
`action=raw`. That form mostly expands to template transclusions and
yields very few bullets — the observed production log was zero-ingest
every cycle. The rendered HTML gives us the bullets directly.
"""

from __future__ import annotations

import asyncio
import json
import re
from html.parser import HTMLParser
from typing import Iterable

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts

_HEADERS = {
    "User-Agent": "NexoPolyBot/0.1 (+https://github.com/local)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
_WS_RE = re.compile(r"\s+")


class _BulletExtractor(HTMLParser):
    """Pulls text out of every <li> element at any nesting depth.

    Good enough for the Current Events Portal, which wraps each day's
    items in <div class="current-events-content"> -> <ul> -> <li>.
    We don't need class-level filtering: trivial <li>s (navigation,
    empty footers) are removed downstream by the length filter.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._buffer: list[str] = []
        self._skip_depth = 0   # inside <style>, <script>, <sup> (citations)
        self.bullets: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in ("style", "script", "sup"):
            self._skip_depth += 1
            return
        if tag == "li":
            if self._depth == 0:
                self._buffer = []
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in ("style", "script", "sup"):
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag == "li" and self._depth > 0:
            self._depth -= 1
            if self._depth == 0:
                text = _WS_RE.sub(" ", "".join(self._buffer)).strip()
                # Strip the trailing citation bracket residue.
                text = re.sub(r"\[\d+\]", "", text).strip()
                if text:
                    self.bullets.append(text)

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth > 0:
            return
        if self._depth > 0:
            self._buffer.append(data)


class WikipediaFeed:
    component = "feed.wikipedia"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "wikipedia") or {}
        if not cfg.get("enabled", True):
            logger.info("[wiki] disabled")
            return
        sources: list[str] = list(cfg.get("sources") or [])
        poll = int(cfg.get("poll_seconds", 1800))
        backoff = Backoff(base=10, cap=600)

        logger.info("[wiki] starting; {} sources poll={}s", len(sources), poll)
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
            while not self._stop.is_set():
                try:
                    new = 0
                    for url in sources:
                        new += await self._poll_one(client, url)
                    if new:
                        logger.info("[wiki] ingested {} new items", new)
                    backoff.reset()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    d = backoff.next_delay()
                    logger.warning(
                        "[wiki] error ({}), sleeping {:.1f}s: {}",
                        type(e).__name__, d, e,
                    )
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

    async def _poll_one(self, client: httpx.AsyncClient, url: str) -> int:
        r = await client.get(url)
        r.raise_for_status()
        new = 0
        for bullet in _bullets_from_html(r.text or ""):
            # Filter out navigation / footer noise. Real news bullets
            # on the Current Events Portal are typically ≥ 30 chars; we
            # keep the floor at 20 to be safe.
            if len(bullet) < 20:
                continue
            h = url_hash(f"wiki:{url}:{bullet}")
            existing = await fetch_one(
                "SELECT id FROM feed_items WHERE url_hash=?", (h,)
            )
            if existing:
                continue
            title = bullet[:200]
            await execute(
                """INSERT OR IGNORE INTO feed_items
                (url_hash, source, title, summary, url, published_at, ingested_at, meta)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    h,
                    "wikipedia",
                    title,
                    bullet[:1000],
                    url,
                    now_ts(),
                    now_ts(),
                    json.dumps({"source_url": url}),
                ),
            )
            new += 1
        return new


def _bullets_from_html(html: str) -> Iterable[str]:
    if not html:
        return []
    parser = _BulletExtractor()
    try:
        parser.feed(html)
    except Exception as e:  # HTMLParser can choke on malformed pages
        logger.debug("[wiki] html parse error: {}", e)
        return parser.bullets
    return parser.bullets
