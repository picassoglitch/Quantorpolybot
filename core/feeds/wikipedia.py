"""Wikipedia / MediaWiki raw-wikitext poller.

Fetches the Portal:Current_events page (and any other configured raw
wikitext URLs), splits the content into bullet items, and stores each
distinct bullet as its own feed_item. Dedup uses a hash of the URL +
trimmed bullet text so the same bullet across days isn't re-stored.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Iterable

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts

_HEADERS = {
    "User-Agent": "Quantorpolybot/0.1 (+https://github.com/local)",
    "Accept": "text/plain, text/wiki, */*",
}
# Strip MediaWiki link markup: [[Target|Display]] -> Display, [[X]] -> X.
_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")
# Strip [[image:...]] / [[file:...]] entirely.
_FILE_LINK_RE = re.compile(r"\[\[(?:file|image):[^\]]+\]\]", re.IGNORECASE)
# Templates: {{...}} — replace with empty.
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
# Bullet at the start of a line (one or more *).
_BULLET_RE = re.compile(r"^\s*\*+\s*(.+)$")


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
                    logger.exception("[wiki] error, sleeping {:.1f}s: {}", d, e)
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
        text = r.text or ""
        new = 0
        for bullet in _bullets(text):
            if len(bullet) < 12:
                continue  # skip empty / wiki-noise lines
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


def _strip_wiki_markup(text: str) -> str:
    text = _FILE_LINK_RE.sub("", text)
    # Repeatedly strip templates so nested ones get reduced.
    for _ in range(3):
        new = _TEMPLATE_RE.sub("", text)
        if new == text:
            break
        text = new

    def _link_repl(m: re.Match) -> str:
        return m.group(2) or m.group(1)

    text = _LINK_RE.sub(_link_repl, text)
    # Strip leading apostrophes used for bold/italic, '''bold''' -> bold.
    text = re.sub(r"'{2,5}", "", text)
    return text.strip()


def _bullets(wikitext: str) -> Iterable[str]:
    """Yield cleaned text for each bullet line."""
    for raw_line in wikitext.splitlines():
        m = _BULLET_RE.match(raw_line)
        if not m:
            continue
        cleaned = _strip_wiki_markup(m.group(1))
        if cleaned:
            yield cleaned
