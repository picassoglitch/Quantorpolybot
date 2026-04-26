"""GDELT 2.0 Doc API connector.

GDELT's public API: https://api.gdeltproject.org/api/v2/doc/doc
No auth, no rate limit token (soft rate limit only). Returns articles
matching a query within a time window, with structured fields:

  - ``url``, ``title``, ``seendate``, ``socialimage``, ``domain``,
    ``language``, ``sourcecountry``
  - ``themes``: GDELT's CAMEO-derived theme tags (we use these for
    category hinting)
  - ``persons``, ``locations``, ``organizations``: extracted entities

We poll on a configurable cadence and write each new article into
``scout_signals`` with ``source="gdelt"`` and ``raw_payload`` carrying
the full JSON record. Dedup is by ``signal_hash`` (sha256 of
source + url).

Deliberately writes to ``scout_signals``, not ``feed_items``: the
existing signals pipeline auto-scores every feed_items row via Ollama,
which would double-process GDELT (the scout has its own heuristic
impact scorer in PR #1; LLM-backed scoring of these articles is on the
roadmap, not in v1).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlencode

import httpx
from loguru import logger

from core.scout.event import EventCategory, Signal
from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.helpers import Backoff, now_ts, safe_float
from core.utils.watchdog import is_degraded

DEGRADED_POLL_MULTIPLIER = 2

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Per-category GDELT search queries. GDELT's query language supports
# boolean OR / AND, exact-phrase quoting, and theme: filters. The
# queries are intentionally broad on the keyword side and lean on the
# Doc API's `mode=ArtList` to give us article-level granularity; the
# normalizer downstream handles fine-grained classification.
#
# Themes list: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
# We use a mix of free-text keywords (caught by GDELT's full-text index)
# and theme: predicates (caught by the entity-extraction layer) so a
# single category hits multiple GDELT signal pathways.
_CATEGORY_QUERIES: dict[EventCategory, str] = {
    EventCategory.SHOOTING: '("shooting" OR "shooter" OR "gunman") -movie',
    EventCategory.ASSASSINATION_ATTEMPT: '"assassination attempt" OR (assassinated AND politician)',
    EventCategory.EVACUATION: '"evacuation" OR "evacuated" OR "evacuating"',
    EventCategory.DEATH_INJURY: '("died" OR "killed" OR "wounded" OR "injured") (politician OR official OR leader)',
    EventCategory.RESIGNATION: '("resignation" OR "resigned" OR "stepping down") (minister OR director OR ceo OR president OR senator)',
    EventCategory.ARREST: '("arrested" OR "in custody" OR "detained") (politician OR ceo OR official)',
    EventCategory.INDICTMENT: '"indicted" OR "indictment" OR "charged with"',
    EventCategory.WAR_ESCALATION: '"airstrike" OR "missile strike" OR "ground offensive" OR "war escalation"',
    EventCategory.CEASEFIRE: '"ceasefire" OR "truce" OR "peace deal"',
    EventCategory.ELECTION_RESULT: '("election" AND ("called" OR "winner" OR "results" OR "concedes" OR "victory"))',
    EventCategory.COURT_RULING: '"supreme court ruling" OR "court ruled" OR "judge ruled" OR "verdict"',
    EventCategory.MACRO_DATA_SURPRISE: '"CPI" OR "jobs report" OR "GDP" OR "Fed decision" OR "rate hike" OR "rate cut"',
    EventCategory.SPORTS_INJURY: '("injured" OR "out for season" OR "ruled out") (NBA OR NFL OR MLB OR Premier League OR Champions League)',
}


class GdeltFeed:
    """Polls the GDELT Doc 2.0 API on a per-category cadence.

    Each `poll_seconds` cycle iterates through `_CATEGORY_QUERIES`,
    running one HTTP GET per category with `timespan=15min`. Articles
    are written to `scout_signals`; the normalizer (run by the scout
    lane) reads from there.

    Soft-disabled by `feeds.gdelt.enabled: false` in config. Backs off
    on any HTTP/parse error (5s..300s) so a temporary GDELT outage
    can't pin a CPU.
    """

    component = "feed.gdelt"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "gdelt") or {}
        if not cfg.get("enabled", True):
            logger.info("[gdelt] disabled")
            return
        poll = int(cfg.get("poll_seconds", 300))
        per_query_max = int(cfg.get("max_records_per_query", 50))
        request_delay = safe_float(cfg.get("request_delay_seconds", 1.0))
        timespan = cfg.get("timespan", "15min")
        backoff = Backoff(base=5, cap=300)

        logger.info(
            "[gdelt] starting; poll={}s queries={} per_query_max={} timespan={}",
            poll, len(_CATEGORY_QUERIES), per_query_max, timespan,
        )
        while not self._stop.is_set():
            try:
                total_new = 0
                for category, query in _CATEGORY_QUERIES.items():
                    if self._stop.is_set():
                        break
                    try:
                        new = await self._poll_category(
                            category, query, per_query_max, timespan,
                        )
                        total_new += new
                    except Exception as e:
                        # Per-category failures don't kill the cycle;
                        # log and continue so a single bad query (e.g.
                        # GDELT 503 on one shard) doesn't blank the
                        # whole scan.
                        logger.warning(
                            "[gdelt] category={} failed: {}",
                            category.value, e,
                        )
                    if request_delay > 0:
                        await self._sleep(request_delay)
                if total_new:
                    logger.info(
                        "[gdelt] ingested {} new signals across {} categories",
                        total_new, len(_CATEGORY_QUERIES),
                    )
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                d = backoff.next_delay()
                logger.warning(
                    "[gdelt] cycle error ({}), sleeping {:.1f}s: {}",
                    type(e).__name__, d, e,
                )
                await self._sleep(d)
                continue
            await self._sleep(
                poll * DEGRADED_POLL_MULTIPLIER if is_degraded() else poll,
            )

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _poll_category(
        self,
        category: EventCategory,
        query: str,
        max_records: int,
        timespan: str,
    ) -> int:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(max_records),
            "timespan": timespan,
            "sort": "DateDesc",
        }
        url = f"{GDELT_DOC_URL}?{urlencode(params)}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers={"User-Agent": "Quantorpolybot/0.1 (scout)"})
            r.raise_for_status()
            payload = r.json() if r.content else {}
        articles = payload.get("articles") or []
        new = 0
        for art in articles:
            sig = parse_gdelt_article(art, category)
            if sig is None:
                continue
            if await self._persist(sig):
                new += 1
        return new

    async def _persist(self, sig: Signal) -> bool:
        """INSERT OR IGNORE on signal_hash. Returns True iff a new
        row was created."""
        h = sig.signal_hash()
        existing = await fetch_one(
            "SELECT id FROM scout_signals WHERE signal_hash=?", (h,),
        )
        if existing:
            return False
        try:
            await execute(
                """INSERT OR IGNORE INTO scout_signals
                     (signal_hash, source, source_type, title, body,
                      url, entities, category_hint, published_at,
                      ingested_at, confidence, raw_payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    h,
                    sig.source,
                    sig.source_type,
                    sig.title[:500],
                    sig.body[:2000],
                    sig.url[:500],
                    json.dumps(sig.entities),
                    sig.category_hint,
                    sig.timestamp,
                    now_ts(),
                    sig.confidence,
                    json.dumps(sig.raw_payload, default=str)[:4000],
                ),
            )
            return True
        except Exception as e:
            logger.debug("[gdelt] persist failed for {}: {}", sig.url, e)
            return False


def parse_gdelt_article(art: dict[str, Any], category: EventCategory) -> Signal | None:
    """Parse one record from the GDELT Doc 2.0 ArtList JSON response.

    Pulled out of the connector class so tests can hit it without
    spinning up the async loop. Returns None on missing required
    fields (title or url) — those rows are skipped, not persisted.

    Confidence:
      Domain-tier multiplier × language-coverage multiplier.
      English-language wire/major-news domains score ~0.80; obscure
      blogs ~0.40. This is a conservative trust prior — the
      normalizer's source_count check then layers corroboration on
      top.
    """
    url = (art.get("url") or "").strip()
    title = (art.get("title") or "").strip()
    if not url or not title:
        return None

    domain = (art.get("domain") or "").strip().lower()
    language = (art.get("language") or "").strip().lower()
    seendate = art.get("seendate") or ""

    # GDELT's seendate is "YYYYMMDDhhmmss" (UTC). Convert to unix
    # seconds; fall back to "now" on parse failure so we don't drop
    # the article.
    published_at = _parse_gdelt_seendate(seendate)

    # Confidence prior. Tier-1 wire/major-news ~ 0.85; tier-2 mid ~
    # 0.65; everything else ~ 0.45. English-only bonus (other
    # languages drop to 0.40 max — we can't classify reliably).
    domain_tier = _domain_tier(domain)
    lang_mult = 1.0 if language in ("english", "en", "") else 0.6
    confidence = round(domain_tier * lang_mult, 3)

    # GDELT's per-article entity fields aren't returned by ArtList by
    # default; the connector pulls them from the title and from any
    # `themes` field if present. Cheap NER is plenty for v1.
    entities = _entities_from_title(title)

    return Signal(
        source="gdelt",
        source_type="global_event_db",
        timestamp=published_at,
        title=title,
        body=(art.get("excerpt") or "")[:2000],
        entities=entities,
        category_hint=category.value,
        url=url,
        confidence=confidence,
        raw_payload={
            "domain": domain,
            "language": language,
            "seendate": seendate,
            "sourcecountry": art.get("sourcecountry"),
            "socialimage": art.get("socialimage"),
        },
    )


def _parse_gdelt_seendate(s: str) -> float:
    """`YYYYMMDDhhmmss` -> unix seconds (UTC)."""
    if not s or len(s) < 14:
        return now_ts()
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return now_ts()


# Tier-1: international wire + major newsrooms. Tier-2: respectable
# regional / specialty. Everything else falls into the default low
# tier. Conservative — easier to add reputable domains over time
# than to retract trust.
_TIER1_DOMAINS: frozenset[str] = frozenset({
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "ft.com",
    "wsj.com", "nytimes.com", "washingtonpost.com", "bloomberg.com",
    "theguardian.com", "economist.com", "axios.com", "politico.com",
    "npr.org", "cnn.com", "cnbc.com",
})
_TIER2_DOMAINS: frozenset[str] = frozenset({
    "aljazeera.com", "dw.com", "lemonde.fr", "spiegel.de",
    "japantimes.co.jp", "scmp.com", "hindustantimes.com",
    "timesofindia.indiatimes.com", "abc.net.au", "cbc.ca",
    "thetimes.co.uk", "telegraph.co.uk", "independent.co.uk",
    "cnbc.com", "marketwatch.com", "thehill.com", "thedailybeast.com",
    "espn.com", "espnfc.com",
})


def _domain_tier(domain: str) -> float:
    """0.85 / 0.65 / 0.45 trust prior based on domain reputation."""
    if not domain:
        return 0.45
    # Strip leading "www." and any subdomain prefixes for matching.
    bare = domain.split("//")[-1].lstrip("www.").strip("/")
    # Match the rightmost N-1 segments — handles both "bbc.co.uk" and
    # "news.bbc.co.uk".
    parts = bare.split(".")
    candidates = {bare}
    for i in range(1, len(parts) - 1):
        candidates.add(".".join(parts[i:]))
    if candidates & _TIER1_DOMAINS:
        return 0.85
    if candidates & _TIER2_DOMAINS:
        return 0.65
    return 0.45


# Crude NER: pull single Title-Cased tokens of length >= 3 from the
# title and drop a stop-list of common headline verbs / honorifics /
# generic nouns. Single tokens (rather than greedy multi-word matches)
# work better for downstream Jaccard-overlap clustering — two
# articles about the same event are more likely to share `Trump` than
# `Trump Survives Shooting`. Good enough for v1 clustering; the
# mapper then matches against market.question tokens anyway.
import re

_TITLE_TOKEN_RE = re.compile(r"\b([A-Z][a-zA-Z'\-]+)\b")
_STOP_TITLES: frozenset[str] = frozenset({
    # Articles + honorifics
    "The", "A", "An", "President", "Senator", "Senators", "Governor",
    "Mr", "Mrs", "Ms", "Dr", "Sir", "Lord", "Lady", "Reverend",
    "Republican", "Democrat", "Republicans", "Democrats",
    "Police", "Officials", "Officer", "Reports", "Report",
    # Headline verbs commonly title-cased in news
    "Says", "Said", "Meets", "Met", "Survives", "Survived", "Faces",
    "Vows", "Wins", "Won", "Loses", "Lost", "Calls", "Called",
    "Ends", "Ended", "Begins", "Began", "Holds", "Held",
    "Announces", "Announced", "Defends", "Slams", "Strikes",
    # Generic event nouns that surface in many categories
    "Shooting", "Attack", "Rally", "Crisis", "Statement",
    "Election", "Vote", "Hearing", "Trial", "Speech",
    # Connectives that can start a clause
    "After", "Before", "While", "When", "During",
})


def _entities_from_title(title: str) -> list[str]:
    """Return up to 8 deduped Title-Cased single tokens from `title`."""
    if not title:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _TITLE_TOKEN_RE.findall(title):
        m = match.strip()
        if m in _STOP_TITLES:
            continue
        if len(m) < 3:
            continue
        key = m.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(m)
        if len(found) >= 8:
            break
    return found
