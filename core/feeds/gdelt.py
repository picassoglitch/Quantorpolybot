"""GDELT 2.0 Doc API connector.

GDELT's public API: https://api.gdeltproject.org/api/v2/doc/doc
No auth, no rate limit token (soft rate limit only). Returns articles
matching a query within a time window, with structured fields:

  - ``url``, ``title``, ``seendate``, ``socialimage``, ``domain``,
    ``language``, ``sourcecountry``

We poll on a configurable cadence and write each new article into
``scout_signals`` with ``source="gdelt"`` and ``raw_payload`` carrying
the full JSON record. Dedup is by ``signal_hash`` (sha256 of
source + url).

Deliberately writes to ``scout_signals``, not ``feed_items``: the
existing signals pipeline auto-scores every feed_items row via Ollama,
which would double-process GDELT (the scout has its own heuristic
impact scorer in PR #1; LLM-backed scoring of these articles is on
the roadmap, not in v1).

Resilience contract (post-incident, 2026-04-26):

  GDELT enforces a hard "1 request every 5 seconds" rate limit and
  responds to violations with HTTP 429 + a plain-text body
  ("Please limit requests to one every 5 seconds..."). Earlier
  versions of this connector had `request_delay_seconds: 1.0` which
  hit that limit on every cycle, then crashed `r.json()` on the
  text body, then surfaced as a vague `"category=X failed: ..."`
  log. The fix is the union of:

    1. ``_fetch_gdelt`` always returns a ``GdeltFetchResult`` — never
       raises. The result records status / content-type / body excerpt
       / exception class so triage logs are actionable.
    2. Default ``request_delay_seconds`` is 5.5 (above the 5s ask).
    3. Per-category exponential backoff: a category that fails is
       skipped for `min(60s × 2^N, 30min)` before the next attempt,
       so one bad query never re-triggers the same 429 storm next
       cycle.
    4. Split httpx timeout (connect=10, read=20). Single 15s deadlines
       were getting eaten in SSL handshake on rate-limited IPs.
    5. Startup canary: one simple-keyword fetch so the operator sees
       loud structured logs IF the credential/network/path is broken
       before the lane runs the full 13-category sweep.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
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

# Per-category backoff bounds. 60s start, double per consecutive
# failure, cap at 30 min. Tuned for GDELT's 5s rate limit + the
# observation that once GDELT 429s us, it tarpits SSL handshakes
# for ~minutes — we want to back off LONGER than the rate-limit
# floor so the IP cools down.
_BACKOFF_BASE_SECONDS = 60.0
_BACKOFF_CAP_SECONDS = 30 * 60.0

# Per-category GDELT search queries. GDELT's query language supports
# boolean OR / AND, exact-phrase quoting. The queries are intentionally
# focused — overly-broad queries get rejected as ambiguous, overly-
# complex queries fail validation. Keep ≤ 5 OR-terms per query.
_CATEGORY_QUERIES: dict[EventCategory, str] = {
    EventCategory.SHOOTING: '"shooting" OR "gunman" OR "active shooter"',
    EventCategory.ASSASSINATION_ATTEMPT: '"assassination attempt" OR "attempted assassination"',
    EventCategory.EVACUATION: '"evacuation" OR "evacuated" OR "evacuate"',
    EventCategory.DEATH_INJURY: '"killed in" OR "wounded in" OR "fatally shot"',
    EventCategory.RESIGNATION: '"resigned" OR "resignation" OR "stepping down"',
    EventCategory.ARREST: '"arrested" OR "in custody" OR "detained by police"',
    EventCategory.INDICTMENT: '"indicted" OR "indictment" OR "charged with"',
    EventCategory.WAR_ESCALATION: '"airstrike" OR "missile strike" OR "invasion"',
    EventCategory.CEASEFIRE: '"ceasefire" OR "truce" OR "peace deal"',
    EventCategory.ELECTION_RESULT: '"wins election" OR "election results" OR "concedes"',
    EventCategory.COURT_RULING: '"supreme court ruling" OR "court ruled" OR "verdict"',
    EventCategory.MACRO_DATA_SURPRISE: '"jobs report" OR "rate cut" OR "rate hike" OR "Fed decision"',
    EventCategory.SPORTS_INJURY: '"out for season" OR "ruled out" OR "torn ACL"',
}

# Used by the startup canary. Plain, broad, won't be rejected — the
# point is to verify GDELT is reachable and returning JSON before we
# launch into the 13-category sweep.
_CANARY_QUERY = '"climate change"'

# GDELT 429 body always begins with this phrase. Detected explicitly so
# we can short-circuit the "looks like JSON" check and emit a clearly
# tagged log line.
_RATE_LIMIT_PREFIX = "Please limit requests"

# Tier-1 / tier-2 domain reputation tables. Confidence prior for
# parsed articles. Conservative — easier to add reputable domains
# over time than to retract trust.
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


# ============================================================
# Structured fetch + parse
# ============================================================


@dataclass
class GdeltFetchResult:
    """Result of a single GDELT HTTP call. ``ok=True`` iff the call
    returned a 2xx with a parseable JSON body. All other paths set
    ``ok=False`` and populate the diagnostic fields so the caller can
    log a single structured line that names the failure mode.
    """

    ok: bool
    url: str
    status_code: int = 0
    content_type: str = ""
    bytes_len: int = 0
    body_excerpt: str = ""           # first 300 chars of response body
    exception_class: str = ""        # type(exc).__name__, empty on success
    exception_msg: str = ""          # str(exc)[:300], empty on success
    failure_mode: str = ""           # human-readable: "rate_limit"|"timeout"|"non_json"|"http_error"|"network"|"unknown"
    articles: list[dict[str, Any]] = field(default_factory=list)

    def diagnostic(self) -> str:
        """Compact one-line diagnostic for log emission."""
        if self.ok:
            return (
                f"ok status={self.status_code} ct={self.content_type} "
                f"bytes={self.bytes_len} articles={len(self.articles)}"
            )
        bits = [f"FAIL mode={self.failure_mode}"]
        if self.status_code:
            bits.append(f"status={self.status_code}")
        if self.content_type:
            bits.append(f"ct={self.content_type}")
        if self.bytes_len:
            bits.append(f"bytes={self.bytes_len}")
        if self.exception_class:
            bits.append(f"exc={self.exception_class}")
        if self.exception_msg:
            bits.append(f"msg={self.exception_msg!r}")
        if self.body_excerpt:
            bits.append(f"body={self.body_excerpt!r}")
        return " ".join(bits)


def _build_url(query: str, *, max_records: int, timespan: str, sort: str = "DateDesc") -> str:
    return f"{GDELT_DOC_URL}?" + urlencode({
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "timespan": timespan,
        "sort": sort,
    })


def _classify_failure(
    *,
    status_code: int,
    content_type: str,
    body_excerpt: str,
    exception_class: str,
) -> str:
    """Coarse-grained label for the failure mode. Used by the per-
    category backoff and the operator log."""
    if exception_class.endswith("TimeoutException") or "Timeout" in exception_class:
        return "timeout"
    if exception_class in ("ConnectError", "ReadError", "RemoteProtocolError",
                           "NetworkError", "TransportError"):
        return "network"
    if status_code == 429:
        return "rate_limit"
    if status_code and 400 <= status_code < 500:
        return "http_error"
    if status_code and 500 <= status_code < 600:
        return "server_error"
    if status_code == 200 and "json" not in content_type.lower():
        # Most common 200-but-not-json case for GDELT is the rate-limit
        # text body served with status 200 in some configurations, or
        # an HTML error page from a CDN edge. Detect explicitly.
        if body_excerpt.startswith(_RATE_LIMIT_PREFIX):
            return "rate_limit"
        return "non_json"
    if status_code == 200:
        return "json_decode"
    if exception_class:
        return f"exception:{exception_class}"
    return "unknown"


async def _fetch_gdelt(
    query: str,
    *,
    max_records: int = 50,
    timespan: str = "15min",
    timeout_connect: float = 10.0,
    timeout_read: float = 20.0,
    user_agent: str = "Quantorpolybot/0.1 (scout)",
) -> GdeltFetchResult:
    """No-throw fetch. Returns a fully-populated GdeltFetchResult on
    every code path so the caller can log a single structured line
    and update its backoff state without try/except gymnastics.

    Pulled out of the connector class so:
      - tests can hit it without an event loop fixture for the lane,
      - the smoke probe script can call it directly.
    """
    url = _build_url(query, max_records=max_records, timespan=timespan)
    timeout = httpx.Timeout(connect=timeout_connect, read=timeout_read,
                            write=10.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, headers={"User-Agent": user_agent})
    except httpx.TimeoutException as e:
        return GdeltFetchResult(
            ok=False, url=url,
            exception_class=type(e).__name__, exception_msg=str(e)[:300],
            failure_mode=_classify_failure(
                status_code=0, content_type="", body_excerpt="",
                exception_class=type(e).__name__,
            ),
        )
    except httpx.HTTPError as e:
        return GdeltFetchResult(
            ok=False, url=url,
            exception_class=type(e).__name__, exception_msg=str(e)[:300],
            failure_mode=_classify_failure(
                status_code=0, content_type="", body_excerpt="",
                exception_class=type(e).__name__,
            ),
        )
    except Exception as e:
        return GdeltFetchResult(
            ok=False, url=url,
            exception_class=type(e).__name__, exception_msg=str(e)[:300],
            failure_mode=f"exception:{type(e).__name__}",
        )

    content_type = r.headers.get("content-type", "")
    body_text = r.text or ""
    body_excerpt = body_text[:300].replace("\n", " ").replace("\r", " ")
    bytes_len = len(r.content or b"")

    # ---- Non-2xx ----
    if r.status_code >= 400:
        return GdeltFetchResult(
            ok=False, url=url,
            status_code=r.status_code, content_type=content_type,
            bytes_len=bytes_len, body_excerpt=body_excerpt,
            failure_mode=_classify_failure(
                status_code=r.status_code, content_type=content_type,
                body_excerpt=body_excerpt, exception_class="",
            ),
        )

    # ---- 2xx but obviously rate-limited (text body that begins with
    # the rate-limit phrase, served with 200 in some GDELT routes) ----
    if body_excerpt.startswith(_RATE_LIMIT_PREFIX):
        return GdeltFetchResult(
            ok=False, url=url,
            status_code=r.status_code, content_type=content_type,
            bytes_len=bytes_len, body_excerpt=body_excerpt,
            failure_mode="rate_limit",
        )

    # ---- 2xx empty body — treat as zero-articles success, NOT failure.
    # GDELT serves empty body for queries with no matches in window. ----
    if bytes_len == 0:
        return GdeltFetchResult(
            ok=True, url=url,
            status_code=r.status_code, content_type=content_type,
            bytes_len=0, body_excerpt="", articles=[],
        )

    # ---- 2xx with body — must be JSON ----
    try:
        payload = r.json()
    except (json.JSONDecodeError, ValueError) as e:
        return GdeltFetchResult(
            ok=False, url=url,
            status_code=r.status_code, content_type=content_type,
            bytes_len=bytes_len, body_excerpt=body_excerpt,
            exception_class=type(e).__name__, exception_msg=str(e)[:300],
            failure_mode=_classify_failure(
                status_code=r.status_code, content_type=content_type,
                body_excerpt=body_excerpt, exception_class="",
            ),
        )

    articles = payload.get("articles") if isinstance(payload, dict) else None
    if not isinstance(articles, list):
        articles = []
    return GdeltFetchResult(
        ok=True, url=url,
        status_code=r.status_code, content_type=content_type,
        bytes_len=bytes_len, body_excerpt=body_excerpt[:120],  # trim on success
        articles=articles,
    )


# ============================================================
# Per-category backoff state
# ============================================================


@dataclass
class _CategoryState:
    """Per-category fetch health. Lives on the GdeltFeed instance.

    `next_retry_ts` gates whether the cycle attempts this category
    at all. After ``consecutive_failures`` failures the next attempt
    is delayed by ``min(BASE × 2^(N-1), CAP)`` from the failure
    timestamp.
    """
    consecutive_failures: int = 0
    next_retry_ts: float = 0.0
    last_failure_mode: str = ""
    last_status: int = 0
    last_attempt_ts: float = 0.0

    def is_ready(self, now: float) -> bool:
        return now >= self.next_retry_ts

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.next_retry_ts = 0.0
        self.last_failure_mode = ""
        self.last_status = 0

    def record_failure(self, *, mode: str, status: int, now: float) -> float:
        """Apply exponential backoff. Returns the new next_retry_ts so
        the caller can log it."""
        self.consecutive_failures += 1
        self.last_failure_mode = mode
        self.last_status = status
        delay = min(
            _BACKOFF_BASE_SECONDS * (2 ** (self.consecutive_failures - 1)),
            _BACKOFF_CAP_SECONDS,
        )
        self.next_retry_ts = now + delay
        return self.next_retry_ts


# ============================================================
# Connector class
# ============================================================


class GdeltFeed:
    """Polls the GDELT Doc 2.0 API on a per-category cadence with
    per-category exponential backoff.
    """

    component = "feed.gdelt"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._cat_state: dict[EventCategory, _CategoryState] = {
            cat: _CategoryState() for cat in _CATEGORY_QUERIES
        }
        self._canary_done = False

    async def run(self) -> None:
        cfg = get_config().get("feeds", "gdelt") or {}
        if not cfg.get("enabled", True):
            logger.info("[gdelt] disabled")
            return
        poll = int(cfg.get("poll_seconds", 300))
        per_query_max = int(cfg.get("max_records_per_query", 50))
        # GDELT explicitly asks for 1 request per 5s. 5.5s gives a
        # safety margin against jitter. The legacy 1.0s default WAS
        # the root cause of the 2026-04-26 incident — never go back
        # under 5.0 without GDELT loosening the published rate limit.
        request_delay = safe_float(cfg.get("request_delay_seconds", 5.5))
        if request_delay < 5.0:
            logger.warning(
                "[gdelt] request_delay_seconds={} is below GDELT's "
                "documented 5s minimum — expect 429s",
                request_delay,
            )
        timespan = cfg.get("timespan", "15min")
        connect_t = safe_float(cfg.get("timeout_connect_seconds", 10.0))
        read_t = safe_float(cfg.get("timeout_read_seconds", 20.0))
        cycle_backoff = Backoff(base=5, cap=300)

        logger.info(
            "[gdelt] starting; poll={}s queries={} per_query_max={} "
            "timespan={} request_delay={}s timeout=connect{}/read{}",
            poll, len(_CATEGORY_QUERIES), per_query_max, timespan,
            request_delay, connect_t, read_t,
        )

        # ---- One-shot startup canary so an operator sees a clear
        # diagnostic line BEFORE the 13-category sweep starts. Doesn't
        # block the loop — failure logs but the main cycle still runs
        # (categories will hit their own backoffs if GDELT is broken). ----
        if not self._canary_done:
            await self._startup_canary(connect_t, read_t)
            self._canary_done = True

        while not self._stop.is_set():
            try:
                summary = await self._cycle(
                    per_query_max=per_query_max,
                    timespan=timespan,
                    request_delay=request_delay,
                    timeout_connect=connect_t,
                    timeout_read=read_t,
                )
                if summary["new"] or summary["fail"] or summary["skipped"]:
                    logger.info(
                        "[gdelt] cycle ok={} fail={} skipped_in_backoff={} "
                        "new_signals={}",
                        summary["ok"], summary["fail"], summary["skipped"],
                        summary["new"],
                    )
                cycle_backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                d = cycle_backoff.next_delay()
                logger.warning(
                    "[gdelt] cycle outer error ({}), sleeping {:.1f}s: {}",
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

    async def _startup_canary(self, connect_t: float, read_t: float) -> None:
        """Single broad-keyword fetch to surface env/network/path
        issues at startup with a single loud log line. Failure does
        NOT abort startup — categories will discover the same
        breakage via their own backoffs."""
        result = await _fetch_gdelt(
            _CANARY_QUERY,
            max_records=5,
            timespan="1h",
            timeout_connect=connect_t,
            timeout_read=read_t,
        )
        if result.ok:
            logger.info(
                "[gdelt] startup canary ok status={} articles={} bytes={} "
                "ct={}",
                result.status_code, len(result.articles), result.bytes_len,
                result.content_type,
            )
        else:
            logger.warning(
                "[gdelt] startup canary FAILED — {}",
                result.diagnostic(),
            )

    async def _cycle(
        self,
        *,
        per_query_max: int,
        timespan: str,
        request_delay: float,
        timeout_connect: float,
        timeout_read: float,
    ) -> dict[str, int]:
        """One iteration of the per-category sweep. Returns a counter
        dict — used by the cycle log line and by tests."""
        ok = fail = skipped = new = 0
        now_t = now_ts()
        for category, query in _CATEGORY_QUERIES.items():
            if self._stop.is_set():
                break
            state = self._cat_state[category]
            if not state.is_ready(now_t):
                skipped += 1
                # DEBUG only — INFO would spam during a long backoff.
                logger.debug(
                    "[gdelt] category={} skipped (in backoff for {:.0f}s)",
                    category.value, state.next_retry_ts - now_t,
                )
                continue

            state.last_attempt_ts = now_t
            result = await _fetch_gdelt(
                query,
                max_records=per_query_max,
                timespan=timespan,
                timeout_connect=timeout_connect,
                timeout_read=timeout_read,
            )

            if not result.ok:
                next_retry = state.record_failure(
                    mode=result.failure_mode,
                    status=result.status_code,
                    now=now_t,
                )
                # Single structured line per failure — the log is the
                # debug interface here. body_excerpt + content_type +
                # exception class together pinpoint the failure mode.
                logger.warning(
                    "[gdelt] category={} {} backoff={:.0f}s consecutive={}",
                    category.value, result.diagnostic(),
                    next_retry - now_t, state.consecutive_failures,
                )
                fail += 1
            else:
                state.record_success()
                category_new = await self._persist_articles(
                    result.articles, category,
                )
                new += category_new
                ok += 1
                if result.articles:
                    logger.debug(
                        "[gdelt] category={} {} new={}",
                        category.value, result.diagnostic(), category_new,
                    )

            # Pace requests irrespective of success/fail — the rate
            # limit applies to attempts, not just successes.
            if request_delay > 0 and not self._stop.is_set():
                await self._sleep(request_delay)
            now_t = now_ts()  # advance for the next category's
                              # is_ready() check
        return {"ok": ok, "fail": fail, "skipped": skipped, "new": new}

    async def _persist_articles(
        self, articles: list[dict[str, Any]], category: EventCategory,
    ) -> int:
        new = 0
        for art in articles:
            sig = parse_gdelt_article(art, category)
            if sig is None:
                continue
            if await self._persist(sig):
                new += 1
        return new

    async def _persist(self, sig: Signal) -> bool:
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
                    h, sig.source, sig.source_type,
                    sig.title[:500], sig.body[:2000], sig.url[:500],
                    json.dumps(sig.entities), sig.category_hint,
                    sig.timestamp, now_ts(), sig.confidence,
                    json.dumps(sig.raw_payload, default=str)[:4000],
                ),
            )
            return True
        except Exception as e:
            logger.debug("[gdelt] persist failed for {}: {}", sig.url, e)
            return False


# ============================================================
# Article parser (unchanged from PR #1, kept here for parity)
# ============================================================


def parse_gdelt_article(art: dict[str, Any], category: EventCategory) -> Signal | None:
    url = (art.get("url") or "").strip()
    title = (art.get("title") or "").strip()
    if not url or not title:
        return None

    domain = (art.get("domain") or "").strip().lower()
    language = (art.get("language") or "").strip().lower()
    seendate = art.get("seendate") or ""

    published_at = _parse_gdelt_seendate(seendate)

    domain_tier = _domain_tier(domain)
    lang_mult = 1.0 if language in ("english", "en", "") else 0.6
    confidence = round(domain_tier * lang_mult, 3)

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
    if not s or len(s) < 14:
        return now_ts()
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return now_ts()


def _domain_tier(domain: str) -> float:
    if not domain:
        return 0.45
    bare = domain.split("//")[-1].lstrip("www.").strip("/")
    parts = bare.split(".")
    candidates = {bare}
    for i in range(1, len(parts) - 1):
        candidates.add(".".join(parts[i:]))
    if candidates & _TIER1_DOMAINS:
        return 0.85
    if candidates & _TIER2_DOMAINS:
        return 0.65
    return 0.45


_TITLE_TOKEN_RE = re.compile(r"\b([A-Z][a-zA-Z'\-]+)\b")
_STOP_TITLES: frozenset[str] = frozenset({
    "The", "A", "An", "President", "Senator", "Senators", "Governor",
    "Mr", "Mrs", "Ms", "Dr", "Sir", "Lord", "Lady", "Reverend",
    "Republican", "Democrat", "Republicans", "Democrats",
    "Police", "Officials", "Officer", "Reports", "Report",
    "Says", "Said", "Meets", "Met", "Survives", "Survived", "Faces",
    "Vows", "Wins", "Won", "Loses", "Lost", "Calls", "Called",
    "Ends", "Ended", "Begins", "Began", "Holds", "Held",
    "Announces", "Announced", "Defends", "Slams", "Strikes",
    "Shooting", "Attack", "Rally", "Crisis", "Statement",
    "Election", "Vote", "Hearing", "Trial", "Speech",
    "After", "Before", "While", "When", "During",
})


def _entities_from_title(title: str) -> list[str]:
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
