"""GDELT 2.0 Doc API connector.

GDELT's public API: https://api.gdeltproject.org/api/v2/doc/doc
No auth, no rate limit token (soft rate limit only). Returns articles
matching a query within a time window.

We poll on a configurable cadence and write each new article into
``scout_signals`` with ``source="gdelt"``. Dedup is by ``signal_hash``
(sha256 of source + url).

Deliberately writes to ``scout_signals``, not ``feed_items``: the
existing signals pipeline auto-scores every feed_items row via Ollama,
which would double-process GDELT (the scout has its own heuristic
impact scorer in PR #1; LLM-backed scoring of these articles is on
the roadmap, not in v1).

Resilience contract (post-incidents, 2026-04-26):

  GDELT enforces a hard "1 request every 5 seconds" rate limit and
  responds to violations with HTTP 429 + a plain-text body. Once an
  IP has been rate-limited, GDELT also tarpits SSL handshakes for
  several minutes — observed as a wave of ConnectTimeouts even after
  spacing requests >5s. Recovery requires a fully serial, globally-
  rate-limited request stream, not just per-category spacing.

  v3 (this revision):

    1. ``_fetch_gdelt`` always returns a ``GdeltFetchResult`` —
       never raises. Records status / content-type / body excerpt /
       exception class so triage logs are actionable.
    2. PROCESS-WIDE ``GdeltRateLimiter`` (singleton). Every request
       — startup canary, category fetches, smoke probe — calls
       ``await GLOBAL_LIMITER.acquire()`` first. min_interval
       defaults to 6.0s (above GDELT's 5s ask). On a 429 the
       limiter widens to 12s for 60s, then narrows back. This is
       the primary fix for the 429 storms: even with two coroutines
       calling concurrently the limiter serializes them.
    3. Per-category exponential backoff (kept from v2): a category
       that fails is skipped for ``min(60s × 2^N, 30min)``.
    4. Categories whitelist (``feeds.gdelt.categories`` config) —
       defaults to a small set (shooting / election_result /
       macro_data_surprise) until the connector demonstrates
       stability. Operator can promote to "all" once the cycle log
       shows consistent ok>0.
    5. Bumped split httpx timeout (connect=20, read=30). On
       rate-limited IPs the SSL handshake alone can eat 10s+.
    6. Retry-once on timeout: the first ConnectTimeout / ReadTimeout
       gets one retry after a 5s wait (still going through the
       rate limiter). A persistent timeout backs off the category;
       a transient one is forgiven.
    7. INFO-level success log per category fetch — operators need
       to SEE that the connector is doing work, not only when it
       fails.
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

# Default categories whitelist for the initial-stability phase. The
# connector ships covering 13 categories but only RUNS the ones in
# this list until the operator promotes the config. With one request
# every ~6s and 3 categories, a cycle is ~18s — well under any
# reasonable rate limit and easy to validate.
_DEFAULT_ENABLED_CATEGORIES: tuple[EventCategory, ...] = (
    EventCategory.SHOOTING,
    EventCategory.ELECTION_RESULT,
    EventCategory.MACRO_DATA_SURPRISE,
)


# ============================================================
# Process-wide rate limiter (singleton)
# ============================================================


class GdeltRateLimiter:
    """Process-wide async rate limiter for GDELT requests.

    Every code path that hits GDELT (lane cycle, startup canary,
    smoke probe) MUST call ``await acquire()`` before the HTTP
    request — that's the contract. Only one request can pass per
    ``min_interval_seconds``.

    On a 429 response, the caller invokes ``penalize(seconds)`` to
    widen ``min_interval_seconds`` for that duration, then it
    automatically narrows back. This is the second-order rate-limit
    response: not just spacing requests further apart for the
    immediate retry, but pushing the overall rate down for a window
    so the IP can cool off.

    Process-local — runs in whichever event loop calls it first. The
    smoke probe and the lane both reuse the same module-level
    ``GLOBAL_LIMITER``.
    """

    def __init__(self, *, min_interval_seconds: float = 6.0) -> None:
        self._base_interval = float(min_interval_seconds)
        self._current_interval = float(min_interval_seconds)
        self._penalty_until_ts: float = 0.0
        self._last_request_ts: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def base_interval_seconds(self) -> float:
        return self._base_interval

    @property
    def current_interval_seconds(self) -> float:
        # If penalty window has expired, narrow back to the base
        # interval transparently.
        if self._penalty_until_ts and now_ts() >= self._penalty_until_ts:
            self._current_interval = self._base_interval
            self._penalty_until_ts = 0.0
        return self._current_interval

    def configure(self, *, min_interval_seconds: float) -> None:
        """Update the base interval. Used by the connector when the
        operator changes config without restart."""
        self._base_interval = float(min_interval_seconds)
        if not self._penalty_until_ts:
            self._current_interval = self._base_interval

    def penalize(self, *, widen_to_seconds: float, hold_for_seconds: float) -> None:
        """Widen the interval to ``widen_to_seconds`` for the next
        ``hold_for_seconds`` (relative to now). Multiple concurrent
        penalties take the widest interval and the latest expiry —
        a 429 storm naturally hardens the limiter.
        """
        new_interval = max(self._current_interval, float(widen_to_seconds))
        new_until = max(self._penalty_until_ts, now_ts() + float(hold_for_seconds))
        self._current_interval = new_interval
        self._penalty_until_ts = new_until

    async def acquire(self) -> float:
        """Block until at least ``current_interval_seconds`` has
        passed since the last request returned. Returns the actual
        wait time (for log/test inspection)."""
        async with self._lock:
            interval = self.current_interval_seconds
            wait = max(0.0, (self._last_request_ts + interval) - now_ts())
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = now_ts()
            return wait


# Module-level singleton. The probe script and the lane share this
# instance — that's the whole point of "GLOBAL". Initialised with the
# v3 default; the connector calls .configure() at run() if config has
# a different value.
GLOBAL_LIMITER = GdeltRateLimiter(min_interval_seconds=6.0)

# 429 response: widen the limiter to 12s for 60s. These are
# conservative — we'd rather over-cool than re-trigger.
_PENALTY_WIDEN_TO_SECONDS = 12.0
_PENALTY_HOLD_SECONDS = 60.0

# Retry-once on transient timeout: how long to wait between the
# timeout and the retry. Goes through the rate limiter on the second
# attempt as well.
_TIMEOUT_RETRY_DELAY_SECONDS = 5.0

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
    timeout_connect: float = 20.0,
    timeout_read: float = 30.0,
    user_agent: str = "Quantorpolybot/0.1 (scout)",
    limiter: GdeltRateLimiter | None = GLOBAL_LIMITER,
    retry_on_timeout: bool = True,
) -> GdeltFetchResult:
    """No-throw fetch. Returns a fully-populated GdeltFetchResult on
    every code path so the caller can log a single structured line
    and update its backoff state without try/except gymnastics.

    All requests go through ``limiter`` (default: the process-wide
    ``GLOBAL_LIMITER``) — every call site of this function MUST
    share the limiter so we can't accidentally exceed GDELT's
    1-req/5s rate from concurrent code paths. Pass ``limiter=None``
    only in tests where rate-limiting isn't being verified.

    On a 429, the limiter is automatically widened (penalize) for
    a hold window so subsequent calls space themselves further apart
    without each caller having to know.

    On a transient timeout (Connect or Read) AND ``retry_on_timeout``
    is True, ONE retry is attempted after a short wait. A second
    timeout records the failure.

    Pulled out of the connector class so:
      - tests can hit it without an event loop fixture for the lane,
      - the smoke probe script can call it directly through the same
        rate limiter.
    """
    url = _build_url(query, max_records=max_records, timespan=timespan)
    timeout = httpx.Timeout(connect=timeout_connect, read=timeout_read,
                            write=10.0, pool=10.0)

    async def _attempt() -> tuple[GdeltFetchResult, bool]:
        """Single fetch attempt. Returns ``(result, is_timeout)``.

        ``is_timeout=True`` signals the result was caused by a
        ``httpx.TimeoutException`` and the caller may retry. The
        result is still a fully populated ``GdeltFetchResult`` (with
        the original exception_class preserved) so a no-retry caller
        gets clean diagnostics.
        """
        if limiter is not None:
            await limiter.acquire()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, headers={"User-Agent": user_agent})
        except httpx.TimeoutException as e:
            return GdeltFetchResult(
                ok=False, url=url,
                exception_class=type(e).__name__, exception_msg=str(e)[:300],
                failure_mode="timeout",
            ), True
        except httpx.HTTPError as e:
            return GdeltFetchResult(
                ok=False, url=url,
                exception_class=type(e).__name__, exception_msg=str(e)[:300],
                failure_mode=_classify_failure(
                    status_code=0, content_type="", body_excerpt="",
                    exception_class=type(e).__name__,
                ),
            ), False
        except Exception as e:
            return GdeltFetchResult(
                ok=False, url=url,
                exception_class=type(e).__name__, exception_msg=str(e)[:300],
                failure_mode=f"exception:{type(e).__name__}",
            ), False
        # Successful HTTP — process inline (parse) and return a final
        # result. Not retry-eligible regardless of failure_mode.
        return _process_response(r, url), False

    # Attempt 1.
    result, was_timeout = await _attempt()
    if was_timeout and retry_on_timeout:
        # Transient timeout. One retry after a short wait.
        await asyncio.sleep(_TIMEOUT_RETRY_DELAY_SECONDS)
        retry_result, retry_was_timeout = await _attempt()
        if not retry_was_timeout:
            # Retry produced a definitive answer (success OR non-
            # timeout failure). Use it.
            result = retry_result
        else:
            # Both attempts timed out — annotate the original result
            # so logs make clear it wasn't a single transient blip.
            original_class = result.exception_class
            result = GdeltFetchResult(
                ok=False, url=url,
                exception_class=original_class,
                exception_msg=(
                    f"timeout (2 attempts, "
                    f"connect={timeout_connect}s read={timeout_read}s)"
                ),
                failure_mode="timeout",
            )
    elif was_timeout:
        # No-retry policy. Annotate but preserve the original
        # exception_class so the operator sees what kind of timeout.
        result = GdeltFetchResult(
            ok=False, url=url,
            exception_class=result.exception_class,
            exception_msg=(
                f"timeout (no retry, "
                f"connect={timeout_connect}s read={timeout_read}s)"
            ),
            failure_mode="timeout",
        )

    # On 429 / 200-with-rate-limit-text-body, widen the global limiter
    # so the NEXT caller spaces further apart.
    if not result.ok and result.failure_mode == "rate_limit" and limiter is not None:
        limiter.penalize(
            widen_to_seconds=_PENALTY_WIDEN_TO_SECONDS,
            hold_for_seconds=_PENALTY_HOLD_SECONDS,
        )
    return result


def _process_response(r: httpx.Response, url: str) -> GdeltFetchResult:
    """Pull out of ``_fetch_gdelt`` so the retry path doesn't
    duplicate the parse / classify logic."""
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
        # `request_delay_seconds` is now the GLOBAL rate-limiter
        # interval (was per-category in v2). Defaults to 6.0 — the
        # limiter widens to 12s for 60s after any 429 so the IP
        # cools off before subsequent calls.
        min_interval = safe_float(cfg.get("request_delay_seconds", 6.0))
        if min_interval < 5.0:
            logger.warning(
                "[gdelt] request_delay_seconds={} is below GDELT's "
                "documented 5s minimum — expect 429s",
                min_interval,
            )
        GLOBAL_LIMITER.configure(min_interval_seconds=min_interval)

        timespan = cfg.get("timespan", "15min")
        # Bumped defaults: 20s connect / 30s read. The previous
        # 10/20 was getting eaten in SSL handshake on a tarpitted IP.
        connect_t = safe_float(cfg.get("timeout_connect_seconds", 20.0))
        read_t = safe_float(cfg.get("timeout_read_seconds", 30.0))
        retry_on_timeout = bool(cfg.get("retry_on_timeout", True))

        # Categories whitelist. v3 starts with a 3-category subset for
        # stability (shooting / election_result / macro_data_surprise);
        # operator can promote to "all" or a custom list once the
        # cycle log shows consistent ok>0.
        enabled = self._resolve_enabled_categories(cfg.get("categories"))
        cycle_backoff = Backoff(base=5, cap=300)

        logger.info(
            "[gdelt] starting; poll={}s active_categories={}/{} per_query_max={} "
            "timespan={} min_interval={}s retry_on_timeout={} "
            "timeout=connect{}/read{}",
            poll, len(enabled), len(_CATEGORY_QUERIES), per_query_max,
            timespan, min_interval, retry_on_timeout, connect_t, read_t,
        )

        # ---- One-shot startup canary so an operator sees a clear
        # diagnostic line BEFORE the category sweep starts. Goes
        # through the same global limiter as everything else. ----
        if not self._canary_done:
            await self._startup_canary(connect_t, read_t, retry_on_timeout)
            self._canary_done = True

        while not self._stop.is_set():
            try:
                summary = await self._cycle(
                    enabled_categories=enabled,
                    per_query_max=per_query_max,
                    timespan=timespan,
                    timeout_connect=connect_t,
                    timeout_read=read_t,
                    retry_on_timeout=retry_on_timeout,
                )
                # ALWAYS log the cycle summary at INFO so operators
                # can see the connector is doing work even when no
                # signals come in (vs. silent and possibly stuck).
                logger.info(
                    "[gdelt] cycle ok={} fail={} skipped_in_backoff={} "
                    "articles_seen={} new_signals={} limiter_interval={:.1f}s",
                    summary["ok"], summary["fail"], summary["skipped"],
                    summary["articles_seen"], summary["new"],
                    GLOBAL_LIMITER.current_interval_seconds,
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

    def _resolve_enabled_categories(
        self, raw: Any,
    ) -> list[EventCategory]:
        """Resolve the operator's `categories` config into the actual
        list of enabled EventCategory enums.

        Accepted values:
          - None / missing → v3 default subset
          - "all" → every key in `_CATEGORY_QUERIES`
          - list of strings → filter by name (unknown names logged + dropped)
        """
        if raw is None:
            return list(_DEFAULT_ENABLED_CATEGORIES)
        if isinstance(raw, str) and raw.strip().lower() == "all":
            return list(_CATEGORY_QUERIES.keys())
        if not isinstance(raw, (list, tuple)):
            logger.warning(
                "[gdelt] feeds.gdelt.categories must be 'all' or a list; "
                "got {!r} — falling back to v3 default subset",
                raw,
            )
            return list(_DEFAULT_ENABLED_CATEGORIES)
        out: list[EventCategory] = []
        seen: set[EventCategory] = set()
        for name in raw:
            try:
                cat = EventCategory(str(name).strip().lower())
            except ValueError:
                logger.warning(
                    "[gdelt] unknown category in config: {!r}", name,
                )
                continue
            if cat not in _CATEGORY_QUERIES:
                logger.warning(
                    "[gdelt] category {!r} has no GDELT query registered",
                    cat.value,
                )
                continue
            if cat in seen:
                continue
            seen.add(cat)
            out.append(cat)
        if not out:
            logger.warning(
                "[gdelt] no valid categories after filtering — "
                "falling back to v3 default subset",
            )
            return list(_DEFAULT_ENABLED_CATEGORIES)
        return out

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _startup_canary(
        self, connect_t: float, read_t: float, retry_on_timeout: bool,
    ) -> None:
        """Single broad-keyword fetch to surface env/network/path
        issues at startup with a single loud log line. Failure does
        NOT abort startup — categories will discover the same
        breakage via their own backoffs.

        Goes through ``GLOBAL_LIMITER`` like everything else, so the
        canary counts as the first request of the cycle's rate
        budget.
        """
        result = await _fetch_gdelt(
            _CANARY_QUERY,
            max_records=5,
            timespan="1h",
            timeout_connect=connect_t,
            timeout_read=read_t,
            retry_on_timeout=retry_on_timeout,
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
        enabled_categories: list[EventCategory],
        per_query_max: int,
        timespan: str,
        timeout_connect: float,
        timeout_read: float,
        retry_on_timeout: bool,
    ) -> dict[str, int]:
        """One iteration of the per-category sweep. Returns a counter
        dict — used by the cycle log line and by tests.

        Strictly serial — each category's request goes through
        ``GLOBAL_LIMITER.acquire()`` so they never burst, even if a
        future caller invokes the connector from a different
        coroutine.
        """
        ok = fail = skipped = new = articles_seen = 0
        now_t = now_ts()
        for category in enabled_categories:
            if self._stop.is_set():
                break
            query = _CATEGORY_QUERIES.get(category)
            if not query:
                continue
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
            # Limiter is invoked inside _fetch_gdelt — no need to
            # sleep manually here, the limiter spaces calls.
            result = await _fetch_gdelt(
                query,
                max_records=per_query_max,
                timespan=timespan,
                timeout_connect=timeout_connect,
                timeout_read=timeout_read,
                retry_on_timeout=retry_on_timeout,
            )

            if not result.ok:
                next_retry = state.record_failure(
                    mode=result.failure_mode,
                    status=result.status_code,
                    now=now_t,
                )
                logger.warning(
                    "[gdelt] category={} {} backoff={:.0f}s consecutive={} "
                    "limiter_interval={:.1f}s",
                    category.value, result.diagnostic(),
                    next_retry - now_t, state.consecutive_failures,
                    GLOBAL_LIMITER.current_interval_seconds,
                )
                fail += 1
            else:
                state.record_success()
                category_new = await self._persist_articles(
                    result.articles, category,
                )
                new += category_new
                articles_seen += len(result.articles)
                ok += 1
                # INFO — operators need to SEE successful work, not
                # only failures. Even an empty-articles response is
                # signal that the connector + GDELT are talking.
                logger.info(
                    "[gdelt] category={} ok status={} articles={} new={} "
                    "bytes={} ct={}",
                    category.value, result.status_code,
                    len(result.articles), category_new,
                    result.bytes_len, result.content_type,
                )

            now_t = now_ts()  # advance for the next category's
                              # is_ready() check (the limiter sleep
                              # happened inside _fetch_gdelt)
        return {
            "ok": ok, "fail": fail, "skipped": skipped, "new": new,
            "articles_seen": articles_seen,
        }

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
