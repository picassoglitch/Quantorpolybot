"""GDELT fetch + backoff tests.

Pins the resilience contract: `_fetch_gdelt` must NEVER raise to the
caller, must populate `GdeltFetchResult.diagnostic()` with enough detail
to triage a failure from one log line, and `_CategoryState` must apply
exponential backoff that caps at 30 minutes.

Mocks httpx so no network call ever happens during tests.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from core.feeds.gdelt import (
    _BACKOFF_BASE_SECONDS,
    _BACKOFF_CAP_SECONDS,
    _PENALTY_HOLD_SECONDS,
    _PENALTY_WIDEN_TO_SECONDS,
    GdeltRateLimiter,
    _CategoryState,
    _classify_failure,
    _fetch_gdelt,
)


# ============================================================
# _fetch_gdelt — happy path + every failure mode
# ============================================================


class _MockResponse:
    """Minimal stand-in for httpx.Response that supports .text /
    .content / .headers / .status_code / .json()."""

    def __init__(self, *, status_code: int, body: bytes,
                 content_type: str = "application/json") -> None:
        self.status_code = status_code
        self.content = body
        self.text = body.decode("utf-8", errors="replace")
        self.headers = {"content-type": content_type}

    def json(self):
        return json.loads(self.text)


class _MockClient:
    """httpx.AsyncClient stand-in. Returns a single canned response."""

    def __init__(self, response=None, raise_exc: BaseException | None = None) -> None:
        self._response = response
        self._raise = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *args, **kwargs):
        if self._raise is not None:
            raise self._raise
        return self._response


def _patch_client(monkeypatch, response=None, raise_exc=None):
    """Replace httpx.AsyncClient with our mock for this test."""
    def _factory(*a, **k):
        return _MockClient(response=response, raise_exc=raise_exc)
    monkeypatch.setattr(httpx, "AsyncClient", _factory)


@pytest.mark.asyncio
async def test_fetch_ok_returns_articles(monkeypatch):
    body = json.dumps({"articles": [
        {"url": "https://x.com/1", "title": "t1", "domain": "x.com",
         "language": "english", "seendate": "20260426010000"},
        {"url": "https://x.com/2", "title": "t2", "domain": "x.com",
         "language": "english", "seendate": "20260426010000"},
    ]}).encode()
    _patch_client(monkeypatch, response=_MockResponse(status_code=200, body=body))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is True
    assert r.status_code == 200
    assert len(r.articles) == 2
    assert "ok" in r.diagnostic()


@pytest.mark.asyncio
async def test_fetch_429_with_text_body_classified_as_rate_limit(monkeypatch):
    body = b"Please limit requests to one every 5 seconds or contact ..."
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=429, body=body, content_type="text/plain",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "rate_limit"
    assert r.status_code == 429
    assert "Please limit" in r.body_excerpt
    # Diagnostic must name the failure mode + status + body excerpt.
    diag = r.diagnostic()
    assert "rate_limit" in diag
    assert "429" in diag
    assert "Please limit" in diag


@pytest.mark.asyncio
async def test_fetch_200_with_rate_limit_text_body_also_classified_as_rate_limit(monkeypatch):
    """GDELT sometimes returns the rate-limit message with status 200
    + text/plain content-type. The classifier must catch this case
    too — otherwise we'd treat it as a `non_json` and miss the
    rate-limit signal."""
    body = b"Please limit requests to one every 5 seconds..."
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=200, body=body, content_type="text/plain",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "rate_limit"


@pytest.mark.asyncio
async def test_fetch_200_html_body_classified_as_non_json(monkeypatch):
    body = b"<html><body>Internal error</body></html>"
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=200, body=body, content_type="text/html",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "non_json"
    assert "html" in r.content_type
    assert "Internal error" in r.body_excerpt


@pytest.mark.asyncio
async def test_fetch_200_invalid_json_with_json_content_type(monkeypatch):
    """JSON content-type but body is malformed JSON → json_decode."""
    body = b'{"articles": [malformed'
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=200, body=body, content_type="application/json",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "json_decode"
    assert r.exception_class in ("JSONDecodeError", "ValueError")


@pytest.mark.asyncio
async def test_fetch_200_empty_body_returns_ok_with_zero_articles(monkeypatch):
    """GDELT returns empty body for queries with no matches in window
    — that's a valid zero-result, NOT a failure."""
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=200, body=b"", content_type="application/json",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is True
    assert r.articles == []


@pytest.mark.asyncio
async def test_fetch_5xx_classified_as_server_error(monkeypatch):
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=503, body=b"upstream broken", content_type="text/plain",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "server_error"
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_fetch_403_classified_as_http_error(monkeypatch):
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=403, body=b"forbidden",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "http_error"


@pytest.mark.asyncio
async def test_fetch_connect_timeout_classified_as_timeout(monkeypatch):
    _patch_client(monkeypatch, raise_exc=httpx.ConnectTimeout("timed out"))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "timeout"
    assert r.exception_class == "ConnectTimeout"


@pytest.mark.asyncio
async def test_fetch_read_timeout_classified_as_timeout(monkeypatch):
    _patch_client(monkeypatch, raise_exc=httpx.ReadTimeout("read timed out"))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "timeout"


@pytest.mark.asyncio
async def test_fetch_network_error_classified_as_network(monkeypatch):
    _patch_client(monkeypatch, raise_exc=httpx.ConnectError("dns failed"))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "network"


@pytest.mark.asyncio
async def test_fetch_diagnostic_truncates_body_at_300_chars(monkeypatch):
    body = ("X" * 1000).encode()
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=429, body=body, content_type="text/plain",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    # Excerpt cap is 300 — never log a full multi-KB body.
    assert len(r.body_excerpt) <= 300


@pytest.mark.asyncio
async def test_fetch_never_raises_even_on_unexpected_exception(monkeypatch):
    """The whole point of the resilience contract: caller must never
    have to wrap _fetch_gdelt in try/except."""
    class _Boom(Exception):
        pass
    _patch_client(monkeypatch, raise_exc=_Boom("something weird"))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)  # MUST NOT RAISE
    assert r.ok is False
    assert r.exception_class == "_Boom"
    assert "something weird" in r.exception_msg


# ============================================================
# _classify_failure — direct unit test of the labeller
# ============================================================


def test_classify_429_is_rate_limit():
    assert _classify_failure(
        status_code=429, content_type="text/plain",
        body_excerpt="Please limit", exception_class="",
    ) == "rate_limit"


def test_classify_200_with_rate_limit_text_is_rate_limit():
    assert _classify_failure(
        status_code=200, content_type="text/plain",
        body_excerpt="Please limit requests to one every 5 seconds",
        exception_class="",
    ) == "rate_limit"


def test_classify_200_with_html_is_non_json():
    assert _classify_failure(
        status_code=200, content_type="text/html",
        body_excerpt="<html>", exception_class="",
    ) == "non_json"


def test_classify_5xx_is_server_error():
    assert _classify_failure(
        status_code=502, content_type="text/html",
        body_excerpt="bad gateway", exception_class="",
    ) == "server_error"


def test_classify_4xx_is_http_error():
    assert _classify_failure(
        status_code=403, content_type="",
        body_excerpt="", exception_class="",
    ) == "http_error"


def test_classify_timeout_exception_is_timeout():
    assert _classify_failure(
        status_code=0, content_type="", body_excerpt="",
        exception_class="ConnectTimeout",
    ) == "timeout"


def test_classify_network_error_is_network():
    assert _classify_failure(
        status_code=0, content_type="", body_excerpt="",
        exception_class="ConnectError",
    ) == "network"


# ============================================================
# query_invalid classification — GDELT validation errors
# ============================================================


def test_classify_timespan_too_short_is_query_invalid():
    """The GDELT message that triggered the v3 fix. Must be classified
    as `query_invalid` so the operator log shouts 'fix config' instead
    of generic non_json."""
    assert _classify_failure(
        status_code=200, content_type="text/html; charset=utf-8",
        body_excerpt="Timespan is too short. ",
        exception_class="",
    ) == "query_invalid"


def test_classify_other_gdelt_validation_messages_are_query_invalid():
    for body in (
        "Timespan is invalid",
        "Mode is invalid",
        "Query is invalid",
        "Format is invalid",
        "Query must contain at least one keyword",
    ):
        assert _classify_failure(
            status_code=200, content_type="text/html",
            body_excerpt=body, exception_class="",
        ) == "query_invalid", f"failed for body={body!r}"


def test_classify_unrelated_html_body_is_non_json_not_query_invalid():
    """A genuine HTML error page (CDN edge, etc.) should still be
    classified as non_json, not as a config-bug query_invalid."""
    assert _classify_failure(
        status_code=200, content_type="text/html",
        body_excerpt="<html><body>504 Gateway Timeout</body></html>",
        exception_class="",
    ) == "non_json"


@pytest.mark.asyncio
async def test_fetch_query_invalid_classified_via_response(monkeypatch):
    """End-to-end: a GDELT 'Timespan is too short' response surfaces
    as failure_mode=query_invalid in the GdeltFetchResult."""
    body = b"Timespan is too short. "
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=200, body=body, content_type="text/html; charset=utf-8",
    ))
    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "query_invalid"
    assert "Timespan" in r.body_excerpt


# ============================================================
# _CategoryState backoff
# ============================================================


def test_state_starts_ready():
    s = _CategoryState()
    assert s.is_ready(now=0.0)
    assert s.consecutive_failures == 0


def test_failure_pushes_next_retry_forward():
    s = _CategoryState()
    next_retry = s.record_failure(mode="rate_limit", status=429, now=1000.0)
    assert next_retry == 1000.0 + _BACKOFF_BASE_SECONDS
    assert s.is_ready(now=1000.0) is False
    assert s.is_ready(now=next_retry) is True


def test_consecutive_failures_double_the_backoff():
    s = _CategoryState()
    t1 = s.record_failure(mode="rate_limit", status=429, now=1000.0)
    delay1 = t1 - 1000.0
    t2 = s.record_failure(mode="rate_limit", status=429, now=2000.0)
    delay2 = t2 - 2000.0
    t3 = s.record_failure(mode="rate_limit", status=429, now=3000.0)
    delay3 = t3 - 3000.0
    assert delay2 == 2 * delay1
    assert delay3 == 4 * delay1


def test_backoff_caps_at_30_minutes():
    s = _CategoryState()
    # Force the failure count high enough that 60s × 2^N > cap.
    s.consecutive_failures = 20
    t = s.record_failure(mode="rate_limit", status=429, now=0.0)
    delay = t - 0.0
    assert delay <= _BACKOFF_CAP_SECONDS + 0.01
    assert delay == _BACKOFF_CAP_SECONDS  # 60 × 2^20 way over cap


def test_success_resets_state():
    s = _CategoryState()
    s.record_failure(mode="rate_limit", status=429, now=1000.0)
    assert s.consecutive_failures == 1
    s.record_success()
    assert s.consecutive_failures == 0
    assert s.next_retry_ts == 0.0
    assert s.is_ready(now=0.0) is True
    assert s.last_failure_mode == ""


def test_query_invalid_jumps_straight_to_long_backoff():
    """Spec: re-running the same broken query every cycle wastes
    rate-limit budget. The first query_invalid failure should push
    backoff to the long (30 min) bucket, not the 60s normal start."""
    from core.feeds.gdelt import _QUERY_INVALID_BACKOFF_SECONDS
    s = _CategoryState()
    next_retry = s.record_failure(mode="query_invalid", status=200, now=1000.0)
    assert next_retry - 1000.0 == _QUERY_INVALID_BACKOFF_SECONDS
    assert s.is_ready(now=1000.0 + 60) is False
    assert s.is_ready(now=next_retry) is True


def test_query_invalid_does_not_compound_with_consecutive_failures():
    """Even on the 5th query_invalid in a row, the backoff stays at
    the long flat value — it's not exponentially compounding."""
    from core.feeds.gdelt import _QUERY_INVALID_BACKOFF_SECONDS
    s = _CategoryState()
    last = 1000.0
    for _ in range(5):
        last = s.record_failure(mode="query_invalid", status=200, now=last)
    # last - (1000 + 4 incremental nows) should still be the flat 30 min
    # — but easier: assert the raw delay between the last failure's
    # `now` and `next_retry`.
    s2 = _CategoryState()
    s2.consecutive_failures = 4  # pretend 4 prior fails
    next_retry = s2.record_failure(mode="query_invalid", status=200, now=2000.0)
    assert next_retry - 2000.0 == _QUERY_INVALID_BACKOFF_SECONDS


# ============================================================
# GdeltRateLimiter — global serialization + 429 widening
# ============================================================


@pytest.mark.asyncio
async def test_limiter_first_acquire_does_not_wait():
    lim = GdeltRateLimiter(min_interval_seconds=6.0)
    wait = await lim.acquire()
    assert wait == 0.0


@pytest.mark.asyncio
async def test_limiter_second_acquire_sleeps_until_interval_elapsed(monkeypatch):
    """Don't actually sleep 6s in tests — patch asyncio.sleep + the
    clock so we can prove the wait is computed correctly."""
    from core.feeds import gdelt as gdelt_mod
    fake_now = [1000.0]
    sleeps: list[float] = []

    def _now():
        return fake_now[0]

    async def _sleep(s):
        sleeps.append(s)
        # Advance fake clock so the next acquire sees time has passed.
        fake_now[0] += s

    monkeypatch.setattr(gdelt_mod, "now_ts", _now)
    monkeypatch.setattr("asyncio.sleep", _sleep)

    lim = GdeltRateLimiter(min_interval_seconds=6.0)
    await lim.acquire()  # t=1000, no wait
    assert sleeps == []
    # No real time has passed in fake-clock land, so second acquire
    # must wait the full 6.0s.
    await lim.acquire()
    assert len(sleeps) == 1
    assert abs(sleeps[0] - 6.0) < 0.01


@pytest.mark.asyncio
async def test_limiter_widens_after_penalize(monkeypatch):
    from core.feeds import gdelt as gdelt_mod
    fake_now = [1000.0]
    monkeypatch.setattr(gdelt_mod, "now_ts", lambda: fake_now[0])

    lim = GdeltRateLimiter(min_interval_seconds=6.0)
    assert lim.current_interval_seconds == 6.0
    lim.penalize(widen_to_seconds=12.0, hold_for_seconds=60.0)
    assert lim.current_interval_seconds == 12.0
    # Inside hold window: still wide.
    fake_now[0] += 30.0
    assert lim.current_interval_seconds == 12.0
    # After hold window: narrows back to base.
    fake_now[0] += 60.0
    assert lim.current_interval_seconds == 6.0


@pytest.mark.asyncio
async def test_limiter_penalize_does_not_narrow_existing_wide_window(monkeypatch):
    """If a 429 hits during an existing penalty, the limiter takes
    the WIDER interval and the LATER expiry — never narrows
    accidentally."""
    from core.feeds import gdelt as gdelt_mod
    fake_now = [1000.0]
    monkeypatch.setattr(gdelt_mod, "now_ts", lambda: fake_now[0])

    lim = GdeltRateLimiter(min_interval_seconds=6.0)
    lim.penalize(widen_to_seconds=20.0, hold_for_seconds=120.0)
    assert lim.current_interval_seconds == 20.0
    # A second penalty with smaller widen / shorter hold must NOT
    # narrow us down or shorten the cool-off.
    lim.penalize(widen_to_seconds=8.0, hold_for_seconds=30.0)
    assert lim.current_interval_seconds == 20.0
    fake_now[0] += 60.0
    # Still inside the original 120s window.
    assert lim.current_interval_seconds == 20.0


# ============================================================
# Fetch ↔ Limiter integration: 429 widens limiter automatically
# ============================================================


@pytest.mark.asyncio
async def test_fetch_429_widens_the_limiter(monkeypatch):
    """Spec: on 429 the GLOBAL_LIMITER must auto-widen so the next
    caller spaces further apart without each caller having to know."""
    body = b"Please limit requests to one every 5 seconds..."
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=429, body=body, content_type="text/plain",
    ))
    lim = GdeltRateLimiter(min_interval_seconds=6.0)
    assert lim.current_interval_seconds == 6.0
    r = await _fetch_gdelt("test", limiter=lim, retry_on_timeout=False)
    assert r.failure_mode == "rate_limit"
    assert lim.current_interval_seconds == _PENALTY_WIDEN_TO_SECONDS


@pytest.mark.asyncio
async def test_fetch_non_429_failure_does_not_penalize_limiter(monkeypatch):
    """Server errors / network errors must NOT widen the limiter —
    those failures aren't caused by request rate."""
    _patch_client(monkeypatch, response=_MockResponse(
        status_code=503, body=b"down",
    ))
    lim = GdeltRateLimiter(min_interval_seconds=6.0)
    r = await _fetch_gdelt("test", limiter=lim, retry_on_timeout=False)
    assert r.failure_mode == "server_error"
    assert lim.current_interval_seconds == 6.0


# ============================================================
# Retry-once on timeout
# ============================================================


@pytest.mark.asyncio
async def test_retry_on_timeout_succeeds_on_second_attempt(monkeypatch):
    """First attempt times out, second returns 200 + valid JSON.
    Caller gets the success, not the timeout."""
    body = json.dumps({"articles": [
        {"url": "https://x.com/1", "title": "t", "domain": "x.com"},
    ]}).encode()
    ok_response = _MockResponse(status_code=200, body=body)

    call_count = [0]

    class _Flaky:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, *a, **k):
            call_count[0] += 1
            if call_count[0] == 1:
                raise httpx.ConnectTimeout("first attempt timed out")
            return ok_response

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Flaky())
    # Patch asyncio.sleep so the 5s retry delay doesn't slow tests.
    async def _no_sleep(_):
        return
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=True)
    assert r.ok is True
    assert call_count[0] == 2
    assert len(r.articles) == 1


@pytest.mark.asyncio
async def test_retry_on_timeout_records_failure_after_two_attempts(monkeypatch):
    """Both attempts time out → caller sees a clean 'timeout (2 attempts)'
    failure, not an unraised exception."""
    _patch_client(monkeypatch, raise_exc=httpx.ConnectTimeout("persistent"))
    async def _no_sleep(_):
        return
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=True)
    assert r.ok is False
    assert r.failure_mode == "timeout"
    assert "2 attempts" in r.exception_msg


@pytest.mark.asyncio
async def test_retry_on_timeout_disabled_means_one_attempt(monkeypatch):
    """With retry_on_timeout=False, a timeout records failure
    immediately — no second attempt."""
    call_count = [0]

    class _Counting:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def get(self, *a, **k):
            call_count[0] += 1
            raise httpx.ConnectTimeout("once")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Counting())

    r = await _fetch_gdelt("test", limiter=None, retry_on_timeout=False)
    assert r.ok is False
    assert r.failure_mode == "timeout"
    assert call_count[0] == 1
    assert "no retry" in r.exception_msg


# ============================================================
# Categories whitelist resolution
# ============================================================


def test_resolve_enabled_categories_default_subset():
    from core.feeds.gdelt import GdeltFeed, _DEFAULT_ENABLED_CATEGORIES
    f = GdeltFeed()
    assert f._resolve_enabled_categories(None) == list(_DEFAULT_ENABLED_CATEGORIES)


def test_resolve_enabled_categories_all():
    from core.feeds.gdelt import _CATEGORY_QUERIES, GdeltFeed
    f = GdeltFeed()
    out = f._resolve_enabled_categories("all")
    assert set(out) == set(_CATEGORY_QUERIES.keys())


def test_resolve_enabled_categories_explicit_list():
    from core.feeds.gdelt import GdeltFeed
    from core.scout.event import EventCategory
    f = GdeltFeed()
    out = f._resolve_enabled_categories(["shooting", "ceasefire"])
    assert out == [EventCategory.SHOOTING, EventCategory.CEASEFIRE]


def test_resolve_enabled_categories_drops_unknown_names():
    from core.feeds.gdelt import GdeltFeed
    from core.scout.event import EventCategory
    f = GdeltFeed()
    out = f._resolve_enabled_categories(["shooting", "fake_category"])
    assert out == [EventCategory.SHOOTING]


def test_resolve_enabled_categories_falls_back_when_all_invalid():
    from core.feeds.gdelt import GdeltFeed, _DEFAULT_ENABLED_CATEGORIES
    f = GdeltFeed()
    out = f._resolve_enabled_categories(["fake_a", "fake_b"])
    assert out == list(_DEFAULT_ENABLED_CATEGORIES)


def test_resolve_enabled_categories_dedupes():
    from core.feeds.gdelt import GdeltFeed
    from core.scout.event import EventCategory
    f = GdeltFeed()
    out = f._resolve_enabled_categories(["shooting", "shooting", "ceasefire"])
    assert out == [EventCategory.SHOOTING, EventCategory.CEASEFIRE]
