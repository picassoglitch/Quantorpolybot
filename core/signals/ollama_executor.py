"""Hard thread-pool isolation for Ollama HTTP calls.

The April 2026 soak made it clear that even a single in-flight deep
call was affecting event-loop responsiveness — per-tier asyncio
semaphores were not enough. The async httpx path has subtle sync hot
spots (DNS resolution on Windows, JSON serialization, response parsing)
plus a shared connection pool whose state can entangle tiers via the
client's lock. Cancellation of an awaited POST also has to propagate
through httpx's pool, which on Windows can take several seconds while
the kernel closes a half-open socket.

This module sidesteps all of that by running every Ollama HTTP call
on a per-tier ``concurrent.futures.ThreadPoolExecutor`` with a sync
``httpx.Client``. The asyncio loop then awaits a single ``Future``
(via ``loop.run_in_executor``) and uses ``asyncio.wait_for`` to enforce
a hard wall-clock budget. Even if the worker thread is stuck in
``socket.recv``, the loop returns immediately on timeout — the thread
keeps running until the underlying socket times out and is GC'd by
the executor.

Architecture invariants:

  - One ``ThreadPoolExecutor`` per tier (fast, deep, validator). The
    fast tier's threads NEVER block on a deep call's thread. This is
    the strongest possible tier isolation — at the OS thread level.
  - One sync ``httpx.Client`` per tier (kept inside the worker via a
    ``threading.local``) so connection pools don't cross tiers either.
  - The asyncio loop holds nothing but the ``Future`` from
    ``run_in_executor``. No httpx state, no sockets, no DNS calls.
  - On hard timeout the future is cancelled; the worker thread is
    abandoned (Python can't pre-empt sync code from another thread).
    The executor will reclaim the thread when the underlying socket
    eventually times out (sync httpx Client has its own
    ``connect``/``read`` timeouts, so the worst case is bounded).
  - The module exposes ``submit_generate`` (the one entry point used
    by ``OllamaClient._generate``), ``shutdown`` (called from run.py's
    graceful-stop path), and ``in_flight_snapshot`` (read by the
    watchdog to detect stuck workers).

The sync httpx client uses a generous READ timeout because the per-call
wait_for budget is the operative deadline — the sync timeout is a
safety net so the worker thread doesn't leak indefinitely if the
asyncio side has already abandoned the future.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

from core.utils.config import get_config

_TIERS: tuple[str, ...] = ("fast", "deep", "validator")

_TIER_DEFAULT_WORKERS: dict[str, int] = {
    "fast": 2,
    "deep": 1,
    "validator": 1,
}

# Module-level state. ``_lock`` guards lazy executor / client creation
# so the first concurrent call from two coroutines doesn't race to
# build two pools.
_lock = threading.Lock()
_executors: dict[str, ThreadPoolExecutor] = {}
_thread_local = threading.local()
# In-flight tracking: per-tier dict[request_id, started_monotonic].
# Read by the watchdog; written under ``_inflight_lock`` so iteration
# is consistent. Latency on this path is sub-microsecond — even with a
# lock, we're orders of magnitude below the network call itself.
_inflight_lock = threading.Lock()
_inflight: dict[str, dict[str, float]] = {t: {} for t in _TIERS}
# Abandoned tracking: when the asyncio side gives up via wait_for, the
# request is moved here so the worker thread (which Python cannot
# pre-empt) doesn't keep counting against in-flight metrics. The
# watchdog reads this separately so "5 abandoned workers still
# draining" doesn't get misreported as "5 calls in flight, system
# busy". Worker auto-removes its own entry when the sync side
# eventually finishes.
_abandoned: dict[str, dict[str, float]] = {t: {} for t in _TIERS}


def _ollama_cfg() -> dict[str, Any]:
    return get_config().get("ollama") or {}


def _workers_for(tier: str) -> int:
    cfg = _ollama_cfg()
    raw = cfg.get(f"{tier}_max_concurrent")
    if raw is None and tier == "fast":
        # Backwards-compat: respect the legacy max_concurrent_calls key
        # for the fast tier (most callers are fast-tier today).
        raw = cfg.get("max_concurrent_calls")
    n = int(raw) if raw is not None else _TIER_DEFAULT_WORKERS.get(tier, 1)
    return n if n >= 1 else 1


def _get_executor(tier: str) -> ThreadPoolExecutor:
    ex = _executors.get(tier)
    if ex is not None:
        return ex
    with _lock:
        ex = _executors.get(tier)
        if ex is None:
            workers = _workers_for(tier)
            ex = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=f"ollama-{tier}",
            )
            _executors[tier] = ex
            logger.debug(
                "[ollama-executor] tier={} created executor max_workers={}",
                tier, workers,
            )
    return ex


def _get_thread_client() -> httpx.Client:
    """Return the calling worker's sync ``httpx.Client``. One client per
    worker thread keeps the connection pool thread-local — no shared
    pool means no cross-thread lock contention inside httpx."""
    client = getattr(_thread_local, "client", None)
    if client is not None and not client.is_closed:
        return client
    # Generous READ timeout: the wait_for on the asyncio side is the
    # operative deadline, this is just a safety net so a stuck worker
    # eventually unblocks itself even if asyncio has abandoned it.
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=5.0)
    client = httpx.Client(timeout=timeout)
    _thread_local.client = client
    return client


@dataclass
class GenerateResult:
    """Returned by ``submit_generate``'s blocking call. ``data`` is the
    parsed Ollama response on success, ``None`` on any failure (HTTP
    error, network error, unparseable JSON). ``error`` carries the
    exception class name for telemetry; ``latency_ms`` covers the full
    round trip including queue wait inside the executor."""

    data: dict[str, Any] | None
    error: str
    latency_ms: float
    request_id: str
    tier: str
    queue_depth_at_dispatch: int = 0


def _do_post(
    tier: str,
    request_id: str,
    url: str,
    body: dict[str, Any],
    sync_timeout: float,
) -> GenerateResult:
    """Worker-thread entry point. Performs one synchronous Ollama POST
    and returns the parsed body or an error marker. NEVER raises — the
    asyncio side relies on this returning a structured result so it
    can convert errors into a clean ``None`` on the loop side."""
    started = time.perf_counter()
    queue_depth = _inflight_count(tier)
    with _inflight_lock:
        _inflight[tier][request_id] = started
    try:
        client = _get_thread_client()
        # Use an explicit per-call timeout so a misconfigured
        # thread-local default can't extend a stuck call indefinitely.
        r = client.post(url, json=body, timeout=sync_timeout)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            # Ollama returned non-JSON (e.g. plain "Internal error").
            return GenerateResult(
                data=None,
                error="JSONDecodeError",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                request_id=request_id,
                tier=tier,
                queue_depth_at_dispatch=queue_depth,
            )
        return GenerateResult(
            data=data,
            error="",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_id=request_id,
            tier=tier,
            queue_depth_at_dispatch=queue_depth,
        )
    except httpx.HTTPStatusError as e:
        return GenerateResult(
            data=None,
            error=f"HTTPStatusError:{e.response.status_code}",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_id=request_id,
            tier=tier,
            queue_depth_at_dispatch=queue_depth,
        )
    except Exception as e:
        # Catch-all: connection refused, read timeout (sync), DNS
        # failure. These all map to "Ollama call failed" from the
        # caller's perspective.
        return GenerateResult(
            data=None,
            error=type(e).__name__,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_id=request_id,
            tier=tier,
            queue_depth_at_dispatch=queue_depth,
        )
    finally:
        # Always clear from BOTH trackers in the worker's finally so
        # late-arriving completions from previously-abandoned calls
        # don't leave stale entries.
        with _inflight_lock:
            _inflight[tier].pop(request_id, None)
            _abandoned[tier].pop(request_id, None)


def abandon(tier: str, request_id: str) -> None:
    """Called from the asyncio side when ``wait_for`` fires before the
    worker thread completes. Moves the request from in-flight to
    abandoned so metrics reflect "the loop has given up; the thread
    is still draining" rather than "this call is still active".

    The worker thread itself doesn't know it was abandoned and keeps
    running until its sync_timeout. When it finishes (success OR
    failure), the ``finally`` in ``_do_post`` removes the entry from
    both maps so no stale state lingers."""
    with _inflight_lock:
        started = _inflight[tier].pop(request_id, None)
        if started is not None:
            _abandoned[tier][request_id] = started


def abandoned_count(tier: str | None = None) -> int:
    """Watchdog helper: total abandoned workers (across all tiers
    unless one is named) currently draining in the background.
    A growing count means Ollama is timing out a lot AND the sync
    timeout hasn't fired yet — useful for "the GPU is hung" triage."""
    with _inflight_lock:
        if tier is None:
            return sum(len(v) for v in _abandoned.values())
        return len(_abandoned.get(tier, {}))


def submit_generate(
    *,
    tier: str,
    url: str,
    body: dict[str, Any],
    sync_timeout: float,
    request_id: str | None = None,
):
    """Submit one Ollama call to the tier's executor and return the
    ``concurrent.futures.Future``. The asyncio side wraps this in
    ``asyncio.wrap_future`` + ``asyncio.wait_for`` to bound the
    wall-clock wait without touching any HTTP state on the loop.
    """
    rid = request_id or uuid.uuid4().hex[:8]
    ex = _get_executor(tier)
    return ex.submit(_do_post, tier, rid, url, body, sync_timeout), rid


def _inflight_count(tier: str) -> int:
    with _inflight_lock:
        return len(_inflight.get(tier, {}))


@dataclass
class InFlightSnapshot:
    """Watchdog-friendly view of the executor state.

    - ``per_tier_in_flight``: calls the asyncio side is still awaiting.
    - ``per_tier_abandoned``: calls the asyncio side has given up on
      (``wait_for`` fired) but whose worker thread is still draining
      until its sync timeout. These DON'T count as active work — they're
      a "dead but not yet buried" cohort.
    - ``stuck``: in-flight calls older than ``stuck_age_seconds`` —
      candidates for "the loop says X is running but it's really hung
      on the GPU" triage.
    """

    per_tier_in_flight: dict[str, int] = field(default_factory=dict)
    per_tier_abandoned: dict[str, int] = field(default_factory=dict)
    stuck: list[tuple[str, str, float]] = field(default_factory=list)


def in_flight_snapshot(stuck_age_seconds: float = 30.0) -> InFlightSnapshot:
    now = time.perf_counter()
    snap = InFlightSnapshot()
    with _inflight_lock:
        for tier, rid_to_started in _inflight.items():
            snap.per_tier_in_flight[tier] = len(rid_to_started)
            for rid, started in rid_to_started.items():
                age = now - started
                if age >= stuck_age_seconds:
                    snap.stuck.append((tier, rid, age))
        for tier, rid_to_started in _abandoned.items():
            snap.per_tier_abandoned[tier] = len(rid_to_started)
    return snap


def reset_for_tests() -> None:
    """Test helper: drop all executors so the next call rebuilds them
    against the freshly-monkeypatched config. Tests SHOULD call this
    in an autouse fixture if they touch concurrency settings."""
    global _executors
    with _lock:
        for ex in _executors.values():
            ex.shutdown(wait=False, cancel_futures=True)
        _executors = {}
    with _inflight_lock:
        for tier in _TIERS:
            _inflight[tier].clear()
            _abandoned[tier].clear()


def shutdown(wait: bool = True) -> None:
    """Cleanly stop every tier's executor and close every per-thread
    httpx.Client. Called from run.py's shutdown path so a Ctrl-C
    leaves no zombie threads behind."""
    with _lock:
        executors = dict(_executors)
        _executors.clear()
    for tier, ex in executors.items():
        try:
            ex.shutdown(wait=wait, cancel_futures=True)
        except Exception as e:
            logger.warning("[ollama-executor] shutdown {} failed: {}", tier, e)
    # Per-thread clients close themselves when their thread dies; nothing
    # else to do here.
