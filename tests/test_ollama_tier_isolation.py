"""Per-tier isolation regression tests (executor edition).

Step #1 of the April 2026 unblocking work tried per-tier asyncio
semaphores. The 10-minute soak failed: even a single in-flight deep
call still affected event-loop responsiveness, because async httpx
has subtle sync hot spots (DNS, JSON parsing, pool locks) and
cancellation propagation through the pool can stall on Windows.

Step #1.5 moved every Ollama call onto a per-tier
``ThreadPoolExecutor`` with a sync ``httpx.Client``. The asyncio loop
now only awaits a ``concurrent.futures.Future`` — no sockets, no DNS,
no pool state on the loop thread.

These tests verify the new architecture:

  1. Loop responsiveness: while a worker thread is blocked, the
     asyncio loop can still execute hundreds of small tasks promptly.
  2. Tier isolation: fast and deep workers run in separate executors
     so a deep stall doesn't even occupy a thread the fast tier needs.
  3. Hard timeout: ``asyncio.wait_for`` returns within the configured
     budget regardless of whether the worker thread is still running.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.signals import ollama_client as ollama_mod
from core.signals import ollama_executor
from core.signals.ollama_client import OllamaClient


@pytest.fixture(autouse=True)
def _reset_module_state():
    # Per-tier executors and circuit state are module-level — drop
    # them between tests so prior state doesn't bleed.
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0
    yield
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0


def _patch_do_post(monkeypatch, handler):
    """Replace the worker-thread HTTP call with ``handler``. The
    handler runs on a real worker thread (not the asyncio loop), which
    is exactly the property we want to verify."""
    def fake_do_post(tier, request_id, url, body, sync_timeout):
        # Re-use the executor's tracking + result type.
        from core.signals.ollama_executor import (
            GenerateResult, _inflight, _inflight_lock,
        )
        started = time.perf_counter()
        with _inflight_lock:
            _inflight[tier][request_id] = started
        try:
            return handler(tier, request_id, url, body, sync_timeout)
        finally:
            with _inflight_lock:
                _inflight[tier].pop(request_id, None)

    monkeypatch.setattr(ollama_executor, "_do_post", fake_do_post)


# --- 1. The headline regression: loop stays responsive while a deep
#       worker is blocked ------------------------------------------------


@pytest.mark.asyncio
async def test_loop_remains_responsive_during_blocking_deep_call(monkeypatch):
    """While a deep worker thread is sleeping on a synchronous call,
    the asyncio loop must continue to execute small tasks promptly.
    This is the headline regression — async httpx blocked the loop
    via internal sync hot spots even with per-tier semaphores."""
    import threading

    deep_started = threading.Event()
    release_deep = threading.Event()

    def handler(tier, request_id, url, body, sync_timeout):
        from core.signals.ollama_executor import GenerateResult
        if tier == "deep":
            deep_started.set()
            # Block the worker thread for up to 5s. The loop should
            # keep running fine because this is on a thread, not the
            # loop itself.
            for _ in range(50):
                if release_deep.is_set():
                    break
                time.sleep(0.1)
        return GenerateResult(
            data={"response": '{"implied_prob":0.5,"confidence":0.5}'},
            error="", latency_ms=0.0, request_id=request_id, tier=tier,
        )

    _patch_do_post(monkeypatch, handler)

    # Tighter wait_for so the test cleans up fast.
    monkeypatch.setattr(
        OllamaClient, "_timeout_for",
        lambda self, t: 8.0 if t == "deep" else 1.0,
    )

    client = OllamaClient()
    deep_task = asyncio.create_task(client.deep_score("hold-the-thread"))
    # Wait for the worker thread to enter handler() — poll the
    # threading.Event from the loop without blocking.
    for _ in range(40):
        if deep_started.is_set():
            break
        await asyncio.sleep(0.05)
    assert deep_started.is_set(), "deep worker did not start"

    # Now fire 200 tiny coroutines and time how long they take to
    # complete. Old async-httpx pattern would stall these whenever the
    # loop was processing the deep call's pending I/O.
    t0 = time.perf_counter()
    await asyncio.gather(*(asyncio.sleep(0) for _ in range(200)))
    elapsed = time.perf_counter() - t0
    # Generous bound — even a slow CI box shouldn't take more than 100ms
    # to drain 200 no-op tasks. With the old async-httpx pattern under
    # load this could blow past 1s.
    assert elapsed < 0.5, (
        f"loop processed 200 no-op tasks in {elapsed*1000:.0f}ms "
        "while a deep worker was blocked — expected <500ms"
    )

    # And firing a fast call should also complete promptly because the
    # fast tier has its own executor + httpx.Client.
    t0 = time.perf_counter()
    fast_result = await client.fast_score("hi")
    fast_elapsed = time.perf_counter() - t0
    assert fast_result is not None
    assert fast_elapsed < 1.0, (
        f"fast call took {fast_elapsed:.2f}s while deep worker was blocked"
    )

    # Cleanup.
    release_deep.set()
    await asyncio.wait_for(deep_task, timeout=10.0)


# --- 2. Hard timeout fires cleanly even when the worker is stuck -------


@pytest.mark.asyncio
async def test_hard_timeout_returns_none_quickly(monkeypatch):
    """If the worker thread doesn't return within the wait_for budget,
    the asyncio side must give up promptly and yield None — even if
    the worker is still stuck."""
    def handler(tier, request_id, url, body, sync_timeout):
        # Sleep WAY past the wait_for budget. The asyncio side should
        # not wait for us.
        time.sleep(2.0)
        from core.signals.ollama_executor import GenerateResult
        return GenerateResult(
            data={"response": "{}"}, error="", latency_ms=0.0,
            request_id=request_id, tier=tier,
        )

    _patch_do_post(monkeypatch, handler)
    monkeypatch.setattr(
        OllamaClient, "_timeout_for", lambda self, t: 0.2,
    )

    client = OllamaClient()
    t0 = time.perf_counter()
    result = await client.fast_score("hello")
    elapsed = time.perf_counter() - t0
    assert result is None
    # Generous slack: must be well under the 2s worker-thread sleep.
    assert elapsed < 1.0, (
        f"wait_for took {elapsed:.2f}s — should fire near the 0.2s budget"
    )


# --- 3. fast_queue_saturated reflects either source (executor or class) -


def test_fast_queue_saturated_threshold_with_class_attr(monkeypatch):
    from core.utils.config import get_config

    cfg = get_config()
    cfg._data.setdefault("ollama", {})["queue_depth_alert"] = 4

    OllamaClient.pending_fast = 3
    assert OllamaClient.fast_queue_saturated() is False
    OllamaClient.pending_fast = 4
    assert OllamaClient.fast_queue_saturated() is True


def test_fast_queue_saturated_reads_executor_in_flight(monkeypatch):
    """In production the executor's in-flight tracker is what matters.
    Even with the class counter at 0, a real running call must show
    up in the saturation calculation."""
    from core.utils.config import get_config

    cfg = get_config()
    cfg._data.setdefault("ollama", {})["queue_depth_alert"] = 2

    # Simulate two in-flight fast calls without using the real handler.
    from core.signals.ollama_executor import _inflight, _inflight_lock
    with _inflight_lock:
        _inflight["fast"]["req-a"] = time.perf_counter()
        _inflight["fast"]["req-b"] = time.perf_counter()

    OllamaClient.pending_fast = 0  # class counter empty
    assert OllamaClient.fast_queue_saturated() is True

    with _inflight_lock:
        _inflight["fast"].clear()


# --- 4. Per-tier ThreadPoolExecutor honors config ----------------------


def test_per_tier_executor_uses_config(monkeypatch):
    from core.utils.config import get_config

    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    ollama["fast_max_concurrent"] = 3
    ollama["deep_max_concurrent"] = 1
    ollama["validator_max_concurrent"] = 2
    cfg._data["ollama"] = ollama
    ollama_executor.reset_for_tests()

    fast_ex = ollama_executor._get_executor("fast")
    deep_ex = ollama_executor._get_executor("deep")
    val_ex = ollama_executor._get_executor("validator")
    assert fast_ex._max_workers == 3
    assert deep_ex._max_workers == 1
    assert val_ex._max_workers == 2


def test_legacy_max_concurrent_calls_used_for_fast_default():
    from core.utils.config import get_config

    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    ollama.pop("fast_max_concurrent", None)
    ollama["max_concurrent_calls"] = 5
    cfg._data["ollama"] = ollama
    ollama_executor.reset_for_tests()

    fast_ex = ollama_executor._get_executor("fast")
    assert fast_ex._max_workers == 5


# --- 5. Watchdog stuck-task detection ----------------------------------


def test_in_flight_snapshot_marks_old_calls_as_stuck():
    from core.signals.ollama_executor import _inflight, _inflight_lock
    now = time.perf_counter()
    with _inflight_lock:
        _inflight["deep"]["old-1"] = now - 60.0  # 60s old
        _inflight["fast"]["new-1"] = now - 1.0   # 1s old

    snap = ollama_executor.in_flight_snapshot(stuck_age_seconds=30.0)
    stuck_ids = [rid for _, rid, _ in snap.stuck]
    assert "old-1" in stuck_ids
    assert "new-1" not in stuck_ids
    assert snap.per_tier_in_flight.get("deep", 0) >= 1

    with _inflight_lock:
        _inflight["deep"].clear()
        _inflight["fast"].clear()
