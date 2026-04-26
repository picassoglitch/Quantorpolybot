"""Abandoned-worker tracking tests.

When the asyncio side of an Ollama call gives up via ``wait_for``, we
move the request from in-flight to abandoned so metrics reflect "the
loop has given up; the thread is still draining". The April 2026
log showed deep:1 in-flight for ~30s after the asyncio timeout fired,
because ``_inflight`` was only cleared on worker completion. This
made the watchdog falsely believe Ollama was still processing.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.signals import ollama_client as ollama_mod
from core.signals import ollama_executor
from core.signals.ollama_client import OllamaClient


@pytest.fixture(autouse=True)
def _reset_state():
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0
    yield
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()


@pytest.mark.asyncio
async def test_abandon_clears_in_flight_immediately(monkeypatch):
    """A timed-out call must vanish from in_flight() at once and
    appear in abandoned() instead."""
    started_event = __import__("threading").Event()
    release_event = __import__("threading").Event()

    def slow_handler(tier, request_id, url, body, sync_timeout):
        from core.signals.ollama_executor import (
            GenerateResult, _inflight, _inflight_lock,
        )
        with _inflight_lock:
            _inflight[tier][request_id] = time.perf_counter()
        started_event.set()
        try:
            # Hold the worker thread well past the asyncio timeout so
            # the abandon path has time to fire.
            for _ in range(50):
                if release_event.is_set():
                    break
                time.sleep(0.05)
            return GenerateResult(
                data={"response": "{}"}, error="", latency_ms=0.0,
                request_id=request_id, tier=tier,
            )
        finally:
            from core.signals.ollama_executor import _abandoned
            with _inflight_lock:
                _inflight[tier].pop(request_id, None)
                _abandoned[tier].pop(request_id, None)

    monkeypatch.setattr(ollama_executor, "_do_post", slow_handler)
    monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 0.1)

    client = OllamaClient()
    result = await client.fast_score("hello")
    assert result is None  # timed out

    # Wait for the worker to actually start so the timing is real.
    assert started_event.is_set() or started_event.wait(timeout=1.0)

    snap = ollama_executor.in_flight_snapshot()
    # The async timeout fired, so in-flight for fast must be 0.
    assert snap.per_tier_in_flight.get("fast", 0) == 0, (
        "abandoned worker should not count as in-flight"
    )
    # And abandoned should reflect the still-running thread.
    assert snap.per_tier_abandoned.get("fast", 0) == 1, (
        "timed-out worker should be tracked as abandoned"
    )

    # Clean up: let the worker finish so its finally block clears state.
    release_event.set()
    # Wait briefly for the thread to drain.
    for _ in range(40):
        snap = ollama_executor.in_flight_snapshot()
        if snap.per_tier_abandoned.get("fast", 0) == 0:
            break
        await asyncio.sleep(0.05)
    snap = ollama_executor.in_flight_snapshot()
    assert snap.per_tier_abandoned.get("fast", 0) == 0


@pytest.mark.asyncio
async def test_abandoned_count_helper(monkeypatch):
    """``abandoned_count`` reads the same map without the snapshot
    overhead — useful for the watchdog's per-tick check."""
    from core.signals.ollama_executor import _abandoned, _inflight_lock

    with _inflight_lock:
        _abandoned["deep"]["x"] = time.perf_counter()
        _abandoned["fast"]["y"] = time.perf_counter()
        _abandoned["fast"]["z"] = time.perf_counter()

    assert ollama_executor.abandoned_count() == 3
    assert ollama_executor.abandoned_count("fast") == 2
    assert ollama_executor.abandoned_count("deep") == 1

    with _inflight_lock:
        _abandoned["deep"].clear()
        _abandoned["fast"].clear()
