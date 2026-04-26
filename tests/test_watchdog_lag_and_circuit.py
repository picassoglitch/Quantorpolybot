"""Watchdog correctness tests (April 2026 soak hardening).

The previous watchdog inferred "loop blocked" from indirect signals
(pending-task count + Ollama silence). Both produced false positives
during normal operation — at boot we legitimately have ~25 background
coroutines, and when the Ollama circuit is intentionally OPEN we have
no in-flight calls AND no successful responses to refresh
``LAST_SUCCESS_TS``, so the watchdog used to escalate to ERROR for
something that's working as designed.

These tests verify:

  1. ``_measure_loop_lag_ms`` returns a small value when the loop is
     responsive and a large value when it's blocked by sync work.
  2. The single tick line carries ``loop_lag_ms``, per-tier circuit
     state, and ``heuristic_only=True`` whenever any tier is OPEN.
  3. With circuit=OPEN, an aged ``LAST_SUCCESS_TS`` does NOT trigger
     ``ollama_unresponsive`` and does NOT enter DEGRADED.
  4. With ALL circuits closed but lag > threshold, the tick logs
     ``event_loop_blocked`` and DEGRADED flips True.
  5. Abandoned workers count separately from in-flight in the
     watchdog snapshot.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from loguru import logger

from core.signals import ollama_client as ollama_mod
from core.signals import ollama_executor
from core.utils import watchdog as wd_mod


@pytest.fixture(autouse=True)
def _reset_watchdog_state():
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()
    wd_mod.DEGRADED = False
    yield
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()
    wd_mod.DEGRADED = False


@pytest.fixture
def loguru_sink():
    """Capture loguru records into a list. ``caplog`` only sees stdlib
    logging; the watchdog uses loguru directly so we install our own
    sink for assertions."""
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(str(message)), level="INFO")
    yield captured
    logger.remove(sink_id)


# ---- 1. Loop-lag probe ------------------------------------------------


@pytest.mark.asyncio
async def test_loop_lag_low_when_loop_idle():
    w = wd_mod.Watchdog()
    lag_ms = await w._measure_loop_lag_ms()
    # On any healthy machine an idle loop should see <50ms lag — we
    # leave huge slack so CI noise doesn't flake this.
    assert lag_ms < 100.0, f"idle loop lag was {lag_ms:.0f}ms — should be ~0"


@pytest.mark.asyncio
async def test_loop_lag_high_when_loop_blocked(monkeypatch):
    """If a coroutine holds the loop synchronously for longer than the
    probe budget, the next probe must report a large lag. We force
    the block by sleeping the loop synchronously (time.sleep, not
    asyncio.sleep) on a separate task that's scheduled BEFORE the
    probe fires."""
    monkeypatch.setattr(wd_mod, "LOOP_LAG_PROBE_DELAY", 0.1)

    w = wd_mod.Watchdog()

    async def block_loop_for(ms: float) -> None:
        # Single blocking call, on the loop thread. This is what we'd
        # diagnose as "event_loop_blocked" in production.
        time.sleep(ms / 1000.0)

    # Fire the probe and the block concurrently. The probe schedules
    # itself for ~100ms in the future; the block holds the loop for
    # 400ms; the probe fires AFTER the block finishes -> lag ~300ms.
    probe_task = asyncio.create_task(w._measure_loop_lag_ms())
    await asyncio.sleep(0)  # let the probe install its timer
    block_task = asyncio.create_task(block_loop_for(400))
    await block_task
    lag_ms = await probe_task
    assert lag_ms >= 200.0, f"blocked loop reported only {lag_ms:.0f}ms lag"


# ---- 2. Tick line includes circuit state + heuristic_only -----------


@pytest.mark.asyncio
async def test_tick_logs_circuit_states(loguru_sink):
    # Open the deep circuit explicitly. The April 2026 step-1.7 state
    # machine requires ``is_closed=False`` to flip to OPEN — cooldown
    # alone no longer implies open. Tests must mirror that contract.
    state = ollama_mod._circuit["deep"]
    state.consecutive_failures = 99
    state.open_until = time.monotonic() + 30.0
    state.is_closed = False

    w = wd_mod.Watchdog()
    await w._tick()

    msgs = "\n".join(loguru_sink)
    assert "ollama_state=" in msgs
    assert "deep:OPEN" in msgs
    assert "heuristic_only=True" in msgs


# ---- 3. Open circuit silences Ollama-silent escalation --------------


@pytest.mark.asyncio
async def test_open_circuit_does_not_trigger_ollama_unresponsive(loguru_sink):
    """The circuit being OPEN means we INTENTIONALLY don't call
    Ollama. ``LAST_SUCCESS_TS`` ages naturally. The watchdog must
    not interpret this as ``ollama_unresponsive`` and must not enter
    DEGRADED."""
    # Pretend we last had a success 999s ago — well past the silent
    # error threshold.
    ollama_mod.LAST_SUCCESS_TS = time.time() - 999.0
    # Open every tier.
    for tier in ("fast", "deep", "validator"):
        state = ollama_mod._circuit[tier]
        state.consecutive_failures = 99
        state.open_until = time.monotonic() + 60.0
        state.is_closed = False

    w = wd_mod.Watchdog()
    await w._tick()

    msgs = "\n".join(loguru_sink)
    assert "ollama_unresponsive" not in msgs, (
        "open circuits must not trigger ollama_unresponsive"
    )
    assert wd_mod.DEGRADED is False


# ---- 4. Real loop lag DOES escalate ---------------------------------


@pytest.mark.asyncio
async def test_real_loop_lag_triggers_degraded(monkeypatch, loguru_sink):
    """Genuine loop block should escalate to ``event_loop_blocked``
    + DEGRADED — but only after CONSECUTIVE_LAG_BREACHES_TO_DEGRADE
    consecutive ticks (default 2) AND outside the startup grace
    window. The April 2026 step-1.7 soak showed that a single 267ms
    blip is NOT a trading-bot degradation event, so the previous
    fire-on-first-breach behaviour was wrong."""
    monkeypatch.setattr(wd_mod, "LOOP_LAG_BLOCKED_MS", 50.0)

    async def fake_probe(self):
        return 500.0

    monkeypatch.setattr(wd_mod.Watchdog, "_measure_loop_lag_ms", fake_probe)

    w = wd_mod.Watchdog()
    # Skip the startup grace window so the breach actually counts.
    w._started_at = time.time() - 10000.0

    # First tick: breach 1/2 — WARN, not yet DEGRADED.
    await w._tick()
    msgs_after_one = "\n".join(loguru_sink)
    assert "event_loop_blocked" not in msgs_after_one
    assert wd_mod.DEGRADED is False
    assert "single-tick breach" in msgs_after_one or "1/2" in msgs_after_one

    # Second consecutive breach: ERROR + DEGRADED.
    await w._tick()
    msgs = "\n".join(loguru_sink)
    assert "event_loop_blocked" in msgs
    assert wd_mod.DEGRADED is True


@pytest.mark.asyncio
async def test_startup_grace_suppresses_lag_breach(monkeypatch, loguru_sink):
    """Within ``STARTUP_GRACE_SECONDS`` of watchdog start, a lag
    breach must NOT count toward DEGRADED — the very first tick
    after boot legitimately measures 1-2s of lag while every
    subsystem ramps up."""
    monkeypatch.setattr(wd_mod, "LOOP_LAG_BLOCKED_MS", 50.0)
    monkeypatch.setattr(wd_mod, "STARTUP_GRACE_SECONDS", 60.0)

    async def fake_probe(self):
        return 2000.0  # 2s spike, like the real boot lag

    monkeypatch.setattr(wd_mod.Watchdog, "_measure_loop_lag_ms", fake_probe)

    w = wd_mod.Watchdog()
    # _started_at is now() — well within the grace window.
    await w._tick()
    await w._tick()  # even two ticks shouldn't degrade during grace
    msgs = "\n".join(loguru_sink)
    assert "event_loop_blocked" not in msgs
    assert "during startup grace" in msgs
    assert wd_mod.DEGRADED is False
    # Counter must not accumulate during grace, otherwise the first
    # post-grace breach would immediately trigger.
    assert w._consecutive_lag_breaches == 0


# ---- 5. Abandoned workers don't count as in-flight ------------------


@pytest.mark.asyncio
async def test_abandoned_workers_reported_separately(loguru_sink):
    """When an Ollama call times out asyncio-side, ``abandon`` moves
    the worker from in-flight to abandoned. The watchdog tick must
    distinguish the two — abandoned workers shouldn't make the
    watchdog think Ollama is busy."""
    from core.signals.ollama_executor import _inflight, _abandoned, _inflight_lock

    with _inflight_lock:
        _abandoned["deep"]["abandoned-1"] = time.perf_counter()

    w = wd_mod.Watchdog()
    await w._tick()

    msgs = "\n".join(loguru_sink)
    # In-flight summary should show 0 deep, abandoned should show 1 deep.
    assert "in_flight=fast:0/deep:0/validator:0" in msgs
    assert "deep:1" in msgs  # abandoned summary
    # And abandoned alone (no real lag) shouldn't trigger DEGRADED.
    assert wd_mod.DEGRADED is False

    with _inflight_lock:
        _abandoned["deep"].clear()
