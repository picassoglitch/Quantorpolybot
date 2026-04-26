"""Process-wide circuit breaker tests.

The breaker opens after N consecutive failures on a tier, denies all
calls during the cooldown window, then admits ONE half-open probe.
Probe success closes the circuit; probe failure re-opens it for a
longer window.

These tests stub the executor's worker function so we can deterministically
control success/failure outcomes without a real Ollama.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.signals import ollama_client as ollama_mod
from core.signals import ollama_executor
from core.signals.ollama_client import OllamaClient


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0
    # Lower the breaker thresholds + cooldown so tests run fast.
    from core.utils.config import get_config

    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    ollama["circuit_open_threshold"] = 3
    ollama["circuit_open_seconds"] = 0.2
    cfg._data["ollama"] = ollama
    yield
    ollama_executor.reset_for_tests()
    ollama_mod.reset_circuits_for_tests()


def _patch_outcomes(monkeypatch, outcomes):
    """``outcomes`` is a callable (tier, body) -> GenerateResult or
    raises. Lets each test script the sequence of failures/successes."""
    def fake_do_post(tier, request_id, url, body, sync_timeout):
        from core.signals.ollama_executor import (
            GenerateResult, _inflight, _inflight_lock,
        )
        with _inflight_lock:
            _inflight[tier][request_id] = time.perf_counter()
        try:
            return outcomes(tier, body)
        finally:
            with _inflight_lock:
                _inflight[tier].pop(request_id, None)

    monkeypatch.setattr(ollama_executor, "_do_post", fake_do_post)


def _ok_result(request_id="r"):
    from core.signals.ollama_executor import GenerateResult
    return GenerateResult(
        data={"response": '{"implied_prob":0.5,"confidence":0.5}'},
        error="", latency_ms=10.0, request_id=request_id, tier="x",
    )


def _err_result(error="ConnectError", request_id="r", tier="x"):
    from core.signals.ollama_executor import GenerateResult
    return GenerateResult(
        data=None, error=error, latency_ms=10.0,
        request_id=request_id, tier=tier,
    )


# --- Open after N failures ---------------------------------------------


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_failures(monkeypatch):
    _patch_outcomes(monkeypatch, lambda tier, body: _err_result(tier=tier))
    monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 1.0)

    client = OllamaClient()
    # Three failures -> open.
    for _ in range(3):
        result = await client.fast_score("hello")
        assert result is None
    assert ollama_mod.circuit_is_open("fast") is True


@pytest.mark.asyncio
async def test_circuit_open_denies_calls_without_dispatch(monkeypatch):
    """Once the circuit is open, calls must return None instantly with
    no executor dispatch — no thread spin-up, no log spam."""
    dispatched: list[int] = []

    def outcomes(tier, body):
        dispatched.append(1)
        return _err_result(tier=tier)

    _patch_outcomes(monkeypatch, outcomes)
    monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 1.0)

    client = OllamaClient()
    # Drive the circuit open (3 failures + dispatch).
    for _ in range(3):
        await client.fast_score("x")
    assert len(dispatched) == 3
    assert ollama_mod.circuit_is_open("fast") is True

    # Now five more calls should NOT dispatch.
    for _ in range(5):
        result = await client.fast_score("x")
        assert result is None
    assert len(dispatched) == 3, "open circuit must not dispatch new calls"


# --- Half-open probe + recovery ----------------------------------------


@pytest.mark.asyncio
async def test_circuit_half_open_probe_success_closes_circuit(monkeypatch):
    call_count = {"n": 0}

    def outcomes(tier, body):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            return _err_result(tier=tier)
        return _ok_result()

    _patch_outcomes(monkeypatch, outcomes)
    monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 1.0)

    client = OllamaClient()
    for _ in range(3):
        await client.fast_score("x")
    assert ollama_mod.circuit_is_open("fast") is True

    # Wait for the cooldown window (set to 0.2s in fixture) to elapse.
    await asyncio.sleep(0.3)
    # Next call is the half-open probe — should dispatch and succeed.
    result = await client.fast_score("recover")
    assert result is not None
    # Circuit is now closed.
    assert ollama_mod.circuit_is_open("fast") is False
    state = ollama_mod._circuit["fast"]
    assert state.consecutive_failures == 0
    assert state.open_until == 0.0


@pytest.mark.asyncio
async def test_circuit_half_open_probe_failure_reopens(monkeypatch):
    """Probe failure must re-open the circuit (with a longer window) so
    a flaky Ollama doesn't oscillate."""
    def outcomes(tier, body):
        return _err_result(tier=tier)  # always fails

    _patch_outcomes(monkeypatch, outcomes)
    monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 1.0)

    client = OllamaClient()
    for _ in range(3):
        await client.fast_score("x")
    state_before = ollama_mod._circuit["fast"]
    assert state_before.open_count == 1

    await asyncio.sleep(0.3)
    # Probe attempt — fails.
    result = await client.fast_score("probe")
    assert result is None
    state_after = ollama_mod._circuit["fast"]
    assert state_after.open_count == 2
    assert ollama_mod.circuit_is_open("fast") is True


# --- Tier independence -------------------------------------------------


@pytest.mark.asyncio
async def test_open_circuit_on_one_tier_does_not_affect_other(monkeypatch):
    """A deep-tier circuit-open event must not block fast-tier calls.
    This is the per-tier isolation requirement at the breaker level."""
    def outcomes(tier, body):
        if tier == "deep":
            return _err_result(tier="deep")
        return _ok_result()

    _patch_outcomes(monkeypatch, outcomes)
    monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 1.0)

    client = OllamaClient()
    # Trip deep open.
    for _ in range(3):
        await client.deep_score("trip")
    assert ollama_mod.circuit_is_open("deep") is True
    assert ollama_mod.circuit_is_open("fast") is False

    # Fast tier still works.
    result = await client.fast_score("still alive")
    assert result is not None


# --- State-machine invariants (April 2026 step-1.7 fix) ---------------


@pytest.mark.asyncio
async def test_circuit_stays_open_after_cooldown_until_probe(monkeypatch):
    """Cooldown elapsing on its own does NOT close the circuit. Step
    1.7 invariant: only a successful half-open probe transitions back
    to CLOSED. Until then ``circuit_state`` reports OPEN.

    The April 2026 soak hit this: at 00:07:12 the watchdog logged
    ``deep:CLOSED`` while Ollama was still timing out on every call,
    because the previous implementation used ``time.monotonic() <
    open_until`` as the OPEN test."""
    def outcomes(tier, body):
        return _err_result(tier=tier)

    _patch_outcomes(monkeypatch, outcomes)
    monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 1.0)

    client = OllamaClient()
    for _ in range(3):
        await client.fast_score("trip")
    assert ollama_mod.circuit_state("fast") == "OPEN"

    # Wait for cooldown (0.2s in fixture).
    await asyncio.sleep(0.3)

    # Cooldown has elapsed — ``circuit_cooldown_remaining`` reads 0.
    assert ollama_mod.circuit_cooldown_remaining("fast") == 0.0
    # But state is STILL OPEN — no probe has run yet.
    assert ollama_mod.circuit_state("fast") == "OPEN"


@pytest.mark.asyncio
async def test_circuit_closed_only_after_successful_probe(
    monkeypatch, caplog,
):
    """Only a half-open probe success closes the circuit, and the
    transition is logged with the explicit ``after successful probe``
    marker."""
    from loguru import logger as loguru_logger

    captured: list[str] = []
    sink_id = loguru_logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        call_count = {"n": 0}

        def outcomes(tier, body):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                return _err_result(tier=tier)
            return _ok_result()

        _patch_outcomes(monkeypatch, outcomes)
        monkeypatch.setattr(OllamaClient, "_timeout_for", lambda self, t: 1.0)

        client = OllamaClient()
        for _ in range(3):
            await client.fast_score("trip")
        assert ollama_mod.circuit_state("fast") == "OPEN"

        await asyncio.sleep(0.3)
        # Probe succeeds → CLOSED with the explicit log marker.
        result = await client.fast_score("recover")
        assert result is not None
        assert ollama_mod.circuit_state("fast") == "CLOSED"

        msgs = "\n".join(captured)
        assert "after successful probe" in msgs, (
            "expected explicit 'after successful probe' in CLOSED transition log"
        )
    finally:
        loguru_logger.remove(sink_id)
