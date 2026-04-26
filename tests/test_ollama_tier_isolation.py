"""Per-tier isolation regression tests.

Step #1 of the April 2026 unblocking work split the single global Ollama
semaphore into per-tier semaphores. The headline regression is: a stuck
deep call must not prevent fast-tier scoring from running. Without the
fix, `OllamaClient.fast_score()` would queue behind a 60s deep call on
the shared 2-slot semaphore and burn its own 45s ``wait_for`` budget
before getting a slot — even though the GPU itself wasn't busy with
fast-tier work.

These tests bypass the network entirely by monkey-patching the shared
httpx client's ``post`` to return either a synthetic OK response or a
controllable delay, so they're safe to run anywhere (no real Ollama).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.signals import ollama_client as ollama_mod
from core.signals.ollama_client import OllamaClient


@pytest.fixture(autouse=True)
def _reset_module_state():
    # Per-tier sems and queue counters are module/class level — reset
    # between tests so saturation/cooldown state doesn't bleed.
    ollama_mod._reset_tier_semaphores()
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0
    yield
    ollama_mod._reset_tier_semaphores()
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0


def _ok_response_factory(payload: str = '{"implied_prob":0.5,"confidence":0.5,"reasoning":"x"}'):
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": payload}

    return _Resp()


class _StubClient:
    """Drop-in for the shared httpx client. ``post`` invokes a
    user-supplied async callable so each test can simulate a slow deep
    call, an instant fast call, or whatever it needs."""

    def __init__(self, post_handler):
        self._handler = post_handler
        self.is_closed = False

    async def post(self, url, json=None):  # noqa: A002 - matches httpx API
        return await self._handler(url, json)


# --- 1. The headline regression: stuck deep doesn't starve fast --------


@pytest.mark.asyncio
async def test_stuck_deep_call_does_not_starve_fast(monkeypatch):
    """A deep call hangs for longer than the fast tier's wait_for
    budget. The fast call must still complete promptly because it has
    its own per-tier semaphore — not the previous shared one."""
    deep_started = asyncio.Event()
    release_deep = asyncio.Event()

    async def post_handler(url, body):
        if body.get("model", "") and body["options"].get("num_predict", 300) >= 1:
            # Fast-vs-deep is determined by which model name was sent.
            # Inspect the request to decide what to do.
            pass
        # Distinguish deep from fast by a flag we plant on the body
        # via the prompt prefix below.
        if body.get("prompt", "").startswith("DEEP:"):
            deep_started.set()
            await release_deep.wait()
            return _ok_response_factory()
        # Fast path: respond instantly.
        return _ok_response_factory()

    stub = _StubClient(post_handler)

    async def fake_get_client():
        return stub

    monkeypatch.setattr(ollama_mod, "_get_shared_client", fake_get_client)

    client = OllamaClient()
    # Tighten the fast timeout so the test doesn't have to actually
    # wait the production budget; we just need fast < deep.
    monkeypatch.setattr(
        OllamaClient, "_timeout_for",
        lambda self, t: 0.5 if t == "fast" else 30.0,
    )

    deep_task = asyncio.create_task(client.deep_score("DEEP: stall me"))
    # Make sure the deep call has acquired its slot before we kick off fast.
    await asyncio.wait_for(deep_started.wait(), timeout=2.0)

    # If the fast tier shared a slot with deep, this would either time
    # out (fast budget burned waiting for deep) or queue indefinitely.
    t0 = time.perf_counter()
    fast_result = await client.fast_score("FAST: hello")
    elapsed = time.perf_counter() - t0
    assert fast_result is not None, "fast call must succeed while deep is stuck"
    # Fast call should have completed in well under the deep stall.
    # Allow generous slack for CI scheduling but assert it didn't wait
    # behind the deep slot.
    assert elapsed < 0.5, f"fast call took {elapsed:.2f}s — likely starved by deep slot"

    # Cleanup: release the deep call so the task drains.
    release_deep.set()
    await asyncio.wait_for(deep_task, timeout=5.0)


# --- 2. Per-tier cooldown isolation ------------------------------------


@pytest.mark.asyncio
async def test_deep_cooldown_does_not_block_fast(monkeypatch):
    """Five deep failures trip the deep cooldown. A subsequent fast call
    must still attempt the network — only deep is paused, not the whole
    client."""
    fast_calls: list[int] = []
    deep_calls: list[int] = []

    async def post_handler(url, body):
        prompt = body.get("prompt", "")
        if prompt.startswith("DEEP:"):
            deep_calls.append(1)
            raise RuntimeError("simulated deep failure")
        fast_calls.append(1)
        return _ok_response_factory()

    stub = _StubClient(post_handler)

    async def fake_get_client():
        return stub

    monkeypatch.setattr(ollama_mod, "_get_shared_client", fake_get_client)
    monkeypatch.setattr(
        OllamaClient, "_timeout_for", lambda self, t: 5.0,
    )

    client = OllamaClient()
    # Drive deep into cooldown via 5 consecutive failures.
    for _ in range(5):
        await client.deep_score("DEEP: fail")
    assert client._in_cooldown("deep") is True
    # A 6th deep call short-circuits inside the cooldown without hitting
    # the network — verify by counting deep_calls before/after.
    deep_count_before = len(deep_calls)
    await client.deep_score("DEEP: fail again")
    assert len(deep_calls) == deep_count_before, "deep cooldown must skip the network"

    # Critically: fast tier is unaffected.
    assert client._in_cooldown("fast") is False
    fast_count_before = len(fast_calls)
    result = await client.fast_score("FAST: hello")
    assert result is not None
    assert len(fast_calls) == fast_count_before + 1


# --- 3. fast_queue_saturated reflects the configured threshold ---------


def test_fast_queue_saturated_threshold(monkeypatch):
    from core.utils.config import get_config

    cfg = get_config()
    cfg._data.setdefault("ollama", {})["queue_depth_alert"] = 4

    OllamaClient.pending_fast = 3
    assert OllamaClient.fast_queue_saturated() is False
    OllamaClient.pending_fast = 4
    assert OllamaClient.fast_queue_saturated() is True
    OllamaClient.pending_fast = 100
    assert OllamaClient.fast_queue_saturated() is True


# --- 4. Per-tier semaphore size honors config --------------------------


@pytest.mark.asyncio
async def test_per_tier_semaphores_use_config(monkeypatch):
    from core.utils.config import get_config

    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    ollama["fast_max_concurrent"] = 3
    ollama["deep_max_concurrent"] = 1
    ollama["validator_max_concurrent"] = 2
    cfg._data["ollama"] = ollama
    ollama_mod._reset_tier_semaphores()

    fast_sem = await ollama_mod._get_tier_semaphore("fast")
    deep_sem = await ollama_mod._get_tier_semaphore("deep")
    val_sem = await ollama_mod._get_tier_semaphore("validator")
    # asyncio.Semaphore exposes _value (initial = configured cap before
    # any acquires); use it for assertion since there's no public getter.
    assert fast_sem._value == 3
    assert deep_sem._value == 1
    assert val_sem._value == 2


@pytest.mark.asyncio
async def test_legacy_max_concurrent_calls_used_for_fast_default():
    """Old configs that only set max_concurrent_calls=N must still
    cap the fast tier at N (back-compat for users who haven't migrated
    config.yaml yet)."""
    from core.utils.config import get_config

    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    ollama.pop("fast_max_concurrent", None)
    ollama["max_concurrent_calls"] = 5
    cfg._data["ollama"] = ollama
    ollama_mod._reset_tier_semaphores()

    fast_sem = await ollama_mod._get_tier_semaphore("fast")
    assert fast_sem._value == 5
