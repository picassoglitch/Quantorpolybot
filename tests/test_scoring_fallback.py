"""score_with_fallback contract tests.

The scalping lane (and any other always-on caller) needs scoring that
NEVER returns None — a stalled GPU should yield a heuristic Score with
``source`` set so the caller can tell apart "real LLM said this" from
"the keyword fallback fired because Ollama was slow / saturated".

These tests exercise:

  1. Healthy Ollama -> Score(source="ollama").
  2. Ollama returns malformed/None -> Score(source="heuristic").
  3. Ollama call exceeds wait_for timeout -> Score(source="timeout").
  4. Ollama raises -> Score(source="heuristic").
  5. Fast queue saturated + tier="fast" -> Score(source="heuristic"),
     and crucially the underlying Ollama method is NEVER called (the
     whole point of preempting is to take pressure off a saturated
     queue).

The tests stub ``build_market_context`` to avoid the DB and stub the
client's tier method to control Ollama outcomes deterministically.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.markets.cache import Market
from core.signals.ollama_client import OllamaClient
from core.strategies import scoring


@pytest.fixture(autouse=True)
def _reset_queue_counters():
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0
    yield
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0


@pytest.fixture(autouse=True)
def _stub_context(monkeypatch):
    async def fake_context(market_id):
        return {}

    monkeypatch.setattr(scoring, "build_market_context", fake_context)


def _market(mid: float = 0.40) -> Market:
    spread = 0.02
    return Market(
        market_id="m-test",
        question="Will the test market resolve YES?",
        slug="m-test",
        category="politics",
        active=True,
        close_time="2099-01-01T00:00:00Z",
        token_ids=["yes-tok", "no-tok"],
        best_bid=mid - spread / 2,
        best_ask=mid + spread / 2,
        last_price=mid,
        liquidity=50_000.0,
        updated_at=time.time(),
    )


def _bullish_text() -> str:
    # Heuristic looks for keywords like "wins", "confirms" etc.; bake
    # in a few so the heuristic produces a clear directional Score.
    return "Candidate wins primary; campaign confirms breakthrough"


# --- 1. Healthy ollama -> source=ollama --------------------------------


@pytest.mark.asyncio
async def test_returns_ollama_when_call_succeeds(monkeypatch):
    client = OllamaClient()

    async def fake_fast(prompt, *, context=None, tag=""):
        return {
            "implied_prob": 0.62, "confidence": 0.71, "reasoning": "ok",
        }

    monkeypatch.setattr(client, "fast_score", fake_fast)

    result = await scoring.score_with_fallback(
        _market(), _bullish_text(), client=client, tier="fast",
    )
    assert result.source == "ollama"
    assert result.true_prob == pytest.approx(0.62)
    assert result.confidence == pytest.approx(0.71)


# --- 2. Ollama returns None -> heuristic -------------------------------


@pytest.mark.asyncio
async def test_returns_heuristic_when_ollama_returns_none(monkeypatch):
    client = OllamaClient()

    async def fake_fast(prompt, *, context=None, tag=""):
        return None  # cooldown / unparseable / etc.

    monkeypatch.setattr(client, "fast_score", fake_fast)

    result = await scoring.score_with_fallback(
        _market(), _bullish_text(), client=client, tier="fast",
    )
    assert result.source == "heuristic"
    # Heuristic on bullish text should produce a positive nudge above mid.
    assert result.true_prob > 0.40
    assert 0.0 < result.confidence <= 0.70


# --- 3. wait_for fires -> source=timeout -------------------------------


@pytest.mark.asyncio
async def test_returns_timeout_marker_when_ollama_stalls(monkeypatch):
    client = OllamaClient()

    async def slow_fast(prompt, *, context=None, tag=""):
        await asyncio.sleep(2.0)  # longer than timeout below
        return {"implied_prob": 0.5, "confidence": 0.5, "reasoning": "x"}

    monkeypatch.setattr(client, "fast_score", slow_fast)

    result = await scoring.score_with_fallback(
        _market(), _bullish_text(),
        client=client, tier="fast", timeout_seconds=0.05,
    )
    # Heuristic ran (always returns something) but the source flag
    # preserves the cause so scan logs / downstream can distinguish it.
    assert result.source == "timeout"


# --- 4. Ollama raises -> heuristic -------------------------------------


@pytest.mark.asyncio
async def test_returns_heuristic_when_ollama_raises(monkeypatch):
    client = OllamaClient()

    async def boom_fast(prompt, *, context=None, tag=""):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(client, "fast_score", boom_fast)

    result = await scoring.score_with_fallback(
        _market(), _bullish_text(), client=client, tier="fast",
    )
    assert result.source == "heuristic"


# --- 5. Saturated fast queue -> ollama is NOT called -------------------


@pytest.mark.asyncio
async def test_saturated_fast_queue_skips_ollama(monkeypatch):
    """The whole point of the saturation preempt is to take pressure off
    a queue that's already backed up. The Ollama method must not be
    called when the saturation gate fires."""
    from core.utils.config import get_config

    cfg = get_config()
    cfg._data.setdefault("ollama", {})["queue_depth_alert"] = 3

    client = OllamaClient()
    forbidden_calls: list[int] = []

    async def forbidden_fast(prompt, *, context=None, tag=""):
        forbidden_calls.append(1)
        raise AssertionError(
            "fast_score must not be invoked when the fast queue is saturated"
        )

    monkeypatch.setattr(client, "fast_score", forbidden_fast)

    OllamaClient.pending_fast = 99  # well above the alert threshold
    assert client.fast_queue_saturated() is True

    result = await scoring.score_with_fallback(
        _market(), _bullish_text(), client=client, tier="fast",
    )
    assert result.source == "heuristic"
    assert forbidden_calls == [], "saturation gate should preempt the Ollama call"


# --- 6. Deep tier ignores fast saturation ------------------------------


@pytest.mark.asyncio
async def test_deep_tier_does_not_preempt_on_fast_saturation(monkeypatch):
    """The fast queue is the only tier with a saturation alert wired up.
    A deep-tier caller must continue past the gate even when the fast
    queue is full — deep work has its own (independent) semaphore and
    its own back-pressure model."""
    client = OllamaClient()
    deep_calls: list[int] = []

    async def fake_deep(prompt, *, context=None, tag=""):
        deep_calls.append(1)
        return {"implied_prob": 0.55, "confidence": 0.65, "reasoning": "ok"}

    monkeypatch.setattr(client, "deep_score", fake_deep)

    OllamaClient.pending_fast = 99  # saturate fast — should be ignored

    result = await scoring.score_with_fallback(
        _market(), _bullish_text(), client=client, tier="deep",
    )
    assert deep_calls == [1]
    assert result.source == "ollama"
