"""Deep-tier realtime kill-switch tests.

When ``ollama.deep_realtime_enabled`` is false (the CPU-only / no-GPU
deployment shape that surfaced in the April 2026 soak), the realtime
hot path must not call the deep tier:

  1. ``scoring.score(tier="deep")`` returns None without dispatching.
  2. ``scoring.score_with_fallback(tier="deep")`` returns a heuristic
     ``Score`` without dispatching.
  3. The signal pipeline branches to a heuristic-only path that uses
     ``core.strategies.heuristic.score`` against the linked market
     (or top heuristic candidate) and writes a real signal row with
     ``prompt_version='heuristic'``.

Background callers (``generate_text``, batch enrichment) are NOT
affected — the gate only intercepts the realtime hot path. Those
paths are out of scope for this test file.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from core.markets.cache import Market
from core.risk.rules import RiskEngine
from core.signals import ollama_client as ollama_mod
from core.signals import pipeline as pipeline_mod
from core.signals.ollama_client import OllamaClient, deep_realtime_enabled
from core.signals.pipeline import SignalPipeline
from core.strategies import scoring
from core.utils import db as db_module
from core.utils.config import get_config
from core.utils.db import execute, fetch_all


@pytest.fixture(autouse=True)
def _force_deep_realtime_off():
    """Default true (other tests assume deep is allowed). Flip false
    for the duration of this module's tests, then restore."""
    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    saved = ollama.get("deep_realtime_enabled")
    ollama["deep_realtime_enabled"] = False
    cfg._data["ollama"] = ollama
    yield
    if saved is None:
        ollama.pop("deep_realtime_enabled", None)
    else:
        ollama["deep_realtime_enabled"] = saved
    cfg._data["ollama"] = ollama


@pytest.fixture(autouse=True)
def _stub_context(monkeypatch):
    async def fake_context(market_id):
        return {}

    monkeypatch.setattr(scoring, "build_market_context", fake_context)


def _market(market_id: str = "m-test", mid: float = 0.40) -> Market:
    spread = 0.02
    return Market(
        market_id=market_id,
        question=f"Will {market_id} win?",
        slug=market_id,
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


# --- 1. scoring helpers gate the deep tier ----------------------------


@pytest.mark.asyncio
async def test_score_with_deep_tier_returns_none_when_disabled(monkeypatch):
    assert deep_realtime_enabled() is False

    client = OllamaClient()
    forbidden_calls: list[int] = []

    async def forbidden_deep(prompt, *, context=None, tag=""):
        forbidden_calls.append(1)
        raise AssertionError("deep_score must not be invoked when disabled")

    monkeypatch.setattr(client, "deep_score", forbidden_deep)

    result = await scoring.score(
        _market(), "Candidate wins primary", client=client, tier="deep",
    )
    assert result is None
    assert forbidden_calls == []


@pytest.mark.asyncio
async def test_score_with_fallback_deep_returns_heuristic_when_disabled(monkeypatch):
    client = OllamaClient()

    async def forbidden_deep(prompt, *, context=None, tag=""):
        raise AssertionError("deep_score must not be invoked when disabled")

    monkeypatch.setattr(client, "deep_score", forbidden_deep)

    score = await scoring.score_with_fallback(
        _market(),
        "Candidate wins primary; campaign confirms breakthrough",
        client=client, tier="deep",
    )
    # Heuristic fired; deep was skipped entirely.
    assert score.source == "heuristic"


# --- 2. Pipeline routes through heuristic when disabled --------------


@pytest.fixture
def _pipeline_db(tmp_path, monkeypatch):
    db_path = tmp_path / "pipeline_deep_off.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


@pytest.mark.asyncio
async def test_pipeline_uses_heuristic_when_deep_realtime_disabled(
    _pipeline_db, monkeypatch,
):
    """Drive a feed item through the pipeline with deep_realtime off
    and confirm:
      - generate_json was NOT called
      - a signals row was inserted with prompt_version='heuristic'
      - the row references the linked market_id from the feed meta
    """
    market_id = "m-deep-off-1"
    close_time = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30 * 86400),
    )
    await execute(
        """INSERT INTO markets
           (market_id, question, slug, category, active, close_time,
            token_ids, best_bid, best_ask, last_price, liquidity, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            market_id,
            "Will the candidate win the primary?",
            market_id, "politics", 1, close_time,
            json.dumps(["yes-tok", "no-tok"]),
            0.39, 0.41, 0.40, 50_000.0, time.time(),
        ),
    )
    item_id = await execute(
        """INSERT INTO feed_items
           (url_hash, source, title, summary, url, ingested_at, meta)
           VALUES (?,?,?,?,?,?,?)""",
        (
            "ph-deep-off", "reuters",
            "Candidate wins primary in shock result",
            "The frontrunner surges ahead and confirms victory.",
            "http://x/1", time.time() - 30,
            json.dumps({"linked_market_id": market_id}),
        ),
    )

    seeded_market = Market(
        market_id=market_id,
        question="Will the candidate win the primary?",
        slug=market_id,
        category="politics",
        active=True,
        close_time=close_time,
        token_ids=["yes-tok", "no-tok"],
        best_bid=0.39, best_ask=0.41, last_price=0.40,
        liquidity=50_000.0, updated_at=time.time(),
    )

    async def fake_scored_candidates_for(text):
        return [(0.9, seeded_market)]

    async def fake_context(market_id):
        return {}

    async def fake_get_market(mid):
        if str(mid) == market_id:
            return seeded_market
        return None

    monkeypatch.setattr(pipeline_mod, "scored_candidates_for", fake_scored_candidates_for)
    monkeypatch.setattr(pipeline_mod, "build_market_context", fake_context)
    monkeypatch.setattr(pipeline_mod, "get_market", fake_get_market)

    pipe = SignalPipeline(RiskEngine())

    # generate_json must NOT be called when deep_realtime is off.
    async def forbidden_generate_json(prompt, *, tag=""):
        raise AssertionError(
            "generate_json must not be called with deep_realtime_enabled=false"
        )

    monkeypatch.setattr(pipe._ollama, "generate_json", forbidden_generate_json)

    item_row = {
        "id": int(item_id),
        "title": "Candidate wins primary in shock result",
        "summary": "The frontrunner surges ahead and confirms victory.",
        "source": "reuters",
        "url": "http://x/1",
        "ingested_at": time.time() - 30,
        "meta": json.dumps({"linked_market_id": market_id}),
    }
    await pipe._process_item(item_row)

    # Assert: at least one signals row exists for this feed item, with
    # prompt_version='heuristic' and the linked market_id.
    rows = await fetch_all(
        "SELECT prompt_version, market_id, status, reasoning, implied_prob "
        "FROM signals WHERE feed_item_id=?",
        (int(item_id),),
    )
    assert rows, "expected pipeline to persist a row via the heuristic path"
    sources = [r["prompt_version"] for r in rows]
    assert "heuristic" in sources, (
        f"expected a heuristic-source row, got prompt_versions={sources}"
    )
    heuristic_row = next(r for r in rows if r["prompt_version"] == "heuristic")
    assert heuristic_row["market_id"] == market_id
    # Reasoning should mention the heuristic.
    assert "heuristic" in (heuristic_row["reasoning"] or "").lower()


# --- 3. Default is True (back-compat) --------------------------------


def test_default_deep_realtime_enabled_is_true():
    """Without the explicit override, the gate must default to True
    so existing GPU deployments and the test suite keep working."""
    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    ollama.pop("deep_realtime_enabled", None)
    cfg._data["ollama"] = ollama
    assert deep_realtime_enabled() is True


def test_string_false_is_honored():
    """YAML can produce string ``"false"`` if the operator quotes it
    by mistake — we treat that as False, not as the truthy string."""
    cfg = get_config()
    ollama = dict(cfg._data.get("ollama") or {})
    ollama["deep_realtime_enabled"] = "false"
    cfg._data["ollama"] = ollama
    assert deep_realtime_enabled() is False
    ollama["deep_realtime_enabled"] = "FALSE"
    cfg._data["ollama"] = ollama
    assert deep_realtime_enabled() is False
