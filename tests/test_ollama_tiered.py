"""Tiered Ollama architecture tests.

Covers the three pieces of the tiered-model wiring:

  1. OllamaClient routes fast/deep/validator to the correct config models
     and temperatures.
  2. fast_queue_saturated() flips based on configured alert threshold and
     a class-level pending_fast counter.
  3. core.signals.validator.cross_validate:
        - 'ok' when drift within tolerance
        - 'halved' when drift > max_prob_drift (size cut 50%)
        - 'direction_skip' when validator flips side relative to mid
        - 'unavailable' when the validator call returns None
  4. shadow.open_position respects each validator decision when the
     size is >= validator_high_stakes_usd: smaller sizes bypass the
     validator entirely.
  5. event_sniper entry uses the heuristic path (bypasses Ollama) once
     OllamaClient.fast_queue_saturated() is True.

The tests monkeypatch OllamaClient._generate (or the high-level fast/
deep/validate methods) instead of hitting a real Ollama server. The DB
migration adds `ollama_stats` + `validator_snapshot` columns; we rely on
those being present in the shared temp DB.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from core.execution import allocator, shadow
from core.markets.cache import Market
from core.signals import validator as validator_mod
from core.signals.ollama_client import OllamaClient
from core.signals.validator import ValidationResult, cross_validate
from core.strategies import event_sniper, scoring
from core.strategies.scoring import Score
from core.utils import db as db_module
from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.prices import PriceSnapshot


# --- Fixtures -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ollama_tier.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


@pytest.fixture(autouse=True)
def _reset_queue_counters():
    # Class-level counters leak across tests otherwise.
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0
    yield
    OllamaClient.pending_fast = 0
    OllamaClient.pending_deep = 0
    OllamaClient.pending_validator = 0


def _fake_market(market_id: str = "m-test", mid: float = 0.40) -> Market:
    spread = 0.02
    return Market(
        market_id=market_id,
        question=f"Will {market_id} happen?",
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


# --- 1. Tiered client routing ------------------------------------------


def test_queue_counters_increment_and_clear():
    OllamaClient._inc("fast")
    OllamaClient._inc("fast")
    OllamaClient._inc("deep")
    OllamaClient._inc("validator")
    depths = OllamaClient.queue_depths()
    assert depths == {"fast": 2, "deep": 1, "validator": 1}
    OllamaClient._dec("fast")
    OllamaClient._dec("fast")
    OllamaClient._dec("deep")
    OllamaClient._dec("validator")
    assert OllamaClient.queue_depths() == {"fast": 0, "deep": 0, "validator": 0}


def test_dec_clamps_at_zero():
    OllamaClient._dec("fast")
    assert OllamaClient.pending_fast == 0


def test_fast_queue_saturated_uses_config(monkeypatch):
    cfg = get_config()
    data = cfg.as_dict()
    data.setdefault("ollama", {})["queue_depth_alert"] = 3
    monkeypatch.setattr(cfg, "as_dict", lambda: data)
    # Reload so the cached get() sees the override.
    cfg._data = data  # noqa: SLF001 - tests poke the cache directly

    OllamaClient.pending_fast = 2
    assert OllamaClient.fast_queue_saturated() is False
    OllamaClient.pending_fast = 3
    assert OllamaClient.fast_queue_saturated() is True


@pytest.mark.asyncio
async def test_fast_score_routes_to_fast_model(monkeypatch):
    seen: list[dict] = []

    async def fake_generate(self, prompt, *, call_type, response_format="json"):
        seen.append({
            "call_type": call_type,
            "model": self._model_for(call_type),
            "timeout": self._timeout_for(call_type),
            "temperature": self._temperature_for(call_type),
        })
        return {"implied_prob": 0.5, "confidence": 0.5, "reasoning": "x"}, {}

    monkeypatch.setattr(OllamaClient, "_generate", fake_generate)

    client = OllamaClient()
    await client.fast_score("hello")
    await client.deep_score("hello")
    await client.validate("hello")

    assert [s["call_type"] for s in seen] == ["fast", "deep", "validator"]
    # Three distinct model identifiers — even if two happen to be the
    # same binary, the tier dispatcher must resolve each independently.
    assert seen[0]["model"] == client.fast_model
    assert seen[1]["model"] == client.deep_model
    assert seen[2]["model"] == client.validator_model
    # Fast tier is the lowest-latency contract: it must be strictly
    # faster-timed than deep.
    assert seen[0]["timeout"] <= seen[1]["timeout"]
    # Validator is the lowest-exploration contract.
    assert seen[2]["temperature"] <= seen[1]["temperature"]


@pytest.mark.asyncio
async def test_generate_json_legacy_routes_to_deep(monkeypatch):
    seen: list[str] = []

    async def fake_generate(self, prompt, *, call_type, response_format="json"):
        seen.append(call_type)
        return {"implied_prob": 0.5, "confidence": 0.5, "reasoning": "x"}, {}

    monkeypatch.setattr(OllamaClient, "_generate", fake_generate)
    await OllamaClient().generate_json("hello")
    assert seen == ["deep"]


# --- 2. Validator decisions --------------------------------------------


class _StubClient:
    """Minimal OllamaClient stand-in. The validator only calls
    .validate(prompt, context=...) — we don't need the full client."""

    def __init__(self, response):
        self._response = response

    async def validate(self, prompt, *, context=None):
        if callable(self._response):
            return self._response(prompt)
        return self._response


@pytest.mark.asyncio
async def test_validator_ok_when_within_tolerance():
    market = _fake_market(mid=0.40)
    # validator says 0.55 vs scorer 0.60, drift 0.05 < max 0.15 default.
    client = _StubClient({
        "implied_prob": 0.55, "confidence": 0.7, "reasoning": "agree-ish",
    })
    result = await cross_validate(
        client=client,
        market=market,
        side="BUY",
        original_true_prob=0.60,
        original_reasoning="breakout",
        evidence_text="headline A\nheadline B",
        size_usd=300.0,
    )
    assert result.decision == "ok"
    assert result.adjusted_size == pytest.approx(300.0)
    assert result.drift == pytest.approx(0.05, abs=1e-6)


@pytest.mark.asyncio
async def test_validator_halves_on_prob_drift():
    market = _fake_market(mid=0.40)
    # validator says 0.42 vs scorer 0.70 -> drift 0.28 > max 0.15.
    # Critically, direction is still BUY (0.42 >= 0.40 mid), so the rule
    # under test is 'halve', not 'direction_skip'.
    client = _StubClient({
        "implied_prob": 0.42, "confidence": 0.5, "reasoning": "weaker",
    })
    result = await cross_validate(
        client=client,
        market=market,
        side="BUY",
        original_true_prob=0.70,
        original_reasoning="overconfident",
        evidence_text="news",
        size_usd=300.0,
    )
    assert result.decision == "halved"
    assert result.adjusted_size == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_validator_skips_on_direction_mismatch():
    market = _fake_market(mid=0.40)
    # Scorer wanted BUY (thought true_prob 0.65). Validator says 0.30 —
    # below mid, so it would want SELL. Direction mismatch -> skip.
    client = _StubClient({
        "implied_prob": 0.30, "confidence": 0.8, "reasoning": "disagree",
    })
    result = await cross_validate(
        client=client,
        market=market,
        side="BUY",
        original_true_prob=0.65,
        original_reasoning="long thesis",
        evidence_text="news",
        size_usd=300.0,
    )
    assert result.decision == "direction_skip"
    assert result.adjusted_size == 0.0


@pytest.mark.asyncio
async def test_validator_unavailable_leaves_size_unchanged():
    market = _fake_market(mid=0.40)
    client = _StubClient(None)  # simulates timeout / bad JSON / cooldown
    result = await cross_validate(
        client=client,
        market=market,
        side="BUY",
        original_true_prob=0.65,
        original_reasoning="long thesis",
        evidence_text="news",
        size_usd=300.0,
    )
    assert result.decision == "unavailable"
    assert result.adjusted_size == pytest.approx(300.0)


# --- 3. shadow.open_position integration with validator ----------------


@pytest.mark.asyncio
async def test_shadow_open_halves_size_on_validator_drift(monkeypatch):
    await allocator.init_lane_capital()
    market = _fake_market("m-halve", mid=0.40)
    snap = PriceSnapshot(
        token_id="yes-tok", bid=0.39, ask=0.41, last=0.40,
        ts=time.time(), source="ticks",
    )

    async def fake_cv(**kwargs):
        return ValidationResult(
            decision="halved",
            adjusted_size=100.0,
            validator_true_prob=0.42,
            validator_confidence=0.5,
            validator_reasoning="weak",
            original_true_prob=0.70,
            drift=0.28,
            notes="drift",
        )

    monkeypatch.setattr(
        "core.signals.validator.cross_validate", fake_cv,
    )
    # Caller-side reservation mirrors how lanes wire this.
    approved = await allocator.reserve("longshot", 200.0)
    assert approved == pytest.approx(200.0)

    pos_id = await shadow.open_position(
        strategy="longshot",
        market=market,
        side="BUY",
        snapshot=snap,
        size_usd=approved,
        true_prob=0.70,
        confidence=0.75,
        entry_reason="test",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=0.0,
        ollama_client=OllamaClient(),
        validator_text="headline",
        validator_reasoning="thesis",
    )
    assert pos_id is not None
    row = await fetch_one(
        "SELECT size_usd, validator_snapshot FROM shadow_positions WHERE id=?",
        (pos_id,),
    )
    assert row["size_usd"] == pytest.approx(100.0)
    snap_blob = json.loads(row["validator_snapshot"])
    assert snap_blob["decision"] == "halved"


@pytest.mark.asyncio
async def test_shadow_open_skipped_on_direction_mismatch(monkeypatch):
    await allocator.init_lane_capital()
    market = _fake_market("m-skip", mid=0.40)
    snap = PriceSnapshot(
        token_id="yes-tok", bid=0.39, ask=0.41, last=0.40,
        ts=time.time(), source="ticks",
    )

    async def fake_cv(**kwargs):
        return ValidationResult(
            decision="direction_skip",
            adjusted_size=0.0,
            validator_true_prob=0.30,
            validator_confidence=0.8,
            validator_reasoning="flip",
            original_true_prob=0.65,
            drift=0.35,
            notes="mismatch",
        )

    monkeypatch.setattr("core.signals.validator.cross_validate", fake_cv)
    approved = await allocator.reserve("longshot", 250.0)
    pos_id = await shadow.open_position(
        strategy="longshot",
        market=market,
        side="BUY",
        snapshot=snap,
        size_usd=approved,
        true_prob=0.65,
        confidence=0.75,
        entry_reason="test",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=0.0,
        ollama_client=OllamaClient(),
        validator_text="headline",
        validator_reasoning="thesis",
    )
    assert pos_id is None
    open_rows = await shadow.open_positions_for("longshot")
    assert open_rows == []


@pytest.mark.asyncio
async def test_shadow_open_below_threshold_skips_validator(monkeypatch):
    await allocator.init_lane_capital()
    market = _fake_market("m-small", mid=0.40)
    snap = PriceSnapshot(
        token_id="yes-tok", bid=0.39, ask=0.41, last=0.40,
        ts=time.time(), source="ticks",
    )
    called: list[bool] = []

    async def fake_cv(**kwargs):
        called.append(True)
        return ValidationResult(
            decision="direction_skip",
            adjusted_size=0.0,
            validator_true_prob=0.30,
            validator_confidence=0.8,
            validator_reasoning="",
            original_true_prob=0.65, drift=0.35, notes="",
        )

    monkeypatch.setattr("core.signals.validator.cross_validate", fake_cv)
    # Scalping-sized entry: $75 is below the $200 validator threshold.
    approved = await allocator.reserve("scalping", 75.0)
    pos_id = await shadow.open_position(
        strategy="scalping",
        market=market,
        side="BUY",
        snapshot=snap,
        size_usd=approved,
        true_prob=0.65,
        confidence=0.75,
        entry_reason="test",
        evidence_ids=[],
        evidence_snapshot=None,
        entry_latency_ms=0.0,
        ollama_client=OllamaClient(),
        validator_text="headline",
        validator_reasoning="thesis",
    )
    assert pos_id is not None
    # Validator must not have been invoked at this size.
    assert called == []


# --- 4. event_sniper heuristic-on-saturation path ----------------------


@pytest.mark.asyncio
async def test_event_lane_falls_back_to_heuristic_when_fast_queue_saturated(
    monkeypatch,
):
    """When fast_queue_saturated() is True, _process_item must NOT call
    scoring.score_with_timeout. It should synthesize a Score from the
    keyword heuristic and proceed (heuristic path enforces the smaller
    fallback size)."""
    await allocator.init_lane_capital()

    market_id = "event-sat-1"
    await execute(
        """INSERT INTO markets
           (market_id, question, slug, category, active, close_time,
            token_ids, best_bid, best_ask, last_price, liquidity, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            market_id,
            "Will saturation-market resolve YES?",
            market_id,
            "politics",
            1,
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3 * 86400),
            ),
            json.dumps(["yes-tok", "no-tok"]),
            0.39, 0.41, 0.40, 50_000.0, time.time(),
        ),
    )
    item_id = await execute(
        """INSERT INTO feed_items
           (url_hash, source, title, summary, url, ingested_at, meta)
           VALUES (?,?,?,?,?,?,?)""",
        (
            "sat-hash", "reuters",
            "Candidate wins primary in shock result",
            "The frontrunner surges ahead and confirms victory.",
            "http://x/sat1", time.time() - 30,
            json.dumps({"linked_market_id": market_id}),
        ),
    )

    # Force the saturation gate open by simulating a full fast queue.
    OllamaClient.pending_fast = 99
    assert OllamaClient.fast_queue_saturated() is True

    # If scoring.score_with_timeout is called, the test should fail —
    # the whole point of the saturation branch is to avoid Ollama.
    async def forbidden_score_with_timeout(*args, **kwargs):
        raise AssertionError("scoring.score_with_timeout must not be called when fast queue is saturated")

    async def fake_volume(market_id):
        return 20_000.0

    async def fake_current_price(market_id, token_id):
        return PriceSnapshot(
            token_id=token_id, bid=0.39, ask=0.41, last=0.40,
            ts=time.time(), source="ticks",
        )

    # Heuristic must return non-zero true_prob + confidence so the edge
    # gate passes; the real heuristic.score does this from the keyword
    # hits already present in the headline above. We stub it for
    # determinism so the edge is unambiguously above the min_edge gate.
    def fake_heuristic_score(text, market):
        from core.strategies.heuristic import HeuristicResult
        return HeuristicResult(
            direction=1, strength=0.6, implied_prob=0.70,
            confidence=0.6, reasoning="stub heuristic",
        )

    monkeypatch.setattr(scoring, "score_with_timeout", forbidden_score_with_timeout)
    monkeypatch.setattr(event_sniper, "volume_24h", fake_volume)
    monkeypatch.setattr(event_sniper, "current_price", fake_current_price)
    monkeypatch.setattr(event_sniper.heuristic, "score", fake_heuristic_score)

    lane = event_sniper.EventSniperLane()
    await lane._init_cursor()
    lane._cursor_id = 0

    processed = await lane._drain_new_items()
    assert processed >= 1

    positions = await shadow.open_positions_for("event_sniping")
    assert len(positions) == 1
    # Heuristic path forces the configured fallback size ($50 default).
    assert positions[0].size_usd == pytest.approx(50.0)
