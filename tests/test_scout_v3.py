"""Scout v3 — observed thresholds, high-severity-solo override,
LLM polarity inference, persistence fields.

All tests are pure / mocked. No real Ollama call. No real DB rows
unless explicitly using the temp_db fixture.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.markets.cache import Market
from core.scout import candidate as scout_candidate
from core.scout.event import Event, EventCategory
from core.scout.impact import (
    _LLM_POLARITY_DEFAULT_SEVERITY_FLOOR,
    _LLM_POLARITY_DEFAULT_TIMEOUT_SECONDS,
    _OBSERVED_MIN_MATCH_SCORE,
    _OBSERVED_MIN_SEVERITY,
    ImpactScore,
    _build_polarity_prompt,
    _llm_infer_polarity,
    _parse_polarity_response,
    score_impact,
    score_impact_async,
)
from core.scout.mapper import MarketMatch
from core.utils import db as db_module


# ============================================================
# Fixtures
# ============================================================


def _market(question: str = "Will war end by Q3?", mid: float = 0.40) -> Market:
    return Market(
        market_id="m-x", question=question, slug="m",
        category="politics", active=True, close_time="",
        token_ids=["yes-tok", "no-tok"],
        best_bid=mid - 0.01, best_ask=mid + 0.01, last_price=mid,
        liquidity=25_000.0, updated_at=time.time(),
    )


def _event(
    category: EventCategory = EventCategory.CEASEFIRE,
    severity: float = 0.85,
    confidence: float = 0.70,
    sources: list[str] | None = None,
    age_seconds: float = 60.0,
) -> Event:
    now = time.time()
    sources = sources or ["src-a", "src-b"]
    return Event(
        event_id="evt-v3", timestamp_detected=now - age_seconds,
        title="High-severity event", category=category,
        severity=severity, confidence=confidence, location="",
        entities=["X"], source_count=len(sources), sources=sources,
        contradiction_score=0.0, raw_signal_ids=[1, 2],
        first_seen_at=now - age_seconds, last_seen_at=now,
    )


def _match(market: Market, score: float = 0.5) -> MarketMatch:
    return MarketMatch(
        market=market, score=score,
        entity_overlap=score, keyword_overlap=score,
        category_alignment=1.0, liquidity_quality=0.8,
        near_resolution_bonus=0.5,
    )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "scout_v3.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


# ============================================================
# 1. Observed thresholds — v3.1: lowered to 0.35 / 0.20
# ============================================================


def test_observed_thresholds_lowered_to_v3_values():
    """Pin the constants so a future change is reviewed deliberately.

    v3.0 was 0.50 / 0.25; v3.1 lowered to 0.35 / 0.20 because
    single-source GDELT events from tier-3 domains cap their
    severity at ~0.38 (SHOOTING) / ~0.45 (ASSASSINATION_ATTEMPT).
    The v3.0 floor was unreachable for these in practice, so the
    lane silently dropped exactly the kind of imperfect-information-
    early signal the scout exists for.
    """
    assert _OBSERVED_MIN_SEVERITY == 0.35
    assert _OBSERVED_MIN_MATCH_SCORE == 0.20


def test_observed_fires_at_new_severity_floor():
    """Single-source tier-3 GDELT events cap at severity ~0.38 for
    SHOOTING. The v3.1 threshold (0.35) catches them; the v3.0
    threshold (0.50) and earlier did not."""
    market = _market("Will candidate X win the Q3 primary?")
    impact = score_impact(
        _event(EventCategory.ELECTION_RESULT, severity=0.38),
        _match(market, score=0.25),
    )
    assert impact.observed is True
    assert impact.confidence > 0.0


def test_observed_fires_at_new_match_score_floor():
    """sev=0.50 + match=0.20 should trigger observed-mode under
    v3.1 (0.35 / 0.20). Under v3.0 (0.50 / 0.25) the match would
    have been below the floor."""
    market = _market("Will candidate X win?")
    impact = score_impact(
        _event(EventCategory.ELECTION_RESULT, severity=0.50),
        _match(market, score=0.20),
    )
    assert impact.observed is True


def test_observed_does_not_fire_below_new_thresholds():
    """Below v3.1 thresholds (sev=0.30 < 0.35) → still NOT observed."""
    market = _market("Will candidate X win?")
    impact = score_impact(
        _event(EventCategory.ELECTION_RESULT, severity=0.30),
        _match(market, score=0.30),
    )
    assert impact.observed is False
    assert impact.confidence == 0.0


# ============================================================
# 2. ImpactScore.polarity_source field
# ============================================================


def test_polarity_source_rules_when_rule_matches():
    market = _market("Will the war end by Q3?", mid=0.40)
    impact = score_impact(_event(EventCategory.CEASEFIRE), _match(market))
    assert impact.direction == 1
    assert impact.polarity_source == "rules"


def test_polarity_source_none_when_polarity_unknown():
    """severity 0.30 < 0.35 (v3.1 floor) AND no polarity rule for
    ELECTION_RESULT → polarity_source 'none', no observed-mode."""
    market = _market("Will candidate X win?")
    impact = score_impact(
        _event(EventCategory.ELECTION_RESULT, severity=0.30),
        _match(market, score=0.30),
    )
    assert impact.direction == 0
    assert impact.observed is False
    assert impact.polarity_source == "none"


def test_polarity_source_set_via_override():
    market = _market("Will candidate X win?")
    impact = score_impact(
        _event(EventCategory.ELECTION_RESULT),
        _match(market),
        direction_override=+1,
        polarity_source_override="llm",
        polarity_reason_override="llm: buy (decisive surprise)",
    )
    assert impact.direction == 1
    assert impact.polarity_source == "llm"
    assert "llm" in impact.polarity_reasoning


# ============================================================
# 3. LLM polarity inference — prompt + parser
# ============================================================


def test_polarity_prompt_includes_event_and_market_context():
    market = _market("Will candidate X win?", mid=0.30)
    prompt = _build_polarity_prompt(_event(), market, mid=0.30)
    assert "EVENT" in prompt and "MARKET" in prompt
    # The prompt MUST request structured JSON to avoid free-text
    # responses that need NLP to parse.
    assert "JSON" in prompt
    assert "buy" in prompt and "sell" in prompt and "unclear" in prompt
    # Must include numeric mid so the model sees the current price.
    assert "0.30" in prompt


def test_parse_polarity_buy_synonyms():
    for word in ("buy", "yes", "bullish", "positive", "+1", "buy_yes"):
        d, _ = _parse_polarity_response({"direction": word})
        assert d == +1, f"failed for {word!r}"


def test_parse_polarity_sell_synonyms():
    for word in ("sell", "no", "bearish", "negative", "-1", "sell_yes", "buy_no"):
        d, _ = _parse_polarity_response({"direction": word})
        assert d == -1, f"failed for {word!r}"


def test_parse_polarity_unclear_returns_zero():
    d, reason = _parse_polarity_response({"direction": "unclear"})
    assert d == 0
    assert "unclear" in reason


def test_parse_polarity_handles_none_and_garbage():
    assert _parse_polarity_response(None)[0] == 0
    assert _parse_polarity_response({})[0] == 0
    assert _parse_polarity_response({"direction": 42})[0] == 0
    assert _parse_polarity_response({"direction": "asdf"})[0] == 0


def test_parse_polarity_includes_reason():
    d, reason = _parse_polarity_response({
        "direction": "buy",
        "reason": "decisive escalation",
    })
    assert d == +1
    assert "decisive escalation" in reason


# ============================================================
# 4. LLM polarity — mock client behavior
# ============================================================


class _MockOllama:
    """Minimal stand-in for OllamaClient.fast_score."""
    def __init__(self, response=None, raise_exc=None, sleep_seconds=0.0):
        self._response = response
        self._raise = raise_exc
        self._sleep = sleep_seconds

    async def fast_score(self, prompt, *, tag=""):
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._raise:
            raise self._raise
        return self._response


@pytest.mark.asyncio
async def test_llm_infer_polarity_returns_buy_on_clean_response():
    client = _MockOllama(response={"direction": "buy", "reason": "x"})
    market = _market()
    d, reason = await _llm_infer_polarity(client, _event(), market, 0.40)
    assert d == +1


@pytest.mark.asyncio
async def test_llm_infer_polarity_returns_zero_on_timeout():
    client = _MockOllama(sleep_seconds=2.0)
    market = _market()
    d, reason = await _llm_infer_polarity(
        client, _event(), market, 0.40, timeout_seconds=0.05,
    )
    assert d == 0
    assert "timeout" in reason


@pytest.mark.asyncio
async def test_llm_infer_polarity_returns_zero_on_exception():
    client = _MockOllama(raise_exc=RuntimeError("ollama crashed"))
    market = _market()
    d, reason = await _llm_infer_polarity(client, _event(), market, 0.40)
    assert d == 0
    assert "RuntimeError" in reason


@pytest.mark.asyncio
async def test_llm_infer_polarity_returns_zero_with_no_client():
    market = _market()
    d, reason = await _llm_infer_polarity(None, _event(), market, 0.40)
    assert d == 0
    assert "no client" in reason


# ============================================================
# 5. score_impact_async — gating + fallthrough
# ============================================================


@pytest.mark.asyncio
async def test_score_impact_async_skips_llm_when_rule_matches():
    """If rules already gave a direction, don't burn an LLM call."""
    client = _MockOllama(response={"direction": "sell", "reason": "x"})
    market = _market("Will the war end by Q3?", mid=0.40)
    impact = await score_impact_async(
        _event(EventCategory.CEASEFIRE), _match(market),
        ollama_client=client,
    )
    # Rule says ceasefire + "war end" → +1. LLM (which would have
    # said -1) should NOT have been consulted.
    assert impact.direction == +1
    assert impact.polarity_source == "rules"


@pytest.mark.asyncio
async def test_score_impact_async_skips_llm_when_severity_below_threshold():
    """ELECTION_RESULT has no rules, but severity 0.65 < 0.70 floor
    → don't call LLM. Must fall through to score_impact (which
    routes to observed-mode if thresholds are met)."""
    client = _MockOllama(response={"direction": "buy", "reason": "x"})
    market = _market("Will candidate X win?")
    impact = await score_impact_async(
        _event(EventCategory.ELECTION_RESULT, severity=0.65),
        _match(market, score=0.4),
        ollama_client=client,
        llm_severity_floor=0.70,
    )
    # Below LLM threshold AND match.score=0.4 ≥ 0.25 + sev≥0.50
    # → observed-mode.
    assert impact.direction == 0
    assert impact.observed is True
    assert impact.polarity_source != "llm"


@pytest.mark.asyncio
async def test_score_impact_async_skips_llm_when_disabled():
    client = _MockOllama(response={"direction": "buy", "reason": "x"})
    market = _market("Will candidate X win?")
    impact = await score_impact_async(
        _event(EventCategory.ELECTION_RESULT, severity=0.85),
        _match(market, score=0.4),
        ollama_client=client,
        llm_enabled=False,
    )
    assert impact.polarity_source != "llm"


@pytest.mark.asyncio
async def test_score_impact_async_uses_llm_when_rules_fail_and_severity_high():
    client = _MockOllama(response={"direction": "buy", "reason": "decisive"})
    market = _market("Will candidate X win Q3?")
    impact = await score_impact_async(
        _event(EventCategory.ELECTION_RESULT, severity=0.85),
        _match(market, score=0.5),
        ollama_client=client,
    )
    assert impact.direction == +1
    assert impact.polarity_source == "llm"
    assert "llm" in impact.polarity_reasoning


@pytest.mark.asyncio
async def test_score_impact_async_falls_back_to_observed_on_llm_unclear():
    """LLM returned 'unclear' → fall through to score_impact, which
    routes to observed-mode given high severity + decent match."""
    client = _MockOllama(response={"direction": "unclear", "reason": "x"})
    market = _market("Will candidate X win Q3?")
    impact = await score_impact_async(
        _event(EventCategory.ELECTION_RESULT, severity=0.85),
        _match(market, score=0.5),
        ollama_client=client,
    )
    assert impact.direction == 0
    assert impact.observed is True


@pytest.mark.asyncio
async def test_score_impact_async_falls_back_on_llm_timeout():
    """LLM call times out → caller still gets a usable
    ImpactScore (observed-mode in this case)."""
    client = _MockOllama(sleep_seconds=2.0)
    market = _market("Will candidate X win Q3?")
    impact = await score_impact_async(
        _event(EventCategory.ELECTION_RESULT, severity=0.85),
        _match(market, score=0.5),
        ollama_client=client,
        llm_timeout_seconds=0.05,
    )
    assert impact.direction == 0
    assert impact.observed is True


@pytest.mark.asyncio
async def test_score_impact_async_falls_back_on_llm_exception():
    client = _MockOllama(raise_exc=RuntimeError("ollama down"))
    market = _market("Will candidate X win Q3?")
    impact = await score_impact_async(
        _event(EventCategory.ELECTION_RESULT, severity=0.85),
        _match(market, score=0.5),
        ollama_client=client,
    )
    assert impact.direction == 0
    assert impact.observed is True


# ============================================================
# 6. High-severity-solo corroboration override
# ============================================================


@pytest.mark.asyncio
async def test_high_sev_solo_passes_corroboration(temp_db):
    """severity=0.85 + 1 non-primary source → the corroboration
    check would have rejected pre-v3. With the override it now
    passes AND gets a 0.30x size multiplier."""
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="rule matched",
        components={}, observed=False, polarity_source="rules",
    )
    decision = await scout_candidate.evaluate(
        _event(severity=0.85, sources=["someblog"]),
        _match(_market()), impact,
        cfg={
            "min_sources": 2, "primary_sources": ["reuters"],
            "high_sev_solo_threshold": 0.80,
            "high_sev_solo_size_multiplier": 0.30,
            "min_edge": 0.05, "min_confidence": 0.40,
            "max_event_age_seconds": 1800.0,
            "require_primary_or_corroboration": True,
        },
    )
    assert decision.accepted is True
    assert decision.high_sev_solo is True
    assert decision.size_multiplier == 0.30
    assert "high_sev_solo" in decision.reason


@pytest.mark.asyncio
async def test_high_sev_solo_does_not_apply_below_threshold(temp_db):
    """severity=0.79 < 0.80 → still rejects on corroboration."""
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="rule matched",
        components={}, observed=False, polarity_source="rules",
    )
    decision = await scout_candidate.evaluate(
        _event(severity=0.79, sources=["someblog"]),
        _match(_market()), impact,
        cfg={"high_sev_solo_threshold": 0.80,
             "min_sources": 2, "primary_sources": ["reuters"],
             "max_event_age_seconds": 1800.0,
             "require_primary_or_corroboration": True},
    )
    assert decision.accepted is False
    assert "corroboration" in decision.reason


@pytest.mark.asyncio
async def test_high_sev_solo_does_not_bypass_safety_gates(temp_db):
    """High-sev-solo is corroboration-only. An OLD event must
    still reject as too old, NOT pass through with reduced size."""
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="rule matched",
        components={}, observed=False, polarity_source="rules",
    )
    decision = await scout_candidate.evaluate(
        _event(severity=0.85, sources=["someblog"], age_seconds=5_000.0),
        _match(_market()), impact,
        cfg={"high_sev_solo_threshold": 0.80,
             "max_event_age_seconds": 1800.0,
             "min_edge": 0.05, "min_confidence": 0.40,
             "require_primary_or_corroboration": True},
    )
    assert decision.accepted is False
    assert "too old" in decision.reason


@pytest.mark.asyncio
async def test_normal_corroboration_path_does_not_set_high_sev_solo(temp_db):
    """Two-source event still goes through the normal path; no
    size penalty."""
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="rule matched",
        components={}, observed=False, polarity_source="rules",
    )
    decision = await scout_candidate.evaluate(
        _event(severity=0.85, sources=["src-a", "src-b"]),
        _match(_market()), impact,
    )
    assert decision.accepted is True
    assert decision.high_sev_solo is False
    assert decision.size_multiplier == 1.0


# ============================================================
# 7. Persistence fields in audit_snapshot
# ============================================================


@pytest.mark.asyncio
async def test_audit_snapshot_includes_first_seen_timestamp(temp_db):
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="rule matched",
        components={}, observed=False, polarity_source="rules",
    )
    event = _event()
    decision = await scout_candidate.evaluate(
        event, _match(_market()), impact,
    )
    snap = decision.impact_snapshot
    assert snap["event"]["first_seen_timestamp"] == event.first_seen_at
    assert snap["event"]["timestamp_detected"] == event.timestamp_detected
    assert snap["event"]["last_seen_timestamp"] == event.last_seen_at


@pytest.mark.asyncio
async def test_audit_snapshot_includes_snapshot_price_and_taken_at(temp_db):
    """The persistence contract for the planned Pattern Discovery
    layer. ``snapshot_price`` + ``snapshot_taken_at`` together are
    the JOIN key against price_ticks for "price after N min"."""
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="rule matched",
        components={}, observed=False, polarity_source="rules",
    )
    decision = await scout_candidate.evaluate(
        _event(), _match(_market(mid=0.42)), impact,
        market_mid=0.42,
    )
    snap = decision.impact_snapshot
    assert snap["market"]["snapshot_price"] == 0.42
    # snapshot_taken_at should be a recent unix timestamp.
    taken = snap["market"]["snapshot_taken_at"]
    assert abs(taken - time.time()) < 5.0
    # yes_token must round-trip (it's the price_ticks JOIN key).
    assert snap["market"]["yes_token"] == "yes-tok"


@pytest.mark.asyncio
async def test_audit_snapshot_polarity_source_field_present(temp_db):
    impact = ImpactScore(
        direction=+1, true_prob=0.55, confidence=0.50,
        expected_nudge=0.10, polarity_reasoning="llm: buy",
        components={}, observed=False, polarity_source="llm",
    )
    decision = await scout_candidate.evaluate(
        _event(), _match(_market()), impact,
    )
    assert decision.impact_snapshot["impact"]["polarity_source"] == "llm"
