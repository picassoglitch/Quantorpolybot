"""Tests for the JSON parser + field normaliser in enrich.py.

The Ollama HTTP call itself is mocked with a fake httpx transport so we
can exercise the parse-and-persist path end-to-end without a running
Ollama server.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.feeds.enrich import NewsEnricher, _normalise_fields, _parse_json


# ---------- _parse_json ----------

def test_parse_json_direct() -> None:
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_with_markdown_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert _parse_json(raw) == {"a": 1}


def test_parse_json_returns_none_on_garbage() -> None:
    assert _parse_json("") is None
    assert _parse_json("not json at all") is None


def test_parse_json_returns_none_on_non_object() -> None:
    # top-level array — enricher expects an object
    assert _parse_json("[1,2,3]") is None


# ---------- _normalise_fields ----------

def test_normalise_clamps_relevance_and_filters_topics() -> None:
    out = _normalise_fields({
        "tickers": ["btc", "eth", "", None, 123],
        "topics": ["regulation", "made_up_topic", "macro"],
        "sentiment": "BULLISH",
        "market_relevance": 1.7,
    })
    assert out["tickers"] == ["BTC", "ETH", "123"]
    assert out["topics"] == ["regulation", "macro"]
    assert out["sentiment"] == "bullish"
    assert out["market_relevance"] == 1.0


def test_normalise_defaults_when_missing() -> None:
    out = _normalise_fields({})
    assert out["tickers"] == []
    assert out["topics"] == []
    assert out["sentiment"] == "neutral"
    assert out["market_relevance"] == 0.0


def test_normalise_invalid_sentiment_becomes_neutral() -> None:
    out = _normalise_fields({"sentiment": "euphoric"})
    assert out["sentiment"] == "neutral"


def test_normalise_relevance_non_numeric_becomes_zero() -> None:
    out = _normalise_fields({"market_relevance": "very high"})
    assert out["market_relevance"] == 0.0


# ---------- end-to-end with mocked Ollama ----------

@pytest.mark.asyncio
async def test_one_calls_ollama_and_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``mark_enriched`` so we can assert the enricher passes the
    parsed JSON through. Stub httpx so no real network is needed."""
    captured: dict = {}

    async def fake_mark(fid, enriched, relevance):
        captured["fid"] = fid
        captured["enriched"] = enriched
        captured["relevance"] = relevance

    monkeypatch.setattr("core.feeds.enrich.mark_enriched", fake_mark)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "response": json.dumps({
                "tickers": ["BTC"],
                "topics": ["etf"],
                "sentiment": "bullish",
                "market_relevance": 0.82,
            }),
        })

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://fake") as client:
        sem = asyncio.Semaphore(1)
        await NewsEnricher()._one(
            client, sem, "fake-model",
            {"id": 42, "title": "SEC greenlights spot BTC ETF", "summary": "...",
             "source": "coindesk", "source_weight": 0.75},
        )
    assert captured["fid"] == 42
    assert captured["relevance"] == 0.82
    assert captured["enriched"]["tickers"] == ["BTC"]
    assert captured["enriched"]["topics"] == ["etf"]


@pytest.mark.asyncio
async def test_one_handles_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def fake_mark(fid, enriched, relevance):
        captured["fid"] = fid
        captured["relevance"] = relevance
        captured["enriched"] = enriched

    monkeypatch.setattr("core.feeds.enrich.mark_enriched", fake_mark)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "I'm just a chatty model 😅"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://fake") as client:
        sem = asyncio.Semaphore(1)
        await NewsEnricher()._one(
            client, sem, "fake-model",
            {"id": 7, "title": "t", "summary": "s", "source": "x", "source_weight": 0.5},
        )
    # Parse fail → relevance 0 and the error field is recorded.
    assert captured["relevance"] == 0.0
    assert captured["enriched"] == {"error": "parse_fail"}


@pytest.mark.asyncio
async def test_run_yields_when_watchdog_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: enrichment must pause when the watchdog flags
    Ollama/event-loop contention, otherwise its batches starve the
    scoring hot path (observed live: 258s Ollama silence). We confirm
    ``pending_enrichment`` is NOT called while degraded, and that it
    resumes as soon as the flag clears."""
    from core.feeds import enrich as enrich_mod

    calls = {"pending": 0}

    async def fake_pending(limit):
        calls["pending"] += 1
        # Signal stop after the first post-degraded call so the loop
        # exits cleanly.
        enricher._stop.set()
        return []

    # Return degraded=True for the first 3 ticks, then healthy.
    states = iter([True, True, True, False, False, False, False])
    monkeypatch.setattr(enrich_mod, "is_degraded", lambda: next(states, False))
    monkeypatch.setattr(enrich_mod, "pending_enrichment", fake_pending)
    # Config: zero-ish polls so the test runs fast.
    monkeypatch.setattr(
        "core.feeds.enrich.get_config",
        lambda: type("C", (), {"get": lambda self, *a, **k: {
            "enabled": True, "model": "m", "concurrency": 1, "batch_size": 1,
            "idle_poll_seconds": 0.01, "busy_poll_seconds": 0.01,
        }})(),
    )

    enricher = enrich_mod.NewsEnricher()
    await asyncio.wait_for(enricher.run(), timeout=2.0)

    # While degraded (3 ticks) no pending query fires; once healthy,
    # exactly one pending call runs before the stop flag trips.
    assert calls["pending"] == 1
