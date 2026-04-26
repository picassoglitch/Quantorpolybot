"""Tests for the PredictIt filter thread-offload (PR #6).

Two layers of coverage:

  1. Unit tests on ``_filter_predictit_markets_sync`` — the pure-CPU
     function that runs in the worker thread. Verifies it produces
     the same accepts / rejects the inline loop did, and that it
     touches NO async / DB / network code (the function is allowed
     to sit on a worker thread because it's pure).

  2. Integration test on ``_poll`` — verifies the function is
     actually dispatched via ``asyncio.to_thread`` (not just inlined),
     so a future refactor that re-inlines it gets caught.

Mocks httpx + the DB execute/fetch_one so no network or disk happens.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from core.feeds.predictit import (
    PredictItFeed,
    _MatchResult,
    _contract_price,
    _filter_predictit_markets_sync,
    _mid_from_row,
)


# ============================================================
# _filter_predictit_markets_sync — unit tests
# ============================================================


def _market(name: str, contracts: list[dict]) -> dict:
    return {"name": name, "contracts": contracts}


def _contract(name: str, price: float = 0.50) -> dict:
    return {"name": name, "bestBuyYesCost": price}


def test_filter_accepts_clean_match():
    """Same country + same election type → accept. Mirrors the
    matcher's `(country=usa, etype=presidential)` strong-anchor path."""
    markets = [
        _market(
            "2028 US Presidential Election",
            [_contract("Donald Trump", price=0.55)],
        ),
    ]
    poly_index = [
        ("poly-1", "Will Donald Trump win the US presidential election in 2028?", 0.50),
    ]
    matches, rejected = _filter_predictit_markets_sync(markets, poly_index)
    assert len(matches) == 1
    assert rejected == 0
    m = matches[0]
    assert m.poly_id == "poly-1"
    assert m.pi_price == 0.55
    assert m.poly_mid == 0.50
    assert abs(m.divergence - 0.05) < 1e-9
    assert "presidential" in m.match_reason or "trump" in m.match_reason.lower()


def test_filter_rejects_country_mismatch():
    """USA presidential vs Quebec presidential → reject."""
    markets = [
        _market(
            "Quebec Election",
            [_contract("Quebec Liberal Party", price=0.30)],
        ),
    ]
    poly_index = [
        ("poly-1", "Will the Republican party win the US presidential election?", 0.40),
    ]
    matches, rejected = _filter_predictit_markets_sync(markets, poly_index)
    assert matches == []
    assert rejected == 1


def test_filter_rejects_phase_mismatch():
    """PredictIt is overall winner; Polymarket asks round 1.
    Matcher's phase-mismatch rule should reject."""
    markets = [
        _market(
            "Colombia 2026 Presidential",
            [_contract("Gustavo Petro will win the election", price=0.60)],
        ),
    ]
    poly_index = [
        ("poly-1", "Will Gustavo Petro win round 1 of Colombia's 2026 presidential?", 0.40),
    ]
    matches, rejected = _filter_predictit_markets_sync(markets, poly_index)
    assert matches == []
    assert rejected == 1


def test_filter_skips_contracts_with_no_price():
    """Missing prices on a contract should silently skip it (not
    reject — the matcher never even runs)."""
    markets = [
        _market("X", [{"name": "Foo"}]),  # no price keys
    ]
    poly_index = [("poly-1", "Will Foo happen?", 0.5)]
    matches, rejected = _filter_predictit_markets_sync(markets, poly_index)
    assert matches == []
    assert rejected == 0


def test_filter_skips_match_when_polymid_is_none():
    """Polymarket entry with no usable mid → skip silently (not
    reject; we just can't compute divergence)."""
    markets = [
        _market(
            "2028 US Presidential",
            [_contract("Donald Trump", price=0.55)],
        ),
    ]
    poly_index = [
        ("poly-1", "Will Donald Trump win the US presidential election in 2028?", None),
    ]
    matches, rejected = _filter_predictit_markets_sync(markets, poly_index)
    assert matches == []
    assert rejected == 0


def test_filter_handles_empty_inputs_gracefully():
    matches, rejected = _filter_predictit_markets_sync([], [])
    assert matches == [] and rejected == 0
    matches, rejected = _filter_predictit_markets_sync([], [("p", "q", 0.5)])
    assert matches == [] and rejected == 0


def test_filter_handles_malformed_market_entries():
    """Non-dict markets / contracts must not crash the loop —
    PredictIt's API has historically returned mixed types. The
    valid contract inside the otherwise-mostly-garbage market
    must still be processed."""
    markets = [
        None,
        "not a dict",
        _market("2028 US Presidential Election",
                [None, "garbage",
                 _contract("Donald Trump", 0.55)]),
    ]
    poly_index = [
        ("poly-1", "Will Donald Trump win the US presidential election in 2028?", 0.50),
    ]
    matches, rejected = _filter_predictit_markets_sync(markets, poly_index)
    # The garbage entries are silently skipped (NOT counted as
    # rejects — they never reached the matcher), and the one valid
    # contract matches.
    assert len(matches) == 1


def test_filter_returns_match_result_dataclass():
    """The thread boundary should return frozen dataclasses, not
    bare tuples — frozen so the receiving coroutine can't accidentally
    mutate them."""
    markets = [
        _market("2028 US Presidential",
                [_contract("Donald Trump", 0.55)]),
    ]
    poly_index = [
        ("poly-1", "Will Donald Trump win the US presidential election in 2028?", 0.50),
    ]
    matches, _ = _filter_predictit_markets_sync(markets, poly_index)
    assert len(matches) == 1
    assert isinstance(matches[0], _MatchResult)
    # Frozen — attribute assignment must raise.
    with pytest.raises((AttributeError, TypeError)):
        matches[0].pi_price = 0.99  # type: ignore[misc]


def test_filter_does_not_touch_event_loop():
    """Contract: the filter must be safe to call from a thread. The
    tightest way to verify is to make sure it runs without an event
    loop AT ALL.

    `asyncio.to_thread` doesn't actually require this (it just runs
    the function in a worker thread), but this test enforces our
    intent: no awaitables, no `asyncio.run`, no `asyncio.get_event_loop()`.
    Done by running the filter inside a fresh thread that has no
    event loop attached.
    """
    import threading

    markets = [_market("2028 US Presidential",
                       [_contract("Donald Trump", 0.55)])]
    poly_index = [
        ("poly-1", "Will Donald Trump win the US presidential election in 2028?", 0.50),
    ]
    result_holder: dict = {}
    error_holder: dict = {}

    def runner():
        try:
            result_holder["m"], result_holder["r"] = (
                _filter_predictit_markets_sync(markets, poly_index)
            )
        except Exception as e:  # pragma: no cover — failure surfaces below
            error_holder["e"] = e

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=5.0)
    assert "e" not in error_holder, f"filter raised in thread: {error_holder['e']}"
    assert len(result_holder["m"]) == 1


# ============================================================
# Integration test — _poll uses asyncio.to_thread
# ============================================================


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def get(self, *args, **kwargs):
        return _FakeResponse(self._payload)


@pytest.mark.asyncio
async def test_poll_dispatches_filter_via_to_thread(monkeypatch):
    """Pin the contract: ``_poll`` must call
    ``asyncio.to_thread(_filter_predictit_markets_sync, ...)``. If a
    future refactor inlines the filter back onto the event loop, this
    test fails."""
    payload = {"markets": [{
        "name": "2028 US Presidential",
        "contracts": [{"name": "Donald Trump", "bestBuyYesCost": 0.55}],
    }]}
    client = _FakeClient(payload)

    # Stub the DB-read so we don't need a tmp DB. Returns one row that
    # matches the canned contract.
    async def fake_fetch_all(_sql):
        return [{
            "market_id": "poly-1",
            "question": "Will Donald Trump win the US presidential election in 2028?",
            "best_bid": 0.49, "best_ask": 0.51, "last_price": 0.50,
        }]

    # Stub the writes so they don't error in the test env.
    async def fake_execute(*a, **k):
        return 0

    async def fake_fetch_one(*a, **k):
        return None

    monkeypatch.setattr("core.feeds.predictit.fetch_all", fake_fetch_all)
    monkeypatch.setattr("core.feeds.predictit.execute", fake_execute)
    monkeypatch.setattr("core.feeds.predictit.fetch_one", fake_fetch_one)

    # Spy on asyncio.to_thread.
    real_to_thread = asyncio.to_thread
    calls: list[str] = []

    async def spy_to_thread(func, *args, **kwargs):
        calls.append(func.__name__)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

    feed = PredictItFeed()
    matched, signaled = await feed._poll(client, threshold=0.4, auto_div=0.01)

    assert "_filter_predictit_markets_sync" in calls, (
        "filter was not dispatched via asyncio.to_thread — main loop "
        "would block on the CPU-heavy match work"
    )
    assert matched == 1
    # Divergence is 0.05, auto_div 0.01 → signaled.
    assert signaled == 1


@pytest.mark.asyncio
async def test_poll_returns_zeros_for_empty_predictit_payload(monkeypatch):
    client = _FakeClient({"markets": []})

    async def fake_fetch_all(_sql):
        return [{"market_id": "x", "question": "y", "best_bid": 0.5,
                 "best_ask": 0.5, "last_price": 0.5}]
    monkeypatch.setattr("core.feeds.predictit.fetch_all", fake_fetch_all)

    feed = PredictItFeed()
    matched, signaled = await feed._poll(client, 0.4, 0.08)
    assert (matched, signaled) == (0, 0)


@pytest.mark.asyncio
async def test_poll_returns_zeros_when_no_active_polymarket(monkeypatch):
    payload = {"markets": [{
        "name": "X", "contracts": [{"name": "Y", "bestBuyYesCost": 0.5}],
    }]}
    client = _FakeClient(payload)

    async def fake_fetch_all(_sql):
        return []
    monkeypatch.setattr("core.feeds.predictit.fetch_all", fake_fetch_all)

    feed = PredictItFeed()
    matched, signaled = await feed._poll(client, 0.4, 0.08)
    assert (matched, signaled) == (0, 0)


# ============================================================
# Helpers — _contract_price + _mid_from_row coverage
# ============================================================


def test_contract_price_prefers_best_buy_then_falls_back():
    assert _contract_price({"bestBuyYesCost": 0.7}) == 0.7
    assert _contract_price({"lastTradePrice": 0.6}) == 0.6
    assert _contract_price({"lastClosePrice": 0.4}) == 0.4
    assert _contract_price({"bestBuyYesCost": 0.7, "lastTradePrice": 0.6}) == 0.7
    assert _contract_price({}) is None
    # Out-of-range values fall through to None.
    assert _contract_price({"bestBuyYesCost": 1.5}) is None
    assert _contract_price({"bestBuyYesCost": -0.1}) is None


def test_mid_from_row_prefers_midpoint_then_last():
    class _Row(dict):
        def __getitem__(self, k):
            return self.get(k)

    assert _mid_from_row(_Row(best_bid=0.4, best_ask=0.6, last_price=0.55)) == 0.5
    assert _mid_from_row(_Row(best_bid=0, best_ask=0, last_price=0.55)) == 0.55
    assert _mid_from_row(_Row(best_bid=0, best_ask=0, last_price=0)) is None
    # Crossed book → ignore the bid/ask, use last.
    assert _mid_from_row(_Row(best_bid=0.6, best_ask=0.4, last_price=0.55)) == 0.55
