"""Price-band pre-filter tests for the scalping lane.

Background: ``min_edge`` is an absolute threshold (default 0.04). On a
market trading at ``mid=0.001`` even a 30x mispricing call only buys
``0.03`` of edge — mathematically below the gate. The April 2026
post-publisher-fix soak showed 100% of strong-tier scoring events
landed at ``mid<0.01`` or ``mid>=0.99`` (lottery-tail "Will X happen
by date Y?" markets), and the lane entered zero trades despite the
LLM producing structured, honest output. The price-band pre-filter
drops those tails BEFORE any LLM call so the lane spends its scoring
budget on markets where 4¢ absolute edge is reachable.

These tests pin the boundary semantics (``min_mid <= mid <= max_mid``,
both inclusive), confirm out-of-band markets never reach the scoring
function, and verify the per-cycle log line emits accurate
kept/dropped counters.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from loguru import logger as loguru_logger

from core.execution import allocator, shadow
from core.strategies import scalping, scoring
from core.strategies.scoring import Score
from core.utils import db as db_module
from core.utils.db import execute
from core.utils.prices import PriceSnapshot


# ---------------------------------------------------------------------
# Fixtures + helpers (mirror tests/test_lane_integration.py shape)
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "scalp_band.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _iso_days_from_now(days: int) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + days * 86400)
    )


async def _insert_market(
    *, market_id: str, best_bid: float, best_ask: float,
    resolve_days: int = 7,
    tokens: tuple[str, str] = ("yes-tok", "no-tok"),
) -> None:
    await execute(
        """INSERT INTO markets
           (market_id, question, slug, category, active, close_time,
            token_ids, best_bid, best_ask, last_price, liquidity, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            market_id, f"Will {market_id} happen?", market_id, "politics",
            1, _iso_days_from_now(resolve_days),
            json.dumps(list(tokens)),
            best_bid, best_ask, (best_bid + best_ask) / 2,
            50_000.0, time.time(),
        ),
    )


async def _insert_two_evidence_rows(market_id: str) -> None:
    """Two distinct sources tag this market for STRONG-tier
    classification; without them the lane falls into WEAK or NONE
    and the price-band semantics get muddled by tier-specific
    thresholds."""
    for i, src in enumerate(("reuters", "bbc")):
        await execute(
            """INSERT INTO feed_items
               (url_hash, source, title, summary, url, ingested_at, meta)
               VALUES (?,?,?,?,?,?,?)""",
            (
                f"{market_id}-h-{i}", src, f"Headline {i} for {market_id}",
                "Supporting summary",
                f"http://x/{market_id}/{i}", time.time() - 60,
                json.dumps({"linked_market_id": market_id}),
            ),
        )


def _patch_downstream(monkeypatch, *, score_calls: list[str]) -> None:
    """Wire all the I/O-shaped helpers in scalping to deterministic
    in-band-friendly returns. The score mock records the market_id of
    every call so tests can assert that out-of-band markets never
    reach scoring (the price-band filter sits well upstream of it)."""

    async def fake_score_with_fallback(
        market, text, *, client=None, tier="fast", timeout_seconds=None,
    ):
        score_calls.append(market.market_id)
        # Pick a true_prob 0.05 away from the market mid so |edge|=0.05
        # >= min_edge=0.04 regardless of where mid is in [0,1]. Direction
        # flips near the upper end so we don't ever push true_prob > 1.
        mid = market.mid
        delta = 0.05 if mid <= 0.5 else -0.05
        return Score(
            true_prob=max(0.0, min(1.0, mid + delta)),
            confidence=0.75, reasoning="mock", source="ollama",
        )

    async def fake_volume(market_id):
        return 50_000.0

    async def fake_current_price(market_id, token_id):
        return PriceSnapshot(
            token_id=token_id, bid=0.49, ask=0.51, last=0.50, ts=time.time(),
            source="ticks",
        )

    monkeypatch.setattr(scoring, "score_with_fallback", fake_score_with_fallback)
    monkeypatch.setattr(scalping, "volume_24h", fake_volume)
    monkeypatch.setattr(scalping, "current_price", fake_current_price)


# ---------------------------------------------------------------------
# Boundary semantics: <= min_mid filtered, >= max_mid filtered
# ---------------------------------------------------------------------


@pytest.mark.parametrize("market_id, best_bid, best_ask, expected_entered", [
    # Inside the band — should reach scoring + enter.
    ("midband",       0.49,   0.51,   1),
    # Lower boundary inclusive: mid == min_mid → passes.
    ("low_boundary",  0.04,   0.06,   1),  # mid = 0.05
    # Upper boundary inclusive: mid == max_mid → passes.
    ("high_boundary", 0.94,   0.96,   1),  # mid = 0.95
    # Just below lower boundary → filtered.
    ("just_below",    0.03,   0.05,   0),  # mid = 0.04
    # Just above upper boundary → filtered.
    ("just_above",    0.95,   0.97,   0),  # mid = 0.96
    # Deep tail (lottery): mid ≈ 0.001 → filtered.
    ("deep_low",      0.001,  0.002,  0),  # mid = 0.0015
    # Near-settled: mid ≈ 0.999 → filtered.
    ("deep_high",     0.998,  0.999,  0),  # mid = 0.9985
])
@pytest.mark.asyncio
async def test_price_band_boundary_semantics(
    monkeypatch, market_id, best_bid, best_ask, expected_entered,
):
    """One market per parameter set: assert the filter accepts/rejects
    by the correct boundary rule (inclusive on both ends) and that the
    lane returns ``entered=expected_entered``. Out-of-band markets must
    NOT reach the scoring function (the filter is an upstream gate, not
    a post-scoring veto)."""
    await allocator.init_lane_capital()
    await _insert_market(
        market_id=market_id, best_bid=best_bid, best_ask=best_ask,
    )
    await _insert_two_evidence_rows(market_id)

    score_calls: list[str] = []
    _patch_downstream(monkeypatch, score_calls=score_calls)

    lane = scalping.ScalpingLane()
    entered = await lane.scan_once()

    assert entered == expected_entered, (
        f"market_id={market_id} mid={(best_bid+best_ask)/2:.4f} "
        f"expected entered={expected_entered}, got {entered}"
    )
    if expected_entered == 0:
        # The filter is upstream of scoring — out-of-band markets must
        # never reach the LLM call. (Otherwise we're saving zero
        # inference budget.)
        assert market_id not in score_calls, (
            f"out-of-band market {market_id} reached scoring — filter "
            "is positioned downstream of where it should be"
        )
    else:
        assert market_id in score_calls


# ---------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_override_narrows_band(monkeypatch):
    """Setting ``price_band: {min_mid: 0.10, max_mid: 0.80}`` in config
    should drop a market at mid=0.05 (which would pass the default
    band) and keep a market at mid=0.50."""
    await allocator.init_lane_capital()
    await _insert_market(
        market_id="below_narrow", best_bid=0.04, best_ask=0.06,
    )  # mid = 0.05
    await _insert_market(
        market_id="inside_narrow", best_bid=0.49, best_ask=0.51,
    )  # mid = 0.50
    await _insert_two_evidence_rows("below_narrow")
    await _insert_two_evidence_rows("inside_narrow")

    score_calls: list[str] = []
    _patch_downstream(monkeypatch, score_calls=score_calls)

    # Override the scalping config with a narrower band by mutating
    # the loaded data dict in-place. monkeypatch's setattr restores
    # the original value at teardown.
    from core.utils import config as config_module
    cfg = config_module.get_config()
    original_data = cfg._data
    new_data = dict(original_data)
    new_scalping = dict(new_data.get("scalping") or {})
    new_scalping["price_band"] = {"min_mid": 0.10, "max_mid": 0.80}
    new_data["scalping"] = new_scalping
    monkeypatch.setattr(cfg, "_data", new_data)

    lane = scalping.ScalpingLane()
    entered = await lane.scan_once()

    assert entered == 1, (
        f"narrowed band should drop mid=0.05 and keep mid=0.50; "
        f"got entered={entered}, score_calls={score_calls}"
    )
    assert "below_narrow" not in score_calls
    assert "inside_narrow" in score_calls


# ---------------------------------------------------------------------
# Per-cycle log line
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_line_reports_kept_and_dropped_counts(monkeypatch):
    """The INFO log line ``[scalping] price_band_filter: kept=X
    dropped_low_mid=Y dropped_high_mid=Z`` must reflect the true
    in-window pool — including drops past ``scan_cap`` so the
    operator gets a full-fidelity signal, not just the prefix the
    loop would otherwise have looked at."""
    await allocator.init_lane_capital()
    # 2 in-band, 3 below band, 1 above band — all date-window-eligible.
    await _insert_market(market_id="kept_a",  best_bid=0.49, best_ask=0.51)
    await _insert_market(market_id="kept_b",  best_bid=0.39, best_ask=0.41)
    await _insert_market(market_id="low_a",   best_bid=0.001, best_ask=0.002)
    await _insert_market(market_id="low_b",   best_bid=0.005, best_ask=0.010)
    await _insert_market(market_id="low_c",   best_bid=0.020, best_ask=0.030)
    await _insert_market(market_id="high_a",  best_bid=0.998, best_ask=0.999)

    for mid in ("kept_a", "kept_b"):
        await _insert_two_evidence_rows(mid)

    score_calls: list[str] = []
    _patch_downstream(monkeypatch, score_calls=score_calls)

    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda m: captured.append(str(m)), level="INFO",
    )
    try:
        lane = scalping.ScalpingLane()
        await lane.scan_once()
    finally:
        loguru_logger.remove(sink_id)

    band_lines = [m for m in captured if "price_band_filter" in m]
    assert band_lines, (
        f"expected a price_band_filter log line, got captured={captured!r}"
    )
    line = band_lines[0]
    assert "kept=2" in line, line
    assert "dropped_low_mid=3" in line, line
    assert "dropped_high_mid=1" in line, line


# ---------------------------------------------------------------------
# Sanity: empty pool is silent (don't spam logs when there's nothing)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_line_omitted_when_pool_empty(monkeypatch):
    """If list_active returns nothing (or nothing in the date window),
    we shouldn't bother emitting the price-band log line. Operator
    has bigger problems to look at."""
    await allocator.init_lane_capital()
    # Insert a market OUTSIDE the date window (resolve_days=1, lane
    # window starts at min_resolve_days=2). Pool has it, but the
    # date filter drops it before the band filter sees it.
    await _insert_market(
        market_id="too_close", best_bid=0.49, best_ask=0.51, resolve_days=1,
    )
    await _insert_two_evidence_rows("too_close")

    score_calls: list[str] = []
    _patch_downstream(monkeypatch, score_calls=score_calls)

    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda m: captured.append(str(m)), level="INFO",
    )
    try:
        lane = scalping.ScalpingLane()
        await lane.scan_once()
    finally:
        loguru_logger.remove(sink_id)

    band_lines = [m for m in captured if "price_band_filter" in m]
    assert not band_lines, (
        "log line should be suppressed when no market reached the "
        f"price-band stage; got: {band_lines!r}"
    )
