"""End-to-end lane integration tests.

One test per lane that mocks feeds/prices/scoring and walks a single
position through entry -> monitor -> exit, asserting the correct
close_reason fires. Not a full scheduler run — just 'does this lane's
loop actually work end to end when wired up?'
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from core.execution import allocator, shadow
from core.markets.cache import Market
from core.strategies import event_sniper, longshot, scalping, scoring
from core.strategies.scoring import Score
from core.utils import db as db_module
from core.utils.db import execute
from core.utils.prices import PriceSnapshot


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "lanes.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


def _iso_days_from_now(days: int) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + days * 86400)
    )


async def _insert_market(
    *,
    market_id: str,
    best_bid: float,
    best_ask: float,
    resolve_days: int,
    tokens: tuple[str, str] = ("yes-tok", "no-tok"),
) -> None:
    await execute(
        """INSERT INTO markets
           (market_id, question, slug, category, active, close_time,
            token_ids, best_bid, best_ask, last_price, liquidity, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            market_id,
            f"Will {market_id} happen?",
            market_id,
            "politics",
            1,
            _iso_days_from_now(resolve_days),
            json.dumps(list(tokens)),
            best_bid,
            best_ask,
            (best_bid + best_ask) / 2,
            50_000.0,
            time.time(),
        ),
    )


# =============================================================================
# Scalping integration: tight-spread market -> +8% move -> take_profit exit
# =============================================================================


@pytest.mark.asyncio
async def test_scalping_lane_enters_and_take_profit_exits(monkeypatch):
    await allocator.init_lane_capital()

    mid = 0.50
    market_id = "scalp-m1"
    await _insert_market(
        market_id=market_id, best_bid=0.49, best_ask=0.51, resolve_days=7,
    )

    # Scalp requires >=2 distinct evidence sources.
    for i, src in enumerate(("reuters", "bbc")):
        await execute(
            """INSERT INTO feed_items
               (url_hash, source, title, summary, url, ingested_at, meta)
               VALUES (?,?,?,?,?,?,?)""",
            (
                f"h-{i}", src, f"Headline {i}", "Supporting summary",
                f"http://x/{i}", time.time() - 60,
                json.dumps({"linked_market_id": market_id}),
            ),
        )

    # Mock the slow/external calls: scoring, volume, and price snapshots.
    async def fake_score(market, text, client=None, tier="deep"):
        return Score(true_prob=0.60, confidence=0.75, reasoning="mock", source="ollama")

    async def fake_volume(market_id):
        return 50_000.0

    async def fake_current_price(market_id, token_id):
        return PriceSnapshot(
            token_id=token_id, bid=0.49, ask=0.51, last=0.50, ts=time.time(),
            source="ticks",
        )

    async def fake_live_orderbook(market_id, token_id):
        # Tight spread so the liquidity-exit rule doesn't fire.
        return PriceSnapshot(
            token_id=token_id, bid=0.49, ask=0.51, last=0.50, ts=time.time(),
            source="gamma",
        )

    monkeypatch.setattr(scoring, "score", fake_score)
    monkeypatch.setattr(scalping, "volume_24h", fake_volume)
    monkeypatch.setattr(scalping, "current_price", fake_current_price)
    monkeypatch.setattr(scalping, "live_orderbook_snapshot", fake_live_orderbook)

    lane = scalping.ScalpingLane()
    entered = await lane.scan_once()
    assert entered == 1

    positions = await shadow.open_positions_for("scalping")
    assert len(positions) == 1
    pos = positions[0]
    assert pos.side == "BUY"
    assert pos.size_usd == pytest.approx(75.0)

    # Now move price up ~12% (ask 0.51 entry -> bid 0.58 exit = +13.7%).
    async def fake_price_up(market_id, token_id):
        return PriceSnapshot(
            token_id=token_id, bid=0.58, ask=0.60, last=0.59, ts=time.time(),
            source="ticks",
        )

    monkeypatch.setattr(scalping, "current_price", fake_price_up)

    closed = await lane.monitor_once()
    assert closed == 1
    positions_after = await shadow.open_positions_for("scalping")
    assert positions_after == []

    # Confirm close_reason is take_profit.
    rows = await shadow.all_open_positions()
    # Fetch the now-closed record directly.
    from core.utils.db import fetch_one
    row = await fetch_one(
        "SELECT close_reason FROM shadow_positions WHERE id=?", (pos.id,),
    )
    assert row["close_reason"].startswith("take_profit")


# =============================================================================
# Event sniping integration: fresh feed item -> heuristic fallback -> TP exit
# =============================================================================


@pytest.mark.asyncio
async def test_event_lane_heuristic_fallback_entry_and_tp_exit(monkeypatch):
    await allocator.init_lane_capital()

    market_id = "event-m1"
    await _insert_market(
        market_id=market_id, best_bid=0.39, best_ask=0.41, resolve_days=3,
    )

    # Feed item with linked_market_id -> direct match path.
    feed_id = await execute(
        """INSERT INTO feed_items
           (url_hash, source, title, summary, url, ingested_at, meta)
           VALUES (?,?,?,?,?,?,?)""",
        (
            "ev-hash-1", "reuters",
            "Candidate wins primary in shock result",
            "The frontrunner surges ahead and confirms victory.",
            "http://x/ev1",
            time.time() - 30,   # very fresh
            json.dumps({"linked_market_id": market_id}),
        ),
    )

    # Force the scoring to return the heuristic-source path (Ollama "timed out").
    # The event lane allows heuristic entries regardless of min_confidence.
    async def fake_score_with_timeout(market, text, timeout_seconds, client=None, tier="fast"):
        return Score(
            true_prob=0.60,  # edge = 0.60 - 0.40 = 0.20 > min_edge 0.10
            confidence=0.65,
            reasoning="heuristic fallback used",
            source="heuristic",
        )

    async def fake_volume(market_id):
        return 20_000.0

    async def fake_current_price(market_id, token_id):
        return PriceSnapshot(
            token_id=token_id, bid=0.39, ask=0.41, last=0.40, ts=time.time(),
            source="ticks",
        )

    monkeypatch.setattr(scoring, "score_with_timeout", fake_score_with_timeout)
    monkeypatch.setattr(event_sniper, "volume_24h", fake_volume)
    monkeypatch.setattr(event_sniper, "current_price", fake_current_price)

    lane = event_sniper.EventSniperLane()
    await lane._init_cursor()
    # Push cursor back so the newly inserted item is picked up.
    lane._cursor_id = 0

    processed = await lane._drain_new_items()
    assert processed >= 1

    positions = await shadow.open_positions_for("event_sniping")
    assert len(positions) == 1
    pos = positions[0]
    # Heuristic path forces the configured fallback size ($50).
    assert pos.size_usd == pytest.approx(50.0)
    assert pos.side == "BUY"

    # Move price up enough to trigger the +20% TP.
    async def fake_price_up(market_id, token_id):
        # BUY at 0.41 entry, bid jumps to 0.55 -> (0.55-0.41)/0.41 = 34%
        return PriceSnapshot(
            token_id=token_id, bid=0.55, ask=0.57, last=0.56, ts=time.time(),
            source="ticks",
        )

    monkeypatch.setattr(event_sniper, "current_price", fake_price_up)

    closed = await lane.monitor_once()
    assert closed == 1

    from core.utils.db import fetch_one
    row = await fetch_one(
        "SELECT close_reason FROM shadow_positions WHERE id=?", (pos.id,),
    )
    assert row["close_reason"].startswith("take_profit")


# =============================================================================
# Longshot integration: cheap mispricing -> 2x surge -> take_profit exit
# =============================================================================


@pytest.mark.asyncio
async def test_longshot_lane_enters_and_price_surge_tp_exits(monkeypatch):
    await allocator.init_lane_capital()

    market_id = "longshot-m1"
    # Cheap market: mid = 0.09. 45-day resolution sits comfortably inside
    # the configured [30, 180] longshot window.
    await _insert_market(
        market_id=market_id, best_bid=0.08, best_ask=0.10, resolve_days=45,
    )

    # Hardened longshot gates require ≥2 independent evidence sources
    # linked to the market. Insert two.
    for i, src in enumerate(("reuters", "bbc")):
        await execute(
            """INSERT INTO feed_items
               (url_hash, source, title, summary, url, ingested_at, meta)
               VALUES (?,?,?,?,?,?,?)""",
            (
                f"ls-h-{i}", src, f"Longshot headline {i}",
                "Supporting summary for the thesis.",
                f"http://x/ls/{i}", time.time() - 60,
                json.dumps({"linked_market_id": market_id}),
            ),
        )

    async def fake_score(market, text, client=None, tier="deep"):
        # True prob 0.25 >> 2x * 0.09 = 0.18. Well above the edge multiple.
        # Reasoning includes a named catalyst ("March 15 FOMC") so the
        # catalyst guard passes.
        return Score(
            true_prob=0.25, confidence=0.65,
            reasoning="catalyst: March 15 FOMC ruling drives repricing",
            source="ollama",
        )

    async def fake_volume(market_id):
        return 2_000.0

    async def fake_current_price(market_id, token_id):
        return PriceSnapshot(
            token_id=token_id, bid=0.08, ask=0.10, last=0.09, ts=time.time(),
            source="ticks",
        )

    monkeypatch.setattr(scoring, "score", fake_score)
    monkeypatch.setattr(longshot, "volume_24h", fake_volume)
    monkeypatch.setattr(longshot, "current_price", fake_current_price)

    lane = longshot.LongshotLane()
    entered = await lane.scan_once()
    assert entered == 1

    positions = await shadow.open_positions_for("longshot")
    assert len(positions) == 1
    pos = positions[0]
    assert pos.size_usd == pytest.approx(25.0)
    assert pos.side == "BUY"

    # Simulate a price surge to 2.2x entry: entry ~0.10 -> mid 0.22+.
    async def fake_price_surge(market_id, token_id):
        return PriceSnapshot(
            token_id=token_id, bid=0.22, ask=0.24, last=0.23, ts=time.time(),
            source="ticks",
        )

    monkeypatch.setattr(longshot, "current_price", fake_price_surge)

    closed = await lane.monitor_once()
    assert closed == 1

    from core.utils.db import fetch_one
    row = await fetch_one(
        "SELECT close_reason FROM shadow_positions WHERE id=?", (pos.id,),
    )
    assert row["close_reason"].startswith("take_profit")
