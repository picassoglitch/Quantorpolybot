"""Evidence-tier classifier + scan_skips persistence tests.

Pure-function classify_evidence covers all four tier outputs. Async
record_skip / purge_skips_older_than tests round-trip through a tmp DB.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.strategies.evidence_tier import (
    EvidenceTier,
    classify_evidence,
    purge_skips_older_than,
    record_skip,
)
from core.utils import db as db_module


# ---------------- classify_evidence (pure) ----------------


def _item(source: str, age_seconds: float, *, now: float) -> dict:
    return {
        "id": 1,
        "source": source,
        "title": "x",
        "summary": "y",
        "url": "u",
        "ingested_at": now - age_seconds,
    }


def test_strong_with_two_distinct_fresh_sources():
    now = time.time()
    items = [
        _item("rss:bbc", 60, now=now),
        _item("polymarket_news", 120, now=now),
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.STRONG
    assert c.distinct_sources == 2
    assert c.freshest_age_seconds is not None and c.freshest_age_seconds <= 65


def test_weak_when_only_one_distinct_source():
    now = time.time()
    items = [_item("rss:bbc", 60, now=now), _item("rss:bbc", 30, now=now)]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.WEAK
    assert c.distinct_sources == 1
    assert c.total_items == 2


def test_weak_when_two_sources_but_all_stale():
    """Multi-source but freshest item is older than fresh_within_seconds
    -> WEAK (price has digested)."""
    now = time.time()
    items = [
        _item("rss:bbc", 8 * 3600, now=now),
        _item("polymarket_news", 9 * 3600, now=now),
    ]
    c = classify_evidence(
        items,
        strong_min_sources=2,
        weak_min_sources=1,
        fresh_within_seconds=6 * 3600.0,
        now=now,
    )
    assert c.tier is EvidenceTier.WEAK
    assert "stale" in c.reasoning or "old" in c.reasoning


def test_none_when_zero_sources():
    c = classify_evidence([], strong_min_sources=2, weak_min_sources=1)
    assert c.tier is EvidenceTier.NONE
    assert c.distinct_sources == 0
    assert c.total_items == 0


def test_items_without_source_are_ignored():
    """The classifier counts distinct *named* sources only — an item
    without a source field can't corroborate anything."""
    now = time.time()
    items = [
        {"id": 1, "title": "x", "summary": "y"},  # no source
        _item("rss:bbc", 60, now=now),
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.WEAK
    assert c.distinct_sources == 1


def test_missing_ingested_at_does_not_crash_strong_path():
    """Items lacking ingested_at: distinct-source count still works,
    but they can't make the strong-tier freshness bar."""
    items = [
        {"id": 1, "source": "rss:bbc", "title": "x"},
        {"id": 2, "source": "polymarket_news", "title": "y"},
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1)
    # No ingested_at = no freshness data = NOT strong (can't prove fresh).
    assert c.tier is EvidenceTier.WEAK


def test_strong_threshold_configurable():
    """Operator should be able to require 3 sources for strong."""
    now = time.time()
    items = [
        _item("a", 60, now=now),
        _item("b", 60, now=now),
    ]
    c = classify_evidence(items, strong_min_sources=3, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.WEAK


# ---------------- record_skip + purge (DB) ----------------


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "skips.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


@pytest.mark.asyncio
async def test_record_skip_writes_one_row(temp_db):
    await record_skip(
        lane="scalping",
        market_id="m-1",
        tier_attempted="strong",
        reject_reason="confidence 0.50 < 0.60",
        evidence_tier="strong",
        watchlist=False,
    )
    rows = await db_module.fetch_all("SELECT * FROM scan_skips")
    assert len(rows) == 1
    r = rows[0]
    assert r["lane"] == "scalping"
    assert r["market_id"] == "m-1"
    assert r["tier_attempted"] == "strong"
    assert r["reject_reason"].startswith("confidence")
    assert r["evidence_tier"] == "strong"
    assert r["watchlist"] == 0


@pytest.mark.asyncio
async def test_record_skip_serializes_score_snapshot(temp_db):
    await record_skip(
        lane="scalping",
        market_id="m-2",
        tier_attempted="microstructure",
        reject_reason="microstructure: insufficient signal",
        evidence_tier="none",
        watchlist=False,
        score_snapshot={"strength": 0.42, "direction": 1},
    )
    rows = await db_module.fetch_all("SELECT * FROM scan_skips")
    snap = rows[0]["score_snapshot"]
    assert snap is not None
    assert "strength" in snap and "0.42" in snap


@pytest.mark.asyncio
async def test_record_skip_does_not_throw_on_db_error(temp_db, monkeypatch):
    """If the INSERT fails the scan loop must NOT raise — just log and
    move on. This is a contract test: scoring path stability beats
    perfect skip-row coverage."""
    async def boom(*a, **k):
        raise RuntimeError("db gone")
    monkeypatch.setattr("core.strategies.evidence_tier.execute", boom)
    # Should not raise.
    await record_skip(
        lane="scalping",
        market_id="m-3",
        tier_attempted="strong",
        reject_reason="x",
        evidence_tier="strong",
    )


@pytest.mark.asyncio
async def test_watchlist_flag_persists_as_one(temp_db):
    await record_skip(
        lane="scalping",
        market_id="m-4",
        tier_attempted="weak",
        reject_reason="confidence 0.50 < 0.60",
        evidence_tier="weak",
        watchlist=True,
    )
    row = await db_module.fetch_one(
        "SELECT watchlist FROM scan_skips WHERE market_id=?", ("m-4",),
    )
    assert row["watchlist"] == 1


@pytest.mark.asyncio
async def test_purge_removes_old_rows_only(temp_db):
    now = time.time()
    # Old row: 10 days ago.
    await db_module.execute(
        """INSERT INTO scan_skips
             (scan_ts, lane, market_id, tier_attempted, reject_reason,
              evidence_tier, watchlist, score_snapshot)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (now - 10 * 86400, "scalping", "old", "strong", "x", "strong", 0, None),
    )
    # Fresh row: 1 hour ago.
    await db_module.execute(
        """INSERT INTO scan_skips
             (scan_ts, lane, market_id, tier_attempted, reject_reason,
              evidence_tier, watchlist, score_snapshot)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (now - 3600, "scalping", "new", "strong", "x", "strong", 0, None),
    )
    deleted = await purge_skips_older_than(7 * 86400.0)
    assert deleted == 1
    rows = await db_module.fetch_all("SELECT market_id FROM scan_skips")
    assert {r["market_id"] for r in rows} == {"new"}
