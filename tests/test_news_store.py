"""Integration tests for news_store against an in-memory-ish SQLite.

Uses the project's own db.execute/fetch_* helpers pointed at a temp
file so the schema migration runs end to end and the enrichment
columns land exactly as they would in production.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.feeds import news_store
from core.utils import db as db_module
from core.utils.db import execute, fetch_one, init_db


@pytest.fixture
def temp_db(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point the DB helper at a fresh file for this test."""
    path = tmp_path / "t.db"
    monkeypatch.setattr(db_module, "_DB_PATH", path)
    monkeypatch.setattr(db_module, "_resolve_db_path", lambda: path)
    asyncio.run(init_db())
    return path


async def _insert_row(source: str, title: str, weight: float | None, ingested_at: float) -> int:
    rid = await execute(
        """INSERT INTO feed_items
           (url_hash, source, title, summary, url, published_at, ingested_at, meta, source_weight)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (f"h-{title}", source, title, title, f"http://x/{title}", ingested_at,
         ingested_at, "{}", weight),
    )
    return rid


def test_pending_and_mark_enriched(temp_db) -> None:
    async def go():
        await _insert_row("coindesk", "a", 0.75, 1000.0)
        await _insert_row("coindesk", "b", 0.75, 1001.0)
        pending = await news_store.pending_enrichment(limit=10)
        ids = [r["id"] for r in pending]
        assert len(ids) == 2
        # Rows must be plain dicts — the enricher calls .get() on them,
        # and sqlite3.Row doesn't support .get(). (Regression: live bot
        # spammed AttributeError on every cycle until this was enforced.)
        assert all(isinstance(r, dict) for r in pending)
        assert pending[0].get("title") == "a"
        assert pending[0].get("missing_key", "sentinel") == "sentinel"
        await news_store.mark_enriched(
            ids[0], {"tickers": ["BTC"], "sentiment": "bullish"}, 0.8,
        )
        remaining = await news_store.pending_enrichment(limit=10)
        assert len(remaining) == 1
        row = await fetch_one(
            "SELECT market_relevance, enriched_json FROM feed_items WHERE id=?",
            (ids[0],),
        )
        assert row["market_relevance"] == pytest.approx(0.8)
        assert json.loads(row["enriched_json"])["tickers"] == ["BTC"]

    asyncio.run(go())


def test_recent_enriched_filters_by_relevance(temp_db) -> None:
    async def go():
        a = await _insert_row("coindesk", "hi-rel", 0.75, 5000.0)
        b = await _insert_row("coindesk", "lo-rel", 0.75, 5001.0)
        c = await _insert_row("coindesk", "not-yet", 0.75, 5002.0)
        await news_store.mark_enriched(a, {"tickers": ["BTC"]}, 0.9)
        await news_store.mark_enriched(b, {"tickers": []}, 0.1)
        # unenriched `c` must never appear regardless of threshold
        hi = await news_store.recent_enriched(since_ts=0.0, min_relevance=0.5)
        assert [x["id"] for x in hi] == [a]
        all_ = await news_store.recent_enriched(since_ts=0.0, min_relevance=0.0)
        assert sorted(x["id"] for x in all_) == sorted([a, b])
        # enriched json round-trips back into a dict
        assert hi[0]["enriched"]["tickers"] == ["BTC"]

    asyncio.run(go())


def test_enrichment_summary_counts(temp_db) -> None:
    async def go():
        now = 10_000.0
        a = await _insert_row("coindesk", "one", 0.75, now - 10)
        b = await _insert_row("coindesk", "two", 0.75, now - 20)
        c = await _insert_row("coindesk", "three", 0.75, now - 30)  # stays NULL
        await news_store.mark_enriched(a, {}, 0.7)
        await news_store.mark_enriched(b, {}, 0.2)
        summary = await news_store.enrichment_summary_24h(now)
        assert summary["total"] == 3
        assert summary["enriched"] == 2
        assert summary["high_rel"] == 1
        assert summary["backlog"] == 1  # row c stays NULL; backfill happens in background

    asyncio.run(go())
