"""Tests for rss.py normalisation + poll_one with a mocked feedparser."""

from __future__ import annotations

import asyncio

import pytest

from core.feeds.rss import FeedSpec, RSSFeed, _normalise, configured_feeds
from core.utils import db as db_module
from core.utils.db import execute, fetch_all, init_db


def test_normalise_bare_url() -> None:
    spec = _normalise("https://example.com/feed.xml")
    assert isinstance(spec, FeedSpec)
    assert spec.url == "https://example.com/feed.xml"
    assert spec.name == "example.com"
    assert spec.weight == 0.5


def test_normalise_dict_clamps_weight() -> None:
    spec = _normalise({"name": "x", "url": "https://x", "weight": 3.0})
    assert spec.weight == 1.0
    spec = _normalise({"name": "x", "url": "https://x", "weight": -1})
    assert spec.weight == 0.0


def test_configured_feeds_returns_defaults_when_config_missing() -> None:
    feeds = configured_feeds()
    assert len(feeds) >= 10
    assert all(isinstance(f, FeedSpec) for f in feeds)
    # Ensure the name+weight scheme applied (no raw URL names leaking through)
    assert any(f.name == "coindesk" for f in feeds)
    assert all(0.0 <= f.weight <= 1.0 for f in feeds)


@pytest.fixture
def temp_db(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "t.db"
    monkeypatch.setattr(db_module, "_DB_PATH", path)
    monkeypatch.setattr(db_module, "_resolve_db_path", lambda: path)
    asyncio.run(init_db())
    return path


class _FakeEntry(dict):
    def __init__(self, **kw):
        super().__init__(**kw)

    def get(self, key, default=None):  # feedparser entries are dict-like
        return super().get(key, default)


def _fake_parsed(entries: list[dict]):
    class _Parsed:
        pass
    p = _Parsed()
    p.entries = [_FakeEntry(**e) for e in entries]
    return p


def test_poll_one_dedups_and_writes_weight(temp_db, monkeypatch: pytest.MonkeyPatch) -> None:
    async def go():
        feed = RSSFeed()
        spec = FeedSpec(name="coindesk", url="https://fake", weight=0.75)

        parsed = _fake_parsed([
            {"link": "https://fake/a", "title": "A", "summary": "summary A" * 100},
            {"link": "https://fake/b", "title": "B", "summary": "short"},
            {"link": "", "title": "no-link"},  # dropped
        ])
        # Patch feedparser.parse (called via run_in_executor)
        monkeypatch.setattr(
            "core.feeds.rss.feedparser.parse", lambda url: parsed,
        )

        new = await feed._poll_one(spec)
        assert new == 2

        # Re-polling the same feed inserts zero (dedup via url_hash).
        monkeypatch.setattr(
            "core.feeds.rss.feedparser.parse", lambda url: parsed,
        )
        again = await feed._poll_one(spec)
        assert again == 0

        rows = await fetch_all(
            "SELECT source, source_weight, summary FROM feed_items ORDER BY id"
        )
        assert len(rows) == 2
        assert all(r["source"] == "coindesk" for r in rows)
        assert all(r["source_weight"] == pytest.approx(0.75) for r in rows)
        # Summary A exceeds 500 chars — must be truncated with ellipsis.
        long_row = next(r for r in rows if len(r["summary"]) > 100)
        assert long_row["summary"].endswith("…")
        assert len(long_row["summary"]) <= 500

    asyncio.run(go())


def test_poll_one_failures_dont_kill_cycle(temp_db, monkeypatch: pytest.MonkeyPatch) -> None:
    """One dead feed must not take down the others. We call poll_one
    directly with a raising parser and confirm the exception bubbles —
    the supervisor loop in run() is what swallows it in production."""
    async def go():
        feed = RSSFeed()
        spec = FeedSpec(name="dead", url="https://dead", weight=0.1)

        def boom(url):
            raise ConnectionError("simulated network down")
        monkeypatch.setattr("core.feeds.rss.feedparser.parse", boom)

        with pytest.raises(ConnectionError):
            await feed._poll_one(spec)
        # No rows written.
        rows = await fetch_all("SELECT COUNT(*) AS n FROM feed_items")
        assert rows[0]["n"] == 0

    asyncio.run(go())
