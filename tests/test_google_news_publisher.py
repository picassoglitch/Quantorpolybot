"""Publisher extraction from Google News titles + classifier
distinct-count behaviour with the new ``meta.publisher`` key.

Google News appends the source outlet to every title with a literal
ASCII ``" - "`` separator (verified across a 12-title sample on
2026-04-26). Splitting it out at ingest time and storing the result
in ``meta.publisher`` is what lets the evidence classifier count
distinct *publishers* instead of the single feed name
``"google_news"`` — the bug that was collapsing every market to
``distinct_sources=1`` and starving the scalping lane of STRONG-tier
candidates.
"""

from __future__ import annotations

import time

import pytest

from core.feeds.google_news import _split_title_publisher
from core.strategies.evidence_tier import EvidenceTier, classify_evidence


# ============================================================
# _split_title_publisher
# ============================================================


@pytest.mark.parametrize("title, expected_title, expected_pub", [
    # Standard cases (these are real shapes seen in the live feed).
    (
        "World Cup draw: Profiles of England's group stage opponents - BBC",
        "World Cup draw: Profiles of England's group stage opponents",
        "BBC",
    ),
    (
        "Iraq 2-1 UAE (18 Nov, 2025) Game Analysis - ESPN",
        "Iraq 2-1 UAE (18 Nov, 2025) Game Analysis",
        "ESPN",
    ),
    (
        "BOJ keeps rate-hike door open even as Iran war squeezes firms - Reuters",
        "BOJ keeps rate-hike door open even as Iran war squeezes firms",
        "Reuters",
    ),
    # Multi-word publisher (whitespace inside the suffix is fine).
    (
        "Suriname hold Panama in Concacaf qualifying - Jamaica Observer",
        "Suriname hold Panama in Concacaf qualifying",
        "Jamaica Observer",
    ),
    # Title contains its own " - " — rsplit on the LAST one, not the first.
    (
        "Foo - Bar - Baz Publisher",
        "Foo - Bar",
        "Baz Publisher",
    ),
    # Embedded em-dash inside the headline; the separator we split on
    # is still the trailing ASCII " - ". (We confirmed in the live
    # sample that em-dash never appears as the separator itself.)
    (
        "BOJ keeps rate‑hike door open — even as firms squeeze - Reuters",
        "BOJ keeps rate‑hike door open — even as firms squeeze",
        "Reuters",
    ),
])
def test_split_title_publisher_extracts_expected_pair(title, expected_title, expected_pub):
    cleaned, publisher = _split_title_publisher(title)
    assert cleaned == expected_title
    assert publisher == expected_pub


@pytest.mark.parametrize("title", [
    "Headline with no separator at all",
    "EmDashOnly—NotASeparator",  # em-dash without surrounding spaces
    "Hyphen-only-but-no-spaces",
    "",
])
def test_split_title_publisher_returns_none_when_no_separator(title):
    cleaned, publisher = _split_title_publisher(title)
    assert cleaned == title
    assert publisher is None


def test_split_title_publisher_rejects_overlong_suffix():
    """A 70-char tail almost certainly means the ' - ' was inside the
    headline body, not a publisher attribution. Bail out and leave the
    title alone rather than chopping it."""
    long_suffix = "x" * 70
    title = f"Real headline body - {long_suffix}"
    cleaned, publisher = _split_title_publisher(title)
    assert publisher is None
    assert cleaned == title  # untouched


def test_split_title_publisher_handles_empty_suffix():
    """Trailing ' - ' with whitespace-only tail → no publisher."""
    cleaned, publisher = _split_title_publisher("Headline body - ")
    assert publisher is None
    assert cleaned == "Headline body - "


def test_split_title_publisher_handles_empty_prefix():
    """Defensive: a title that's nothing but ' - Publisher' shouldn't
    leave us with an empty headline. Keep the original."""
    cleaned, publisher = _split_title_publisher(" - Reuters")
    assert publisher is None
    assert cleaned == " - Reuters"


def test_split_title_publisher_strips_whitespace():
    """The split shouldn't leak surrounding whitespace into either
    side — common when an entry has stray padding."""
    cleaned, publisher = _split_title_publisher("  Headline  -   BBC  ")
    # rpartition matches " - " literally; the inner whitespace
    # collapses on the .strip() calls but the leading/trailing
    # whitespace on the title is preserved by rpartition's prefix.
    assert cleaned == "Headline"
    assert publisher == "BBC"


# ============================================================
# classify_evidence — distinctness now keys on meta.publisher
# ============================================================


def _gnews(publisher: str | None, age_seconds: float = 60, *, now: float) -> dict:
    """Build a synthetic feed_items row as it would land after the
    JSON-decode step in :func:`_recent_evidence_for`. ``meta`` is a
    dict (already decoded), matching what classify_evidence will see
    in production."""
    meta: dict = {"linked_market_id": "MKT_X", "query": "q"}
    if publisher is not None:
        meta["publisher"] = publisher
    return {
        "id": id(publisher) & 0xFFFF,  # arbitrary unique-ish int
        "source": "google_news",
        "title": "x",
        "summary": "y",
        "url": "u",
        "ingested_at": now - age_seconds,
        "meta": meta,
    }


def _other(source: str, age_seconds: float = 60, *, now: float) -> dict:
    """Non-google-news item: no meta.publisher, classifier should fall
    back to item.source."""
    return {
        "id": id(source) & 0xFFFF,
        "source": source,
        "title": "x",
        "summary": "y",
        "url": "u",
        "ingested_at": now - age_seconds,
        "meta": {"linked_market_id": "MKT_X"},
    }


def test_classifier_counts_five_distinct_publishers_as_five():
    """Five google_news items with five different publishers → STRONG
    with distinct=5. Without the publisher key these would have all
    collapsed to distinct=1 (the feed name)."""
    now = time.time()
    items = [
        _gnews("BBC", now=now),
        _gnews("Reuters", now=now),
        _gnews("Bloomberg", now=now),
        _gnews("CNBC", now=now),
        _gnews("ESPN", now=now),
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.STRONG
    assert c.distinct_sources == 5
    assert c.total_items == 5


def test_classifier_dedupes_repeat_publishers():
    """3 items, 2 from Reuters + 1 from Bloomberg → distinct=2."""
    now = time.time()
    items = [
        _gnews("Reuters", now=now),
        _gnews("Reuters", now=now),
        _gnews("Bloomberg", now=now),
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.STRONG
    assert c.distinct_sources == 2
    assert c.total_items == 3


def test_classifier_does_not_normalize_publisher_against_other_source():
    """Per spec: a google_news item with publisher 'Reuters' and a
    separate rss feed with source 'reuters_rss' should stay distinct
    (we don't normalize publisher names — that's a separate concern).

    Two items, two different keys → distinct=2."""
    now = time.time()
    items = [
        _gnews("Reuters", now=now),
        _other("reuters_rss", now=now),
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.STRONG
    assert c.distinct_sources == 2
    assert c.total_items == 2


def test_classifier_falls_back_to_source_when_no_publisher():
    """A google_news item without meta.publisher (legacy row that
    pre-dates this PR, or a title that didn't parse) falls back to
    item.source = 'google_news'. Three such items → distinct=1."""
    now = time.time()
    items = [
        _gnews(None, now=now),
        _gnews(None, now=now),
        _gnews(None, now=now),
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.WEAK  # only 1 distinct → weak, not strong
    assert c.distinct_sources == 1
    assert c.total_items == 3


def test_classifier_handles_string_meta_gracefully():
    """Defensive: if a caller passes the raw row from sqlite (meta as
    a JSON string, not a dict), the classifier must not crash — it
    just falls back to source."""
    now = time.time()
    items = [{
        "id": 1,
        "source": "google_news",
        "title": "x",
        "summary": "y",
        "url": "u",
        "ingested_at": now - 60,
        "meta": '{"publisher": "Reuters"}',  # str, not dict
    }]
    # Should not raise; with meta-as-string the publisher path is
    # skipped and we fall through to source = 'google_news'.
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.distinct_sources == 1
    assert c.total_items == 1


def test_classifier_mixed_google_and_rss_with_publishers_count_distinct():
    """Realistic mixed pull: 2 google_news (Reuters, BBC) + 1 rss feed
    (rss:nyt). All three count as distinct → STRONG."""
    now = time.time()
    items = [
        _gnews("Reuters", now=now),
        _gnews("BBC", now=now),
        _other("rss:nyt", now=now),
    ]
    c = classify_evidence(items, strong_min_sources=2, weak_min_sources=1, now=now)
    assert c.tier is EvidenceTier.STRONG
    assert c.distinct_sources == 3
    assert c.total_items == 3
