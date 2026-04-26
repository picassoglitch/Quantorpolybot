"""GDELT Doc 2.0 article parser tests.

Pure parse function — no HTTP, no async loop. Verifies field extraction,
domain-tier confidence, language penalty, entity extraction from title,
and graceful handling of missing fields.
"""

from __future__ import annotations

from core.feeds.gdelt import (
    _entities_from_title,
    _domain_tier,
    _parse_gdelt_seendate,
    parse_gdelt_article,
)
from core.scout.event import EventCategory


def test_parse_full_article():
    art = {
        "url": "https://www.reuters.com/world/some-story",
        "title": "Trump Survives Shooting at Pittsburgh Rally",
        "domain": "reuters.com",
        "language": "english",
        "seendate": "20260426010000",
        "sourcecountry": "US",
    }
    sig = parse_gdelt_article(art, EventCategory.SHOOTING)
    assert sig is not None
    assert sig.source == "gdelt"
    assert sig.url.startswith("https://www.reuters.com")
    assert sig.title.startswith("Trump")
    assert sig.category_hint == "shooting"
    # Tier-1 reuters.com -> 0.85; English -> no penalty.
    assert sig.confidence == 0.85
    assert "Trump" in sig.entities
    assert "Pittsburgh" in sig.entities or "Pittsburgh Rally" in sig.entities


def test_parse_returns_none_on_missing_url():
    art = {"title": "Some headline", "domain": "x.com"}
    assert parse_gdelt_article(art, EventCategory.SHOOTING) is None


def test_parse_returns_none_on_missing_title():
    art = {"url": "https://x.com/a", "domain": "x.com"}
    assert parse_gdelt_article(art, EventCategory.SHOOTING) is None


def test_non_english_article_gets_confidence_penalty():
    art = {
        "url": "https://www.reuters.com/x",
        "title": "Some headline",
        "domain": "reuters.com",
        "language": "spanish",
        "seendate": "20260426010000",
    }
    sig = parse_gdelt_article(art, EventCategory.SHOOTING)
    assert sig is not None
    # 0.85 (tier1) * 0.6 (non-english) = 0.51
    assert sig.confidence < 0.85
    assert sig.confidence == round(0.85 * 0.6, 3)


def test_unknown_domain_falls_to_low_tier():
    art = {
        "url": "https://obscureblog.example/p",
        "title": "X happened today",
        "domain": "obscureblog.example",
        "language": "english",
        "seendate": "20260426010000",
    }
    sig = parse_gdelt_article(art, EventCategory.SHOOTING)
    assert sig is not None
    # Default tier ~ 0.45.
    assert sig.confidence == 0.45


def test_domain_tier_handles_subdomains():
    """Tier matching must work on `news.bbc.co.uk` as well as `bbc.co.uk`."""
    assert _domain_tier("news.bbc.co.uk") == 0.85
    assert _domain_tier("bbc.co.uk") == 0.85
    assert _domain_tier("www.reuters.com") == 0.85


def test_seendate_parser_handles_malformed_input():
    """Bad seendates fall back to now() — never raises."""
    assert _parse_gdelt_seendate("") > 0
    assert _parse_gdelt_seendate("notadate") > 0
    assert _parse_gdelt_seendate("20260426010000") > 0


def test_entity_extractor_pulls_title_phrases():
    ents = _entities_from_title("Trump Meets President Biden in Washington")
    # "President" alone is a stop-title; the regex picks "President Biden"
    # though, which is acceptable. We assert at least one of the
    # interesting entities is present.
    assert "Trump" in ents
    assert any("Biden" in e for e in ents)
    assert "Washington" in ents


def test_entity_extractor_drops_pure_stop_titles():
    ents = _entities_from_title("The President spoke today")
    # Neither "The" nor "President" alone should appear; "President"
    # IS in the stop list. "Today" is lowercase → not extracted.
    assert "The" not in ents
    assert "President" not in ents
