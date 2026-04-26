"""Category inference from market question text."""

from __future__ import annotations

import pytest

from core.markets.categorize import infer_category


@pytest.mark.parametrize(
    "text",
    [
        "Will the Lakers win the NBA Finals this year?",
        "NFL MVP: Patrick Mahomes vs Jalen Hurts",
        "Will Manchester City win the Premier League?",
        "Will Novak Djokovic win Wimbledon 2026?",
    ],
)
def test_sports(text: str) -> None:
    assert infer_category(text) == "sports"


@pytest.mark.parametrize(
    "text",
    [
        "Will Donald Trump be the Republican presidential nominee?",
        "Democratic primary: who wins the Iowa caucus?",
    ],
)
def test_politics(text: str) -> None:
    assert infer_category(text) == "politics"


@pytest.mark.parametrize(
    "text",
    [
        "Will Bitcoin close above $100,000 by year-end?",
        "ETH ETF approval by SEC before March 2026?",
    ],
)
def test_crypto(text: str) -> None:
    assert infer_category(text) == "crypto"


@pytest.mark.parametrize(
    "text",
    [
        "Will the Fed cut interest rates at the next FOMC meeting?",
        "CPI inflation above 3.5% for March 2026?",
    ],
)
def test_macro(text: str) -> None:
    assert infer_category(text) == "macro"


def test_unknown_returns_other() -> None:
    # A question with no matching keyword bucket should fall into 'other',
    # not be misclassified.
    assert infer_category("Will a new species of beetle be discovered in 2026?") == "other"


def test_empty_returns_empty_string() -> None:
    # Empty input -> empty string. The risk engine uses this to SKIP the
    # correlation cap entirely rather than lump the market into a fake
    # bucket.
    assert infer_category("") == ""
    assert infer_category(None) == ""  # type: ignore[arg-type]
