"""Unit tests for helpers."""

from core.utils.helpers import Backoff, clamp, jaccard, keywords, safe_float, safe_int


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


def test_safe_float_handles_garbage():
    assert safe_float("nope") == 0.0
    assert safe_float(None, 1.5) == 1.5
    assert safe_float("3.14") == 3.14


def test_safe_int():
    assert safe_int("3.7") == 3
    assert safe_int(None, 9) == 9


def test_keywords_extracts_words():
    assert keywords("Trump wins 2024 election!") == {"trump", "wins", "2024", "election"}


def test_jaccard_basic():
    a = {"trump", "election", "2024"}
    b = {"trump", "biden", "2024"}
    s = jaccard(a, b)
    assert 0 < s < 1


def test_jaccard_empty():
    assert jaccard(set(), {"a"}) == 0.0


def test_backoff_grows():
    b = Backoff(base=1, cap=8, factor=2)
    d1 = b.next_delay()
    d2 = b.next_delay()
    d3 = b.next_delay()
    assert d1 < d2 < d3
    b.reset()
    assert b.next_delay() < d2
