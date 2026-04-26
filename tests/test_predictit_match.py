"""Stricter PredictIt ↔ Polymarket matcher.

The old Jaccard-only matcher produced the two false-positive bugs
captured below — they drove losing auto-signals until the rules were
tightened. Negative tests pin down the new behaviour.
"""

from __future__ import annotations

import pytest

from core.feeds.predictit_match import match


# ---- Negative cases (must reject under the new rules) ----

def test_different_countries_reject():
    """Bulgaria vs Québec both share the word 'election' — strict rules
    must reject because the country signals disagree."""
    pi = "Which party wins the Bulgaria parliamentary election?"
    poly = "Which party wins the Québec general election?"
    ok, reason = match(pi, poly)
    assert ok is False
    assert "countries" in reason or "country" in reason


def test_different_candidates_same_district_reject():
    """CA-11 primary: the old matcher would map a PredictIt contract on
    one candidate to a Polymarket contract on a different candidate
    because they shared 'CA-11' and 'primary'."""
    pi = "Will Rudy Melendez win the CA-11 Republican primary?"
    poly = "Will Michelle Steel win the CA-11 Republican primary?"
    ok, reason = match(pi, poly)
    assert ok is False
    assert "candidates" in reason


def test_completely_unrelated_reject():
    pi = "Will Donald Trump win the Republican nomination?"
    poly = "Will Bitcoin reach $150,000 by year-end?"
    ok, _ = match(pi, poly)
    assert ok is False


# ---- Positive cases (must still accept) ----

def test_same_candidate_same_race_accept():
    pi = "Will Donald Trump win the 2024 Republican presidential nomination?"
    poly = "Will Donald Trump win the 2024 Republican presidential nomination?"
    ok, _ = match(pi, poly)
    assert ok is True


def test_shared_entities_accept():
    """Multiple shared named entities (Trump + Iowa) — no country conflict,
    no candidate conflict — must accept."""
    pi = "Donald Trump margin of victory in Iowa caucus"
    poly = "Will Donald Trump win the Iowa caucus?"
    ok, _ = match(pi, poly)
    assert ok is True
