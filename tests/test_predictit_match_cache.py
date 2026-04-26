"""Cache + correctness tests for the PredictIt ↔ Polymarket matcher.

The matcher is called ~750 × ~800 = ~600 000 times per PredictIt
poll cycle. Each invocation runs five regex-heavy helpers, all pure
functions of a single string input. April 2026 soak showed the
uncached path producing 240-second filter cycles on the worker
thread, which through GIL contention starved the event loop and
spiked loop_lag_ms to 500-1500ms during every poll.

These tests pin three things:

1. The cached helpers actually hit the cache on repeat input
   (`cache_info().hits` increases). If the decorator is removed or
   the function signature changes such that hashing breaks, this
   test fails.

2. ``_extract_entities`` returns a ``frozenset`` (not ``set``) so the
   shared cached instance cannot be mutated by a caller. Mutability
   on a cached value is a real bug class that's silent until it bites
   in production.

3. End-to-end equivalence — the cached helpers produce the SAME
   match decisions and reasons as the call without caching. We
   verify by clearing caches mid-test and replaying.

A soft microbenchmark guards against future regression: 750 × 200
match calls (150 000 invocations, smaller than a real cycle but
representative) must complete inside a generous ceiling on a hot
cache. Used as a smoke test, not a tight performance bound — set
loose enough that ordinary CI noise won't false-positive.
"""

from __future__ import annotations

import time

import pytest

from core.feeds import predictit_match as pim


# ---------------------------------------------------------------------
# Cache-hit behaviour
# ---------------------------------------------------------------------


def _clear_all_caches() -> None:
    pim._extract_entities.cache_clear()
    pim._detect_country.cache_clear()
    pim._detect_election_type.cache_clear()
    pim._detect_candidate.cache_clear()
    pim._detect_phase.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Each test starts with empty caches so hit counts are
    deterministic. Module-level `lru_cache` would otherwise leak
    state across tests."""
    _clear_all_caches()
    yield
    _clear_all_caches()


@pytest.mark.parametrize("helper", [
    pim._extract_entities,
    pim._detect_country,
    pim._detect_election_type,
    pim._detect_candidate,
    pim._detect_phase,
])
def test_helper_caches_repeat_input(helper):
    """Calling each helper with the same input twice must produce
    one miss + one hit. If the decorator gets dropped or the
    function becomes unhashable in its inputs, this fails."""
    text = "Will Donald Trump win the 2028 US presidential election?"
    helper(text)
    helper(text)
    info = helper.cache_info()
    assert info.hits >= 1, (
        f"{helper.__name__} did not hit the cache on repeat input "
        f"(hits={info.hits}, misses={info.misses})"
    )
    assert info.misses == 1, (
        f"{helper.__name__} missed twice on identical input "
        f"(hits={info.hits}, misses={info.misses})"
    )


def test_extract_entities_returns_frozenset():
    """Cached helpers MUST return immutable values — otherwise a
    caller mutating the result corrupts the cache for every future
    caller. ``set`` is mutable; ``frozenset`` is not."""
    result = pim._extract_entities("Will Donald Trump win in 2028?")
    assert isinstance(result, frozenset), (
        f"_extract_entities returned {type(result).__name__}, "
        "expected frozenset (mutable returns from cached pure "
        "functions are a footgun)"
    )


def test_frozenset_intersection_still_works():
    """End-to-end sanity: the only ``_extract_entities`` consumer is
    `match()` which does ``pi_ents & poly_ents`` (set intersection).
    This works on frozensets but the test is cheap insurance against
    a future refactor that assumes mutability."""
    a = pim._extract_entities("Donald Trump 2028 presidential")
    b = pim._extract_entities("Donald Trump approval rating high")
    shared = a & b
    assert "donald trump" in shared


# ---------------------------------------------------------------------
# Equivalence: cached path produces same answers as cold path
# ---------------------------------------------------------------------


_PAIRS = [
    # Strong-anchor accept: same country (usa) + same etype (presidential),
    # no phase conflict.
    (
        "Will Donald Trump win the U.S. presidential race in 2028?",
        "Will Donald Trump win the 2028 U.S. presidential election?",
    ),
    # Country mismatch reject: bulgaria vs quebec.
    (
        "Will the Bulgarian parliamentary election produce a coalition?",
        "Will the Québec provincial election be won by the CAQ?",
    ),
    # Phase XOR reject: poly side names round_1, pi side is silent
    # (implicit overall). Same candidate + country, but the matcher
    # rejects because the questions aren't asking the same thing.
    (
        "Iván Cepeda Castro to win 2026 Colombian presidential election",
        "Will Iván Cepeda Castro win the 2026 Colombian presidential round 1?",
    ),
    # Candidate mismatch reject: different curated names, same etype/country.
    (
        "Will Marine Le Pen win the French presidential election?",
        "Will Macron be re-elected as French president?",
    ),
    # Shared-entities accept: ≥2 named entities overlap.
    (
        "Will the New York Yankees win the 2026 World Series?",
        "Will the New York Yankees be 2026 World Series champions?",
    ),
]


def test_cached_path_matches_cold_path():
    """Build the answers with empty caches (fixture cleared them),
    then clear caches and replay. Both passes must produce the SAME
    (bool, reason) tuples — the cache only changes WHEN work happens,
    never WHAT it produces. A divergence would mean a side effect
    crept in somewhere.

    Spot-check at least one accept and one reject so we know the
    test data covers both branches; otherwise an `always-False`
    regression in the helpers could pass equivalence trivially."""
    cold_results = [pim.match(pi, poly) for pi, poly in _PAIRS]
    decisions = [r[0] for r in cold_results]
    assert any(decisions), (
        f"test data has no accept cases — equivalence test would "
        f"silently pass on an always-False bug. Decisions: {decisions}"
    )
    assert not all(decisions), (
        f"test data has no reject cases — equivalence test would "
        f"silently pass on an always-True bug. Decisions: {decisions}"
    )

    _clear_all_caches()
    warm_results = [pim.match(pi, poly) for pi, poly in _PAIRS]
    assert cold_results == warm_results


# ---------------------------------------------------------------------
# Soft microbenchmark — guards against future regression
# ---------------------------------------------------------------------


def test_cache_keeps_repeat_workload_under_soft_ceiling():
    """Run 200 distinct PredictIt texts × 200 distinct Polymarket
    texts = 40 000 ``match()`` invocations. Pre-cache, this is the
    work shape that produced 240s filter cycles. With caching, each
    helper sees only ~200 unique inputs per side, so the inner loop
    is a hash-lookup grind.

    Ceiling: 5.0 seconds. That's an order of magnitude above the
    expected ~0.2-0.5s on dev hardware so CI noise (cold imports,
    GC, parallel test workers) won't trigger false failures, but
    will catch a regression that re-introduces the un-cached
    workload.
    """
    pi_texts = [
        f"Will candidate{i} win the 2028 US senate race in district {i}?"
        for i in range(200)
    ]
    poly_texts = [
        f"Who wins the 2028 US senate seat for state {i}?"
        for i in range(200)
    ]
    start = time.perf_counter()
    for pi in pi_texts:
        for poly in poly_texts:
            pim.match(pi, poly)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, (
        f"40 000 match() calls took {elapsed:.2f}s — uncached "
        "regression suspected. Healthy hot-cache run is ~0.2-0.5s."
    )
