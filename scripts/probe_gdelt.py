"""Smoke probe for the GDELT connector — debug outside the full bot run.

Usage:

    # Default: run the canary query.
    python scripts/probe_gdelt.py

    # All 13 category queries, paced at 5.5s like the connector.
    python scripts/probe_gdelt.py --all

    # One specific category.
    python scripts/probe_gdelt.py --category shooting

    # Custom query.
    python scripts/probe_gdelt.py --query "ceasefire"

For each fetch, prints the structured GdeltFetchResult diagnostic
(status / content-type / bytes / body excerpt / exception class /
articles count) so a single command tells you whether GDELT is
reachable, rate-limiting us, or returning a non-JSON body.

When --normalize is passed and a fetch succeeded, also runs
core.scout.normalizer.normalize() on the parsed signals and prints
the resulting Event count + categories. Confirms the normalizer
isn't silently dropping live GDELT shapes.

This script is intentionally async/standalone — does NOT spin up the
DB, the lane, or the dashboard. Safe to run while the main bot is
soaking.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Make the project root importable when running `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feeds.gdelt import (  # noqa: E402
    GLOBAL_LIMITER,
    _CANARY_QUERY,
    _CATEGORY_QUERIES,
    _fetch_gdelt,
    parse_gdelt_article,
)
from core.scout.event import EventCategory  # noqa: E402
from core.scout.normalizer import normalize  # noqa: E402


def _print_result(label: str, result, *, verbose: bool = False) -> None:
    print(f"\n=== {label} ===")
    print(f"url: {result.url}")
    print(f"  {result.diagnostic()}")
    if result.body_excerpt and verbose:
        print(f"  body[:300]: {result.body_excerpt!r}")
    if result.ok and result.articles:
        first = result.articles[0]
        print(
            f"  first.article: domain={first.get('domain')!r} "
            f"title={(first.get('title') or '')[:80]!r}"
        )


async def _probe_one(category: EventCategory, query: str, *, max_records: int,
                     timespan: str, verbose: bool, normalize_too: bool) -> None:
    result = await _fetch_gdelt(query, max_records=max_records, timespan=timespan)
    _print_result(f"{category.value} ({query!r})", result, verbose=verbose)

    if normalize_too and result.ok:
        sigs: list = []
        for art in result.articles:
            sig = parse_gdelt_article(art, category)
            if sig is not None:
                sigs.append((len(sigs) + 1, sig))
        if sigs:
            events = normalize(sigs)
            print(f"  normalize: {len(sigs)} signals -> {len(events)} events")
            for ev in events[:3]:
                print(
                    f"    event_id={ev.event_id} cat={ev.category.value} "
                    f"sev={ev.severity:.2f} conf={ev.confidence:.2f} "
                    f"sources={ev.sources} title={ev.title[:60]!r}"
                )
        else:
            print("  normalize: no parseable signals")


async def main() -> int:
    p = argparse.ArgumentParser(description="GDELT connector smoke probe")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true",
                   help="Run all 13 category queries (paced at 5.5s).")
    g.add_argument("--category", type=str,
                   help="Run one category by name (e.g. 'shooting').")
    g.add_argument("--query", type=str,
                   help="Run a custom GDELT query string (no category context).")
    p.add_argument("--max-records", type=int, default=10)
    p.add_argument("--timespan", type=str, default="1h")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip running normalize() on the parsed signals.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full body excerpt and per-article details.")
    args = p.parse_args()

    normalize_too = not args.no_normalize

    if args.query:
        # Use a synthetic OTHER category for the run — the parser
        # tags signals with whatever category we pass in.
        result = await _fetch_gdelt(
            args.query, max_records=args.max_records, timespan=args.timespan,
        )
        _print_result(f"custom query ({args.query!r})", result, verbose=args.verbose)
        return 0 if result.ok else 1

    if args.all:
        # No manual sleep — the GLOBAL_LIMITER inside _fetch_gdelt
        # spaces calls. Sharing the limiter between this script and
        # any concurrent bot run prevents the probe from burning the
        # bot's rate budget.
        for cat, q in _CATEGORY_QUERIES.items():
            await _probe_one(
                cat, q, max_records=args.max_records, timespan=args.timespan,
                verbose=args.verbose, normalize_too=normalize_too,
            )
        print(
            f"\nlimiter: base={GLOBAL_LIMITER.base_interval_seconds:.1f}s "
            f"current={GLOBAL_LIMITER.current_interval_seconds:.1f}s"
        )
        return 0

    if args.category:
        try:
            cat = EventCategory(args.category.lower())
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Valid: {[c.value for c in _CATEGORY_QUERIES.keys()]}")
            return 2
        if cat not in _CATEGORY_QUERIES:
            print(f"Category not registered with GDELT queries: {cat.value}")
            return 2
        await _probe_one(
            cat, _CATEGORY_QUERIES[cat], max_records=args.max_records,
            timespan=args.timespan, verbose=args.verbose,
            normalize_too=normalize_too,
        )
        return 0

    # Default: canary.
    print(f"Canary query: {_CANARY_QUERY!r}")
    result = await _fetch_gdelt(
        _CANARY_QUERY, max_records=args.max_records, timespan=args.timespan,
    )
    _print_result("canary", result, verbose=args.verbose)
    return 0 if result.ok else 1


if __name__ == "__main__":
    t0 = time.perf_counter()
    rc = asyncio.run(main())
    print(f"\nelapsed: {time.perf_counter() - t0:.2f}s")
    sys.exit(rc)
