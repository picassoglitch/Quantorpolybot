"""Pattern Discovery Engine CLI — read-only analytics.

Usage:

    # Default: rebuild signal_outcomes from current DB, then print
    # the standard report set.
    python scripts/analyze_patterns.py

    # Just rebuild (no report).
    python scripts/analyze_patterns.py --rebuild-only

    # Just report (re-use existing signal_outcomes — fast).
    python scripts/analyze_patterns.py --no-rebuild

    # Specific report sections.
    python scripts/analyze_patterns.py --report sources
    python scripts/analyze_patterns.py --report categories
    python scripts/analyze_patterns.py --report missed
    python scripts/analyze_patterns.py --report noisy
    python scripts/analyze_patterns.py --report tiers

    # Limit to recent candidates (default: all).
    python scripts/analyze_patterns.py --since-hours 24

The script does NOT need the bot to be running. It opens its own
DB connection (read + write to `signal_outcomes` only) and exits
cleanly. Trading behavior is never consulted, never modified.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analytics import aggregations, signal_outcomes, trust_tiers  # noqa: E402
from core.analytics.aggregations import (  # noqa: E402
    CategoryMetrics,
    SourceMetrics,
)
from core.utils.db import init_db  # noqa: E402


# ============================================================
# Pretty-print helpers
# ============================================================


def _h(text: str) -> str:
    line = "=" * max(40, len(text))
    return f"\n{line}\n{text}\n{line}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:5.1f}%"


def _fmt_move(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+5.2f}c"


def _fmt_int(value: int) -> str:
    return f"{value:>6d}"


def _print_source_table(metrics: list[SourceMetrics], *, limit: int) -> None:
    """Source-performance table. Sorted descending by sample_size."""
    if not metrics:
        print("  (no source rows in signal_outcomes)")
        return
    header = (
        f"  {'source':<28} {'n':>6} "
        f"{'hit':>7} {'fp':>7} "
        f"{'5m abs':>9} {'15m abs':>9} "
        f"{'acc':>5} {'obs':>5} {'rej':>5}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for m in metrics[:limit]:
        print(
            f"  {m.source[:28]:<28} {_fmt_int(m.sample_size)} "
            f"{_fmt_pct(m.hit_rate):>7} {_fmt_pct(m.false_positive_rate):>7} "
            f"{_fmt_move(m.avg_move_5m_abs):>9} "
            f"{_fmt_move(m.avg_move_15m_abs):>9} "
            f"{m.accepted_count:>5} {m.observed_count:>5} {m.rejected_count:>5}"
        )


def _print_category_table(metrics: list[CategoryMetrics], *, limit: int) -> None:
    if not metrics:
        print("  (no category rows in signal_outcomes)")
        return
    header = (
        f"  {'category':<28} {'n':>6} "
        f"{'hit':>7} "
        f"{'5m abs':>9} {'15m abs':>9} "
        f"{'acc':>5} {'obs':>5} {'rej':>5}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for m in metrics[:limit]:
        print(
            f"  {m.category[:28]:<28} {_fmt_int(m.sample_size)} "
            f"{_fmt_pct(m.hit_rate):>7} "
            f"{_fmt_move(m.avg_move_5m_abs):>9} "
            f"{_fmt_move(m.avg_move_15m_abs):>9} "
            f"{m.accepted_count:>5} {m.observed_count:>5} {m.rejected_count:>5}"
        )


def _print_missed(rows: list[dict], *, limit: int) -> None:
    if not rows:
        print("  (no missed-edge candidates — either no rejected/observed rows "
              "had non-null estimated_edge_missed, or signal_outcomes is empty)")
        return
    header = (
        f"  {'source':<22} {'category':<14} {'market':<14} "
        f"{'fav 15m':>10} {'adv 15m':>10} {'status':>10} reason"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for r in rows[:limit]:
        print(
            f"  {(r['source'] or '?')[:22]:<22} "
            f"{(r['category'] or '?')[:14]:<14} "
            f"{(r['market_id'] or '?')[:14]:<14} "
            f"{_fmt_move(r['max_favorable_move_15m']):>10} "
            f"{_fmt_move(r['max_adverse_move_15m']):>10} "
            f"{(r['final_status'] or '?')[:10]:>10} "
            f"{(r['reject_reason'] or '')[:60]}"
        )


def _print_tier_table(
    assignments: list[trust_tiers.TrustTierAssignment], *, limit: int,
) -> None:
    if not assignments:
        print("  (no source rows in signal_outcomes)")
        return
    header = (
        f"  {'tier':<18} {'source':<24} {'n':>6} "
        f"{'hit':>7} {'fp':>7} {'5m abs':>9} reasoning"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for a in assignments[:limit]:
        print(
            f"  {a.tier:<18} {a.source[:24]:<24} {_fmt_int(a.sample_size)} "
            f"{_fmt_pct(a.hit_rate):>7} "
            f"{_fmt_pct(a.false_positive_rate):>7} "
            f"{_fmt_move(a.avg_move_5m_abs):>9} "
            f"{a.reasoning[:80]}"
        )


# ============================================================
# CLI entry
# ============================================================


_REPORTS = ("sources", "categories", "missed", "noisy", "tiers", "all")


async def main() -> int:
    p = argparse.ArgumentParser(description="Pattern Discovery Engine — read-only analytics")
    p.add_argument(
        "--rebuild-only", action="store_true",
        help="Rebuild signal_outcomes and exit without printing reports.",
    )
    p.add_argument(
        "--no-rebuild", action="store_true",
        help="Skip the rebuild — re-use the existing signal_outcomes table.",
    )
    p.add_argument(
        "--report", choices=_REPORTS, default="all",
        help="Which report section to print (default: all).",
    )
    p.add_argument(
        "--since-hours", type=float, default=None,
        help="Only consider candidates from the last N hours (default: all).",
    )
    p.add_argument(
        "--limit", type=int, default=20,
        help="Max rows per table (default: 20).",
    )
    p.add_argument(
        "--min-sample-tier", type=int,
        default=trust_tiers._DEFAULT_MIN_SAMPLE_NEW,
        help=(
            "Minimum sample size before a source is scoreable (below this "
            f"-> NEW; default: {trust_tiers._DEFAULT_MIN_SAMPLE_NEW})."
        ),
    )
    args = p.parse_args()

    # Ensure the schema exists. init_db is idempotent.
    await init_db()

    # ---- Rebuild ----
    if not args.no_rebuild:
        since_seconds = (
            args.since_hours * 3600.0 if args.since_hours else None
        )
        n = await signal_outcomes.rebuild(since_seconds=since_seconds)
        print(f"\nrebuilt signal_outcomes: {n} rows")
    if args.rebuild_only:
        return 0

    # ---- Reports ----
    section = args.report
    show_all = section == "all"

    if show_all or section == "sources":
        print(_h("source_performance — all sources"))
        metrics = await aggregations.source_performance()
        _print_source_table(metrics, limit=args.limit)

        print(_h("top sources by 5m move (qualifying min sample)"))
        top = await aggregations.top_sources_by_5m_move(
            min_sample_size=args.min_sample_tier, limit=args.limit,
        )
        _print_source_table(top, limit=args.limit)

    if show_all or section == "categories":
        print(_h("category_performance — all categories"))
        cats = await aggregations.category_performance()
        _print_category_table(cats, limit=args.limit)
        print(_h("top categories by hit rate"))
        topcat = await aggregations.top_categories_by_hit_rate(
            min_sample_size=args.min_sample_tier, limit=args.limit,
        )
        _print_category_table(topcat, limit=args.limit)

    if show_all or section == "missed":
        print(_h("missed-edge candidates (largest favorable move on a "
                 "rejected/observed row)"))
        missed = await aggregations.missed_edge_candidates(limit=args.limit)
        _print_missed(missed, limit=args.limit)

    if show_all or section == "noisy":
        print(_h("noisy sources (high false-positive rate, qualifying sample)"))
        noisy = await aggregations.noisy_sources(
            min_sample_size=args.min_sample_tier,
        )
        _print_source_table(noisy, limit=args.limit)

    if show_all or section == "tiers":
        print(_h("trust tiers — read-only; bot never auto-applies these"))
        all_metrics = await aggregations.source_performance()
        cfg = trust_tiers.TrustTierConfig(
            min_sample_new=args.min_sample_tier,
        )
        assignments = trust_tiers.classify_all(all_metrics, cfg)
        _print_tier_table(assignments, limit=args.limit)

    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
