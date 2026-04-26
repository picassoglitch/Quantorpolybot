"""Pattern Discovery Engine — read-only analytics.

This package is **never imported by the trading path**. It powers the
``scripts/analyze_patterns.py`` CLI and any future dashboard surfaces.
Importing it has no side effects on the bot. Running it does not
modify any production table other than ``signal_outcomes`` itself
(which the CLI rebuilds idempotently).

Layers:

  - ``signal_outcomes``   — extract candidate rows from
    event_market_candidates / scan_skips / shadow_positions, JOIN
    against price_ticks, fan out per contributing source name,
    persist to the ``signal_outcomes`` table.
  - ``source_performance`` / ``category_performance`` /
    ``market_mapper_performance`` — aggregations over
    ``signal_outcomes`` for the CLI report.
  - ``trust_tiers`` — NEW / WATCH / TRUSTED / LATE_CONFIRMATION /
    NOISY / BLACKLIST classifier with minimum-sample-size guard.

PR #1 contract (per spec):

  - On-demand CLI only. No cron, no background job.
  - Read-only. No strategy changes, no auto-trust, no real orders.
  - Graceful when price_ticks is missing — outcome columns are NULL
    in the rebuilt table, downstream aggregations skip those rows.
"""
