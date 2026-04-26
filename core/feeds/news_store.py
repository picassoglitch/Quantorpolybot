"""Thin read/write helpers for the news enrichment columns on
``feed_items``.

Kept as a separate module from ``rss.py`` so the enricher and the news
API routes don't reach into RSS internals. Queries only — no network.
"""

from __future__ import annotations

import json
from typing import Any

from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import now_ts


async def pending_enrichment(limit: int = 50) -> list[dict[str, Any]]:
    """Oldest rows that the enricher hasn't touched yet. Returns plain
    dicts (not sqlite3.Row) so callers can use .get() freely."""
    rows = await fetch_all(
        """SELECT id, source, title, summary, url, published_at, ingested_at,
                  source_weight
           FROM feed_items
           WHERE enriched_at IS NULL
           ORDER BY id ASC
           LIMIT ?""",
        (limit,),
    )
    return [dict(r) for r in rows]


async def mark_enriched(
    feed_item_id: int,
    enriched: dict[str, Any],
    market_relevance: float,
) -> None:
    await execute(
        """UPDATE feed_items
           SET enriched_json = ?, market_relevance = ?, enriched_at = ?
           WHERE id = ?""",
        (json.dumps(enriched), float(market_relevance), now_ts(), int(feed_item_id)),
    )


async def set_source_weight(source: str, weight: float) -> None:
    """Propagate a trust weight to every row from this source. Cheap —
    sources appear a few thousand times at most."""
    await execute(
        "UPDATE feed_items SET source_weight = ? WHERE source = ?",
        (float(weight), source),
    )


async def recent_enriched(
    since_ts: float,
    min_relevance: float = 0.0,
    limit: int = 200,
) -> list[dict[str, Any]]:
    rows = await fetch_all(
        """SELECT id, source, source_weight, title, summary, url,
                  published_at, ingested_at, enriched_at, enriched_json,
                  market_relevance
           FROM feed_items
           WHERE ingested_at >= ?
             AND market_relevance IS NOT NULL
             AND market_relevance >= ?
           ORDER BY market_relevance DESC, ingested_at DESC
           LIMIT ?""",
        (float(since_ts), float(min_relevance), int(limit)),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["enriched"] = json.loads(d.pop("enriched_json") or "{}") or {}
        except (TypeError, json.JSONDecodeError):
            d["enriched"] = {}
        out.append(d)
    return out


async def source_stats_24h(now: float) -> list[dict[str, Any]]:
    """Per-source summary for the last 24h: total, hourly, last fetch ts,
    and the weight currently attached to those rows (median suffices —
    all writes use the same weight)."""
    since = now - 86400
    hour_ago = now - 3600
    rows = await fetch_all(
        """SELECT source,
                  COUNT(*)                                    AS total,
                  SUM(CASE WHEN ingested_at >= ? THEN 1 ELSE 0 END) AS last_hour,
                  MAX(ingested_at)                            AS last_fetch,
                  MAX(source_weight)                          AS weight
           FROM feed_items
           WHERE ingested_at >= ?
           GROUP BY source
           ORDER BY total DESC""",
        (hour_ago, since),
    )
    return [dict(r) for r in rows]


async def enrichment_backlog() -> int:
    row = await fetch_one(
        "SELECT COUNT(*) AS n FROM feed_items WHERE enriched_at IS NULL"
    )
    return int(row["n"] if row else 0)


async def enrichment_summary_24h(now: float) -> dict[str, Any]:
    since = now - 86400
    totals = await fetch_one(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN market_relevance IS NOT NULL THEN 1 ELSE 0 END) AS enriched,
                  SUM(CASE WHEN market_relevance >= 0.5 THEN 1 ELSE 0 END) AS high_rel
           FROM feed_items WHERE ingested_at >= ?""",
        (since,),
    )
    totals_d = dict(totals) if totals else {}
    return {
        "total": int(totals_d.get("total") or 0),
        "enriched": int(totals_d.get("enriched") or 0),
        "high_rel": int(totals_d.get("high_rel") or 0),
        "backlog": await enrichment_backlog(),
    }
