"""Evidence tiering for the scalping lane.

Replaces the binary evidence gate (``len(distinct_sources) >= 2 -> proceed,
else skip``) with a four-tier classifier so the lane can keep doing
useful work on markets that fall short of the strong-evidence bar.

Tiers:

  - ``STRONG``  — enough corroboration to fund the full LLM scoring
                  path with nominal sizing. Today: ``>= strong_min_sources``
                  distinct sources, with at least one item fresher than
                  ``fresh_within_seconds``.
  - ``WEAK``    — at least one usable source but below the strong bar.
                  Lane scores via the keyword heuristic only (no LLM
                  call) and sizes down by ``weak_size_multiplier``. Logged
                  with ``watchlist=true`` so post-hoc review can promote
                  recurring weak signals into strong-source list.
  - ``MICRO``   — no usable news evidence, but the market itself is
                  liquid and the orderbook/tick stream gives a
                  directional read (see ``core.strategies.microstructure``).
                  Sized by ``microstructure_size_multiplier``; the
                  microstructure module's own confidence cap (0.55)
                  prevents this tier from sizing up regardless of
                  config.
  - ``NONE``    — neither news nor microstructure can support a trade.
                  Lane skips with ``reject_reason="no_evidence_no_microstructure"``.

The classifier returns the tier *and* a small explanation string so the
scan_skips table records exactly what the lane saw.

Persistence: ``record_skip`` writes a row to the ``scan_skips`` table.
Cheap (single INSERT, no transaction wrap), no-throw — a logging
failure must not break the scan loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from loguru import logger

from core.utils.db import connect, execute
from core.utils.helpers import now_ts, safe_float


class EvidenceTier(str, Enum):
    """Classifier output. Stored as a string in scan_skips for greppability."""
    STRONG = "strong"
    WEAK = "weak"
    MICRO = "microstructure"
    NONE = "none"


@dataclass(frozen=True)
class EvidenceClassification:
    tier: EvidenceTier
    distinct_sources: int
    total_items: int
    freshest_age_seconds: float | None
    reasoning: str


def classify_evidence(
    items: Iterable[dict[str, Any]],
    *,
    strong_min_sources: int = 2,
    weak_min_sources: int = 1,
    fresh_within_seconds: float = 6 * 3600.0,
    now: float | None = None,
) -> EvidenceClassification:
    """Pure function. Doesn't touch microstructure — the lane decides
    whether to fall through to the MICRO path when this returns NONE.

    `fresh_within_seconds` controls the staleness ceiling for the STRONG
    tier specifically: 3 sources from a week ago is *not* strong because
    the price has long since digested it. Default 6h matches the
    scalping lane's `max_age_hours: 24` budget halved — fresh-enough
    that the LLM call is worth the cost.

    Items are dicts straight from `feed_items` (`id`, `source`,
    `title`, `summary`, `url`, `ingested_at`, optionally ``meta``).
    Items without a usable source key are ignored (can't count toward
    a source bar).

    Distinctness key: ``meta["publisher"]`` when present, falling back
    to ``item["source"]``. The Google News feed all comes in under one
    source name (``"google_news"``) but each entry's ``meta.publisher``
    is the actual outlet ("BBC", "Reuters", "ESPN", ...). Counting by
    publisher when available — and by feed name otherwise — is what
    lets a market with 30 articles from 10 different outlets classify
    as STRONG instead of being collapsed to ``distinct=1``.
    """
    items_list = list(items)
    sources: set[str] = set()
    total = 0
    freshest_age: float | None = None
    now_t = now if now is not None else now_ts()

    for item in items_list:
        meta = item.get("meta")
        publisher = meta.get("publisher") if isinstance(meta, dict) else None
        src = publisher or item.get("source")
        if not src:
            continue
        total += 1
        sources.add(str(src))
        ingested_raw = item.get("ingested_at")
        if ingested_raw is None:
            continue
        age = now_t - safe_float(ingested_raw)
        if age < 0:
            age = 0.0
        if freshest_age is None or age < freshest_age:
            freshest_age = age

    distinct = len(sources)

    has_fresh = (
        freshest_age is not None and freshest_age <= fresh_within_seconds
    )

    if distinct >= strong_min_sources and has_fresh:
        reason = (
            f"strong: {distinct} distinct sources, freshest "
            f"{freshest_age:.0f}s old (<= {fresh_within_seconds:.0f}s)"
        )
        return EvidenceClassification(
            tier=EvidenceTier.STRONG,
            distinct_sources=distinct,
            total_items=total,
            freshest_age_seconds=freshest_age,
            reasoning=reason,
        )

    if distinct >= weak_min_sources:
        # Stale-but-multi-source still counts as weak — the lane will
        # heuristic-score it. The "watchlist" treatment is the main
        # value-add: recurring weak hits are signal for the operator
        # to look at adding the source to the strong-source list.
        if not has_fresh and freshest_age is not None:
            reason = (
                f"weak: {distinct} source(s), freshest {freshest_age:.0f}s "
                f"old (> strong staleness {fresh_within_seconds:.0f}s)"
            )
        elif freshest_age is None:
            reason = (
                f"weak: {distinct} source(s), no ingested_at on items"
            )
        else:
            reason = (
                f"weak: {distinct} source(s) (need {strong_min_sources} for strong)"
            )
        return EvidenceClassification(
            tier=EvidenceTier.WEAK,
            distinct_sources=distinct,
            total_items=total,
            freshest_age_seconds=freshest_age,
            reasoning=reason,
        )

    return EvidenceClassification(
        tier=EvidenceTier.NONE,
        distinct_sources=distinct,
        total_items=total,
        freshest_age_seconds=freshest_age,
        reasoning=(
            f"none: {distinct} source(s) and {total} item(s); "
            "below weak threshold"
        ),
    )


async def record_skip(
    *,
    lane: str,
    market_id: str,
    tier_attempted: str,
    reject_reason: str,
    evidence_tier: str,
    watchlist: bool = False,
    score_snapshot: dict[str, Any] | None = None,
    scan_ts: float | None = None,
) -> None:
    """INSERT one row into ``scan_skips``. No-throw: log on failure and
    return, the scan loop must not be derailed by a persistence hiccup.

    `score_snapshot` is serialized to single-line JSON if provided —
    keep it small; we want histograms, not transcripts.
    """
    payload: str | None = None
    if score_snapshot is not None:
        try:
            payload = json.dumps(score_snapshot, default=str)
        except (TypeError, ValueError):
            payload = None
    try:
        await execute(
            """INSERT INTO scan_skips
                 (scan_ts, lane, market_id, tier_attempted,
                  reject_reason, evidence_tier, watchlist, score_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan_ts if scan_ts is not None else now_ts(),
                lane,
                market_id,
                tier_attempted,
                reject_reason,
                evidence_tier,
                1 if watchlist else 0,
                payload,
            ),
        )
    except Exception as e:
        logger.debug(
            "[scan_skips] persist failed (lane={} market={}): {}",
            lane, market_id, e,
        )


async def purge_skips_older_than(seconds: float) -> int:
    """Delete scan_skips rows older than ``seconds``. Returns rowcount.

    Wrapped here (rather than inlined in the scheduler) so callers can
    swap the TTL policy without touching the cron wiring. No-throw.

    Uses a direct cursor (not ``db.execute``) because ``db.execute``
    returns ``lastrowid`` — fine for INSERTs, useless for DELETEs which
    have no last-row concept. The scheduler logs this rowcount so the
    operator can see the table size at a glance.
    """
    try:
        cutoff = now_ts() - seconds
        async with connect() as conn:
            cur = await conn.execute(
                "DELETE FROM scan_skips WHERE scan_ts < ?", (cutoff,)
            )
            deleted = cur.rowcount
            await conn.commit()
            return int(deleted) if deleted is not None else 0
    except Exception as e:
        logger.warning("[scan_skips] purge failed: {}", e)
        return 0
