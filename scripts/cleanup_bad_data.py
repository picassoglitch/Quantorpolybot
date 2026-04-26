"""One-shot cleanup for data generated under the pre-fix rules.

Run once after the bug-fix directive lands:

    python scripts/cleanup_bad_data.py

Does three things:
  1. Deletes `signals` rows that were written without a category (those
     were all generated under the old "uncategorised" bucket bug that
     caused spurious category-cap rejections).
  2. Deletes recent (last 7 days) `feed_items` whose source is
     `predictit_xref` — all synthesised under the loose Jaccard match
     that now gets replaced by the strict named-entity matcher.
  3. Resets the `longshot` lane's capital allocator row so the lane
     starts fresh with an empty deployed balance (the lane itself is
     paused in config until we re-enable it).

Safe to re-run — all operations are idempotent enough that a second
invocation won't do anything harmful.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make `core.*` imports resolve when run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from core.execution import allocator  # noqa: E402
from core.utils.db import execute, fetch_one, init_db  # noqa: E402
from core.utils.helpers import now_ts  # noqa: E402


async def _delete_empty_category_signals() -> int:
    row = await fetch_one(
        "SELECT COUNT(*) AS n FROM signals WHERE category IS NULL OR category=''"
    )
    n = int(row["n"] if row else 0)
    if n == 0:
        logger.info("[cleanup] no empty-category signals to delete")
        return 0
    await execute("DELETE FROM signals WHERE category IS NULL OR category=''")
    logger.info("[cleanup] deleted {} signals with empty category", n)
    return n


async def _delete_recent_predictit_xref(days: int = 7) -> int:
    cutoff = now_ts() - days * 86400.0
    row = await fetch_one(
        "SELECT COUNT(*) AS n FROM feed_items WHERE source='predictit_xref' AND ingested_at >= ?",
        (cutoff,),
    )
    n = int(row["n"] if row else 0)
    if n == 0:
        logger.info("[cleanup] no recent predictit_xref feed_items to delete")
        return 0
    await execute(
        "DELETE FROM feed_items WHERE source='predictit_xref' AND ingested_at >= ?",
        (cutoff,),
    )
    # Also drop the cross_references rows from the same window so the
    # dashboard doesn't show stale divergences.
    await execute(
        "DELETE FROM cross_references WHERE source='predictit' AND fetched_at >= ?",
        (cutoff,),
    )
    logger.info(
        "[cleanup] deleted {} predictit_xref feed_items + cross_references in last {} days",
        n, days,
    )
    return n


async def _reset_longshot_lane() -> None:
    # Re-init the allocator so all lane rows exist, then zero the
    # longshot row's deployed/available back to its starting budget.
    await allocator.init_lane_capital()
    await execute(
        """UPDATE lane_capital
           SET deployed=0.0,
               available=total_budget,
               paused_until=NULL,
               updated_at=?
           WHERE lane='longshot'""",
        (now_ts(),),
    )
    logger.info("[cleanup] reset longshot lane capital (deployed=0, unpaused)")


async def main() -> None:
    await init_db()
    deleted_signals = await _delete_empty_category_signals()
    deleted_feeds = await _delete_recent_predictit_xref(days=7)
    await _reset_longshot_lane()
    logger.info(
        "[cleanup] done — signals -{}, predictit_xref -{}, longshot capital reset",
        deleted_signals, deleted_feeds,
    )


if __name__ == "__main__":
    asyncio.run(main())
