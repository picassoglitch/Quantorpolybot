"""One-shot cleanup: close open positions that violate current policy.

Context (Apr 2026 incident): event_sniping filled up on 2026 World Cup
futures ($10.15 of $10.15 budget deployed, 65-88 days out) after Ollama
hallucinated big edges on low-liquidity dark-horse markets. The new
duration filter (config: event_sniping.max_days_to_resolution=14) stops
new entries of that shape, but the positions already on the book still
need to be cleared so the capital can rotate into short-horizon plays.

Also closes microscalp positions — the lane is disabled in config.yaml
while the YES/NO token-ordering fix soaks in, so no exit logic is running
on them; they'd sit open until resolution otherwise.

Policy applied here:
  - event_sniping: close if days_until_resolve > 14 (matches new config)
  - microscalp:    close unconditionally (lane disabled)
  - everything else: leave alone

Runs at current bid (or mid if the book is one-sided), which is how
:func:`shadow.close_position` already handles it. Safe to re-run: if
nothing matches the policy the script does nothing.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from core.execution import allocator, shadow
from core.utils.db import fetch_all, fetch_one
from core.utils.prices import current_price, days_until_resolve


MAX_DAYS_EVENT_SNIPING = 14
REASON_EVENT = "manual_cleanup_long_horizon_futures"
REASON_MICROSCALP = "manual_cleanup_lane_disabled"


async def _snapshot_lanes(label: str) -> None:
    logger.info("[cleanup] allocator state {}:", label)
    for lane in allocator.LANES:
        s = await allocator.get_state(lane)
        if s is None:
            continue
        logger.info(
            "  {}: tb={:.2f} dep={:.2f} avail={:.2f}",
            lane, s.total_budget, s.deployed, s.available,
        )


async def main() -> None:
    await _snapshot_lanes("BEFORE")

    rows = await fetch_all(
        """SELECT id FROM shadow_positions WHERE status=?""",
        ("OPEN",),
    )
    closed_count = 0
    total_realized = 0.0
    for r in rows:
        # Re-load through shadow's row->dataclass path so close_position
        # gets exactly the shape it expects (avoids mismatches in
        # computed fields like hold_seconds or is_real coercion).
        pos_row = await fetch_one(
            "SELECT * FROM shadow_positions WHERE id=?", (r["id"],),
        )
        pos = shadow._row_to_position(pos_row)

        if pos.strategy == "event_sniping":
            mk = await fetch_one(
                "SELECT close_time FROM markets WHERE market_id=?",
                (pos.market_id,),
            )
            days = days_until_resolve(mk["close_time"]) if mk else None
            if days is None:
                reason = f"{REASON_EVENT}_unknown_close_time"
            elif days > MAX_DAYS_EVENT_SNIPING:
                reason = f"{REASON_EVENT}_{int(days)}d"
            else:
                logger.info(
                    "[cleanup] keep event_sniping id={} (resolves in {:.1f}d)",
                    pos.id, days,
                )
                continue
        elif pos.strategy == "microscalp":
            reason = REASON_MICROSCALP
        else:
            logger.info(
                "[cleanup] keep {} id={} (not in scope)",
                pos.strategy, pos.id,
            )
            continue

        snap = await current_price(pos.market_id, pos.token_id)
        if snap is None:
            logger.warning(
                "[cleanup] skip id={} — no price snapshot available", pos.id,
            )
            continue

        realized = await shadow.close_position(pos, snap, reason)
        total_realized += realized
        closed_count += 1
        logger.info(
            "[cleanup] closed id={} {} @ reason={} realized={:+.2f}",
            pos.id, pos.market_id, reason, realized,
        )

    logger.info(
        "[cleanup] closed {} positions, total realized {:+.2f} USD",
        closed_count, total_realized,
    )
    await _snapshot_lanes("AFTER")


if __name__ == "__main__":
    asyncio.run(main())
