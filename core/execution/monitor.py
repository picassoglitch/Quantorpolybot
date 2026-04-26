"""Real-mode CLOB order reconciliation.

When a lane opens a position in real mode, `shadow.open_position` posts
a GTC limit order via `clob_client` and parks the row with
`status='PENDING_FILL'`. This monitor polls the CLOB for each pending
order and transitions the row:

  matched / filled  → status='OPEN', entry_price replaced with the fill
                      price, entry_ts bumped to the fill timestamp
  cancelled / rejected → status='CLOSED' with reason='clob_rejected',
                      capital released, no PnL
  older than order_timeout_seconds with no fill → cancel on CLOB, then
                      transition to CLOSED with reason='clob_timeout'

Shadow-mode rows (`is_real=0`) are ignored here — they land directly in
OPEN on creation.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from core.execution import allocator, clob_client
from core.utils.config import get_config
from core.utils.db import execute, fetch_all
from core.utils.helpers import now_ts, safe_float
from core.utils.logging import audit


class OrderMonitor:
    component = "execution.monitor"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("execution") or {}
        poll = float(cfg.get("poll_interval_seconds", 5))
        timeout = float(cfg.get("order_timeout_seconds", 90))
        logger.info("[monitor] started poll={}s timeout={}s", poll, timeout)
        while not self._stop.is_set():
            try:
                await self._reconcile_pending(timeout)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[monitor] tick error: {}", e)
            await self._sleep(poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _reconcile_pending(self, timeout_seconds: float) -> None:
        rows = await fetch_all(
            """SELECT id, strategy, market_id, token_id, side, entry_price,
                      size_usd, size_shares, entry_ts, clob_order_id
               FROM shadow_positions
               WHERE status='PENDING_FILL' AND COALESCE(is_real, 0)=1"""
        )
        now = now_ts()
        for r in rows:
            cid = r["clob_order_id"] or ""
            age = now - safe_float(r["entry_ts"])
            if not cid:
                # Submitted without an ID (shouldn't happen, but treat as lost).
                if age > timeout_seconds:
                    await self._fail(r, "clob_no_order_id")
                continue
            status = await clob_client.order_status(cid)
            state = (status.get("status") or status.get("state") or "").lower()
            filled = safe_float(status.get("size_matched") or status.get("filled_size"))
            requested = safe_float(r["size_shares"])
            if state in ("matched", "filled") or (requested > 0 and filled >= requested):
                fill_price = safe_float(status.get("price") or r["entry_price"])
                await execute(
                    """UPDATE shadow_positions
                       SET status='OPEN', entry_price=?, last_price=?,
                           last_price_ts=?
                       WHERE id=?""",
                    (fill_price, fill_price, now, r["id"]),
                )
                audit(
                    "real_fill",
                    position_id=r["id"],
                    market_id=r["market_id"],
                    clob_order_id=cid,
                    fill_price=fill_price,
                    size_usd=safe_float(r["size_usd"]),
                )
                logger.info(
                    "[monitor] FILL pos={} {} {} @ {:.3f}",
                    r["id"], r["side"], r["market_id"], fill_price,
                )
                continue
            if state in ("cancelled", "canceled", "rejected", "expired"):
                await self._fail(r, f"clob_{state}")
                continue
            if age > timeout_seconds:
                ok = await clob_client.cancel_order(cid)
                await self._fail(r, "clob_timeout" if ok else "clob_timeout_cancel_failed")

    async def _fail(self, row, reason: str) -> None:
        """Transition a PENDING_FILL row to CLOSED, release capital."""
        await execute(
            """UPDATE shadow_positions
               SET status='CLOSED', close_ts=?, close_reason=?,
                   realized_pnl_usd=0
               WHERE id=?""",
            (now_ts(), reason, row["id"]),
        )
        await allocator.release(
            row["strategy"], safe_float(row["size_usd"]), 0.0, mode="real",
        )
        audit(
            "real_open_aborted",
            position_id=row["id"],
            market_id=row["market_id"],
            reason=reason,
        )
        logger.warning(
            "[monitor] aborted pending pos={} market={} reason={}",
            row["id"], row["market_id"], reason,
        )
