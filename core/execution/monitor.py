"""Order monitor.

For LIVE orders:
  - Polls CLOB for fills, records executions, updates positions.
  - Cancels orders that exceed `order_timeout_seconds`.

For DRY_RUN orders:
  - Pretends fill at the limit price after a small delay so the rest of
    the system (positions, risk caps, learning loop) sees realistic data.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from core.execution import clob_client
from core.state.positions import upsert_from_fill
from core.utils.config import get_config
from core.utils.db import execute, fetch_all
from core.utils.helpers import now_ts, safe_float


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
                await self._poll_dry_run()
                await self._poll_live(timeout)
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

    async def _poll_dry_run(self) -> None:
        rows = await fetch_all(
            "SELECT * FROM orders WHERE dry_run=1 AND status='DRY_RUN' AND created_at <= ?",
            (now_ts() - 1.0,),
        )
        for r in rows:
            price = safe_float(r["price"])
            size = safe_float(r["size"])
            await upsert_from_fill(r["market_id"], r["token_id"], r["side"], price, size)
            await execute(
                "INSERT INTO executions (order_id, clob_trade_id, market_id, token_id, side, price, size, ts) VALUES (?,?,?,?,?,?,?,?)",
                (r["id"], "dry-run", r["market_id"], r["token_id"], r["side"], price, size, now_ts()),
            )
            await execute(
                "UPDATE orders SET status=?, updated_at=? WHERE id=?",
                ("FILLED_DRY", now_ts(), r["id"]),
            )

    async def _poll_live(self, timeout_seconds: float) -> None:
        rows = await fetch_all(
            "SELECT * FROM orders WHERE dry_run=0 AND status IN ('OPEN','PENDING')"
        )
        now = now_ts()
        for r in rows:
            order_id = r["id"]
            cid = r["clob_order_id"] or ""
            age = now - safe_float(r["created_at"])
            if not cid:
                if age > timeout_seconds:
                    await execute(
                        "UPDATE orders SET status=?, updated_at=? WHERE id=?",
                        ("TIMEOUT", now, order_id),
                    )
                continue
            status = await clob_client.order_status(cid)
            state = (status.get("status") or status.get("state") or "").lower()
            filled = safe_float(status.get("size_matched") or status.get("filled_size"))
            if state in ("matched", "filled") or filled >= safe_float(r["size"]) > 0:
                price = safe_float(status.get("price") or r["price"])
                size = safe_float(status.get("size_matched") or r["size"])
                await execute(
                    "INSERT INTO executions (order_id, clob_trade_id, market_id, token_id, side, price, size, ts) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        order_id,
                        status.get("trade_id") or cid,
                        r["market_id"],
                        r["token_id"],
                        r["side"],
                        price,
                        size,
                        now,
                    ),
                )
                await upsert_from_fill(r["market_id"], r["token_id"], r["side"], price, size)
                await execute(
                    "UPDATE orders SET status=?, updated_at=? WHERE id=?",
                    ("FILLED", now, order_id),
                )
            elif age > timeout_seconds:
                ok = await clob_client.cancel_order(cid)
                await execute(
                    "UPDATE orders SET status=?, updated_at=? WHERE id=?",
                    ("CANCELLED" if ok else "CANCEL_FAILED", now, order_id),
                )
                logger.info("[monitor] cancelled stale order {} ({:.0f}s old)", cid, age)
