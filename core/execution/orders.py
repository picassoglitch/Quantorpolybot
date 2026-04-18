"""Order engine. Handles dry-run vs live, persistence, and the safety
gate (`dry_run: false` AND `live_trading_enabled: true`).
"""

from __future__ import annotations

import asyncio

from loguru import logger

from core.execution import clob_client
from core.markets.cache import Market
from core.utils.config import get_config
from core.utils.db import execute, fetch_one
from core.utils.helpers import now_ts
from core.utils.logging import audit


class OrderEngine:
    component = "execution.orders"

    def __init__(self) -> None:
        self._submit_lock = asyncio.Lock()

    @staticmethod
    def live_trading_active() -> bool:
        cfg = get_config()
        return (not cfg.get("dry_run", default=True)) and bool(
            cfg.get("live_trading_enabled", default=False)
        )

    async def submit_signal(
        self,
        signal_id: int,
        market: Market,
        side: str,
        implied_prob: float,
        confidence: float,
        edge: float,
        size_usd: float,
        target_price: float,
    ) -> int | None:
        token_id = market.yes_token() if side == "BUY" else market.no_token() or market.yes_token()
        if not token_id:
            logger.warning("[orders] no token id for market {}", market.market_id)
            return None
        if target_price <= 0 or target_price >= 1:
            logger.warning("[orders] bad target price {}", target_price)
            return None
        size = round(size_usd / target_price, 4)
        async with self._submit_lock:
            return await self._submit(
                signal_id=signal_id,
                market=market,
                token_id=token_id,
                side=side,
                price=target_price,
                size=size,
                size_usd=size_usd,
                implied_prob=implied_prob,
                confidence=confidence,
                edge=edge,
            )

    async def _submit(
        self,
        signal_id: int,
        market: Market,
        token_id: str,
        side: str,
        price: float,
        size: float,
        size_usd: float,
        implied_prob: float,
        confidence: float,
        edge: float,
    ) -> int:
        live = self.live_trading_active()
        ts = now_ts()
        order_id = await execute(
            """INSERT INTO orders
            (signal_id, market_id, token_id, side, price, size, size_usd,
             clob_order_id, status, created_at, updated_at, dry_run)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal_id,
                market.market_id,
                token_id,
                side,
                price,
                size,
                size_usd,
                "",
                "PENDING" if live else "DRY_RUN",
                ts,
                ts,
                0 if live else 1,
            ),
        )
        await execute(
            "UPDATE signals SET status=?, size_usd=? WHERE id=?",
            ("SUBMITTED" if live else "DRY_RUN", size_usd, signal_id),
        )
        if not live:
            logger.warning(
                "[orders] DRY RUN - would have placed {} {} @ {:.3f} size={:.2f} "
                "USD on {} ({}). conf={:.2f} edge={:+.3f}",
                side,
                token_id[:10],
                price,
                size_usd,
                market.market_id,
                market.question[:60],
                confidence,
                edge,
            )
            audit(
                "dry_run_order",
                signal_id=signal_id,
                order_id=order_id,
                market_id=market.market_id,
                side=side,
                price=price,
                size=size,
                size_usd=size_usd,
                confidence=confidence,
                edge=edge,
            )
            return order_id

        # ---- LIVE ----
        result = await clob_client.place_limit_order(token_id, side, price, size)
        if result.ok:
            await execute(
                "UPDATE orders SET clob_order_id=?, status=?, updated_at=? WHERE id=?",
                (result.clob_order_id, "OPEN", now_ts(), order_id),
            )
            logger.info(
                "[orders] LIVE submitted {} {} @ {:.3f} size={:.4f} -> {}",
                side, token_id[:10], price, size, result.clob_order_id,
            )
            audit(
                "live_order_submitted",
                order_id=order_id,
                clob_order_id=result.clob_order_id,
                market_id=market.market_id,
                side=side,
                price=price,
                size=size,
            )
        else:
            await execute(
                "UPDATE orders SET status=?, updated_at=? WHERE id=?",
                ("FAILED", now_ts(), order_id),
            )
            logger.error("[orders] LIVE submit failed: {}", result.error)
            audit("live_order_failed", order_id=order_id, error=result.error)
        return order_id

    async def cancel_all_open(self) -> int:
        from core.utils.db import fetch_all

        rows = await fetch_all(
            "SELECT id, clob_order_id FROM orders WHERE status='OPEN' AND dry_run=0"
        )
        cancelled = 0
        for r in rows:
            ok = await clob_client.cancel_order(r["clob_order_id"]) if r["clob_order_id"] else False
            await execute(
                "UPDATE orders SET status=?, updated_at=? WHERE id=?",
                ("CANCELLED" if ok else "CANCEL_FAILED", now_ts(), r["id"]),
            )
            cancelled += 1 if ok else 0
        # also flip dry-run open orders to cancelled for cleanliness
        await execute(
            "UPDATE orders SET status=?, updated_at=? WHERE status='PENDING' OR status='OPEN'",
            ("CANCELLED", now_ts()),
        )
        logger.info("[orders] cancelled {} live open orders", cancelled)
        return cancelled
