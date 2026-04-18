"""Polymarket CLOB WebSocket subscriber. Streams live mid prices for all
active markets and writes price_ticks rows. Auto-reconnects with backoff.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Iterable

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosed

from core.utils.config import get_config
from core.utils.db import execute, fetch_all
from core.utils.helpers import Backoff, now_ts, safe_float

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolymarketWS:
    component = "feed.polymarket_ws"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "polymarket_ws") or {}
        if not cfg.get("enabled", True):
            logger.info("[poly_ws] disabled")
            return
        backoff = Backoff(base=float(cfg.get("reconnect_seconds", 5)), cap=120)
        logger.info("[poly_ws] starting")
        while not self._stop.is_set():
            try:
                token_ids = await self._active_token_ids()
                if not token_ids:
                    await self._sleep(30)
                    continue
                await self._connect_and_stream(token_ids)
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = backoff.next_delay()
                logger.exception("[poly_ws] error, sleeping {:.1f}s: {}", delay, e)
                await self._sleep(delay)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _active_token_ids(self) -> list[str]:
        rows = await fetch_all(
            "SELECT token_ids FROM markets WHERE active=1 AND token_ids IS NOT NULL"
        )
        ids: list[str] = []
        for row in rows:
            try:
                parsed = json.loads(row["token_ids"]) if row["token_ids"] else []
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                ids.extend(str(t) for t in parsed if t)
        # de-dupe, cap to a reasonable batch (CLOB WS supports many but
        # huge subs are slow on reconnect)
        unique = list(dict.fromkeys(ids))[:500]
        return unique

    async def _connect_and_stream(self, token_ids: Iterable[str]) -> None:
        sub = {"type": "Market", "assets_ids": list(token_ids)}
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
            await ws.send(json.dumps(sub))
            logger.info("[poly_ws] connected, subscribed to {} tokens", len(sub["assets_ids"]))
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    await ws.ping()
                    continue
                except ConnectionClosed:
                    logger.warning("[poly_ws] connection closed by server")
                    return
                await self._handle_message(raw)

    async def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        events: list[dict[str, Any]] = data if isinstance(data, list) else [data]
        ts = now_ts()
        rows = []
        for ev in events:
            asset_id = ev.get("asset_id") or ev.get("token_id")
            market_id = ev.get("market") or ev.get("market_id")
            bid = safe_float(ev.get("best_bid") or ev.get("bid"))
            ask = safe_float(ev.get("best_ask") or ev.get("ask"))
            last = safe_float(ev.get("price") or ev.get("last_price"))
            if not asset_id and not market_id:
                continue
            rows.append((market_id, asset_id, bid, ask, last, ts))
            if market_id:
                await execute(
                    """UPDATE markets SET best_bid=?, best_ask=?, last_price=?, updated_at=?
                       WHERE market_id=?""",
                    (bid, ask, last, ts, market_id),
                )
        if rows:
            from core.utils.db import executemany

            await executemany(
                "INSERT INTO price_ticks (market_id, token_id, bid, ask, last, ts) VALUES (?,?,?,?,?,?)",
                rows,
            )
