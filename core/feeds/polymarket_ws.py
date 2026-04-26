"""Polymarket CLOB WebSocket subscriber. Streams live mid prices for all
active markets and writes price_ticks rows. Auto-reconnects with
jittered backoff.

The connect path is hard-bounded on every dimension so a flaky CLOB
host can't starve the event loop: handshake via ``asyncio.wait_for``,
subscription batched in small chunks with an ``asyncio.sleep`` between
each (so other coroutines get time), recv() on a 60s timeout that
pings + continues on idle.
"""

from __future__ import annotations

import asyncio
import json
import random
from contextlib import suppress
from typing import Any, Iterable

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosed, InvalidHandshake, WebSocketException

from core.utils.config import get_config
from core.utils.db import fetch_all
from core.utils.helpers import now_ts, safe_float

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Module-level timestamp of the last successful message. Read by the
# event-loop watchdog to decide whether to trigger a forced reconnect.
LAST_MESSAGE_TS: float = 0.0

# Network errors that are expected in the wild (partial outages, ISP
# NAT timeouts, CLOB deploys). Logged as WARNING one-liners — tracebacks
# for these are pure noise.
_EXPECTED_WS_ERRORS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionClosed,
    InvalidHandshake,
    WebSocketException,
    OSError,
)


class PolymarketWS:
    component = "feed.polymarket_ws"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        # Setting this interrupts the current _connect_and_stream so the
        # outer reconnect loop re-establishes the socket. Used by the
        # watchdog when messages have stopped arriving.
        self._force_reconnect = asyncio.Event()
        # Count of consecutive failed connect attempts since the last
        # successful stream. The watchdog reads this to back off after
        # repeated force-reconnects land on a dead endpoint.
        self._consecutive_failures = 0

    def request_reconnect(self) -> None:
        """External trigger — called by the watchdog when messages have
        gone silent for too long. Safe to call from any task."""
        self._force_reconnect.set()

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    async def run(self) -> None:
        cfg = get_config().get("feeds", "polymarket_ws") or {}
        if not cfg.get("enabled", True):
            logger.info("[poly_ws] disabled")
            return
        self._batch_size = int(cfg.get("subscribe_batch", 25))
        self._batch_sleep = float(cfg.get("subscribe_batch_sleep_seconds", 0.5))
        self._connect_timeout = float(cfg.get("connect_timeout_seconds", 10.0))
        self._backoff_cap = float(cfg.get("backoff_cap_seconds", 60.0))
        # Keepalive tuning. Defaults (20/20) were too aggressive for the
        # Polymarket CLOB: with ~1600 tokens and 25/frame subscribe
        # batches, the initial subscribe phase alone can span 30+ seconds,
        # during which the server's pong reply to our ping slips past the
        # 20s window and we get ``1011 keepalive ping timeout`` before we
        # ever reach the recv loop. 30/60 is the safe middle ground; set
        # ``ping_interval_seconds: null`` to disable client-initiated
        # pings entirely and rely on server-initiated keepalives + the
        # watchdog's silence detector.
        ping_interval_raw = cfg.get("ping_interval_seconds", 30)
        self._ping_interval: float | None = (
            None if ping_interval_raw is None else float(ping_interval_raw)
        )
        self._ping_timeout = float(cfg.get("ping_timeout_seconds", 60))
        self._attempt = 0
        # Render ping_interval=None as "off" so the banner doesn't say
        # "ping=Nones/..." when client-initiated pings are disabled.
        ping_label = "off" if self._ping_interval is None else f"{self._ping_interval}s"
        logger.info(
            "[poly_ws] starting (batch={}, batch_sleep={}s, connect_timeout={}s, "
            "ping={}/{}s)",
            self._batch_size, self._batch_sleep, self._connect_timeout,
            ping_label, self._ping_timeout,
        )
        while not self._stop.is_set():
            token_ids = await self._active_token_ids()
            if not token_ids:
                await self._sleep(30)
                continue
            try:
                # Shield keeps a stray CancelledError on the parent task
                # from tearing down mid-message. Outer stop() still wakes
                # because _stop.is_set() is checked inside.
                await asyncio.shield(self._connect_and_stream(token_ids))
                # Clean return — socket closed cleanly or stop was set.
                self._attempt = 0
                self._consecutive_failures = 0
            except asyncio.CancelledError:
                raise
            except _EXPECTED_WS_ERRORS as e:
                self._consecutive_failures += 1
                self._attempt += 1
                delay = self._backoff_delay()
                logger.warning(
                    "[poly_ws] disconnected ({}), reconnect in {:.1f}s: {}",
                    type(e).__name__, delay, e,
                )
                await self._sleep(delay)
            except Exception as e:
                self._consecutive_failures += 1
                self._attempt += 1
                delay = self._backoff_delay()
                logger.exception(
                    "[poly_ws] unexpected error, reconnect in {:.1f}s: {}", delay, e,
                )
                await self._sleep(delay)

    def _backoff_delay(self) -> float:
        """Jittered exponential: base = min(cap, 2 * attempt); jitter 0-5s.
        Jitter is *non-deterministic* on purpose — two parallel failures
        shouldn't retry in lock-step against the same server."""
        base = min(self._backoff_cap, 2.0 * self._attempt)
        jitter = random.uniform(0.0, 5.0)
        return base + jitter

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
        # de-dupe; the CLOB WS rejects oversized subscriptions and just
        # closes the socket, so caller batches in self._batch_size chunks.
        return list(dict.fromkeys(ids))

    async def _connect_and_stream(self, token_ids: Iterable[str]) -> None:
        all_ids = list(token_ids)
        # Hard-bounded handshake. Without asyncio.wait_for, a TLS stall
        # can pin this task for the full socket-level timeout while the
        # rest of the event loop starves.
        ws_cm = websockets.connect(
            WS_URL,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
        )
        ws = await asyncio.wait_for(ws_cm.__aenter__(), timeout=self._connect_timeout)
        self._force_reconnect.clear()
        global LAST_MESSAGE_TS
        try:
            await self._subscribe_in_batches(ws, all_ids)
            logger.info(
                "[poly_ws] connected; subscribed to {} tokens in batches of {}",
                len(all_ids), self._batch_size,
            )
            while not self._stop.is_set():
                if self._force_reconnect.is_set():
                    logger.warning("[poly_ws] force_reconnect requested by watchdog")
                    return
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                except asyncio.TimeoutError:
                    # No message in 60s — send a ping and keep looping.
                    # Don't update LAST_MESSAGE_TS; the watchdog treats
                    # an empty channel as a stall, which is accurate.
                    with suppress(Exception):
                        await ws.ping()
                    continue
                except ConnectionClosed:
                    logger.warning("[poly_ws] connection closed by server")
                    return
                LAST_MESSAGE_TS = now_ts()
                await self._handle_message(raw)
        finally:
            with suppress(Exception):
                await ws_cm.__aexit__(None, None, None)

    async def _subscribe_in_batches(self, ws: Any, token_ids: list[str]) -> None:
        """Send assets_ids in smaller frames. 25/frame with 500ms sleeps
        gives the loop time to service Ollama/DB tasks while the CLOB
        wires up each batch on its end."""
        if not token_ids:
            return
        batch = self._batch_size
        for i in range(0, len(token_ids), batch):
            chunk = token_ids[i : i + batch]
            sub = {"type": "MARKET", "assets_ids": chunk}
            await ws.send(json.dumps(sub))
            if i + batch < len(token_ids) and self._batch_sleep > 0:
                await asyncio.sleep(self._batch_sleep)

    async def _handle_message(self, raw: str) -> None:
        # One WS frame can carry dozens to hundreds of events. Doing one
        # execute() per event spawned one aiosqlite connection per event
        # and pinned the event loop long enough that the websockets
        # library's keepalive task starved — the server eventually
        # closed us with `1011 keepalive ping timeout`, which we were
        # reconnecting from every 60-90s. Coalescing into two
        # executemany() calls drops that to two connections per frame
        # regardless of event count.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        events: list[dict[str, Any]] = data if isinstance(data, list) else [data]
        ts = now_ts()
        tick_rows: list[tuple[Any, ...]] = []
        update_rows: list[tuple[Any, ...]] = []
        for ev in events:
            asset_id = ev.get("asset_id") or ev.get("token_id")
            market_id = ev.get("market") or ev.get("market_id")
            bid = safe_float(ev.get("best_bid") or ev.get("bid"))
            ask = safe_float(ev.get("best_ask") or ev.get("ask"))
            last = safe_float(ev.get("price") or ev.get("last_price"))
            if not asset_id and not market_id:
                continue
            tick_rows.append((market_id, asset_id, bid, ask, last, ts))
            if market_id:
                update_rows.append((bid, ask, last, ts, market_id))
        if update_rows or tick_rows:
            from core.utils.db import executemany

            if update_rows:
                await executemany(
                    """UPDATE markets SET best_bid=?, best_ask=?, last_price=?, updated_at=?
                       WHERE market_id=?""",
                    update_rows,
                )
            if tick_rows:
                await executemany(
                    "INSERT INTO price_ticks (market_id, token_id, bid, ask, last, ts) VALUES (?,?,?,?,?,?)",
                    tick_rows,
                )
