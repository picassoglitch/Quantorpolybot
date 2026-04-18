"""aiosqlite wrapper. Schema migration on first connect; thin helpers for
the modules that don't want raw SQL boilerplate everywhere.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterable, Sequence

import aiosqlite
from loguru import logger

from core.utils.config import get_config, root_dir

_DB_LOCK = asyncio.Lock()
_DB_PATH: Path | None = None

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS markets (
        market_id TEXT PRIMARY KEY,
        question TEXT,
        slug TEXT,
        category TEXT,
        active INTEGER DEFAULT 1,
        close_time TEXT,
        token_ids TEXT,
        best_bid REAL,
        best_ask REAL,
        last_price REAL,
        liquidity REAL,
        updated_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active)",
    "CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category)",
    """
    CREATE TABLE IF NOT EXISTS feed_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url_hash TEXT UNIQUE,
        source TEXT,
        title TEXT,
        summary TEXT,
        url TEXT,
        published_at REAL,
        ingested_at REAL,
        meta TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_feed_items_ingested ON feed_items(ingested_at)",
    """
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feed_item_id INTEGER,
        market_id TEXT,
        implied_prob REAL,
        confidence REAL,
        edge REAL,
        mid_price REAL,
        side TEXT,
        size_usd REAL,
        reasoning TEXT,
        prompt_version TEXT,
        created_at REAL,
        outcome REAL,
        pnl_usd REAL,
        status TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_id)",
    "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)",
    """
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        market_id TEXT,
        token_id TEXT,
        side TEXT,
        price REAL,
        size REAL,
        size_usd REAL,
        clob_order_id TEXT,
        status TEXT,
        created_at REAL,
        updated_at REAL,
        dry_run INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_market ON orders(market_id)",
    """
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT,
        token_id TEXT,
        side TEXT,
        size REAL,
        avg_price REAL,
        opened_at REAL,
        closed_at REAL,
        realized_pnl_usd REAL,
        status TEXT,
        UNIQUE(market_id, token_id, status)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        clob_trade_id TEXT,
        market_id TEXT,
        token_id TEXT,
        side TEXT,
        price REAL,
        size REAL,
        ts REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS price_ticks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT,
        token_id TEXT,
        bid REAL,
        ask REAL,
        last REAL,
        ts REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_price_market_ts ON price_ticks(market_id, ts)",
    """
    CREATE TABLE IF NOT EXISTS system_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        level TEXT,
        component TEXT,
        message TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS config_overrides (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        reason TEXT,
        old_yaml TEXT,
        new_yaml TEXT,
        sharpe_before REAL,
        sharpe_after REAL,
        pnl_before REAL,
        pnl_after REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS health_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        component TEXT,
        ok INTEGER,
        detail TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prompt_evals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL,
        version TEXT,
        score REAL,
        sample_size INTEGER,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cross_references (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        polymarket_id TEXT,
        source TEXT,
        source_market_name TEXT,
        source_price REAL,
        poly_price REAL,
        divergence REAL,
        fetched_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cross_refs_market ON cross_references(polymarket_id)",
    "CREATE INDEX IF NOT EXISTS idx_cross_refs_div ON cross_references(divergence)",
    "CREATE INDEX IF NOT EXISTS idx_cross_refs_fetched ON cross_references(fetched_at)",
    "CREATE INDEX IF NOT EXISTS idx_feed_items_source ON feed_items(source)",
]


def _resolve_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is not None:
        return _DB_PATH
    cfg = get_config()
    p = Path(cfg.get("db_path", default="polybot.db"))
    if not p.is_absolute():
        p = root_dir() / p
    _DB_PATH = p
    return p


async def init_db() -> None:
    path = _resolve_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        for stmt in SCHEMA:
            await conn.execute(stmt)
        await conn.commit()
    logger.info("DB initialised at {}", path)


def connect():
    """Open a fresh aiosqlite connection.

    Returns the proxy from ``aiosqlite.connect``; use it as
    ``async with connect() as conn:``. Do NOT pre-await it before the
    ``async with`` — aiosqlite would try to start its background thread
    twice and raise ``RuntimeError: threads can only be started once``.
    """
    return aiosqlite.connect(_resolve_db_path())


async def execute(sql: str, params: Sequence[Any] | None = None) -> int:
    """Execute one statement; returns lastrowid."""
    async with _DB_LOCK:
        async with connect() as conn:
            conn.row_factory = aiosqlite.Row
            cur = await conn.execute(sql, params or ())
            await conn.commit()
            return cur.lastrowid or 0


async def executemany(sql: str, rows: Iterable[Sequence[Any]]) -> None:
    async with _DB_LOCK:
        async with connect() as conn:
            await conn.executemany(sql, list(rows))
            await conn.commit()


async def fetch_all(sql: str, params: Sequence[Any] | None = None) -> list[aiosqlite.Row]:
    async with connect() as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql, params or ())
        rows = await cur.fetchall()
        await cur.close()
        return list(rows)


async def fetch_one(sql: str, params: Sequence[Any] | None = None) -> aiosqlite.Row | None:
    async with connect() as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql, params or ())
        row = await cur.fetchone()
        await cur.close()
        return row


async def log_system(level: str, component: str, message: str) -> None:
    from core.utils.helpers import now_ts

    await execute(
        "INSERT INTO system_log (ts, level, component, message) VALUES (?,?,?,?)",
        (now_ts(), level, component, message),
    )
