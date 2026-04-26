"""aiosqlite wrapper. Schema migration on first connect; thin helpers for
the modules that don't want raw SQL boilerplate everywhere.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Sequence

import aiosqlite
from loguru import logger

from core.utils.config import get_config, root_dir

# SQLite busy_timeout: how long the engine waits on a locked DB before
# raising SQLITE_BUSY. WAL mode handles concurrent readers natively;
# this covers the write-write race across coroutines / threads / loops.
# Applied per-connection (see _PRAGMAS + connect()) because these are
# all connection-local pragmas. 30s absorbs the discovery refresh's
# bulk-UPDATE window without bleeding into feed/monitor writers.
_BUSY_TIMEOUT_MS = 30_000

# Pragmas applied on EVERY fresh connection. synchronous=NORMAL + WAL
# is the safe fast-commit combo; default synchronous=FULL forces an
# fsync per commit which is brutal on Windows and was the main source
# of lock-hold time in the previous setup. temp_store=MEMORY keeps the
# discovery module's TEMP tables off disk. journal_mode=WAL is
# persistent at the file level, so this is a no-op after init_db
# but costs nothing to re-assert.
_PRAGMAS: tuple[str, ...] = (
    f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}",
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-8000",
)

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
        status TEXT,
        category TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_id)",
    "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)",
    # Existing databases predate the category column; add it idempotently.
    # SQLite's ADD COLUMN is a no-op if the column already exists — but it
    # errors rather than silently succeeds, so we swallow OperationalError.
    "__SAFE_ALTER__ ALTER TABLE signals ADD COLUMN category TEXT",
    # The legacy `orders`, `positions`, `executions` tables are gone — the
    # three shadow lanes are the sole entry path now and both shadow and
    # real fills land in `shadow_positions` (distinguished by `is_real`).
    # Drop them on old DBs so queries can't quietly hit stale rows.
    "__SAFE_ALTER__ DROP TABLE IF EXISTS orders",
    "__SAFE_ALTER__ DROP TABLE IF EXISTS positions",
    "__SAFE_ALTER__ DROP TABLE IF EXISTS executions",
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
    # ---- News enrichment columns (feeds/enrich.py) ----
    # Added for the self-hosted news pipeline (RSS + local Ollama).
    # ALTER TABLE ADD COLUMN with a NULL default is O(1) on SQLite — it
    # only rewrites the schema header, not per-row data. Existing rows
    # stay NULL, which the enrich loop treats as "unenriched" and will
    # drain in the background without blocking startup. No explicit
    # backfill migration is needed; the enricher backfills opportunistically.
    "__SAFE_ALTER__ ALTER TABLE feed_items ADD COLUMN enriched_json TEXT",
    "__SAFE_ALTER__ ALTER TABLE feed_items ADD COLUMN market_relevance REAL",
    "__SAFE_ALTER__ ALTER TABLE feed_items ADD COLUMN enriched_at REAL",
    "__SAFE_ALTER__ ALTER TABLE feed_items ADD COLUMN source_weight REAL",
    # Partial index: only rows the enricher is yet to touch. Keeps the
    # drain query fast even when feed_items grows past 100k rows.
    "CREATE INDEX IF NOT EXISTS idx_feed_items_enrich_queue "
    "ON feed_items(id) WHERE enriched_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_feed_items_relevance "
    "ON feed_items(market_relevance, ingested_at)",
    # ---- Three-lane trading (shadow + real share this table) ----
    # Primary key is (lane, mode) so both modes keep independent budgets.
    # Old databases keyed on `lane` alone — drop so the new PK takes effect
    # cleanly. Losing lane_capital state is safe: `init_lane_capital()`
    # recomputes `deployed` from open shadow_positions on every boot.
    "__SAFE_ALTER__ DROP TABLE IF EXISTS lane_capital",
    """
    CREATE TABLE IF NOT EXISTS lane_capital (
        lane TEXT NOT NULL,
        mode TEXT NOT NULL,
        total_budget REAL,
        deployed REAL,
        available REAL,
        paused_until REAL,
        updated_at REAL,
        PRIMARY KEY (lane, mode)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS shadow_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        strategy TEXT NOT NULL,
        market_id TEXT NOT NULL,
        token_id TEXT,
        side TEXT,
        entry_price REAL,
        size_usd REAL,
        size_shares REAL,
        entry_ts REAL,
        entry_reason TEXT,
        entry_latency_ms REAL,
        cited_evidence_ids TEXT,
        evidence_snapshot TEXT,
        conviction_trajectory TEXT,
        true_prob_entry REAL,
        confidence_entry REAL,
        last_rescored_ts REAL,
        last_price REAL,
        last_price_ts REAL,
        unrealized_pnl_usd REAL,
        status TEXT,
        close_price REAL,
        close_ts REAL,
        close_reason TEXT,
        realized_pnl_usd REAL,
        what_if_held_pnl_usd REAL,
        resolved_outcome REAL,
        is_real INTEGER DEFAULT 0,
        clob_order_id TEXT,
        clob_close_order_id TEXT,
        peak_pnl_pct REAL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_shadow_strategy ON shadow_positions(strategy, status)",
    "CREATE INDEX IF NOT EXISTS idx_shadow_market ON shadow_positions(market_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_shadow_status_ts ON shadow_positions(status, close_ts)",
    # Validator cross-check snapshot (JSON): original scorer prob/reasoning
    # alongside validator prob/reasoning + the size adjustment that was
    # applied ("halved", "direction_skip", "ok"). Migrated idempotently.
    # ALTERs MUST run before the is_real index below or pre-existing DBs
    # that lack the column fail to create the index.
    "__SAFE_ALTER__ ALTER TABLE shadow_positions ADD COLUMN validator_snapshot TEXT",
    "__SAFE_ALTER__ ALTER TABLE shadow_positions ADD COLUMN is_real INTEGER DEFAULT 0",
    "__SAFE_ALTER__ ALTER TABLE shadow_positions ADD COLUMN clob_order_id TEXT",
    "__SAFE_ALTER__ ALTER TABLE shadow_positions ADD COLUMN clob_close_order_id TEXT",
    # Peak PnL % since entry — drives trailing-TP exits. Updated on every
    # price tick via shadow.update_price. Default 0 means "no peak yet"
    # so the trailing exit simply waits until the activate threshold is
    # crossed. Reset to 0 on startup is fine — worst case we re-arm the
    # trailing stop with the current PnL as a new peak baseline.
    "__SAFE_ALTER__ ALTER TABLE shadow_positions ADD COLUMN peak_pnl_pct REAL DEFAULT 0",
    "CREATE INDEX IF NOT EXISTS idx_shadow_is_real ON shadow_positions(is_real, status)",
    # ---- Tiered Ollama call telemetry ----
    # One row per LLM call. Used by the /api/ollama-stats endpoint and
    # the Models section on the overview page to show per-model latency,
    # timeout rate, and call volume. Cheap — a single INSERT per call.
    """
    CREATE TABLE IF NOT EXISTS ollama_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT,
        call_type TEXT,
        latency_ms REAL,
        success INTEGER,
        tokens_in INTEGER,
        tokens_out INTEGER,
        called_at REAL,
        error TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ollama_stats_called ON ollama_stats(called_at)",
    "CREATE INDEX IF NOT EXISTS idx_ollama_stats_model ON ollama_stats(model, called_at)",
    # ---- Per-source trust calibration ----
    # Populated by core.learning.source_trust.calibrate(), read by the
    # hallucination guard so well-calibrated sources count for more
    # than one "raw" source. `trust_weight` is the multiplier applied
    # to each distinct-source count in _count_recent_sources. Keep this
    # table small — one row per source name.
    """
    CREATE TABLE IF NOT EXISTS source_stats (
        source TEXT PRIMARY KEY,
        samples INTEGER,
        wins INTEGER,
        losses INTEGER,
        win_rate REAL,
        weighted_pnl REAL,
        trust_weight REAL,
        last_computed_at REAL
    )
    """,
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
        for pragma in _PRAGMAS:
            await conn.execute(pragma)
        await conn.execute("PRAGMA foreign_keys=ON")
        for stmt in SCHEMA:
            if stmt.lstrip().startswith("__SAFE_ALTER__"):
                # Idempotent ALTER: tolerate "duplicate column name" on
                # databases that were created after the column was added
                # to CREATE TABLE. New DBs silently skip; old DBs migrate.
                real = stmt.replace("__SAFE_ALTER__", "", 1).strip()
                try:
                    await conn.execute(real)
                except aiosqlite.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
                continue
            await conn.execute(stmt)
        await conn.commit()
    logger.info("DB initialised at {}", path)


@asynccontextmanager
async def connect() -> AsyncIterator[aiosqlite.Connection]:
    """Open a fresh aiosqlite connection with the project's pragmas applied.

    Use as ``async with connect() as conn:``. Every connection gets
    ``busy_timeout``, ``synchronous=NORMAL``, ``temp_store=MEMORY`` and
    the WAL assertion before it's handed out — that's what makes writer
    contention wait instead of raising ``database is locked``. A previous
    asyncio.Lock-based serializer was removed because it was loop-bound
    (the dashboard runs in its own thread/loop); SQLite's own
    busy_timeout handles cross-loop serialization correctly.
    """
    async with aiosqlite.connect(_resolve_db_path()) as conn:
        for pragma in _PRAGMAS:
            await conn.execute(pragma)
        yield conn


async def execute(sql: str, params: Sequence[Any] | None = None) -> int:
    """Execute one statement; returns lastrowid."""
    async with connect() as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql, params or ())
        await conn.commit()
        return cur.lastrowid or 0


async def executemany(sql: str, rows: Iterable[Sequence[Any]]) -> None:
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
