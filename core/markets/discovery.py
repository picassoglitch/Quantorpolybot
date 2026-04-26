"""Polymarket Gamma API market discovery. Walks paginated /markets and
upserts active markets into the local SQLite cache.

Every refresh enforces a hard cap on the universe so downstream
subscribers (WS, signal pipeline, lanes) never see more than
``MAX_MARKETS`` markets. Gamma returned ~15k active markets at peak,
which overwhelmed the WS and starved the event loop — the filter
below drops that to O(2k) that actually have enough volume and a
near-enough resolution to be tradable.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import httpx
from loguru import logger

from core.markets.categorize import infer_category
from core.utils.config import get_config
from core.utils.db import connect, execute
from core.utils.helpers import Backoff, now_ts, safe_float

# Defaults; overrideable via markets.* in config.yaml.
MAX_MARKETS = 2000
MIN_VOLUME_24H = 10_000.0
MAX_DAYS_TO_RESOLVE = 120
# Categories we refuse to ingest. 'unknown' never appears in our
# infer_category output but we accept it as a Gamma value too.
_EXCLUDED_CATEGORIES = {"", "unknown"}

# Per-page fetch budget. A whole refresh must complete in under
# TOTAL_REFRESH_BUDGET or we abandon the remainder and let the next
# poll pick up.
PER_PAGE_TIMEOUT = 15.0
TOTAL_REFRESH_BUDGET = 30.0


class MarketDiscovery:
    component = "markets.discovery"

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        # Timestamp of the last successful refresh. Seeded to 0 so a cold
        # start (no boot-time refresh) triggers an immediate refresh in
        # run(). run.py's boot path calls refresh_once() explicitly before
        # starting run() as a task; without this accounting, run() would
        # fire a second back-to-back refresh as its first iteration,
        # pinning the Gamma API + event loop for ~30s and dropping a
        # health_check tick by 15-20s.
        self._last_refresh_ts: float = 0.0

    async def run(self) -> None:
        cfg = get_config().get("markets") or {}
        poll = int(cfg.get("refresh_seconds", 300))
        backoff = Backoff(base=5, cap=300)
        logger.info("[markets] discovery loop started poll={}s", poll)
        while not self._stop.is_set():
            # Only refresh if enough time has passed since the last one.
            # If run.py just did a boot refresh, wait out the remainder
            # of `poll` rather than duplicating the work.
            since = now_ts() - self._last_refresh_ts
            if self._last_refresh_ts > 0 and since < poll:
                await self._sleep(poll - since)
                continue
            try:
                count = await self.refresh_once()
                logger.info("[markets] refreshed {} markets", count)
                backoff.reset()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = backoff.next_delay()
                logger.warning(
                    "[markets] refresh error, sleeping {:.1f}s: {} ({})",
                    delay, type(e).__name__, e,
                )
                await self._sleep(delay)
                continue
            await self._sleep(poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def refresh_once(self) -> int:
        async with self._lock:
            cfg = get_config().get("markets") or {}
            base = cfg.get("gamma_url", "https://gamma-api.polymarket.com/markets")
            page_size = int(cfg.get("page_size", 500))
            max_pages = int(cfg.get("max_pages", 30))
            active_only = bool(cfg.get("active_only", True))
            page_delay = float(cfg.get("page_delay_seconds", 0.2))
            cap = int(cfg.get("max_markets", MAX_MARKETS))
            min_vol = float(cfg.get("min_volume_24h", MIN_VOLUME_24H))
            max_days = float(cfg.get("max_days_to_resolve", MAX_DAYS_TO_RESOLVE))
            total_budget = float(cfg.get("refresh_total_budget_seconds", TOTAL_REFRESH_BUDGET))

            raw_markets: list[dict[str, Any]] = []
            offset = 0
            started = asyncio.get_event_loop().time()

            async with httpx.AsyncClient(timeout=PER_PAGE_TIMEOUT) as client:
                for page_num in range(max_pages):
                    # Abandon the refresh rather than keep paging past the
                    # budget; next poll cycle picks up the rest. Keeps the
                    # event loop from being pinned for minutes on a slow
                    # Gamma response.
                    if asyncio.get_event_loop().time() - started > total_budget:
                        logger.warning(
                            "[markets] refresh budget ({:.0f}s) exceeded after "
                            "{} pages; abandoning remainder",
                            total_budget, page_num,
                        )
                        break
                    params: dict[str, Any] = {"limit": page_size, "offset": offset}
                    if active_only:
                        params["active"] = "true"
                        params["closed"] = "false"
                    try:
                        # Per-page hard wait_for on top of the client
                        # timeout — belt and braces so a connection that
                        # hangs past the read timeout can't pin the task.
                        r = await asyncio.wait_for(
                            client.get(base, params=params), timeout=PER_PAGE_TIMEOUT,
                        )
                        r.raise_for_status()
                        payload = r.json()
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[markets] page {} (offset={}) timed out after {:.0f}s, skipping",
                            page_num, offset, PER_PAGE_TIMEOUT,
                        )
                        offset += page_size
                        if page_delay > 0:
                            await asyncio.sleep(page_delay)
                        continue
                    except (httpx.HTTPError, httpx.TimeoutException) as e:
                        logger.warning(
                            "[markets] page {} (offset={}) failed ({}): {}",
                            page_num, offset, type(e).__name__, e,
                        )
                        offset += page_size
                        if page_delay > 0:
                            await asyncio.sleep(page_delay)
                        continue
                    except Exception as e:
                        logger.warning(
                            "[markets] page {} (offset={}) failed ({}): {}",
                            page_num, offset, type(e).__name__, e,
                        )
                        offset += page_size
                        if page_delay > 0:
                            await asyncio.sleep(page_delay)
                        continue
                    markets = payload if isinstance(payload, list) else payload.get("data", [])
                    if not markets:
                        break
                    raw_markets.extend(markets)
                    if len(markets) < page_size:
                        break
                    offset += page_size
                    # Yield between pages so Ollama / WS tasks get a chance
                    # to run.
                    if page_delay > 0:
                        await asyncio.sleep(page_delay)

            # Filtering + row-build over 15k raw markets is pure CPU
            # (date parse x5 format attempts, regex-based category
            # inference, JSON encoding). On the main event loop this
            # blocked APScheduler jobs by 10+ seconds and pushed the
            # websockets keepalive past its 60s pong budget. Moving it
            # to a worker thread keeps the loop responsive — asyncio.to_thread
            # yields on every GIL switch (~100 bytecodes), so Ollama,
            # DB commits and WS pongs continue to progress.
            kept, dropped, rows = await asyncio.to_thread(
                _build_kept_rows,
                raw_markets,
                min_vol=min_vol,
                max_days=max_days,
                cap=cap,
            )
            logger.info(
                "[markets] kept {} of {} total (filtered: volume={}, date={}, inactive={}, category={})",
                len(kept), len(raw_markets),
                dropped["volume"], dropped["date"], dropped["inactive"],
                dropped["category"],
            )
            if not kept:
                self._last_refresh_ts = now_ts()
                return 0
            if rows:
                await self._commit_universe(rows)
            # Marked after the commit so an exception mid-commit leaves
            # the timestamp stale and triggers an immediate retry on
            # the next run() iteration rather than waiting out the poll.
            self._last_refresh_ts = now_ts()
            return len(rows)

    @staticmethod
    def _row_from_market(m: dict[str, Any]) -> tuple | None:
        market_id = m.get("id") or m.get("conditionId") or m.get("condition_id")
        if not market_id:
            return None
        question = (m.get("question") or m.get("title") or "").strip()
        slug = m.get("slug") or m.get("market_slug") or ""
        category = m.get("category") or (m.get("tags") or [{}])[0].get("label", "") if m.get("tags") else m.get("category", "")
        if isinstance(category, dict):
            category = category.get("label", "")
        # Gamma frequently returns blank / generic categories; infer from
        # the question text so the risk engine's correlation cap actually
        # groups sensibly (sports / politics / crypto / macro / other).
        if not category or not str(category).strip():
            category = infer_category(question)
        active = 1 if m.get("active", True) and not m.get("closed", False) else 0
        close_time = m.get("end_date") or m.get("endDate") or m.get("close_time")
        token_ids = m.get("clobTokenIds") or m.get("tokenIds") or m.get("tokens")
        if isinstance(token_ids, str):
            try:
                token_ids = json.loads(token_ids)
            except json.JSONDecodeError:
                token_ids = [token_ids]
        # Gamma returns `outcomes` in lockstep with `clobTokenIds`, but
        # the Yes outcome is NOT always at index 0 — some markets come
        # back as ["No", "Yes"]. Everything downstream (Market.yes_token,
        # shadow.open_position's token selection, current_price) assumes
        # token_ids[0] IS the Yes outcome. When we got that wrong every
        # SELL landed on the NO token, causing -195%/-267% stop-loss
        # exits because the monitor read the NO-frame ask as the
        # position's exit price. Normalise here so the whole system
        # can keep its "index 0 = Yes" invariant safely.
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                outcomes = None
        if (
            isinstance(token_ids, list)
            and isinstance(outcomes, list)
            and len(token_ids) == len(outcomes) == 2
        ):
            labels = [str(o).strip().lower() for o in outcomes]
            if labels[0] == "no" and labels[1] == "yes":
                token_ids = [token_ids[1], token_ids[0]]
        if isinstance(token_ids, list):
            token_ids_json = json.dumps([str(t) for t in token_ids])
        else:
            token_ids_json = json.dumps([])
        liquidity = safe_float(m.get("liquidity") or m.get("liquidityNum"))
        last_price = safe_float(m.get("lastTradePrice") or m.get("last_price"))
        bid = safe_float(m.get("bestBid") or m.get("best_bid"))
        ask = safe_float(m.get("bestAsk") or m.get("best_ask"))
        return (
            str(market_id),
            question,
            slug,
            str(category or ""),
            active,
            str(close_time or ""),
            token_ids_json,
            bid,
            ask,
            last_price,
            liquidity,
            now_ts(),
        )

    @staticmethod
    async def _commit_universe(rows: list[tuple]) -> None:
        """Apply the new capped universe in one transaction on one connection.

        Previous version split this into two methods (``_upsert`` +
        ``_deactivate_others``) each opening its own aiosqlite connection.
        The deactivate step also used a TEMP table + ``NOT IN (SELECT ...)``
        anti-join, which is the slowest shape SQLite offers for this task
        on a growing markets table. For 268 kept rows we measured ~22s of
        wall-clock held across the two connections; during that window
        every other writer (poly_ws price ticks, signal pipeline) piled
        up on ``busy_timeout``, and the event loop missed APScheduler
        ticks by 7-19s.

        The replacement is two statements in one transaction:
          1. bulk flip: ``UPDATE markets SET active=0 WHERE active=1``
             — indexed on ``active``, single pass, no anti-join.
          2. UPSERT the kept rows — their ``active=excluded.active`` sets
             them back to 1, so the net effect is identical to the old
             two-stage process but holds the write lock once, briefly.
        Readers between the two statements inside the transaction can't
        see the intermediate "everyone inactive" state — SQLite's WAL
        snapshot isolation hides the uncommitted UPDATE.
        """
        if not rows:
            return
        async with connect() as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute("UPDATE markets SET active=0 WHERE active=1")
                await db.executemany(
                    """INSERT INTO markets
                    (market_id, question, slug, category, active, close_time, token_ids,
                     best_bid, best_ask, last_price, liquidity, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(market_id) DO UPDATE SET
                      question=excluded.question,
                      slug=excluded.slug,
                      category=excluded.category,
                      active=excluded.active,
                      close_time=excluded.close_time,
                      token_ids=excluded.token_ids,
                      best_bid=excluded.best_bid,
                      best_ask=excluded.best_ask,
                      last_price=excluded.last_price,
                      liquidity=excluded.liquidity,
                      updated_at=excluded.updated_at
                    """,
                    rows,
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def deactivate_stale(self, max_age_seconds: float = 86400) -> int:
        """Mark markets inactive if we haven't seen them in a refresh cycle."""
        cutoff = now_ts() - max_age_seconds
        await execute("UPDATE markets SET active=0 WHERE updated_at < ?", (cutoff,))
        return 0


def _parse_end_date(raw: Any) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Gamma uses full ISO-8601 with Z; accept the few variants in the wild.
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Fromisoformat is more permissive but can still reject older formats.
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _volume_24h(m: dict[str, Any]) -> float:
    # Gamma surfaces volume under several keys depending on edge version.
    # Fall back gracefully so a schema nudge doesn't drop the whole filter.
    for key in ("volume24hr", "volume_24h", "volumeNum", "volume", "volume24Hr"):
        v = safe_float(m.get(key))
        if v > 0:
            return v
    return 0.0


def _build_kept_rows(
    raw: list[dict[str, Any]],
    *,
    min_vol: float,
    max_days: float,
    cap: int,
) -> tuple[list[dict[str, Any]], dict[str, int], list[tuple]]:
    """Pure-CPU bundle run off the event loop: filter → rank → rows.

    Returns ``(kept, dropped, rows)`` where ``rows`` are already null-
    filtered tuples ready for the upsert. Kept separate from
    ``MarketDiscovery._row_from_market`` so the module-level function
    can be ``asyncio.to_thread``-ed directly without dragging ``self``
    into the worker thread.
    """
    kept, dropped = _apply_universe_filter(
        raw, min_vol=min_vol, max_days=max_days, cap=cap,
    )
    rows = [MarketDiscovery._row_from_market(m) for m in kept]
    rows = [r for r in rows if r is not None]
    return kept, dropped, rows


def _apply_universe_filter(
    raw: list[dict[str, Any]],
    *,
    min_vol: float,
    max_days: float,
    cap: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (kept, dropped_counts). Keeps only active, near-resolving,
    liquid markets in a recognised category and hard-caps the result at
    ``cap`` by a simple (volume / days_to_resolve) rank."""
    now = datetime.now(timezone.utc)
    dropped = {"volume": 0, "date": 0, "inactive": 0, "category": 0}
    scored: list[tuple[float, dict[str, Any]]] = []
    for m in raw:
        active = bool(m.get("active", True)) and not bool(m.get("closed", False))
        if not active:
            dropped["inactive"] += 1
            continue
        vol = _volume_24h(m)
        if vol < min_vol:
            dropped["volume"] += 1
            continue
        end = _parse_end_date(
            m.get("end_date") or m.get("endDate") or m.get("close_time")
        )
        if end is None:
            dropped["date"] += 1
            continue
        delta = end - now
        days = delta.total_seconds() / 86400.0
        if days < 0 or days > max_days:
            dropped["date"] += 1
            continue
        # Determine category; infer from question when Gamma is blank.
        cat_raw = m.get("category") or ""
        if isinstance(cat_raw, dict):
            cat_raw = cat_raw.get("label", "")
        elif isinstance(cat_raw, list):
            cat_raw = (cat_raw[0].get("label", "") if cat_raw and isinstance(cat_raw[0], dict) else "")
        cat = str(cat_raw).strip().lower()
        if not cat:
            cat = infer_category((m.get("question") or m.get("title") or "")).lower()
        if cat in _EXCLUDED_CATEGORIES:
            dropped["category"] += 1
            continue
        # Rank: higher volume and nearer resolution wins. Guard against
        # div-by-zero on same-day resolutions.
        days_safe = max(days, 0.5)
        score = vol / days_safe
        scored.append((score, m))

    scored.sort(key=lambda t: t[0], reverse=True)
    kept = [m for _, m in scored[:cap]]
    return kept, dropped
