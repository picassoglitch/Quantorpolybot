"""PredictIt cross-reference feed.

Pulls the public PredictIt market dump, fuzzy-matches each contract
against our active Polymarket questions by entity / phase / candidate
rules (see ``core.feeds.predictit_match``), and stores every match in
the ``cross_references`` table. Whenever the PredictIt vs. Polymarket
implied-probability divergence exceeds the configured
``auto_signal_divergence`` threshold, an additional synthesized
``feed_item`` is inserted (source=``predictit_xref``) so the existing
Ollama signal pipeline picks it up — keeping the cross-reference logic
decoupled from risk/orders.

PredictIt's dump is a single JSON list (no pagination) and refreshes
every 60s on their side, so polling every 10 minutes is plenty.

Threading note (2026-04-26 incident):

  The matching loop is ``len(predictit_contracts) × len(active_polymarket_markets)``
  invocations of ``predictit_match`` — typically ~750 × ~800 = 600 000
  pure-CPU calls per cycle. Running that synchronously on the event
  loop produced ``loop_lag_ms=2000`` for two consecutive watchdog
  ticks every poll, which tripped the DEGRADED state and missed the
  health_check job by 5–25 seconds.

  Fix: extract the matching as a pure-CPU function
  (``_filter_predictit_markets_sync``) and call it via
  ``asyncio.to_thread``. DB writes stay on the main loop where the
  aiosqlite connection lives — only the CPU-heavy regex / set-ops
  loop runs in the worker thread. ``predictit_match`` is already
  thread-safe (it's a pure function over its arguments).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from core.feeds.predictit_match import match as predictit_match
from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, now_ts, safe_float
from core.utils.watchdog import is_degraded

DEGRADED_POLL_MULTIPLIER = 2

API_URL = "https://www.predictit.org/api/marketdata/all/"
_HEADERS = {
    "User-Agent": "NexoPolyBot/0.1 (+https://github.com/local)",
    "Accept": "application/json",
}


@dataclass(frozen=True)
class _MatchResult:
    """One accepted PredictIt ↔ Polymarket pair, with everything the
    DB-writing code needs after the thread returns. Frozen so the
    container survives the thread boundary unmodifiably."""
    poly_id: str
    poly_q: str
    pi_name: str
    pi_price: float
    poly_mid: float
    divergence: float
    match_reason: str


class PredictItFeed:
    component = "feed.predictit"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        cfg = get_config().get("feeds", "predictit") or {}
        if not cfg.get("enabled", True):
            logger.info("[predictit] disabled")
            return
        poll = int(cfg.get("poll_seconds", 600))
        threshold = float(cfg.get("fuzzy_match_threshold", 0.40))
        auto_div = float(cfg.get("auto_signal_divergence", 0.08))
        backoff = Backoff(base=10, cap=600)

        logger.info(
            "[predictit] starting; poll={}s match>={} signal-div>={}",
            poll, threshold, auto_div,
        )
        async with httpx.AsyncClient(timeout=30.0, headers=_HEADERS) as client:
            while not self._stop.is_set():
                try:
                    matched, signaled = await self._poll(client, threshold, auto_div)
                    if matched or signaled:
                        logger.info(
                            "[predictit] matched {} contracts, {} signaled",
                            matched, signaled,
                        )
                    backoff.reset()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    d = backoff.next_delay()
                    logger.warning(
                        "[predictit] error ({}), sleeping {:.1f}s: {}",
                        type(e).__name__, d, e,
                    )
                    await self._sleep(d)
                    continue
                await self._sleep(poll * DEGRADED_POLL_MULTIPLIER if is_degraded() else poll)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _poll(
        self, client: httpx.AsyncClient, threshold: float, auto_div: float
    ) -> tuple[int, int]:
        # ---- I/O on main loop ----
        r = await client.get(API_URL)
        r.raise_for_status()
        payload = r.json()
        markets = payload.get("markets") or []
        if not markets:
            return 0, 0

        poly_rows = await fetch_all(
            "SELECT market_id, question, best_bid, best_ask, last_price "
            "FROM markets WHERE active=1"
        )
        if not poly_rows:
            return 0, 0
        poly_index: list[tuple[str, str, float | None]] = [
            (str(row["market_id"]), row["question"] or "", _mid_from_row(row))
            for row in poly_rows
        ]

        # ---- Pure CPU on a worker thread ----
        # `threshold` retained for backwards-compat in the call signature
        # so config tweaks don't need a migration; the rule-based matcher
        # ignores it. ``_filter_predictit_markets_sync`` is intentionally
        # synchronous — it must NOT await anything (no DB, no network,
        # no asyncio primitives). See the module docstring for why.
        matches, rejected = await asyncio.to_thread(
            _filter_predictit_markets_sync, markets, poly_index,
        )

        # ---- DB writes back on main loop ----
        ts = now_ts()
        matched = 0
        signaled = 0
        for m in matches:
            await execute(
                """INSERT INTO cross_references
                (polymarket_id, source, source_market_name,
                 source_price, poly_price, divergence, fetched_at)
                VALUES (?,?,?,?,?,?,?)""",
                (
                    m.poly_id, "predictit", m.pi_name[:200],
                    m.pi_price, m.poly_mid, m.divergence, ts,
                ),
            )
            matched += 1
            if m.divergence >= auto_div:
                if await self._emit_signal(
                    m.poly_id, m.poly_q, m.pi_name, m.pi_price, m.poly_mid,
                    m.divergence, m.match_reason,
                ):
                    signaled += 1

        if rejected:
            logger.info(
                "[predictit] filtered {} contracts as non-matches (new stricter rules)",
                rejected,
            )
        return matched, signaled

    async def _emit_signal(
        self,
        poly_id: str,
        poly_q: str,
        pi_name: str,
        pi_price: float,
        poly_mid: float,
        divergence: float,
        match_reason: str,
    ) -> bool:
        # Bucket by ~hour so the same persistent divergence doesn't spam
        # the pipeline every cycle, but a *new* large move still fires.
        bucket = int(now_ts() // 3600)
        h = url_hash(f"predictit_xref:{poly_id}:{bucket}:{round(divergence, 3)}")
        if await fetch_one("SELECT id FROM feed_items WHERE url_hash=?", (h,)):
            return False
        title = (
            f"PredictIt vs Polymarket divergence {divergence:.2%} on: "
            f"{poly_q[:120]}"
        )
        summary = (
            f"PredictIt contract '{pi_name}' trades at {pi_price:.3f} while "
            f"Polymarket mid is {poly_mid:.3f} (divergence {divergence:.3f}, "
            f"match: {match_reason}). Consider whether new info on "
            f"either side justifies the gap."
        )
        meta = {
            "linked_market_id": poly_id,
            "predictit_name": pi_name,
            "predictit_price": pi_price,
            "polymarket_mid": poly_mid,
            "divergence": divergence,
            "match_reason": match_reason,
        }
        await execute(
            """INSERT OR IGNORE INTO feed_items
            (url_hash, source, title, summary, url, published_at, ingested_at, meta)
            VALUES (?,?,?,?,?,?,?,?)""",
            (
                h,
                "predictit_xref",
                title,
                summary[:1000],
                f"https://www.predictit.org/markets/detail/{poly_id}",
                now_ts(),
                now_ts(),
                json.dumps(meta),
            ),
        )
        return True


# ============================================================
# Pure-CPU filter — runs in a worker thread via asyncio.to_thread.
# MUST NOT await anything. MUST NOT touch the DB.
# ============================================================


def _filter_predictit_markets_sync(
    markets: list[dict[str, Any]],
    poly_index: list[tuple[str, str, float | None]],
) -> tuple[list[_MatchResult], int]:
    """Iterate every PredictIt contract and try to match it against the
    Polymarket index. Returns ``(accepted_matches, rejected_count)``.

    Pure CPU. Safe to call from a worker thread because:
      - inputs are plain Python objects (lists/dicts/tuples)
      - ``predictit_match`` is a pure function
      - logging via loguru is thread-safe
      - no shared mutable state is read or written

    Performance shape (April 2026 numbers):
      ~750 contracts × ~800 active polymarket questions = ~600 000
      matcher invocations per cycle. Each invocation runs a small
      regex + a few set ops, ~10–50 µs apiece on the dev machine.
      Running on the main loop spiked ``loop_lag_ms`` to 2000+ for
      multiple consecutive watchdog ticks; on a worker thread the
      main loop sees ~0 ms lag during the same cycle.
    """
    matches: list[_MatchResult] = []
    rejected = 0

    for market in markets:
        if not isinstance(market, dict):
            continue
        market_name = market.get("name") or ""
        for contract in market.get("contracts") or []:
            if not isinstance(contract, dict):
                continue
            pi_name = (contract.get("name") or market_name or "").strip()
            if not pi_name:
                continue
            pi_price = _contract_price(contract)
            if pi_price is None:
                continue
            pi_text = f"{market_name} {pi_name}".strip()

            # First accepted Polymarket question wins. The rule set is
            # strict so multiple candidates rarely tie; if they do, the
            # first active market is fine — the alternative (silently
            # picking one via ranking) loses traceability.
            accepted: tuple[str, str, float | None, str] | None = None
            last_reason = ""
            for poly_id, poly_q, poly_mid in poly_index:
                ok, reason = predictit_match(pi_text, poly_q)
                if not ok:
                    last_reason = reason
                    continue
                accepted = (poly_id, poly_q, poly_mid, reason)
                break
            if accepted is None:
                rejected += 1
                logger.debug(
                    "[predictit] reject {!r}: {}", pi_text[:80], last_reason,
                )
                continue
            poly_id, poly_q, poly_mid, match_reason = accepted
            if poly_mid is None:
                continue

            divergence = abs(pi_price - poly_mid)
            matches.append(_MatchResult(
                poly_id=poly_id,
                poly_q=poly_q,
                pi_name=pi_name,
                pi_price=pi_price,
                poly_mid=poly_mid,
                divergence=divergence,
                match_reason=match_reason,
            ))

    return matches, rejected


def _contract_price(contract: dict[str, Any]) -> float | None:
    """Return the implied YES price for a PredictIt contract, falling
    back across the various price fields PredictIt exposes."""
    for key in ("bestBuyYesCost", "lastTradePrice", "lastClosePrice"):
        v = contract.get(key)
        if v is None:
            continue
        f = safe_float(v, default=-1.0)
        if 0.0 < f < 1.0:
            return f
    return None


def _mid_from_row(row: Any) -> float | None:
    """Best-effort mid price from a markets row. Prefers (bid+ask)/2,
    falls back to last_price; returns None if nothing usable."""
    bid = safe_float(row["best_bid"], default=0.0)
    ask = safe_float(row["best_ask"], default=0.0)
    if bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    last = safe_float(row["last_price"], default=0.0)
    if 0.0 < last < 1.0:
        return last
    return None
