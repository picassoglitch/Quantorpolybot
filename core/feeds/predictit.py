"""PredictIt cross-reference feed.

Pulls the public PredictIt market dump, fuzzy-matches each contract
against our active Polymarket questions by Jaccard keyword overlap,
and stores every match in the ``cross_references`` table. Whenever the
PredictIt vs. Polymarket implied-probability divergence exceeds the
configured ``auto_signal_divergence`` threshold, an additional
synthesized ``feed_item`` is inserted (source=``predictit_xref``) so
the existing Ollama signal pipeline picks it up — keeping the cross-
reference logic decoupled from risk/orders.

PredictIt's dump is a single JSON list (no pagination) and refreshes
every 60s on their side, so polling every 10 minutes is plenty.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.hashing import url_hash
from core.utils.helpers import Backoff, jaccard, keywords, now_ts, safe_float

API_URL = "https://www.predictit.org/api/marketdata/all/"
_HEADERS = {
    "User-Agent": "Quantorpolybot/0.1 (+https://github.com/local)",
    "Accept": "application/json",
}


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
                    logger.exception("[predictit] error, sleeping {:.1f}s: {}", d, e)
                    await self._sleep(d)
                    continue
                await self._sleep(poll)

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
        # Pre-tokenize Polymarket questions once; we'll Jaccard-match
        # every PredictIt contract against this list.
        poly_index: list[tuple[str, str, set[str], float | None]] = []
        for row in poly_rows:
            q = row["question"] or ""
            mid = _mid_from_row(row)
            poly_index.append((str(row["market_id"]), q, keywords(q), mid))

        matched = 0
        signaled = 0
        ts = now_ts()
        for market in markets:
            if not isinstance(market, dict):
                continue
            for contract in market.get("contracts") or []:
                if not isinstance(contract, dict):
                    continue
                pi_name = (contract.get("name") or market.get("name") or "").strip()
                if not pi_name:
                    continue
                pi_price = _contract_price(contract)
                if pi_price is None:
                    continue
                pi_kw = keywords(f"{market.get('name', '')} {pi_name}")
                if not pi_kw:
                    continue
                # Pick the best Polymarket question above the threshold.
                best: tuple[float, str, str, float | None] | None = None
                for poly_id, poly_q, poly_kw, poly_mid in poly_index:
                    score = jaccard(pi_kw, poly_kw)
                    if score < threshold:
                        continue
                    if best is None or score > best[0]:
                        best = (score, poly_id, poly_q, poly_mid)
                if best is None:
                    continue
                score, poly_id, poly_q, poly_mid = best
                if poly_mid is None:
                    continue
                divergence = abs(pi_price - poly_mid)
                await execute(
                    """INSERT INTO cross_references
                    (polymarket_id, source, source_market_name,
                     source_price, poly_price, divergence, fetched_at)
                    VALUES (?,?,?,?,?,?,?)""",
                    (
                        poly_id,
                        "predictit",
                        pi_name[:200],
                        pi_price,
                        poly_mid,
                        divergence,
                        ts,
                    ),
                )
                matched += 1
                if divergence >= auto_div:
                    if await self._emit_signal(
                        poly_id, poly_q, pi_name, pi_price, poly_mid, divergence, score
                    ):
                        signaled += 1
        return matched, signaled

    async def _emit_signal(
        self,
        poly_id: str,
        poly_q: str,
        pi_name: str,
        pi_price: float,
        poly_mid: float,
        divergence: float,
        match_score: float,
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
            f"match score {match_score:.2f}). Consider whether new info on "
            f"either side justifies the gap."
        )
        meta = {
            "linked_market_id": poly_id,
            "predictit_name": pi_name,
            "predictit_price": pi_price,
            "polymarket_mid": poly_mid,
            "divergence": divergence,
            "match_score": match_score,
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
