"""End-to-end signal pipeline:

   feed_items (unprocessed) -> candidate markets -> Ollama JSON ->
   edge calc -> persisted signal -> handed to execution

Runs as a single async loop polling for new feed_items rows.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from core.execution.orders import OrderEngine
from core.markets.cache import get_market
from core.risk.rules import RiskEngine, RiskRejection
from core.signals.candidates import candidates_for, serialize_candidates
from core.signals.ollama_client import OllamaClient
from core.utils.config import get_config, get_prompts
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import clamp, now_ts, safe_float


class SignalPipeline:
    component = "signals.pipeline"

    def __init__(self, order_engine: OrderEngine, risk: RiskEngine) -> None:
        self._stop = asyncio.Event()
        self._ollama = OllamaClient()
        self._orders = order_engine
        self._risk = risk
        self._cursor_id = 0  # last processed feed_items.id

    async def run(self) -> None:
        await self._init_cursor()
        logger.info("[signals] pipeline started; cursor={}", self._cursor_id)
        while not self._stop.is_set():
            try:
                processed = await self._process_batch()
                if processed == 0:
                    await self._sleep(2)
                else:
                    logger.info("[signals] processed {} items", processed)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[signals] loop error: {}", e)
                await self._sleep(5)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _init_cursor(self) -> None:
        # Resume from the latest signal we have (so a restart doesn't
        # re-process the entire history).
        row = await fetch_one(
            "SELECT MAX(feed_item_id) AS m FROM signals"
        )
        self._cursor_id = int((row["m"] or 0) if row else 0)

    async def _process_batch(self) -> int:
        rows = await fetch_all(
            "SELECT * FROM feed_items WHERE id > ? ORDER BY id ASC LIMIT 25",
            (self._cursor_id,),
        )
        if not rows:
            return 0
        for row in rows:
            try:
                await self._process_item(dict(row))
            except Exception as e:
                logger.exception("[signals] item {} failed: {}", row["id"], e)
            self._cursor_id = max(self._cursor_id, int(row["id"]))
        return len(rows)

    async def _process_item(self, item: dict[str, Any]) -> None:
        text = f"{item.get('title', '')}\n{item.get('summary', '')}".strip()
        if not text:
            return
        markets = await candidates_for(text)
        if not markets:
            return
        prompt_version, template = get_prompts().active()
        prompt = template.format(
            news_item=json.dumps(
                {
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "summary": (item.get("summary") or "")[:1000],
                    "url": item.get("url"),
                },
                ensure_ascii=False,
            ),
            candidates=json.dumps(serialize_candidates(markets), ensure_ascii=False),
        )
        result = await self._ollama.generate_json(prompt)
        if not result:
            return
        market_id = result.get("market_id")
        if not market_id or market_id == "null":
            return
        market = await get_market(str(market_id))
        if not market:
            return

        implied = clamp(safe_float(result.get("implied_prob")), 0.0, 1.0)
        confidence = clamp(safe_float(result.get("confidence")), 0.0, 1.0)
        reasoning = (result.get("reasoning") or "")[:500]
        mid = market.mid
        edge = implied - mid
        side = "BUY" if edge > 0 else "SELL"

        signal_id = await execute(
            """INSERT INTO signals
            (feed_item_id, market_id, implied_prob, confidence, edge, mid_price,
             side, size_usd, reasoning, prompt_version, created_at, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                int(item["id"]),
                market.market_id,
                implied,
                confidence,
                edge,
                mid,
                side,
                0.0,
                reasoning,
                prompt_version,
                now_ts(),
                "PENDING",
            ),
        )

        cfg = get_config().get("risk") or {}
        if confidence < float(cfg.get("min_confidence", 0.65)):
            await self._reject(signal_id, "low confidence")
            return
        if abs(edge) < float(cfg.get("min_edge", 0.04)):
            await self._reject(signal_id, "edge below threshold")
            return

        try:
            decision = await self._risk.evaluate(market, side, implied, confidence)
        except RiskRejection as rej:
            await self._reject(signal_id, str(rej))
            return

        await self._orders.submit_signal(
            signal_id=signal_id,
            market=market,
            side=side,
            implied_prob=implied,
            confidence=confidence,
            edge=edge,
            size_usd=decision.size_usd,
            target_price=decision.target_price,
        )

    @staticmethod
    async def _reject(signal_id: int, reason: str) -> None:
        await execute(
            "UPDATE signals SET status=?, reasoning=COALESCE(reasoning,'') || ?  WHERE id=?",
            ("REJECTED", f" | rejected: {reason}", signal_id),
        )
