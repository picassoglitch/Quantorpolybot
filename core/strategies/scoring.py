"""Shared Ollama scoring wrapper used by all three lanes.

Given a market + some evidence text, returns (true_prob, confidence,
reasoning) or None if Ollama can't produce a parseable answer. Reuses
the active prompt template from `prompts.yaml`.

Scalping and longshot lanes call `score` directly. The event lane calls
`score_with_timeout` which races Ollama against the fallback window and
falls back to `heuristic.score` on timeout.

Each call takes a ``tier`` argument that selects which Ollama model
handles the request: ``"fast"`` for latency-sensitive paths (event
sniper, scalping re-scores), ``"deep"`` for analysis-heavy paths
(signal pipeline, longshot). Callers pick their own tier — this module
does not guess.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.markets.cache import Market
from core.signals.context import build_market_context
from core.signals.ollama_client import OllamaClient
from core.strategies import heuristic
from core.utils.config import get_prompts
from core.utils.helpers import clamp, safe_float


@dataclass
class Score:
    true_prob: float
    confidence: float
    reasoning: str
    source: str  # 'ollama' | 'heuristic' | 'timeout'


def _build_prompt(
    market: Market,
    text: str,
    context: dict[str, Any] | None = None,
) -> str:
    version, template = get_prompts().active()
    news_item = json.dumps(
        {"source": "lane_query", "title": text[:300], "summary": text[:1000]},
        ensure_ascii=False,
    )
    candidates = json.dumps(
        [
            {
                "market_id": market.market_id,
                "question": market.question,
                "category": market.category,
                "mid": market.mid,
            }
        ],
        ensure_ascii=False,
    )
    ctx_json = json.dumps(context or {}, ensure_ascii=False)
    return (
        template
        .replace("{news_item}", news_item)
        .replace("{candidates}", candidates)
        .replace("{context}", ctx_json)
    )


def _parse(result: dict[str, Any] | None) -> Score | None:
    if not result:
        return None
    implied = result.get("implied_prob")
    if implied is None:
        return None
    try:
        true_prob = clamp(safe_float(implied), 0.0, 1.0)
        confidence = clamp(safe_float(result.get("confidence")), 0.0, 1.0)
    except (TypeError, ValueError):
        return None
    reasoning = (result.get("reasoning") or "")[:400]
    return Score(true_prob=true_prob, confidence=confidence, reasoning=reasoning, source="ollama")


def _tier_call(client: OllamaClient, tier: str):
    """Return the coroutine factory for the requested tier. Defaults to
    deep for unknown tiers so a config typo doesn't silently downgrade."""
    if tier == "fast":
        return client.fast_score
    if tier == "validator":
        return client.validate
    return client.deep_score


async def _gather_context(market: Market) -> dict[str, Any]:
    """Best-effort context assembly. Never raises — returns {} so the
    prompt still has a `CONTEXT:` block even when the DB is empty."""
    try:
        return await build_market_context(market.market_id)
    except Exception as e:
        logger.debug("[scoring] context build failed: {}", e)
        return {}


async def score(
    market: Market,
    text: str,
    client: OllamaClient | None = None,
    tier: str = "deep",
) -> Score | None:
    """Run Ollama on (market, text). Returns None on any failure — the
    lane decides whether that means skip or use a fallback."""
    client = client or OllamaClient()
    caller = _tier_call(client, tier)
    context = await _gather_context(market)
    try:
        result = await caller(_build_prompt(market, text, context))
    except Exception as e:
        logger.debug("[scoring] ollama error ({}): {}", tier, e)
        return None
    return _parse(result)


async def score_with_timeout(
    market: Market,
    text: str,
    timeout_seconds: float,
    client: OllamaClient | None = None,
    tier: str = "fast",
) -> Score:
    """Event-lane path: race Ollama against timeout, fall back to the
    keyword heuristic so we don't miss the trade. Always returns a
    Score — caller checks `.source` and `.confidence` to decide."""
    client = client or OllamaClient()
    caller = _tier_call(client, tier)
    context = await _gather_context(market)
    try:
        result = await asyncio.wait_for(
            caller(_build_prompt(market, text, context)),
            timeout=timeout_seconds,
        )
        parsed = _parse(result)
        if parsed is not None:
            return parsed
        # Ollama returned but unparseable -> fallback.
        logger.info("[scoring] ollama returned malformed; using heuristic")
    except asyncio.TimeoutError:
        logger.warning(
            "[scoring] ollama timeout (>{:.0f}s) — heuristic fallback", timeout_seconds,
        )
    except Exception as e:
        logger.warning("[scoring] ollama failed: {} — heuristic fallback", e)
    h = heuristic.score(text, market)
    return Score(
        true_prob=h.implied_prob,
        confidence=h.confidence,
        reasoning=h.reasoning,
        source="heuristic",
    )
