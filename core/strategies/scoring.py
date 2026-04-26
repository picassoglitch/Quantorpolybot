"""Shared Ollama scoring wrapper used by all three lanes.

Given a market + some evidence text, returns (true_prob, confidence,
reasoning) or None if Ollama can't produce a parseable answer. Reuses
the active prompt template from `prompts.yaml`.

Three call paths are exposed:

  - ``score``: Ollama-or-None. Original API; callers (longshot,
    resolution_day) check for None and skip on failure.
  - ``score_with_timeout``: race Ollama against an explicit per-call
    budget; on timeout/None fall back to ``heuristic.score``. Returns
    a Score with ``source`` set to ``"ollama"`` / ``"heuristic"`` /
    ``"timeout"``. Used by the event lane.
  - ``score_with_fallback``: same fallback semantics as
    ``score_with_timeout`` PLUS a preemptive heuristic when the fast
    tier's queue is saturated (``OllamaClient.fast_queue_saturated()``).
    Used by the scalping lane so a stalled GPU never produces zero-entry
    scans.

Each call takes a ``tier`` argument that selects which Ollama model
handles the request: ``"fast"`` for latency-sensitive paths (event
sniper, scalping re-scores), ``"deep"`` for analysis-heavy paths
(signal pipeline, longshot). Callers pick their own tier — this module
does not guess.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.markets.cache import Market
from core.signals.context import build_market_context
from core.signals.ollama_client import OllamaClient, deep_realtime_enabled
from core.strategies import heuristic
from core.utils.config import get_prompts
from core.utils.helpers import clamp, safe_float


@dataclass
class Score:
    true_prob: float
    confidence: float
    reasoning: str
    # 'ollama' = real LLM response
    # 'heuristic' = keyword fallback (any cause: no Ollama call attempted,
    #               unparseable response, fast queue saturated, etc.)
    # 'timeout' = wait_for fired before Ollama responded; the heuristic
    #             still ran but the source flag preserves the cause so
    #             scan logs and downstream gates can distinguish a slow
    #             GPU from "no LLM signal available".
    source: str


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


def _heuristic_as_score(text: str, market: Market, source: str) -> Score:
    """Run the keyword heuristic and wrap it in a Score with the given
    ``source`` tag. The heuristic itself never raises; this helper just
    standardises the conversion so all fallback paths agree on field
    semantics."""
    h = heuristic.score(text, market)
    return Score(
        true_prob=h.implied_prob,
        confidence=h.confidence,
        reasoning=h.reasoning,
        source=source,
    )


async def score(
    market: Market,
    text: str,
    client: OllamaClient | None = None,
    tier: str = "deep",
) -> Score | None:
    """Run Ollama on (market, text). Returns None on any failure — the
    lane decides whether that means skip or use a fallback. Most callers
    should prefer ``score_with_fallback`` so a stalled Ollama doesn't
    silently zero out the lane's entries.

    If the caller asks for tier=``deep`` but ``deep_realtime_enabled``
    is False, returns None without dispatching. This is the kill
    switch for CPU-only deployments where a 7B+ model would burn
    20-45s per realtime call. Lanes that need a fallback should use
    ``score_with_fallback`` instead — it produces a heuristic Score
    in this case rather than None.
    """
    if tier == "deep" and not deep_realtime_enabled():
        logger.debug(
            "[scoring] deep tier disabled in realtime; skipping market={}",
            market.market_id,
        )
        return None
    client = client or OllamaClient()
    caller = _tier_call(client, tier)
    context = await _gather_context(market)
    try:
        result = await caller(
            _build_prompt(market, text, context), tag=str(market.market_id),
        )
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
    Score — caller checks `.source` and `.confidence` to decide.

    Does NOT preempt on fast-queue saturation; the event lane checks
    ``OllamaClient.fast_queue_saturated()`` itself before calling so it
    can log the saturation reason explicitly. Use
    ``score_with_fallback`` if you want the saturation check folded in.
    """
    if tier == "deep" and not deep_realtime_enabled():
        logger.debug(
            "[scoring] deep tier disabled in realtime; heuristic for market={}",
            market.market_id,
        )
        return _heuristic_as_score(text, market, source="heuristic")
    client = client or OllamaClient()
    caller = _tier_call(client, tier)
    context = await _gather_context(market)
    start = time.perf_counter()
    market_tag = str(market.market_id)
    try:
        result = await asyncio.wait_for(
            caller(_build_prompt(market, text, context), tag=market_tag),
            timeout=timeout_seconds,
        )
        parsed = _parse(result)
        if parsed is not None:
            logger.debug(
                "[scoring] tier={} market={} source=ollama latency_ms={:.0f}",
                tier, market_tag, (time.perf_counter() - start) * 1000.0,
            )
            return parsed
        # Ollama returned but unparseable -> fallback.
        logger.info(
            "[scoring] tier={} market={} ollama returned malformed JSON — "
            "heuristic fallback",
            tier, market_tag,
        )
        return _heuristic_as_score(text, market, source="heuristic")
    except asyncio.TimeoutError:
        logger.warning(
            "[scoring] tier={} market={} ollama timeout (>{:.0f}s) — "
            "heuristic fallback",
            tier, market_tag, timeout_seconds,
        )
        return _heuristic_as_score(text, market, source="timeout")
    except Exception as e:
        logger.warning(
            "[scoring] tier={} market={} ollama failed ({}) — heuristic fallback",
            tier, market_tag, type(e).__name__,
        )
        return _heuristic_as_score(text, market, source="heuristic")


async def score_with_fallback(
    market: Market,
    text: str,
    *,
    client: OllamaClient | None = None,
    tier: str = "fast",
    timeout_seconds: float | None = None,
) -> Score:
    """Always-returns-a-Score variant for lanes that should never freeze
    on a stalled Ollama.

    Behaviour:

      1. If ``tier == "fast"`` and ``client.fast_queue_saturated()`` is
         True at call time, skip Ollama entirely and return the keyword
         heuristic (``source="heuristic"``). This is the cheap-path fix
         for the April 2026 incident where scalping/event scans logged
         ``scored=0`` for an entire 10-minute window because every fast
         call was queued behind a slow deep call.

      2. Otherwise call the appropriate tier with an ``asyncio.wait_for``
         budget. On timeout the heuristic is returned with
         ``source="timeout"`` so scan summaries can distinguish "slow
         GPU" from "no LLM call attempted".

      3. On parseable Ollama success, return the Ollama Score
         (``source="ollama"``). On unparseable / exception, return the
         heuristic with ``source="heuristic"``.

    ``timeout_seconds=None`` (default) uses the tier's configured budget
    from ``ollama.<tier>_timeout_seconds`` so callers don't have to
    duplicate the per-tier constants.
    """
    client = client or OllamaClient()
    market_tag = str(market.market_id)

    # Master kill-switch: deep tier is structurally too slow for the
    # realtime path on CPU-only deployments. Skip Ollama entirely.
    if tier == "deep" and not deep_realtime_enabled():
        logger.debug(
            "[scoring] tier=deep disabled in realtime; heuristic for market={}",
            market_tag,
        )
        return _heuristic_as_score(text, market, source="heuristic")

    # Preempt on fast-tier saturation. The fast queue is the only one
    # with an alert threshold today (queue_depth_alert in config); we
    # only short-circuit fast-tier callers here. Deep-tier callers under
    # contention still benefit from the wait_for + heuristic fallback
    # below — they just don't get the fully-skip-Ollama optimization.
    if tier == "fast" and client.fast_queue_saturated():
        logger.info(
            "[scoring] tier=fast market={} fast queue saturated "
            "(pending={}) — heuristic fallback (no ollama call)",
            market_tag, OllamaClient.pending_fast,
        )
        return _heuristic_as_score(text, market, source="heuristic")

    if timeout_seconds is None:
        # Tier defaults: fast 25s, deep 60s, validator 60s — matches
        # _timeout_for so the per-call budget here doesn't drift from
        # the underlying Ollama wait_for budget.
        timeout_seconds = client._timeout_for(tier)  # noqa: SLF001
    return await score_with_timeout(
        market=market,
        text=text,
        timeout_seconds=safe_float(timeout_seconds),
        client=client,
        tier=tier,
    )
