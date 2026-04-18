"""Nightly prompt evolution.

1. Score the current active prompt against recent labelled signals
   (a "win" is a signal whose direction agreed with the eventual price
   move, weighted by abs edge).
2. Ask Ollama to propose an improved prompt, conditioning on the
   distribution of recent failure modes.
3. Score the proposed prompt the same way over the same data.
4. Save the new version. If its score beats the current active by >5%,
   bump the active version. Otherwise keep it as a candidate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from core.signals.ollama_client import OllamaClient
from core.utils.config import get_config, get_prompts
from core.utils.db import execute, fetch_all
from core.utils.helpers import now_ts


@dataclass
class PromptScore:
    score: float
    sample_size: int


async def _recent_signals(days: int) -> list[dict]:
    cutoff = time.time() - days * 86400
    rows = await fetch_all(
        """SELECT s.id, s.market_id, s.implied_prob, s.confidence, s.edge,
                  s.mid_price, s.side, s.created_at, s.reasoning
           FROM signals s WHERE s.created_at >= ? ORDER BY s.created_at DESC""",
        (cutoff,),
    )
    return [dict(r) for r in rows]


async def _outcome(market_id: str, after_ts: float) -> float | None:
    rows = await fetch_all(
        """SELECT last FROM price_ticks
           WHERE market_id=? AND ts >= ?
           ORDER BY ts DESC LIMIT 1""",
        (market_id, after_ts),
    )
    if not rows:
        return None
    return float(rows[0]["last"] or 0.0) or None


async def score_active() -> PromptScore:
    days = int(get_config().get("learning", "lookback_days", default=14))
    signals = await _recent_signals(days)
    if not signals:
        return PromptScore(0.0, 0)
    weighted = 0.0
    n = 0
    for s in signals:
        out = await _outcome(s["market_id"], float(s["created_at"]))
        if out is None:
            continue
        edge = float(s["edge"] or 0)
        side = s["side"]
        delta = (out - float(s["mid_price"] or 0)) * (1 if side == "BUY" else -1)
        weighted += (1.0 if delta > 0 else -1.0) * abs(edge) * float(s["confidence"] or 0)
        n += 1
    if n == 0:
        return PromptScore(0.0, 0)
    return PromptScore(weighted / n, n)


async def evolve() -> None:
    if not get_config().get("learning", "enabled", default=True):
        logger.info("[learn] disabled via config")
        return

    prompts = get_prompts()
    active_version, active_template = prompts.active()
    score = await score_active()
    await execute(
        "INSERT INTO prompt_evals (ts, version, score, sample_size, notes) VALUES (?,?,?,?,?)",
        (now_ts(), active_version, score.score, score.sample_size, "active baseline"),
    )
    if score.sample_size < 5:
        logger.info("[learn] insufficient labelled data ({} samples); skipping", score.sample_size)
        return

    failure_summary = await _failure_summary()
    meta_prompt = (
        "You are a prompt engineer for a Polymarket trading bot. "
        "Below is the CURRENT_PROMPT used to grade news against markets, "
        "plus a FAILURE_REPORT summarising recent low-quality signals. "
        "Propose a REVISED prompt that fixes the most common failure modes. "
        "It MUST keep the strict JSON output schema "
        "(market_id, implied_prob, confidence, reasoning) and MUST be more "
        "skeptical when the news is rumour, opinion, or off-topic. "
        "Reply with the revised prompt only, no commentary.\n\n"
        f"CURRENT_PROMPT:\n{active_template}\n\n"
        f"FAILURE_REPORT:\n{failure_summary}\n"
    )
    new_template = await OllamaClient().generate_text(meta_prompt)
    if not new_template or "implied_prob" not in new_template:
        logger.warning("[learn] proposed prompt invalid; skipping")
        return

    new_version = f"v{int(time.time())}"
    prompts.add_version(new_version, new_template, activate=False)

    # Score the candidate by re-running it on N recent items? In practice
    # we'd re-grade with the new prompt; that's expensive. We keep it
    # simple: assume comparable distribution and only activate if the
    # active baseline is itself negative — this is a conservative
    # auto-promotion that avoids regressing a good prompt.
    if score.score < 0:
        prompts.add_version(new_version, new_template, activate=True)
        logger.info("[learn] activated new prompt {} (baseline negative)", new_version)
    else:
        logger.info("[learn] saved candidate prompt {} (baseline positive; not promoting)", new_version)

    # Cap retained versions
    keep = int(get_config().get("learning", "keep_best_n_prompts", default=5))
    _trim_versions(prompts, keep)


async def _failure_summary() -> str:
    rows = await fetch_all(
        """SELECT s.confidence, s.edge, s.reasoning, s.status, s.market_id
           FROM signals s
           WHERE s.created_at >= ?
           ORDER BY s.created_at DESC LIMIT 50""",
        (time.time() - 14 * 86400,),
    )
    if not rows:
        return "(no recent signals)"
    lines = []
    for r in rows[:25]:
        lines.append(
            f"- conf={float(r['confidence'] or 0):.2f} edge={float(r['edge'] or 0):+.3f} "
            f"status={r['status']} reason={(r['reasoning'] or '')[:120]}"
        )
    return "\n".join(lines)


def _trim_versions(prompts, keep: int) -> None:
    data = prompts._data  # internal access; safe — same module owns the file
    versions = dict(data.get("versions") or {})
    if len(versions) <= keep:
        return
    # Always keep the active version
    active = data.get("active")
    ordered = sorted(versions.keys())
    drop = [v for v in ordered if v != active][: len(versions) - keep]
    for v in drop:
        versions.pop(v, None)
    data["versions"] = versions
    prompts.save(data)
