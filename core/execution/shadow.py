"""Execution engine for the three lanes.

Single entry point for both modes:

  * `mode=shadow` → simulated fill at the inside quote (BUY at ask,
    SELL at bid). Position is recorded in `shadow_positions` with
    `is_real=0` and the allocator draws from `shadow_capital`.
  * `mode=real`   → submits a GTC limit order to the Polymarket CLOB
    via `core.execution.clob_client`. On submit-success the position
    is recorded with `is_real=1, status=PENDING_FILL`, `clob_order_id`
    populated. Fill/cancel reconciliation happens in
    `core.execution.monitor.OrderMonitor`. The allocator draws from
    `real_capital`.

Both modes write to the same `shadow_positions` table so the dashboard,
audit trail, learning loop, and risk manager are unchanged — only
`is_real` distinguishes the rows.

Every position carries the full audit trail: strategy, entry_reason,
cited_evidence_ids, evidence_snapshot, conviction_trajectory,
entry_latency_ms, close_reason, and (when the market resolves)
what_if_held_pnl.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

from core.execution import allocator, clob_client
from core.markets.cache import Market
from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import now_ts, safe_float
from core.utils.logging import audit
from core.utils.prices import PriceSnapshot, current_price


async def _total_shadow_exposure() -> float:
    row = await fetch_one(
        """SELECT COALESCE(SUM(size_usd), 0) AS e
           FROM shadow_positions WHERE status IN ('OPEN','PENDING_FILL')"""
    )
    return safe_float(row["e"] if row else 0.0)


async def _global_ceiling_violation(size_usd: float) -> str | None:
    """Belt-and-suspenders check against `risk.max_position_usd` and
    `risk.max_total_exposure_usd`. Returns a reason string if the entry
    would breach the ceiling, or None if cleared."""
    risk_cfg = get_config().get("risk") or {}
    max_position = safe_float(risk_cfg.get("max_position_usd", 500))
    max_total = safe_float(risk_cfg.get("max_total_exposure_usd", 12000))
    if size_usd > max_position:
        return f"per-trade ceiling breached: {size_usd:.2f} > {max_position:.2f}"
    exposure = await _total_shadow_exposure()
    if exposure + size_usd > max_total:
        return (
            f"total exposure ceiling breached: "
            f"{exposure:.2f} + {size_usd:.2f} > {max_total:.2f}"
        )
    return None


@dataclass
class ShadowPosition:
    id: int
    strategy: str
    market_id: str
    token_id: str
    side: str
    entry_price: float
    size_usd: float
    size_shares: float
    entry_ts: float
    entry_reason: str
    entry_latency_ms: float
    true_prob_entry: float
    confidence_entry: float
    last_rescored_ts: float
    last_price: float
    last_price_ts: float
    unrealized_pnl_usd: float
    status: str
    close_price: float
    close_ts: float
    close_reason: str
    realized_pnl_usd: float
    conviction_trajectory: list[list[float]]
    cited_evidence_ids: list[int]
    is_real: bool = False
    clob_order_id: str = ""
    peak_pnl_pct: float = 0.0

    @property
    def age_seconds(self) -> float:
        return now_ts() - self.entry_ts

    @property
    def age_hours(self) -> float:
        return self.age_seconds / 3600.0

    @property
    def mode(self) -> str:
        return "real" if self.is_real else "shadow"


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "null"


def _json_loads(raw: Any, fallback: Any) -> Any:
    if not raw:
        return fallback
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _row_field(row: Any, key: str, default: Any = None) -> Any:
    """Safely pull a column out of an aiosqlite.Row — Row raises IndexError
    for missing keys, so we fall back for schema additions."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _row_to_position(row: Any) -> ShadowPosition:
    return ShadowPosition(
        id=int(row["id"]),
        strategy=row["strategy"],
        market_id=row["market_id"],
        token_id=row["token_id"] or "",
        side=row["side"] or "",
        entry_price=safe_float(row["entry_price"]),
        size_usd=safe_float(row["size_usd"]),
        size_shares=safe_float(row["size_shares"]),
        entry_ts=safe_float(row["entry_ts"]),
        entry_reason=row["entry_reason"] or "",
        entry_latency_ms=safe_float(row["entry_latency_ms"]),
        true_prob_entry=safe_float(row["true_prob_entry"]),
        confidence_entry=safe_float(row["confidence_entry"]),
        last_rescored_ts=safe_float(row["last_rescored_ts"]),
        last_price=safe_float(row["last_price"]),
        last_price_ts=safe_float(row["last_price_ts"]),
        unrealized_pnl_usd=safe_float(row["unrealized_pnl_usd"]),
        status=row["status"] or "",
        close_price=safe_float(row["close_price"]),
        close_ts=safe_float(row["close_ts"]),
        close_reason=row["close_reason"] or "",
        realized_pnl_usd=safe_float(row["realized_pnl_usd"]),
        conviction_trajectory=_json_loads(row["conviction_trajectory"], []),
        cited_evidence_ids=_json_loads(row["cited_evidence_ids"], []),
        is_real=bool(safe_float(_row_field(row, "is_real", 0))),
        clob_order_id=str(_row_field(row, "clob_order_id", "") or ""),
        peak_pnl_pct=safe_float(_row_field(row, "peak_pnl_pct", 0.0)),
    )


_ACTIVE_STATUSES = ("OPEN", "PENDING_FILL")


async def open_positions_for(strategy: str) -> list[ShadowPosition]:
    """Return lane positions that still count against the budget.
    Includes real orders waiting for a fill so re-entry logic doesn't
    double-up on the same market while the CLOB is still pending."""
    rows = await fetch_all(
        """SELECT * FROM shadow_positions
           WHERE strategy=? AND status IN ('OPEN','PENDING_FILL')""",
        (strategy,),
    )
    return [_row_to_position(r) for r in rows]


async def all_open_positions() -> list[ShadowPosition]:
    rows = await fetch_all(
        "SELECT * FROM shadow_positions WHERE status IN ('OPEN','PENDING_FILL')"
    )
    return [_row_to_position(r) for r in rows]


async def count_open(strategy: str) -> int:
    row = await fetch_one(
        """SELECT COUNT(*) AS n FROM shadow_positions
           WHERE strategy=? AND status IN ('OPEN','PENDING_FILL')""",
        (strategy,),
    )
    return int(row["n"] if row else 0)


async def count_open_for_market(market_id: str) -> int:
    row = await fetch_one(
        """SELECT COUNT(*) AS n FROM shadow_positions
           WHERE market_id=? AND status IN ('OPEN','PENDING_FILL')""",
        (market_id,),
    )
    return int(row["n"] if row else 0)


async def count_open_for_market_in_lane(market_id: str, strategy: str) -> int:
    row = await fetch_one(
        """SELECT COUNT(*) AS n FROM shadow_positions
           WHERE market_id=? AND strategy=? AND status IN ('OPEN','PENDING_FILL')""",
        (market_id, strategy),
    )
    return int(row["n"] if row else 0)


async def open_position(
    *,
    strategy: str,
    market: Market,
    side: str,
    snapshot: PriceSnapshot,
    size_usd: float,
    true_prob: float,
    confidence: float,
    entry_reason: str,
    evidence_ids: Iterable[int],
    evidence_snapshot: dict[str, Any] | None,
    entry_latency_ms: float,
    ollama_client: Any = None,
    validator_text: str = "",
    validator_reasoning: str = "",
) -> int | None:
    """Record a simulated entry. Caller must have already `reserve`d
    capital via the allocator — this function does NOT touch the lane
    budget. Returns the new position id or None on failure.

    If ``ollama_client`` and ``validator_text`` are provided and the
    proposed size is >= ``ollama.validator_high_stakes_usd``, the entry
    is cross-validated by a second LLM. The validator can either halve
    the size (on prob drift) or veto the entry outright (on direction
    mismatch). Both the validator's output and the decision are written
    to ``shadow_positions.validator_snapshot``.
    """
    # Account everything in the YES-token frame — callers pass the YES
    # snapshot, so for consistency the stored token_id must also be YES.
    # SELL (strategic short-YES) is tracked as "sold at YES.bid, close by
    # buying back at YES.ask"; compute_pnl_pct's SELL branch is
    # (entry - current)/entry which matches that convention exactly.
    # Previously SELL stored the NO-token id, so the monitor's
    # current_price() call returned NO.ask (~= 1 - YES.bid) and every
    # SELL position hit stop_loss at -200% on the very first tick.
    token_id = market.yes_token() or market.no_token()
    if not token_id:
        logger.warning("[shadow] no token id for market {}", market.market_id)
        return None
    # BUY at ask (taking liquidity), SELL at bid — both on the YES token.
    entry_price = snapshot.ask if side == "BUY" else snapshot.bid
    if entry_price <= 0 or entry_price >= 1:
        logger.warning("[shadow] bad entry price {} for {}", entry_price, market.market_id)
        return None

    # ---- High-stakes cross-validation ----
    # Only fires when size is at/above the configured threshold AND the
    # caller provided a client + evidence text. Callers can opt out by
    # leaving those None (e.g. the scalping lane, whose sizes are always
    # below the threshold anyway). Failures are non-fatal — an
    # unavailable validator must not starve real entries.
    validator_snapshot: dict[str, Any] | None = None
    ollama_cfg = get_config().get("ollama") or {}
    threshold = safe_float(ollama_cfg.get("validator_high_stakes_usd", 200))
    if ollama_client is not None and validator_text and size_usd >= threshold:
        from core.signals.validator import cross_validate
        try:
            result = await cross_validate(
                client=ollama_client,
                market=market,
                side=side,
                original_true_prob=true_prob,
                original_reasoning=validator_reasoning or "",
                evidence_text=validator_text,
                size_usd=size_usd,
            )
        except Exception as e:
            logger.warning("[shadow] validator errored for {}: {}", market.market_id, e)
            result = None
        if result is not None:
            validator_snapshot = result.to_snapshot()
            if result.decision == "direction_skip":
                # Return None per the existing open_position contract —
                # the caller holds the allocator reservation and is
                # responsible for releasing it. We only audit + log.
                audit(
                    "shadow_validator_skip",
                    strategy=strategy,
                    market_id=market.market_id,
                    size_usd=size_usd,
                    notes=result.notes,
                )
                logger.warning(
                    "[shadow] validator vetoed entry on {} ({})",
                    market.market_id, result.notes,
                )
                return None
            if result.decision == "halved":
                # Release the refund portion directly — the caller only
                # knows about the original reservation amount and will
                # bill the lane for the new (smaller) size on close.
                refund = size_usd - result.adjusted_size
                if refund > 0:
                    await allocator.release(strategy, refund, 0.0)
                size_usd = result.adjusted_size

    # Global ceiling check — last line of defense if a lane config
    # sizes up 10x due to a bug or bad override.
    violation = await _global_ceiling_violation(size_usd)
    if violation:
        logger.error("[shadow] rejected by global ceiling: {}", violation)
        audit(
            "shadow_global_ceiling_rejection",
            strategy=strategy, market_id=market.market_id, size_usd=size_usd,
            reason=violation,
        )
        return None
    size_shares = round(size_usd / entry_price, 4)
    evidence_list = list(evidence_ids)
    traj = [[now_ts(), true_prob, snapshot.mid]]

    # ---- Mode branch ----
    # Shadow mode fills instantly at the quoted price. Real mode submits
    # a GTC limit order to the CLOB and parks the row in PENDING_FILL
    # until the monitor reconciles it. On CLOB-submit failure we bail
    # (caller releases the allocator reservation).
    mode = allocator.current_mode()
    is_real = 1 if mode == "real" else 0
    clob_order_id = ""
    if mode == "real":
        # Watchdog degraded-state guard. When the event loop is under
        # heavy contention (ollama silent, WS reconnecting, task
        # backlog climbing), placing a real order is the worst time
        # to submit: the creds-derivation round-trip could stall for
        # tens of seconds and the fill-price we signed against may
        # be stale by the time it lands. Shadow orders still proceed
        # — they fill instantly off the in-memory quote and are what
        # lets us keep measuring strategy performance through the
        # degraded window. Controlled via limits.require_watchdog_healthy.
        # Imported lazily so shadow-only test envs don't need the
        # watchdog module wired up.
        risk_limits = get_config().get("limits") or {}
        require_healthy = bool(risk_limits.get("require_watchdog_healthy", True))
        if require_healthy:
            try:
                from core.utils.watchdog import is_degraded
                if is_degraded():
                    logger.error(
                        "[shadow] real entry blocked: watchdog degraded "
                        "(strategy={} market={} size=${:.2f}) — "
                        "retry after recovery",
                        strategy, market.market_id, size_usd,
                    )
                    audit(
                        "real_order_blocked_degraded",
                        strategy=strategy, market_id=market.market_id,
                        size_usd=size_usd,
                    )
                    return None
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(
                    "[shadow] watchdog check raised ({}) — proceeding with entry",
                    e,
                )
        # Real-mode SELL on YES token assumes we already hold YES shares
        # to sell. A fresh "SELL YES" entry on Polymarket is normally
        # implemented as BUY NO — we do NOT translate that here and will
        # need to revisit before the first real-mode SELL goes live.
        if side == "SELL":
            logger.warning(
                "[shadow] real-mode SELL on {} uses YES-frame accounting; "
                "CLOB submit assumes an existing YES holding — revisit "
                "before flipping real mode on for short strategies.",
                market.market_id,
            )
        # ensure_ready() runs the py-clob-client build off the event loop.
        # Calling is_ready() here would have been fine once the client is
        # cached, but the FIRST real-order entry would block the loop on
        # create_or_derive_api_creds() — the root of the April 2026
        # "watchdog DEGRADED within 2.5 min of boot" symptom.
        if not await clob_client.ensure_ready():
            logger.error(
                "[shadow] real mode selected but CLOB client isn't ready "
                "(missing POLY_PRIVATE_KEY/POLY_FUNDER_ADDRESS?) — entry skipped"
            )
            audit(
                "real_order_blocked_no_client",
                strategy=strategy, market_id=market.market_id, size_usd=size_usd,
            )
            return None
        result = await clob_client.place_limit_order(
            token_id, side, entry_price, size_shares,
        )
        if not result.ok:
            logger.error(
                "[shadow] CLOB submit failed for {} {} {}: {}",
                strategy, side, market.market_id, result.error,
            )
            audit(
                "real_order_submit_failed",
                strategy=strategy, market_id=market.market_id,
                size_usd=size_usd, error=result.error,
            )
            return None
        clob_order_id = result.clob_order_id or ""

    status = "PENDING_FILL" if mode == "real" else "OPEN"
    pos_id = await execute(
        """INSERT INTO shadow_positions
           (strategy, market_id, token_id, side, entry_price, size_usd,
            size_shares, entry_ts, entry_reason, entry_latency_ms,
            cited_evidence_ids, evidence_snapshot, conviction_trajectory,
            true_prob_entry, confidence_entry, last_rescored_ts,
            last_price, last_price_ts, unrealized_pnl_usd, status,
            validator_snapshot, is_real, clob_order_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            strategy,
            market.market_id,
            token_id,
            side,
            entry_price,
            size_usd,
            size_shares,
            now_ts(),
            entry_reason[:500],
            entry_latency_ms,
            _json_dumps(evidence_list),
            _json_dumps(evidence_snapshot or {}),
            _json_dumps(traj),
            true_prob,
            confidence,
            now_ts(),
            snapshot.mid,
            snapshot.ts,
            0.0,
            status,
            _json_dumps(validator_snapshot) if validator_snapshot else None,
            is_real,
            clob_order_id,
        ),
    )
    logger.warning(
        "[{}] ENTER {} {} {} @ {:.3f} size={:.2f} USD conf={:.2f} true_p={:.2f} ({})",
        "real" if is_real else "shadow",
        strategy, side, market.market_id, entry_price, size_usd,
        confidence, true_prob, entry_reason[:80],
    )
    audit(
        "real_open" if is_real else "shadow_open",
        position_id=pos_id,
        strategy=strategy,
        market_id=market.market_id,
        side=side,
        entry_price=entry_price,
        size_usd=size_usd,
        confidence=confidence,
        true_prob=true_prob,
        latency_ms=entry_latency_ms,
        validator=(validator_snapshot or {}).get("decision") if validator_snapshot else None,
        clob_order_id=clob_order_id or None,
    )
    return pos_id


def compute_unrealized_pnl(
    side: str,
    entry_price: float,
    current: float,
    size_shares: float,
) -> float:
    if side == "BUY":
        return (current - entry_price) * size_shares
    return (entry_price - current) * size_shares


def compute_pnl_pct(
    side: str,
    entry_price: float,
    current: float,
) -> float:
    """Percentage change from entry — positive is profit for either side."""
    if entry_price <= 0:
        return 0.0
    if side == "BUY":
        return (current - entry_price) / entry_price * 100.0
    return (entry_price - current) / entry_price * 100.0


async def update_price(position: ShadowPosition, snapshot: PriceSnapshot) -> float:
    """Update last_price + unrealized_pnl + peak_pnl_pct. Returns current
    pnl_pct. Also mutates ``position.peak_pnl_pct`` in place so the caller
    can read the latest peak without a second DB round-trip — the
    trailing-TP exit in each lane's monitor loop relies on that.
    """
    exit_price = snapshot.bid if position.side == "BUY" else snapshot.ask
    if exit_price <= 0:
        exit_price = snapshot.mid
    unrealized = compute_unrealized_pnl(
        position.side, position.entry_price, exit_price, position.size_shares,
    )
    pnl_pct = compute_pnl_pct(position.side, position.entry_price, exit_price)
    peak = max(safe_float(position.peak_pnl_pct), pnl_pct)
    await execute(
        """UPDATE shadow_positions
           SET last_price=?, last_price_ts=?, unrealized_pnl_usd=?,
               peak_pnl_pct=?
           WHERE id=?""",
        (exit_price, snapshot.ts, unrealized, peak, position.id),
    )
    position.peak_pnl_pct = peak
    return pnl_pct


async def append_conviction(
    position: ShadowPosition,
    true_prob: float,
    current_mid: float,
    max_points: int = 500,
) -> list[list[float]]:
    traj = list(position.conviction_trajectory or [])
    traj.append([now_ts(), round(true_prob, 4), round(current_mid, 4)])
    if len(traj) > max_points:
        traj = traj[-max_points:]
    await execute(
        "UPDATE shadow_positions SET conviction_trajectory=?, last_rescored_ts=? WHERE id=?",
        (_json_dumps(traj), now_ts(), position.id),
    )
    return traj


def conviction_is_stable(traj: list[list[float]], eps: float = 0.02) -> bool:
    """True when the last two conviction points differ by <= eps. Used by
    the scheduler to skip Ollama re-scores on stable positions when the
    GPU is hot."""
    if len(traj) < 3:
        return False
    _, p_latest, _ = traj[-1]
    _, p_prev, _ = traj[-2]
    _, p_prev2, _ = traj[-3]
    return abs(p_latest - p_prev) <= eps and abs(p_prev - p_prev2) <= eps


async def close_position(
    position: ShadowPosition,
    snapshot: PriceSnapshot,
    reason: str,
) -> float:
    """Close at the current quote and release capital to the lane.
    Returns the realized PnL in USD.

    Real positions submit a closing order (opposite side) to the CLOB.
    A submit failure is logged but we still mark the row CLOSED locally
    so the lane budget is freed — the user can reconcile manually via
    the dashboard's open-orders panel (py-clob-client retains the order
    ID from `position.clob_close_order_id`).
    """
    exit_price = snapshot.bid if position.side == "BUY" else snapshot.ask
    if exit_price <= 0:
        exit_price = snapshot.mid or position.entry_price
    realized = compute_unrealized_pnl(
        position.side, position.entry_price, exit_price, position.size_shares,
    )

    close_clob_id = ""
    if position.is_real:
        # Flip the side to flatten. BUY position -> SELL to close.
        flip = "SELL" if position.side == "BUY" else "BUY"
        # ensure_ready() is async-safe (see clob_client docstring). Close
        # path is off the hot entry path but still async — same reasoning.
        if await clob_client.ensure_ready():
            result = await clob_client.place_limit_order(
                position.token_id, flip, exit_price, position.size_shares,
            )
            if result.ok:
                close_clob_id = result.clob_order_id or ""
            else:
                logger.error(
                    "[real] close CLOB submit failed for pos={} market={}: {}",
                    position.id, position.market_id, result.error,
                )
                audit(
                    "real_close_submit_failed",
                    position_id=position.id,
                    market_id=position.market_id,
                    error=result.error,
                )
        else:
            logger.error(
                "[real] close skipped — CLOB client not ready; pos={} "
                "market={} stays OPEN until reconciled",
                position.id, position.market_id,
            )
            audit(
                "real_close_blocked_no_client",
                position_id=position.id, market_id=position.market_id,
            )
            # Don't release capital or flip status — caller retries next tick.
            return 0.0

    await execute(
        """UPDATE shadow_positions
           SET status='CLOSED', close_price=?, close_ts=?, close_reason=?,
               realized_pnl_usd=?, unrealized_pnl_usd=0,
               clob_close_order_id=?
           WHERE id=?""",
        (exit_price, now_ts(), reason[:200], realized, close_clob_id, position.id),
    )
    # Credit the bucket the position was drawn from, not the current mode
    # — shadow positions always repay shadow, real always repay real.
    await allocator.release(
        position.strategy, position.size_usd, realized,
        mode="real" if position.is_real else "shadow",
    )
    logger.warning(
        "[{}] EXIT {} {} {} @ {:.3f} pnl={:+.2f} USD reason={}",
        "real" if position.is_real else "shadow",
        position.strategy, position.side, position.market_id,
        exit_price, realized, reason,
    )
    audit(
        "real_close" if position.is_real else "shadow_close",
        position_id=position.id,
        strategy=position.strategy,
        market_id=position.market_id,
        exit_price=exit_price,
        realized_pnl_usd=realized,
        reason=reason,
        clob_close_order_id=close_clob_id or None,
    )
    return realized


# ---- Aggregates for dashboard ----


async def lane_metrics(strategy: str) -> dict[str, Any]:
    """Return per-lane stats: open count, win rate, avg hold, PnL breakdown."""
    rows = await fetch_all(
        "SELECT * FROM shadow_positions WHERE strategy=?",
        (strategy,),
    )
    closed = [r for r in rows if r["status"] == "CLOSED"]
    open_rows = [r for r in rows if r["status"] == "OPEN"]
    wins = [r for r in closed if safe_float(r["realized_pnl_usd"]) > 0]
    losses = [r for r in closed if safe_float(r["realized_pnl_usd"]) < 0]
    realized = sum(safe_float(r["realized_pnl_usd"]) for r in closed)
    unrealized = sum(safe_float(r["unrealized_pnl_usd"]) for r in open_rows)
    hold_times = [
        safe_float(r["close_ts"]) - safe_float(r["entry_ts"])
        for r in closed
        if r["close_ts"] and r["entry_ts"]
    ]
    pnls = [safe_float(r["realized_pnl_usd"]) for r in closed]
    if len(pnls) >= 2:
        mean = sum(pnls) / len(pnls)
        variance = sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1)
        stdev = variance ** 0.5
        sharpe_like = mean / stdev if stdev > 0 else 0.0
    else:
        sharpe_like = 0.0
    avg_win = (sum(safe_float(r["realized_pnl_usd"]) for r in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(safe_float(r["realized_pnl_usd"]) for r in losses) / len(losses)) if losses else 0.0
    return {
        "strategy": strategy,
        "open": len(open_rows),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "avg_hold_seconds": (sum(hold_times) / len(hold_times)) if hold_times else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe_like": sharpe_like,
    }


async def rolling_win_rate(limit: int = 50) -> tuple[float, int]:
    """Global rolling win rate across all lanes over the last N resolved bets."""
    rows = await fetch_all(
        """SELECT realized_pnl_usd FROM shadow_positions
           WHERE status='CLOSED' ORDER BY close_ts DESC LIMIT ?""",
        (limit,),
    )
    if not rows:
        return 0.0, 0
    wins = sum(1 for r in rows if safe_float(r["realized_pnl_usd"]) > 0)
    return wins / len(rows), len(rows)


async def lane_daily_pnl(strategy: str) -> float:
    cutoff = now_ts() - 86400
    row = await fetch_one(
        """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
           FROM shadow_positions
           WHERE strategy=? AND status='CLOSED' AND close_ts >= ?""",
        (strategy, cutoff),
    )
    return safe_float(row["pnl"] if row else 0.0)


async def portfolio_daily_pnl() -> float:
    cutoff = now_ts() - 86400
    row = await fetch_one(
        """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
           FROM shadow_positions
           WHERE status='CLOSED' AND close_ts >= ?""",
        (cutoff,),
    )
    return safe_float(row["pnl"] if row else 0.0)


async def real_portfolio_daily_pnl() -> float:
    """Rolling 24h P&L across real-mode positions only: realized on
    closes within the window PLUS unrealized on currently-open real
    positions. Signed USD — negative means we're down money today.

    Used by the risk manager's real-mode daily-loss-cap (forward-
    looking — real budgets stay at 0 until manually flipped). The
    sibling :func:`portfolio_daily_pnl` stays shadow+real mixed to
    preserve the existing shadow circuit-breaker semantics.
    """
    cutoff = now_ts() - 86400
    realized_row = await fetch_one(
        """SELECT COALESCE(SUM(realized_pnl_usd), 0) AS pnl
           FROM shadow_positions
           WHERE status='CLOSED'
             AND COALESCE(is_real, 0) = 1
             AND close_ts >= ?""",
        (cutoff,),
    )
    unrealized_row = await fetch_one(
        """SELECT COALESCE(SUM(unrealized_pnl_usd), 0) AS pnl
           FROM shadow_positions
           WHERE status IN ('OPEN','PENDING_FILL')
             AND COALESCE(is_real, 0) = 1"""
    )
    realized = safe_float(realized_row["pnl"] if realized_row else 0.0)
    unrealized = safe_float(unrealized_row["pnl"] if unrealized_row else 0.0)
    return realized + unrealized
