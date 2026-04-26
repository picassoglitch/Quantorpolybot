"""Capital allocator for the three lanes.

Two independent budgets run side by side — keyed by `(lane, mode)`:

  * `mode=shadow` uses `shadow_capital` from config.yaml.
  * `mode=real`   uses `real_capital` from config.yaml.

The allocator always consults `get_config().get("mode")` at call time, so
flipping the mode on the dashboard immediately routes new reservations
to the other budget. Open positions already on the books keep their
existing reservation until they close (no reshuffling between buckets).

Each lane has a hard budget. `reserve` deducts from `available` before a
position opens; `release` returns it when the position closes (plus or
minus realized PnL so winners compound and losers shrink the lane).

No lane can borrow from another lane and no mode can borrow from the
other mode.

Dynamic sizing: when `available` is less than the configured size but
above `min_lane_available_usd`, we shrink the position to fit. Below
that floor we skip entirely (the slot isn't worth the overhead).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger

from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import now_ts, safe_float

LANES = ("scalping", "event_sniping", "longshot")
MODES = ("shadow", "real")

_RESERVE_LOCK = asyncio.Lock()

# Strategy → lane-bucket map. Some strategies (microscalp, resolution_day)
# intentionally share another lane's budget bucket so the 60/30/10 capital
# split doesn't need to be re-tuned every time a new lane is added.
# Strategies that don't register here default to their own name (1:1).
#
# The strategy modules also register themselves on import via
# :func:`register_strategy_lane`, but we seed the known-shared mappings
# here so code paths that DON'T import the strategy modules (nightly
# cron jobs, scripted dry-runs, tests) still see the correct mapping.
# Without this, the lane rebalancer's "closes per lane" query missed
# microscalp's 19 closes because its registration hadn't run yet in
# the APScheduler context — the scalping bucket looked idle.
_STRATEGY_LANE_MAP: dict[str, str] = {
    "microscalp": "scalping",
    "resolution_day": "scalping",
}


def register_strategy_lane(strategy: str, lane: str) -> None:
    """Record that positions tagged ``strategy`` draw from lane ``lane``'s
    capital bucket. Called at module import by lanes that intentionally
    reuse another lane's budget. Idempotent — last registration wins."""
    _STRATEGY_LANE_MAP[strategy] = lane


def lane_for_strategy(strategy: str) -> str:
    """Return the lane bucket that funds ``strategy``. Strategies default
    to their own name when not explicitly registered — preserves the 1:1
    mapping for the three primary lanes."""
    return _STRATEGY_LANE_MAP.get(strategy, strategy)


def current_mode() -> str:
    """Read mode from config. Anything other than 'real' is treated as
    'shadow' — defensive default for missing/bogus config values."""
    raw = str(get_config().get("mode", default="shadow") or "shadow").strip().lower()
    return "real" if raw == "real" else "shadow"


def _capital_cfg_for(mode: str) -> dict:
    key = "real_capital" if mode == "real" else "shadow_capital"
    cfg = get_config().get(key) or {}
    return cfg


@dataclass
class LaneState:
    lane: str
    mode: str
    total_budget: float
    deployed: float
    available: float
    paused_until: float
    updated_at: float

    @property
    def is_paused(self) -> bool:
        return self.paused_until > now_ts()


async def init_lane_capital() -> None:
    """Idempotent bootstrap. Creates a row per (lane, mode) on first
    boot, and on subsequent boots reconciles ``deployed`` against open
    positions *without* clobbering ``total_budget`` — that column is
    owned by :func:`release` (per-lane compounding) and the nightly
    :mod:`core.optimization.lane_rebalancer` (cross-lane compounding).
    Overwriting it on every restart would wipe whatever winnings the
    lane accumulated and undo the rebalancer's decisions.

    If the operator needs to re-apply the YAML split (e.g. after funding
    more capital), delete the rows in ``lane_capital`` for the target
    mode and let the next boot re-INSERT from YAML — or call
    :func:`set_lane_budget` directly from the dashboard.
    """
    for mode in MODES:
        cap_cfg = _capital_cfg_for(mode)
        total = safe_float(cap_cfg.get("total_usd", 0.0))
        splits = cap_cfg.get("lane_allocations") or {}
        for lane in LANES:
            pct = safe_float(splits.get(lane, 0.0))
            budget = round(total * pct, 2)  # YAML-computed default
            deployed = await _deployed_for(lane, mode)
            existing = await fetch_one(
                "SELECT * FROM lane_capital WHERE lane=? AND mode=?",
                (lane, mode),
            )
            if existing is None:
                available = max(0.0, budget - deployed)
                await execute(
                    """INSERT INTO lane_capital
                       (lane, mode, total_budget, deployed, available,
                        paused_until, updated_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (lane, mode, budget, deployed, available, 0.0, now_ts()),
                )
                logger.info(
                    "[allocator] {}:{} initialised budget={:.2f} deployed={:.2f}",
                    mode, lane, budget, deployed,
                )
            else:
                # Preserve total_budget — only reconcile deployed+available.
                persisted_tb = safe_float(existing["total_budget"])
                available = max(0.0, persisted_tb - deployed)
                await execute(
                    """UPDATE lane_capital
                       SET deployed=?, available=?, updated_at=?
                       WHERE lane=? AND mode=?""",
                    (deployed, available, now_ts(), lane, mode),
                )
                if abs(persisted_tb - budget) > 0.01:
                    logger.info(
                        "[allocator] {}:{} persisted budget={:.2f} "
                        "(YAML default={:.2f}; not overwriting — "
                        "delete row to force resync)",
                        mode, lane, persisted_tb, budget,
                    )


async def reconcile_deployed(mode: str | None = None) -> dict[str, float]:
    """Re-sync every lane's `deployed` column against the real sum of
    OPEN + PENDING_FILL positions. Returns {lane: drift_corrected_usd}.

    :func:`init_lane_capital` does this on startup, but a long-running
    process can accumulate drift — e.g. a close path that hits an
    exception before :func:`release` lands, or manual DB edits. The
    nightly rebalancer calls this first so it doesn't move budget
    around based on a stale deployed number.
    """
    m = mode or current_mode()
    drift: dict[str, float] = {}
    async with _RESERVE_LOCK:
        for lane in LANES:
            state = await get_state(lane, m)
            if state is None:
                continue
            actual = await _deployed_for(lane, m)
            if abs(actual - state.deployed) < 0.01:
                continue
            new_available = max(0.0, state.total_budget - actual)
            await execute(
                """UPDATE lane_capital
                   SET deployed=?, available=?, updated_at=?
                   WHERE lane=? AND mode=?""",
                (actual, new_available, now_ts(), lane, m),
            )
            drift[lane] = round(state.deployed - actual, 2)
            logger.info(
                "[allocator] reconciled {}:{} deployed {:.2f} -> {:.2f} "
                "(drift {:+.2f})",
                m, lane, state.deployed, actual, drift[lane],
            )
    return drift


async def set_lane_budget(
    lane: str,
    new_total_budget: float,
    *,
    mode: str | None = None,
    reason: str = "",
) -> None:
    """Atomically replace a lane's total_budget, recomputing available
    to preserve `deployed` (open positions keep their reservation).
    Used by the nightly lane rebalancer to shift capital toward
    winning lanes; caller is responsible for coordinating writes
    across lanes so the portfolio total is conserved.
    """
    m = mode or current_mode()
    new_total = max(0.0, round(new_total_budget, 2))
    async with _RESERVE_LOCK:
        state = await get_state(lane, m)
        if state is None:
            logger.warning(
                "[allocator] set_lane_budget on unknown lane {}:{}",
                m, lane,
            )
            return
        new_available = max(0.0, new_total - state.deployed)
        await execute(
            """UPDATE lane_capital
               SET total_budget=?, available=?, updated_at=?
               WHERE lane=? AND mode=?""",
            (new_total, new_available, now_ts(), lane, m),
        )
        logger.info(
            "[allocator] {}:{} total_budget {:.2f} -> {:.2f} "
            "(deployed={:.2f}, reason={})",
            m, lane, state.total_budget, new_total, state.deployed,
            reason or "manual",
        )


async def _deployed_for(lane: str, mode: str) -> float:
    """Sum open size across every strategy that funds from ``lane``.
    Must include shared-bucket strategies (e.g. microscalp, resolution_day)
    — otherwise a restart recomputes the lane's ``deployed`` without their
    rows and `available` comes back inflated."""
    is_real = 1 if mode == "real" else 0
    strategies = {s for s, l in _STRATEGY_LANE_MAP.items() if l == lane}
    strategies.add(lane)  # default 1:1 mapping
    placeholders = ",".join("?" * len(strategies))
    row = await fetch_one(
        f"""SELECT COALESCE(SUM(size_usd), 0) AS s
            FROM shadow_positions
            WHERE strategy IN ({placeholders})
              AND status='OPEN' AND COALESCE(is_real, 0)=?""",
        (*strategies, is_real),
    )
    return safe_float(row["s"] if row else 0.0)


def _row_to_state(row) -> LaneState:
    return LaneState(
        lane=row["lane"],
        mode=row["mode"],
        total_budget=safe_float(row["total_budget"]),
        deployed=safe_float(row["deployed"]),
        available=safe_float(row["available"]),
        paused_until=safe_float(row["paused_until"]),
        updated_at=safe_float(row["updated_at"]),
    )


async def get_state(lane: str, mode: str | None = None) -> LaneState | None:
    m = mode or current_mode()
    row = await fetch_one(
        "SELECT * FROM lane_capital WHERE lane=? AND mode=?", (lane, m),
    )
    if row is None:
        return None
    return _row_to_state(row)


async def all_states(mode: str | None = None) -> list[LaneState]:
    m = mode or current_mode()
    rows = await fetch_all(
        "SELECT * FROM lane_capital WHERE mode=? ORDER BY lane", (m,),
    )
    return [_row_to_state(r) for r in rows]


async def all_states_both_modes() -> dict[str, list[LaneState]]:
    """For the dashboard: return per-mode lane states in one shot."""
    out: dict[str, list[LaneState]] = {}
    for mode in MODES:
        out[mode] = await all_states(mode)
    return out


def clamp_position_size(
    lane: str,
    requested_usd: float,
    available_usd: float,
) -> tuple[float, str]:
    """Return ``(size, skip_reason)``. Size > 0 means dispatch; size == 0
    means skip and the reason is one of:

      * ``"below_min_available"`` — lane has headroom but it's below the
        per-dispatch floor (``min_lane_available_usd``). Not an
        exhaustion: the lane will recover once an open position closes
        and its size is credited back. Common on small budgets where
        the floor is close to the whole lane.
      * ``"exhausted"`` — available dropped to zero. Real exhaustion.

    Positive size: a $200 fill beats skipping a $300 slot (dynamic
    capping).
    """
    mode = current_mode()
    cap_cfg = _capital_cfg_for(mode)
    floor = safe_float(cap_cfg.get("min_lane_available_usd", 50.0))
    if available_usd <= 0:
        return 0.0, "exhausted"
    if available_usd < floor:
        return 0.0, "below_min_available"
    if requested_usd <= available_usd:
        return round(requested_usd, 2), ""
    return round(available_usd, 2), ""


async def compute_position_size(
    lane: str,
    confidence: float,
    lane_cfg: dict,
) -> float:
    """Budget-adaptive sizing. Works at any scale — $15 or $15k — without
    re-tuning lane config, because sizes are a percentage of lane budget.

    Reads lane config for:
      * ``base_position_pct`` (preferred): fraction of lane total budget
        used as the base request (e.g. 0.25 = 25%).
      * ``max_position_pct`` (preferred): fraction used at high confidence.
      * ``base_position`` / ``max_position`` (legacy): absolute dollar
        amounts. Used as fallback when the _pct keys are missing, and
        always as an absolute *upper cap* when present — so a tuned dollar
        limit can protect against over-sizing when the lane grows large.

    Confidence scales sizing smoothly:
      * < 0.60:           base
      * 0.60 .. 0.85:     linear interp base -> max
      * >= 0.85:          max

    The result is floored by ``kelly.min_size_usd`` to avoid dust
    positions, and capped by ``risk.max_position_usd`` for global safety.
    Returns 0.0 if the lane is unknown. Caller still passes the value
    through :func:`reserve`, which applies the per-lane availability floor
    (``min_lane_available_usd``).

    Budget-aware guarantee: if ``total_budget`` drops (PnL or a dashboard
    edit), next call's percentages are applied to the new total. Winners
    compound the base size; losers shrink it. No config reload needed.
    """
    state = await get_state(lane)
    if state is None:
        return 0.0
    total = safe_float(state.total_budget)
    if total <= 0:
        return 0.0

    base_pct = safe_float(lane_cfg.get("base_position_pct", 0.0))
    max_pct = safe_float(lane_cfg.get("max_position_pct", 0.0))
    base_abs = safe_float(lane_cfg.get("base_position", 0.0))
    max_abs = safe_float(lane_cfg.get("max_position", 0.0))
    # Single-value aliases for lanes that don't confidence-scale (longshot,
    # resolution_day). ``position_pct`` = percent of lane budget used for
    # every entry; ``fixed_position`` = absolute dollar alias.
    single_pct = safe_float(lane_cfg.get("position_pct", 0.0))
    single_abs = safe_float(lane_cfg.get("fixed_position", 0.0))
    if single_pct > 0 and base_pct <= 0 and max_pct <= 0:
        base_pct = max_pct = single_pct
    if single_abs > 0 and base_abs <= 0 and max_abs <= 0:
        base_abs = max_abs = single_abs

    # Pct preferred; fall back to absolute dollars for backwards compat.
    base = total * base_pct if base_pct > 0 else base_abs
    top = total * max_pct if max_pct > 0 else max_abs
    # If only one of base/max is set, use it for both ends.
    if base <= 0 and top > 0:
        base = top
    if top <= 0 and base > 0:
        top = base
    if base <= 0 and top <= 0:
        # Nothing configured: fall back to 25% of lane.
        base = total * 0.25
        top = total * 0.40

    # Absolute dollar caps (when present) always win over pct-derived —
    # lets the user set a hard ceiling independent of lane size.
    if max_abs > 0:
        top = min(top, max_abs)
    if base_abs > 0:
        base = min(base, base_abs)

    # Confidence interpolation.
    if confidence >= 0.85:
        wanted = top
    elif confidence >= 0.60:
        t = (confidence - 0.60) / 0.25
        wanted = base + (top - base) * t
    else:
        wanted = base

    # Global safety rails.
    kelly_cfg = get_config().get("kelly") or {}
    risk_cfg = get_config().get("risk") or {}
    min_size = safe_float(kelly_cfg.get("min_size_usd", 0.25))
    max_global = safe_float(risk_cfg.get("max_position_usd", 500))
    if wanted < min_size:
        wanted = min_size
    if wanted > max_global:
        wanted = max_global
    return round(wanted, 2)


async def reserve(lane: str, requested_usd: float) -> float | None:
    """Reserve capital from the *current* mode's bucket. Returns the
    approved size (may be less than requested if the lane is low) or
    None if paused / exhausted / not registered."""
    mode = current_mode()
    async with _RESERVE_LOCK:
        state = await get_state(lane, mode)
        if state is None:
            logger.warning("[allocator] unknown lane '{}:{}'", mode, lane)
            return None
        if state.is_paused:
            logger.info(
                "[allocator] {}:{} paused until {:.0f}",
                mode, lane, state.paused_until,
            )
            return None
        size, skip_reason = clamp_position_size(
            lane, requested_usd, state.available,
        )
        if size <= 0:
            # One log tag per cause — "exhausted" is a real stop, while
            # "below_min_available" is a transient skip that clears as
            # soon as an open position closes. Dashboards / operators
            # react very differently to the two; merging them was
            # noise. See clamp_position_size for the taxonomy.
            logger.info(
                "[allocator] skip_entry mode={} lane={} reason={} "
                "requested={:.2f} available={:.2f}",
                mode, lane, skip_reason or "exhausted",
                requested_usd, state.available,
            )
            return None
        new_deployed = state.deployed + size
        new_available = max(0.0, state.total_budget - new_deployed)
        await execute(
            """UPDATE lane_capital
               SET deployed=?, available=?, updated_at=?
               WHERE lane=? AND mode=?""",
            (new_deployed, new_available, now_ts(), lane, mode),
        )
        return size


async def release(
    lane_or_strategy: str,
    size_usd: float,
    realized_pnl_usd: float,
    *,
    mode: str | None = None,
) -> None:
    """Return capital. PnL adjusts the lane's *total* budget so winners
    compound and losers shrink the lane (matches a real account).

    Accepts either a lane name or a strategy tag; shared-bucket strategies
    (microscalp, resolution_day) close their positions tagged with the
    strategy tag, but need to credit the lane bucket they were reserved
    from. :func:`lane_for_strategy` is a no-op for registered 1:1 lanes,
    so existing callers passing a lane name continue to work unchanged.

    `mode` defaults to the current mode from config. Pass an explicit
    mode when closing a position that was opened in the other mode (e.g.
    the user flipped to real with shadow positions still open) — the
    caller knows which bucket to credit from the position's `is_real`.
    """
    m = mode or current_mode()
    lane = lane_for_strategy(lane_or_strategy)
    async with _RESERVE_LOCK:
        state = await get_state(lane, m)
        if state is None:
            logger.warning(
                "[allocator] release on unknown lane '{}:{}' (from '{}')",
                m, lane, lane_or_strategy,
            )
            return
        new_deployed = max(0.0, state.deployed - size_usd)
        new_total = max(0.0, state.total_budget + realized_pnl_usd)
        new_available = max(0.0, new_total - new_deployed)
        await execute(
            """UPDATE lane_capital
               SET total_budget=?, deployed=?, available=?, updated_at=?
               WHERE lane=? AND mode=?""",
            (new_total, new_deployed, new_available, now_ts(), lane, m),
        )


async def pause(
    lane: str,
    until_ts: float,
    reason: str = "",
    *,
    mode: str | None = None,
) -> None:
    m = mode or current_mode()
    await execute(
        "UPDATE lane_capital SET paused_until=?, updated_at=? WHERE lane=? AND mode=?",
        (until_ts, now_ts(), lane, m),
    )
    logger.warning(
        "[allocator] {}:{} paused until {:.0f} ({})", m, lane, until_ts, reason,
    )


async def unpause(lane: str, *, mode: str | None = None) -> None:
    m = mode or current_mode()
    await execute(
        "UPDATE lane_capital SET paused_until=0, updated_at=? WHERE lane=? AND mode=?",
        (now_ts(), lane, m),
    )
    logger.info("[allocator] {}:{} unpaused", m, lane)


async def pause_all(
    duration_seconds: float,
    reason: str = "",
    *,
    mode: str | None = None,
) -> None:
    """Pause every lane in the given mode (default: current mode)."""
    m = mode or current_mode()
    until = now_ts() + duration_seconds
    for lane in LANES:
        await pause(lane, until, reason, mode=m)
