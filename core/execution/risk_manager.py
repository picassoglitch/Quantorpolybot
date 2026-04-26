"""Shadow risk manager. Runs every 30s and pauses lanes (or the whole
fleet) when circuit breakers trigger.

Triggers:
1. Lane daily drawdown > 5% of lane total_budget  -> pause that lane 24h
2. Portfolio daily drawdown > 3% of total_shadow  -> pause ALL lanes 24h
3. Rolling 50-bet win rate < 40%                   -> alert only (don't pause)
4. Any market with > 3 open positions across lanes -> block new entries via
   `concentration_block_markets` (checked by lanes before entry)
5. Feeds idle > 30 min                             -> pause event lane only
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from core.execution import allocator, shadow
from core.utils.config import get_config
from core.utils.db import fetch_all, fetch_one
from core.utils.helpers import now_ts, safe_float
from core.utils.logging import audit
from core.utils.prices import latest_feed_item_ts

_BLOCKED_MARKETS: set[str] = set()


def concentration_blocked() -> set[str]:
    """Lanes read this set before entry — any market id in here is off-
    limits for new positions (cross-lane concentration cap)."""
    return set(_BLOCKED_MARKETS)


def _cfg() -> dict[str, Any]:
    return get_config().get("shadow_risk") or {}


class ShadowRiskManager:
    component = "execution.risk_manager"

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    async def run(self) -> None:
        interval = safe_float(_cfg().get("check_interval_seconds", 30))
        logger.info("[risk_mgr] started interval={}s", interval)
        while not self._stop.is_set():
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("[risk_mgr] check error: {}", e)
            await self._sleep(interval)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def check_once(self) -> dict[str, Any]:
        """One tick of all circuit breakers. Returns a summary dict
        useful for tests + dashboard."""
        cfg = _cfg()
        lane_dd_pct = safe_float(cfg.get("lane_daily_drawdown_pct", 5))
        port_dd_pct = safe_float(cfg.get("portfolio_daily_drawdown_pct", 3))
        min_wr = safe_float(cfg.get("min_rolling_win_rate", 0.40))
        wr_window = int(cfg.get("rolling_win_rate_window", 50))
        max_per_market = int(cfg.get("max_positions_per_market", 3))
        feed_staleness_s = safe_float(cfg.get("feed_staleness_alert_minutes", 30)) * 60.0

        result: dict[str, Any] = {
            "lane_drawdown_pauses": [],
            "portfolio_pause": False,
            "win_rate_alert": False,
            "catastrophic_pause": False,
            "blocked_markets": 0,
            "feed_stale": False,
            "real_loss_cap_pause": False,
        }

        # ---- Real-mode daily-loss cap ----
        # Rolling 24h realized+unrealized P&L across REAL positions
        # only. Below -daily_loss_cap_usd → pause all real lanes for
        # 24h so we can inspect. Shadow is untouched — the point is
        # for shadow to keep learning while the real books cool off.
        # Forward-looking guardrail: real budgets stay at 0 until the
        # operator flips them, so this is a no-op in shadow mode.
        limits_cfg = get_config().get("limits") or {}
        loss_cap_usd = safe_float(limits_cfg.get("daily_loss_cap_usd", 0))
        if loss_cap_usd > 0:
            real_pnl = await shadow.real_portfolio_daily_pnl()
            if real_pnl <= -loss_cap_usd:
                # Pause only real lanes; leave shadow budgets alone.
                # "Active" = has budget AND isn't already paused. A
                # zero-budget lane can't trade, so pausing it would be
                # cosmetic — skip the audit and pause noise. This keeps
                # the cap a true no-op while real budgets are 0 (ship
                # config).
                any_real_active = False
                for lane in allocator.LANES:
                    lstate = await allocator.get_state(lane, mode="real")
                    if (
                        lstate is not None
                        and not lstate.is_paused
                        and lstate.total_budget > 0
                    ):
                        any_real_active = True
                        break
                if any_real_active:
                    await allocator.pause_all(
                        86400,
                        f"real_daily_loss_cap pnl={real_pnl:+.2f} cap=-{loss_cap_usd:.2f}",
                        mode="real",
                    )
                    audit(
                        "risk_pause_real_daily_loss_cap",
                        pnl_usd=real_pnl,
                        cap_usd=loss_cap_usd,
                    )
                    logger.error(
                        "[risk_mgr] REAL daily-loss cap hit: pnl={:+.2f} "
                        "cap=-{:.2f} — all real lanes paused 24h",
                        real_pnl, loss_cap_usd,
                    )
                    result["real_loss_cap_pause"] = True

        # ---- Per-lane daily drawdown ----
        for lane in allocator.LANES:
            state = await allocator.get_state(lane)
            if state is None or state.total_budget <= 0:
                continue
            daily = await shadow.lane_daily_pnl(lane)
            pct = -daily / state.total_budget * 100.0  # positive = loss %
            if pct >= lane_dd_pct and not state.is_paused:
                until = now_ts() + 86400
                await allocator.pause(
                    lane, until,
                    f"daily_drawdown {pct:.2f}% >= {lane_dd_pct:.1f}%",
                )
                audit(
                    "risk_pause_lane",
                    lane=lane,
                    daily_pnl=daily,
                    drawdown_pct=pct,
                )
                result["lane_drawdown_pauses"].append(lane)

        # ---- Portfolio-level drawdown ----
        total_shadow = safe_float(
            get_config().get("capital", "total_shadow_usd", default=10000)
        )
        port_pnl = await shadow.portfolio_daily_pnl()
        port_pct = -port_pnl / total_shadow * 100.0 if total_shadow > 0 else 0.0
        if port_pct >= port_dd_pct:
            await allocator.pause_all(86400, f"portfolio_drawdown {port_pct:.2f}%")
            audit("risk_pause_all", daily_pnl=port_pnl, drawdown_pct=port_pct)
            result["portfolio_pause"] = True

        # ---- Rolling win rate (alert + catastrophic-brake) ----
        # The alert tier is advisory — it fires when the fleet is
        # underperforming but still within historical variance. The
        # brake tier is a hard stop: if win rate craters below the
        # catastrophic floor over a short window, pause all lanes for
        # a configurable cool-off. That pattern almost always means a
        # structural bug (e.g. every SELL storing against the NO token
        # so monitor reads a mirrored price) rather than a losing
        # streak — and a structural bug doesn't fix itself while we
        # keep feeding it fresh positions.
        wr, n = await shadow.rolling_win_rate(wr_window)
        if n >= 10 and wr < min_wr:
            logger.warning("[risk_mgr] rolling win rate {:.0%} over last {} — alert", wr, n)
            audit("risk_winrate_alert", win_rate=wr, sample_size=n)
            result["win_rate_alert"] = True
        catastrophic_floor = safe_float(cfg.get("catastrophic_win_rate_floor", 0.15))
        catastrophic_window = int(cfg.get("catastrophic_win_rate_window", 20))
        catastrophic_pause_s = safe_float(
            cfg.get("catastrophic_pause_seconds", 1800)
        )
        wr_c, n_c = await shadow.rolling_win_rate(catastrophic_window)
        if n_c >= catastrophic_window and wr_c <= catastrophic_floor:
            # Only pause if we're not already in a pause window — the
            # brake holds for catastrophic_pause_seconds, then lanes
            # auto-resume so the operator can inspect before the bot
            # re-arms itself.
            any_active = False
            for lane in allocator.LANES:
                lstate = await allocator.get_state(lane)
                if lstate is not None and not lstate.is_paused:
                    any_active = True
                    break
            if any_active:
                await allocator.pause_all(
                    catastrophic_pause_s,
                    f"catastrophic_win_rate {wr_c:.0%} over {n_c}",
                )
                audit(
                    "risk_pause_catastrophic",
                    win_rate=wr_c,
                    sample_size=n_c,
                    pause_seconds=catastrophic_pause_s,
                )
                logger.error(
                    "[risk_mgr] catastrophic win rate {:.0%} over last {} — "
                    "all lanes paused for {:.0f}s",
                    wr_c, n_c, catastrophic_pause_s,
                )
                result["catastrophic_pause"] = True

        # ---- Concentration (cross-lane >3 per market) ----
        rows = await fetch_all(
            """SELECT market_id, COUNT(*) AS n
               FROM shadow_positions
               WHERE status='OPEN'
               GROUP BY market_id HAVING n > ?""",
            (max_per_market,),
        )
        new_blocked = {r["market_id"] for r in rows}
        # Mutate the module-level set rather than rebind so lanes'
        # cached reference stays valid.
        _BLOCKED_MARKETS.clear()
        _BLOCKED_MARKETS.update(new_blocked)
        result["blocked_markets"] = len(new_blocked)

        # ---- Feed staleness (pauses event lane only) ----
        latest = await latest_feed_item_ts()
        age = now_ts() - latest if latest else float("inf")
        if age > feed_staleness_s:
            event_state = await allocator.get_state("event_sniping")
            if event_state and not event_state.is_paused:
                await allocator.pause(
                    "event_sniping",
                    now_ts() + 600,  # 10m; event lane should recover fast
                    f"feed_staleness age={age:.0f}s",
                )
                audit("risk_pause_feed_stale", lane="event_sniping", age=age)
            result["feed_stale"] = True
        return result
