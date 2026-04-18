"""Risk engine. Every signal must pass evaluate() before any order goes
out. Hard-blocks on any rule violation. Returns a sizing decision when
the trade is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from core.markets.cache import Market
from core.risk.correlation import exposure_for_category
from core.risk.kelly import kelly_size_usd
from core.state.balances import current_bankroll_usd, daily_pnl_usd, total_open_exposure_usd
from core.state.cooldowns import market_in_cooldown
from core.state.positions import open_position_for
from core.utils.config import get_config
from core.utils.helpers import clamp, now_ts


class RiskRejection(Exception):
    pass


@dataclass
class RiskDecision:
    size_usd: float
    target_price: float
    rationale: str


class RiskEngine:
    component = "risk.engine"

    async def evaluate(
        self,
        market: Market,
        side: str,
        implied_prob: float,
        confidence: float,
    ) -> RiskDecision:
        cfg = get_config().get("risk") or {}
        kelly_cfg = get_config().get("kelly") or {}
        exec_cfg = get_config().get("execution") or {}

        # ---- Stale price ----
        max_stale = float(cfg.get("stale_price_seconds", 60))
        if (now_ts() - market.updated_at) > max_stale:
            raise RiskRejection(f"stale price ({now_ts() - market.updated_at:.0f}s old)")

        # ---- Spread / liquidity ----
        mid = market.mid
        if mid <= 0 or mid >= 1:
            raise RiskRejection("invalid mid price")
        spread_ratio = market.spread / mid if mid else 1.0
        if spread_ratio > float(cfg.get("max_market_spread", 0.05)):
            raise RiskRejection(f"spread {spread_ratio:.3f} > max")
        if market.liquidity < float(cfg.get("min_market_liquidity", 1000)):
            raise RiskRejection(f"liquidity {market.liquidity:.0f} < min")

        # ---- Cooldown ----
        if await market_in_cooldown(market.market_id, float(cfg.get("cooldown_seconds", 600))):
            raise RiskRejection("market in cooldown")

        # ---- Existing position cap per market ----
        max_orders = int(cfg.get("max_orders_per_market", 1))
        if max_orders <= 0:
            raise RiskRejection("max_orders_per_market is 0")
        existing = await open_position_for(market.market_id)
        if existing and existing >= max_orders:
            raise RiskRejection("position cap for this market hit")

        # ---- Daily loss circuit-breaker ----
        max_daily_loss = float(cfg.get("max_daily_loss_usd", 50))
        pnl_today = await daily_pnl_usd()
        if pnl_today <= -abs(max_daily_loss):
            raise RiskRejection(f"daily loss limit reached ({pnl_today:.2f})")

        # ---- Total exposure ----
        max_total = float(cfg.get("max_total_exposure_usd", 500))
        open_exposure = await total_open_exposure_usd()
        if open_exposure >= max_total:
            raise RiskRejection(f"total exposure {open_exposure:.0f} >= cap")

        # ---- Correlated category bucket ----
        cat_cap = float(cfg.get("max_correlated_exposure_usd", 200))
        cat_exposure = await exposure_for_category(market.category or "uncategorised")
        if cat_exposure >= cat_cap:
            raise RiskRejection(f"category '{market.category}' exposure cap hit")

        # ---- Sizing ----
        max_position = float(cfg.get("max_position_usd", 50))
        bankroll = await current_bankroll_usd()
        if kelly_cfg.get("enabled", True):
            size_usd = kelly_size_usd(
                prob_win=implied_prob if side == "BUY" else 1 - implied_prob,
                price=market.best_ask if side == "BUY" else market.best_bid,
                bankroll_usd=bankroll,
                fraction=float(kelly_cfg.get("fraction", 0.25)),
                min_size_usd=float(kelly_cfg.get("min_size_usd", 5)),
                max_size_usd=float(kelly_cfg.get("max_size_usd", 50)),
            )
        else:
            size_usd = float(kelly_cfg.get("max_size_usd", 50))
        size_usd = clamp(size_usd, 0.0, max_position)
        # Confidence damp: high-confidence trades get full size, lower
        # confidence ones get scaled down linearly.
        size_usd *= clamp(confidence, 0.0, 1.0)
        # Don't blow remaining headroom.
        headroom = max(0.0, max_total - open_exposure)
        size_usd = min(size_usd, headroom)
        if size_usd < float(kelly_cfg.get("min_size_usd", 5)):
            raise RiskRejection(f"size {size_usd:.2f} below min")

        # ---- Pricing ----
        edge = abs(implied_prob - mid)
        factor = float(exec_cfg.get("edge_pricing_factor", 0.5))
        if side == "BUY":
            target = clamp(mid + edge * factor, 0.01, 0.99)
        else:
            target = clamp(mid - edge * factor, 0.01, 0.99)

        rationale = (
            f"size={size_usd:.2f} mid={mid:.3f} target={target:.3f} "
            f"conf={confidence:.2f} bankroll={bankroll:.0f}"
        )
        logger.debug("[risk] approved {} {} {}", market.market_id, side, rationale)
        return RiskDecision(size_usd=size_usd, target_price=target, rationale=rationale)
