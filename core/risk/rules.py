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
        *,
        bypass_kelly: bool = False,
        preset_size_usd: float | None = None,
    ) -> RiskDecision:
        """Approve a trade and return a sizing decision.

        ``bypass_kelly`` and ``preset_size_usd`` exist for the three
        shadow lanes: they own their own sizing logic, so the global
        risk engine only enforces absolute caps (max_position,
        max_total_exposure, daily_loss, spread, liquidity, cooldown)
        rather than sizing from Kelly.
        """
        cfg = get_config().get("risk") or {}
        kelly_cfg = get_config().get("kelly") or {}
        exec_cfg = get_config().get("execution") or {}

        # ---- Stale price ----
        # Prefer the new key; fall back to legacy `stale_price_seconds`
        # so older configs still work. Default 300s is intentionally
        # generous so dry-run mode produces enough signals to learn from.
        max_stale = float(
            cfg.get(
                "stale_price_max_age_seconds",
                cfg.get("stale_price_seconds", 300),
            )
        )
        if (now_ts() - market.updated_at) > max_stale:
            raise RiskRejection(f"stale price ({now_ts() - market.updated_at:.0f}s old)")

        # ---- Spread / liquidity ----
        # A pure ratio check was rejecting legitimate longshot entries:
        # on a $0.003 mid market with a $0.002/$0.005 book, the 3¢ absolute
        # spread is tight enough to trade but the ratio is 100%. Accept
        # either an absolute OR relative ceiling; for very low-price
        # markets (mid < 0.05) rely on the absolute test only, since the
        # ratio is inherently huge there.
        mid = market.mid
        if mid <= 0 or mid >= 1:
            raise RiskRejection("invalid mid price")
        spread_abs = market.spread
        spread_rel = (spread_abs / mid) if mid else 1.0
        max_abs = float(cfg.get("max_spread_absolute", 0.03))
        max_rel = float(cfg.get("max_spread_relative", 0.15))
        low_price_threshold = float(cfg.get("low_price_threshold", 0.05))
        if mid < low_price_threshold:
            if spread_abs > max_abs:
                raise RiskRejection(
                    f"spread abs={spread_abs:.3f} > {max_abs:.3f} (low-price market)"
                )
            spread_branch = f"abs<={max_abs:.3f}"
        else:
            if not (spread_abs <= max_abs or spread_rel <= max_rel):
                raise RiskRejection(
                    f"spread abs={spread_abs:.3f} rel={spread_rel:.3f} "
                    f"> max (abs<={max_abs:.3f} or rel<={max_rel:.3f})"
                )
            spread_branch = (
                f"abs<={max_abs:.3f}" if spread_abs <= max_abs else f"rel<={max_rel:.3f}"
            )
        logger.debug(
            "[risk] spread ok {} mid={:.3f} abs={:.3f} rel={:.3f} via {}",
            market.market_id, mid, spread_abs, spread_rel, spread_branch,
        )
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
        # Only enforce the cap when we know which bucket the market is in.
        # Previously everything without a category got lumped into
        # "uncategorised", so once 2-3 stray markets filled that bucket
        # every subsequent non-sports/politics/crypto/macro signal got
        # rejected with "category '' exposure cap hit". If Gamma didn't
        # tell us and our keyword inference punted, skip the cap and move
        # on rather than blocking legitimate trades.
        category = (market.category or "").strip()
        if category:
            cat_cap = float(cfg.get("max_correlated_exposure_usd", 200))
            cat_exposure = await exposure_for_category(category)
            if cat_exposure >= cat_cap:
                raise RiskRejection(f"category '{category}' exposure cap hit")
        else:
            logger.warning(
                "[risk] market {} has no category; skipping correlation cap",
                market.market_id,
            )

        # ---- Sizing ----
        max_position = float(cfg.get("max_position_usd", 50))
        bankroll = await current_bankroll_usd()
        if bypass_kelly or preset_size_usd is not None:
            # Lane-driven trade: caller supplied the size, we only
            # enforce the global ceilings (max_position, headroom).
            size_usd = float(preset_size_usd or 0.0)
        elif kelly_cfg.get("enabled", True):
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
        # Confidence damp: only for Kelly-sized trades. Lane presets are
        # already confidence-scaled by the lane itself.
        if not (bypass_kelly or preset_size_usd is not None):
            size_usd *= clamp(confidence, 0.0, 1.0)
        # Don't blow remaining headroom.
        headroom = max(0.0, max_total - open_exposure)
        size_usd = min(size_usd, headroom)
        min_size = float(kelly_cfg.get("min_size_usd", 5))
        if size_usd < min_size:
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
