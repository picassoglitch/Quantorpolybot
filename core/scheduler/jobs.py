"""APScheduler async job wiring."""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from core.learning.prompt_evolution import evolve as evolve_prompt
from core.learning.source_trust import calibrate as calibrate_sources
from core.markets.discovery import MarketDiscovery
from core.optimization.auto_tune import run as run_auto_tune
from core.optimization.lane_rebalancer import rebalance as rebalance_lanes
from core.state.health import run_all as run_health_checks
from core.strategies.evidence_tier import purge_skips_older_than
from core.utils.config import get_config
from core.utils.helpers import safe_float


class JobScheduler:
    component = "scheduler"

    def __init__(self, discovery: MarketDiscovery) -> None:
        self.scheduler = AsyncIOScheduler()
        self.discovery = discovery

    def start(self) -> None:
        cfg = get_config().get("scheduler") or {}

        # NOTE: market refresh is owned by MarketDiscovery.run()'s internal
        # 300s loop, not by a cron here. Previously this class also ran a
        # `*/5 * * * *` cron that called the same refresh_once; both fire
        # paths shared self._lock so they serialised, but the second one
        # always landed seconds after the first — doubling the window
        # during which price_ticks / pipeline writers waited on the DB
        # busy_timeout. The event loop then missed health_check ticks
        # by 10-20s. One owner for refresh.
        self.scheduler.add_job(
            self._safe(run_health_checks, "health_checks"),
            CronTrigger.from_crontab(cfg.get("health_check_cron", "*/1 * * * *")),
            id="health_checks",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._safe(run_auto_tune, "auto_tune"),
            CronTrigger.from_crontab(cfg.get("optimization_cron", "5 3 * * *")),
            id="auto_tune",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._safe(evolve_prompt, "prompt_evolve"),
            CronTrigger.from_crontab(cfg.get("learning_cron", "30 3 * * *")),
            id="prompt_evolve",
            replace_existing=True,
            max_instances=1,
        )
        # Runs after prompt_evolve so the calibration reflects any
        # signals produced by a freshly-activated prompt in the same
        # window (though in practice evolve() only promotes rarely).
        self.scheduler.add_job(
            self._safe(calibrate_sources, "source_trust"),
            CronTrigger.from_crontab(cfg.get("source_trust_cron", "10 4 * * *")),
            id="source_trust",
            replace_existing=True,
            max_instances=1,
        )
        # Lane rebalancer runs last so both source_trust and auto_tune
        # have already settled for the night. Shifts capital toward
        # the best-earning lane so portfolio-level compounding kicks
        # in on top of the per-lane compounding the allocator already
        # does on every close.
        self.scheduler.add_job(
            self._safe(rebalance_lanes, "lane_rebalance"),
            CronTrigger.from_crontab(cfg.get("lane_rebalance_cron", "20 4 * * *")),
            id="lane_rebalance",
            replace_existing=True,
            max_instances=1,
        )
        # Step #2: nightly purge of scan_skips. Reads TTL from
        # `scan_skips.ttl_days` (default 7d). Wrapped in a thin
        # closure so the cron job factory doesn't need to know the
        # config layout.
        self.scheduler.add_job(
            self._safe(_purge_scan_skips, "scan_skips_purge"),
            CronTrigger.from_crontab(
                cfg.get("scan_skips_purge_cron", "40 4 * * *")
            ),
            id="scan_skips_purge",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info("[scheduler] started with {} jobs", len(self.scheduler.get_jobs()))

    async def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)
        logger.info("[scheduler] stopped")

    @staticmethod
    def _safe(coro_factory, label: str):
        async def runner() -> None:
            try:
                logger.info("[scheduler] running job {}", label)
                await coro_factory()
            except Exception as e:
                logger.exception("[scheduler] job {} failed: {}", label, e)
        return runner


async def _purge_scan_skips() -> None:
    """Read TTL from config and delete older rows. Logs the rowcount so
    the operator sees the purge happened (and how big the table got)."""
    cfg = get_config().get("scan_skips") or {}
    ttl_days = safe_float(cfg.get("ttl_days", 7.0))
    if ttl_days <= 0:
        logger.info("[scan_skips] purge disabled (ttl_days <= 0)")
        return
    seconds = ttl_days * 86400.0
    deleted = await purge_skips_older_than(seconds)
    logger.info(
        "[scan_skips] purged {} rows older than {}d", deleted, ttl_days,
    )
