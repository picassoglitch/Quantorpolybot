"""APScheduler async job wiring."""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from core.learning.prompt_evolution import evolve as evolve_prompt
from core.markets.discovery import MarketDiscovery
from core.optimization.auto_tune import run as run_auto_tune
from core.state.health import run_all as run_health_checks
from core.utils.config import get_config


class JobScheduler:
    component = "scheduler"

    def __init__(self, discovery: MarketDiscovery) -> None:
        self.scheduler = AsyncIOScheduler()
        self.discovery = discovery

    def start(self) -> None:
        cfg = get_config().get("scheduler") or {}

        self.scheduler.add_job(
            self._safe(self.discovery.refresh_once, "market_refresh"),
            CronTrigger.from_crontab(cfg.get("market_refresh_cron", "*/5 * * * *")),
            id="market_refresh",
            replace_existing=True,
            max_instances=1,
        )
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
