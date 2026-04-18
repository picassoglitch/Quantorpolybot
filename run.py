"""Single entrypoint for Quantorpolybot.

  python run.py

Runs everything in one async event loop:
  - DB init
  - Feed manager (RSS, FRED, Metaculus, Polymarket WS)
  - Market discovery loop
  - Signal pipeline (Ollama)
  - Order monitor
  - APScheduler (refreshes, optimization, learning, health)
  - FastAPI dashboard on localhost:8000

Catches all top-level exceptions, logs them, and never lets the main
process crash. Graceful shutdown cancels open orders first.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import suppress

import uvicorn
from loguru import logger

from core.execution.monitor import OrderMonitor
from core.execution.orders import OrderEngine
from core.feeds.manager import FeedManager
from core.markets.discovery import MarketDiscovery
from core.risk.rules import RiskEngine
from core.scheduler.jobs import JobScheduler
from core.signals.pipeline import SignalPipeline
from core.utils.config import env, get_config, load_env
from core.utils.db import init_db
from core.utils.logging import audit, setup_logging


class App:
    def __init__(self) -> None:
        self.feeds = FeedManager()
        self.discovery = MarketDiscovery()
        self.risk = RiskEngine()
        self.orders = OrderEngine()
        self.monitor = OrderMonitor()
        self.signals = SignalPipeline(self.orders, self.risk)
        self.scheduler = JobScheduler(self.discovery)
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._uvicorn_server: uvicorn.Server | None = None

    async def start(self) -> None:
        logger.info("=" * 60)
        logger.info("Quantorpolybot starting (dry_run={}, live_trading_enabled={})",
                    get_config().get("dry_run", default=True),
                    get_config().get("live_trading_enabled", default=False))
        logger.info("=" * 60)
        await init_db()

        # Initial market discovery so we have token ids before WS subscribes.
        try:
            n = await self.discovery.refresh_once()
            logger.info("[boot] initial market refresh: {} markets", n)
        except Exception as e:
            logger.warning("[boot] initial market refresh failed: {}", e)

        await self.feeds.start()

        self._tasks.append(asyncio.create_task(self.discovery.run(), name="markets.discovery"))
        self._tasks.append(asyncio.create_task(self.signals.run(), name="signals.pipeline"))
        self._tasks.append(asyncio.create_task(self.monitor.run(), name="execution.monitor"))

        self.scheduler.start()
        await self._start_dashboard()
        audit("system_start")
        logger.info("[boot] all subsystems running")

    async def _start_dashboard(self) -> None:
        from dashboard.app import create_app

        host = env("DASHBOARD_HOST", get_config().get("dashboard", "host", default="127.0.0.1"))
        port = int(env("DASHBOARD_PORT", str(get_config().get("dashboard", "port", default=8000))))
        config = uvicorn.Config(
            create_app(),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._tasks.append(
            asyncio.create_task(self._uvicorn_server.serve(), name="dashboard")
        )
        logger.info("[dashboard] http://{}:{}", host, port)

    async def shutdown(self) -> None:
        logger.info("[shutdown] cancelling open orders...")
        try:
            await asyncio.wait_for(self.orders.cancel_all_open(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("[shutdown] cancel_all_open timed out")
        except Exception as e:
            logger.warning("[shutdown] cancel error: {}", e)

        await self.feeds.stop()
        await self.signals.stop()
        await self.discovery.stop()
        await self.monitor.stop()
        await self.scheduler.shutdown()
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True

        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        audit("system_stop")
        logger.info("[shutdown] complete")

    async def run_forever(self) -> None:
        await self.start()
        await self._stop.wait()
        await self.shutdown()

    def request_stop(self) -> None:
        self._stop.set()


def _install_signal_handlers(app: App, loop: asyncio.AbstractEventLoop) -> None:
    def _handler() -> None:
        logger.info("[signal] stop requested")
        app.request_stop()

    if sys.platform == "win32":
        # Windows: SIGINT delivered via KeyboardInterrupt in the loop
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handler)


async def _main() -> None:
    load_env()
    setup_logging()
    app = App()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(app, loop)
    try:
        await app.run_forever()
    except KeyboardInterrupt:
        logger.info("[signal] KeyboardInterrupt")
        app.request_stop()
        await app.shutdown()
    except Exception as e:
        logger.exception("[main] fatal: {}", e)
        await app.shutdown()


def main() -> int:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
