"""Single entrypoint for NexoPolyBot.

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
import threading
from contextlib import suppress

import uvicorn
from loguru import logger

from core.execution import allocator
from core.execution.monitor import OrderMonitor
from core.execution.risk_manager import ShadowRiskManager
from core.feeds.manager import FeedManager
from core.feeds.price_refresher import PriceRefresher
from core.markets.discovery import MarketDiscovery
from core.risk.rules import RiskEngine
from core.scheduler.jobs import JobScheduler
from core.signals.ollama_client import OllamaClient, aclose_shared_client
from core.signals.pipeline import SignalPipeline
from core.strategies.breaking_event_scout import BreakingEventScoutLane
from core.strategies.event_sniper import EventSniperLane
from core.strategies.longshot import LongshotLane
from core.strategies.microscalp import MicroscalpLane
from core.strategies.resolution_day import ResolutionDayLane
from core.strategies.scalping import ScalpingLane
from core.utils.config import env, get_config, load_env
from core.utils.db import init_db
from core.utils.logging import audit, setup_logging
from core.utils.sanity import run_startup_checks
from core.utils.watchdog import Watchdog


class App:
    def __init__(self) -> None:
        self.feeds = FeedManager()
        self.discovery = MarketDiscovery()
        self.risk = RiskEngine()
        self.monitor = OrderMonitor()
        self.signals = SignalPipeline(self.risk)
        self.scheduler = JobScheduler(self.discovery)
        self.shadow_risk = ShadowRiskManager()
        self.scalping_lane = ScalpingLane()
        self.event_lane = EventSniperLane()
        self.longshot_lane = LongshotLane()
        self.resolution_day_lane = ResolutionDayLane()
        self.microscalp_lane = MicroscalpLane()
        # Step #3 PR #1: Breaking Event Scout (SHADOW only).
        self.scout_lane = BreakingEventScoutLane()
        self.price_refresher = PriceRefresher()
        # Watchdog needs a handle to poly_ws so it can force a reconnect
        # when the CLOB socket goes silent for too long.
        self.watchdog = Watchdog(poly_ws=self.feeds.poly_ws)
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._uvicorn_server: uvicorn.Server | None = None
        self._dashboard_thread: threading.Thread | None = None
        self._dashboard_loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        from core.brand import print_startup_banner
        print_startup_banner()
        logger.info(
            "NexoPolyBot starting (mode={})",
            str(get_config().get("mode", default="shadow")).upper(),
        )
        await init_db()

        # Cheap reachability pings before heavy subsystem boot, so any
        # misconfigured host/port shows up early in the log instead of
        # being buried under retry spam from 10 different feed loops.
        await run_startup_checks()

        # Force Ollama to swap the deep-tier model into VRAM *before*
        # any lane or the signals pipeline can dispatch a call. Without
        # this, the first ~20s of boot see ConnectTimeouts because
        # Ollama's HTTP listener briefly stalls while mmap'ing the
        # 5 GB GGUF off disk on Windows. Non-blocking on failure —
        # per-call retries still handle a degraded Ollama afterward.
        try:
            await OllamaClient().warmup()
        except Exception as e:
            logger.warning("[boot] Ollama warmup raised: {}", e)

        # Initial market discovery so we have token ids before WS subscribes.
        try:
            n = await self.discovery.refresh_once()
            logger.info("[boot] initial market refresh: {} markets", n)
        except Exception as e:
            logger.warning("[boot] initial market refresh failed: {}", e)

        await self.feeds.start()

        # Bootstrap the three-lane shadow capital before any lane scans.
        await allocator.init_lane_capital()

        self._tasks.append(asyncio.create_task(self.discovery.run(), name="markets.discovery"))
        self._tasks.append(asyncio.create_task(self.price_refresher.run(), name="feeds.price_refresher"))
        self._tasks.append(asyncio.create_task(self.signals.run(), name="signals.pipeline"))
        self._tasks.append(asyncio.create_task(self.monitor.run(), name="execution.monitor"))
        # Shadow trading — three lanes + their risk manager.
        self._tasks.append(asyncio.create_task(self.shadow_risk.run(), name="shadow.risk"))
        self._tasks.append(asyncio.create_task(self.scalping_lane.run(), name="shadow.scalping"))
        self._tasks.append(asyncio.create_task(self.event_lane.run(), name="shadow.event"))
        self._tasks.append(asyncio.create_task(self.longshot_lane.run(), name="shadow.longshot"))
        self._tasks.append(asyncio.create_task(self.resolution_day_lane.run(), name="shadow.resolution_day"))
        self._tasks.append(asyncio.create_task(self.microscalp_lane.run(), name="shadow.microscalp"))
        # Step #3 PR #1: Breaking Event Scout (SHADOW only).
        self._tasks.append(asyncio.create_task(self.scout_lane.run(), name="shadow.breaking_event_scout"))
        self._tasks.append(asyncio.create_task(self.watchdog.run(), name="utils.watchdog"))

        self.scheduler.start()
        await self._start_dashboard()
        audit("system_start")
        logger.info("[boot] all subsystems running")

    async def _start_dashboard(self) -> None:
        """Run uvicorn in its own daemon thread with a dedicated event
        loop so the dashboard doesn't lag when the main loop is busy
        handling WS reconnects, Ollama calls, or market discovery.

        Dashboard handlers talk to SQLite via aiosqlite (WAL mode +
        busy_timeout, see core/utils/db.py). No cross-loop asyncio
        primitives cross the boundary — each loop opens its own
        connections.

        Port-conflict handling: a previous bot process (or anything
        else) may already own the configured port. The April 2026 soak
        showed uvicorn's bind error gets buried in the log and the
        rest of the bot keeps running with no dashboard. We pre-probe
        with a synchronous bind, and on conflict we walk forward up to
        ``DASHBOARD_PORT_RETRIES`` ports before giving up. The chosen
        port is logged loudly so the operator knows where to point
        their browser.
        """
        import socket

        from dashboard.app import create_app

        host = env("DASHBOARD_HOST", get_config().get("dashboard", "host", default="127.0.0.1"))
        configured_port = int(
            env("DASHBOARD_PORT", str(get_config().get("dashboard", "port", default=8000)))
        )
        retries = int(get_config().get("dashboard", "port_retries", default=5))
        port = _find_open_port(host, configured_port, retries=retries)
        if port is None:
            logger.error(
                "[dashboard] could not bind to any port in {}..{} on {} — "
                "is another bot instance already running? "
                "Set DASHBOARD_PORT to an explicit free port to override.",
                configured_port, configured_port + retries, host,
            )
            return
        if port != configured_port:
            logger.warning(
                "[dashboard] port {} was busy on {}; rolled forward to {}",
                configured_port, host, port,
            )

        config = uvicorn.Config(
            create_app(),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._uvicorn_server = uvicorn.Server(config)
        ready = threading.Event()

        def _run_dashboard_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._dashboard_loop = loop
            ready.set()
            try:
                loop.run_until_complete(self._uvicorn_server.serve())
            except Exception as e:
                logger.exception("[dashboard] thread error: {}", e)
            finally:
                with suppress(Exception):
                    loop.close()

        t = threading.Thread(
            target=_run_dashboard_loop, name="dashboard", daemon=True,
        )
        t.start()
        # Wait briefly for the loop reference; prevents a race in
        # shutdown() where we'd try to signal a not-yet-started server.
        ready.wait(timeout=5.0)
        self._dashboard_thread = t
        logger.info("[dashboard] http://{}:{} (in dedicated thread)", host, port)

    async def shutdown(self) -> None:
        logger.info("[shutdown] stopping subsystems...")
        await self.feeds.stop()
        await self.signals.stop()
        await self.discovery.stop()
        await self.price_refresher.stop()
        await self.monitor.stop()
        await self.shadow_risk.stop()
        await self.scalping_lane.stop()
        await self.event_lane.stop()
        await self.longshot_lane.stop()
        await self.resolution_day_lane.stop()
        await self.microscalp_lane.stop()
        await self.watchdog.stop()
        await self.scheduler.shutdown()
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        # Uvicorn runs in its own thread/loop, so its graceful-shutdown
        # hook fires there. Join the thread so we don't race the final
        # "complete" log line.
        if self._dashboard_thread is not None:
            self._dashboard_thread.join(timeout=10.0)

        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # Drop the shared Ollama httpx client and its connection pool.
        await aclose_shared_client()
        audit("system_stop")
        logger.info("[shutdown] complete")

    async def run_forever(self) -> None:
        await self.start()
        await self._stop.wait()
        await self.shutdown()

    def request_stop(self) -> None:
        self._stop.set()


def _find_open_port(host: str, start: int, *, retries: int = 5) -> int | None:
    """Try to bind ``host:start``; on EADDRINUSE walk forward up to
    ``retries`` more ports. Returns the first free port or ``None``
    if all are busy. The probe is synchronous + immediate (no listen
    backlog), so it's safe to call before uvicorn binds for real.

    Uvicorn doesn't give us a clean "did it bind?" hook before the
    server starts serving, so this pre-check is the only reliable way
    to fail loudly on port conflicts. The April 2026 soak showed
    uvicorn's [Errno 10048] just printed to stderr and the bot
    continued without a dashboard, which is silent corruption."""
    import socket

    for offset in range(retries + 1):
        port = start + offset
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        except OSError:
            continue
        else:
            return port
        finally:
            s.close()
    return None


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
