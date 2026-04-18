"""FastAPI dashboard. Read-only views + the kill switch toggle."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.state.balances import (
    daily_pnl_usd,
    equity_curve_points,
    realized_pnl_usd,
    total_open_exposure_usd,
)
from core.state.health import latest as latest_health, run_all as run_all_health
from core.state.positions import list_open
from core.utils.config import get_config, root_dir
from core.utils.db import fetch_all
from core.utils.helpers import now_ts


def create_app() -> FastAPI:
    app = FastAPI(title="Quantorpolybot")
    base = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base / "templates"))
    app.mount("/static", StaticFiles(directory=str(base / "static")), name="static")
    cfg = get_config()
    refresh = int(cfg.get("dashboard", "refresh_seconds", default=10))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        health = await latest_health()
        if not health:
            health = {h.component: h for h in await run_all_health()}
        positions = await list_open()
        ctx = {
            "request": request,
            "title": "Overview",
            "refresh": refresh,
            "dry_run": bool(cfg.get("dry_run", default=True)),
            "live_enabled": bool(cfg.get("live_trading_enabled", default=False)),
            "health": health,
            "positions": positions,
            "exposure": await total_open_exposure_usd(),
            "realized_pnl": await realized_pnl_usd(),
            "daily_pnl": await daily_pnl_usd(),
            "equity_points": await equity_curve_points(),
        }
        return templates.TemplateResponse("index.html", ctx)

    @app.get("/markets", response_class=HTMLResponse)
    async def markets(request: Request) -> HTMLResponse:
        rows = await fetch_all(
            "SELECT * FROM markets WHERE active=1 ORDER BY liquidity DESC LIMIT 100"
        )
        return templates.TemplateResponse(
            "markets.html",
            {"request": request, "title": "Markets", "refresh": refresh, "rows": rows},
        )

    @app.get("/signals", response_class=HTMLResponse)
    async def signals(request: Request) -> HTMLResponse:
        rows = await fetch_all(
            """SELECT s.*, m.question FROM signals s
               LEFT JOIN markets m ON m.market_id = s.market_id
               ORDER BY s.created_at DESC LIMIT 100"""
        )
        return templates.TemplateResponse(
            "signals.html",
            {"request": request, "title": "Signals", "refresh": refresh, "rows": rows},
        )

    @app.get("/orders", response_class=HTMLResponse)
    async def orders(request: Request) -> HTMLResponse:
        rows = await fetch_all(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT 100"
        )
        executions = await fetch_all(
            "SELECT * FROM executions ORDER BY ts DESC LIMIT 50"
        )
        return templates.TemplateResponse(
            "orders.html",
            {
                "request": request,
                "title": "Orders",
                "refresh": refresh,
                "orders": rows,
                "executions": executions,
            },
        )

    @app.get("/logs", response_class=HTMLResponse)
    async def logs(request: Request) -> HTMLResponse:
        rows = await fetch_all(
            "SELECT * FROM system_log ORDER BY ts DESC LIMIT 200"
        )
        log_path = root_dir() / cfg.get("log_dir", default="logs") / "polybot.log"
        tail = ""
        try:
            if log_path.exists():
                with log_path.open("r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                tail = "".join(lines[-200:])
        except Exception:
            tail = "(log file unreadable)"
        return templates.TemplateResponse(
            "logs.html",
            {
                "request": request,
                "title": "Logs",
                "refresh": refresh,
                "rows": rows,
                "tail": tail,
            },
        )

    @app.post("/kill-switch")
    async def kill_switch(enabled: str = Form("")) -> RedirectResponse:
        data = cfg.as_dict()
        data["live_trading_enabled"] = enabled.lower() in ("1", "true", "on", "yes")
        cfg.save(data)
        cfg.reload()
        return RedirectResponse("/", status_code=303)

    @app.get("/api/health")
    async def api_health():
        from core.state.health import run_all
        reports = await run_all()
        return [{"component": r.component, "ok": r.ok, "detail": r.detail} for r in reports]

    @app.get("/api/equity")
    async def api_equity():
        return {"points": await equity_curve_points(), "now": now_ts()}

    return app
