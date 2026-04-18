"""FastAPI dashboard. Read-only views + the kill switch toggle."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
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
from core.utils.config import env, get_config, root_dir
from core.utils.db import fetch_all
from core.utils.helpers import now_ts
from core.utils.secrets import EDITABLE_KEYS, fields_for_dashboard, update_env


async def _ollama_status() -> dict:
    """Probe the local Ollama server and report installed models + whether
    the configured model is one of them. Used by the settings page."""
    host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    configured = env("OLLAMA_MODEL", "mistral")
    out = {
        "host": host,
        "configured_model": configured,
        "reachable": False,
        "tag_count": 0,
        "installed_models": [],
        "model_present": False,
        "error": "",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{host}/api/tags")
            r.raise_for_status()
            data = r.json()
        models = [m.get("name", "") for m in (data.get("models") or [])]
        out["reachable"] = True
        out["installed_models"] = models
        out["tag_count"] = len(models)
        # Match either exact tag ("mistral:latest") or stem ("mistral").
        out["model_present"] = any(
            m == configured or m.split(":", 1)[0] == configured for m in models
        )
    except Exception as e:
        out["error"] = str(e)
    return out


def create_app() -> FastAPI:
    app = FastAPI(title="Quantorpolybot")
    base = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base / "templates"))
    # Disable Jinja2's bytecode/template cache. On Python 3.14 the default
    # LRU cache key construction blows up with
    # "TypeError: cannot use 'tuple' as a dict key (unhashable type: 'dict')"
    # whenever a template is rendered with a context containing dict values.
    templates.env.cache = None
    app.mount("/static", StaticFiles(directory=str(base / "static")), name="static")
    cfg = get_config()
    refresh = int(cfg.get("dashboard", "refresh_seconds", default=10))

    # Starlette >=0.29 requires `request` as the first positional argument
    # to TemplateResponse — the legacy `(name, context)` form is removed.

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        health = await latest_health()
        if not health:
            health = {h.component: h for h in await run_all_health()}
        positions = await list_open()
        ctx = {
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
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/markets", response_class=HTMLResponse)
    async def markets(request: Request) -> HTMLResponse:
        rows = await fetch_all(
            "SELECT * FROM markets WHERE active=1 ORDER BY liquidity DESC LIMIT 100"
        )
        return templates.TemplateResponse(
            request,
            "markets.html",
            {"title": "Markets", "refresh": refresh, "rows": rows},
        )

    @app.get("/signals", response_class=HTMLResponse)
    async def signals(request: Request) -> HTMLResponse:
        rows = await fetch_all(
            """SELECT s.*, m.question FROM signals s
               LEFT JOIN markets m ON m.market_id = s.market_id
               ORDER BY s.created_at DESC LIMIT 100"""
        )
        return templates.TemplateResponse(
            request,
            "signals.html",
            {"title": "Signals", "refresh": refresh, "rows": rows},
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
            request,
            "orders.html",
            {
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
            request,
            "logs.html",
            {
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

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, saved: int = 0) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "settings.html",
            {
                "title": "Settings",
                "refresh": 0,  # don't auto-refresh while editing
                "dry_run": bool(cfg.get("dry_run", default=True)),
                "live_enabled": bool(cfg.get("live_trading_enabled", default=False)),
                "fields": fields_for_dashboard(),
                "ollama_status": await _ollama_status(),
                "saved": saved,
            },
        )

    @app.post("/settings")
    async def settings_save(request: Request) -> RedirectResponse:
        form = await request.form()
        updates: dict[str, str] = {}
        for key, _label, sensitive, _help in EDITABLE_KEYS:
            if key not in form:
                continue
            raw = (form.get(key) or "").strip()
            if sensitive and raw == "":
                # Blank sensitive field = keep existing value, don't overwrite.
                continue
            if raw == "__clear__":
                raw = ""
            updates[key] = raw
        update_env(updates)
        return RedirectResponse(f"/settings?saved={len(updates)}", status_code=303)

    @app.get("/api/health")
    async def api_health():
        from core.state.health import run_all
        reports = await run_all()
        return [{"component": r.component, "ok": r.ok, "detail": r.detail} for r in reports]

    @app.get("/api/equity")
    async def api_equity():
        return {"points": await equity_curve_points(), "now": now_ts()}

    return app
