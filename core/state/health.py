"""Periodic health checks for the dashboard. Pings each major subsystem
and writes the result to the health_checks table.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from loguru import logger

from core.signals.ollama_client import OllamaClient
from core.utils.config import get_config
from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import now_ts


@dataclass
class HealthReport:
    component: str
    ok: bool
    detail: str


async def _check_db() -> HealthReport:
    try:
        row = await fetch_one("SELECT COUNT(*) AS n FROM markets")
        return HealthReport("db", True, f"markets={row['n'] if row else 0}")
    except Exception as e:
        return HealthReport("db", False, str(e))


async def _check_ollama() -> HealthReport:
    client = OllamaClient()
    ok = await client.healthy()
    return HealthReport("ollama", ok, f"host={client.host} model={client.model}")


async def _check_polymarket() -> HealthReport:
    base = get_config().get("markets", "gamma_url", default="https://gamma-api.polymarket.com/markets")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(base, params={"limit": 1})
            return HealthReport("polymarket", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        return HealthReport("polymarket", False, str(e))


async def run_all() -> list[HealthReport]:
    reports = [
        await _check_db(),
        await _check_ollama(),
        await _check_polymarket(),
    ]
    ts = now_ts()
    for r in reports:
        try:
            await execute(
                "INSERT INTO health_checks (ts, component, ok, detail) VALUES (?,?,?,?)",
                (ts, r.component, 1 if r.ok else 0, r.detail),
            )
        except Exception as e:
            logger.warning("[health] persist failed: {}", e)
    return reports


async def latest() -> dict[str, HealthReport]:
    rows = await fetch_all(
        """SELECT component, ok, detail, ts FROM health_checks
           WHERE id IN (SELECT MAX(id) FROM health_checks GROUP BY component)"""
    )
    return {
        r["component"]: HealthReport(
            component=r["component"],
            ok=bool(r["ok"]),
            detail=r["detail"] or "",
        )
        for r in rows
    }
