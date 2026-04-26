"""FastAPI dashboard.

Trader-grade read views for the three lanes plus a single mode switch
(shadow ↔ real). No dry-run flag anywhere; shadow IS the default
simulated-money mode, real is the "take money from the funded wallet"
mode. Flipping to real requires an explicit confirm.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core.execution import allocator, clob_client
from core.execution import shadow as shadow_engine
from core.feeds import news_store
from core.feeds.rss import configured_feeds
from core.signals.ollama_client import OllamaClient
from core.state.balances import (
    daily_pnl_usd,
    equity_curve_points,
    realized_pnl_usd,
    total_open_exposure_usd,
)
from core.state.health import latest as latest_health, run_all as run_all_health
from core.state.positions import list_open
from core.utils.config import env, get_config, root_dir
from core.utils.db import fetch_all, fetch_one
from core.utils.helpers import now_ts
from core.utils.secrets import EDITABLE_KEYS, fields_for_dashboard, update_env


MODE_CHOICES = ("shadow", "real")


def _fmt_age(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def _fmt_age_ago(seconds: float | None) -> str:
    age = _fmt_age(seconds)
    if age in ("?",):
        return "never"
    return f"{age} ago"


def _fmt_datetime(value, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Render a timestamp as a readable UTC string.

    Accepts unix epoch seconds (int/float) or an ISO-8601 string.
    Used by templates as the ``datetimefmt`` Jinja filter so we don't
    show raw epochs like ``1745318400`` in the UI.
    """
    if value is None or value == "":
        return "—"
    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            s = str(value).strip()
            if not s:
                return "—"
            # ISO-8601 with optional trailing Z
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                # Last-ditch: numeric-string epoch
                dt = datetime.fromtimestamp(float(s), tz=timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime(fmt)
    except (ValueError, OSError, TypeError):
        return str(value)


# Human labels for the `status` column on the signals page. The
# pipeline writes machine codes like `market_throttled` or
# `keyword_mismatch`; show traders something they can scan.
_SIGNAL_STATUS_LABELS = {
    "pending": "PENDING",
    "approved": "APPROVED",
    "rejected": "REJECTED",
    "market_throttled": "COOLDOWN",
    "keyword_mismatch": "NO MATCH",
    "below_min_edge": "LOW EDGE",
    "implausible_edge": "IMPLAUSIBLE",
    "stale_market": "STALE MKT",
    "duplicate": "DUPLICATE",
    "long_horizon": "TOO FAR",
}


def _fmt_signal_status(status) -> str:
    if status is None:
        return "—"
    key = str(status).strip().lower()
    return _SIGNAL_STATUS_LABELS.get(key, str(status).upper())


# Statuses where scoring fields (implied/mid/edge/conf/size) are not
# meaningful — the signal was rejected before pricing, or throttled
# before an LLM call. Showing `0.000` in those cells reads as "zero
# edge" which is misleading. Render `—` instead.
_PRESCORE_REJECT_STATUSES = {
    "market_throttled",
    "keyword_mismatch",
    "stale_market",
    "duplicate",
    "long_horizon",
}


def _is_prescore_reject(status) -> bool:
    if status is None:
        return False
    return str(status).strip().lower() in _PRESCORE_REJECT_STATUSES


def _looks_like_rss_name(name: str) -> bool:
    """Heuristic label for the sources page. Named RSS feeds come from
    rss.DEFAULT_FEEDS (`coindesk`, `reuters_markets`, etc.) and don't
    start with a scheme. Non-RSS source names tend to be raw URLs
    (legacy entries) or explicit keys (`fred:DGS10`, `google_news`)."""
    if not name:
        return False
    if name.startswith(("http://", "https://")):
        return True
    # Non-RSS source keys all have a namespace separator.
    return ":" not in name and name not in {
        "google_news", "polymarket_news", "metaculus", "wikipedia",
        "predictit_xref",
    }


def _current_mode() -> str:
    return allocator.current_mode()


def _mode_ctx(cfg) -> dict:
    mode = _current_mode()
    shadow_cap = get_config().get("shadow_capital") or {}
    real_cap = get_config().get("real_capital") or {}
    return {
        "mode": mode,
        "is_real": mode == "real",
        "shadow_total": float(shadow_cap.get("total_usd") or 0.0),
        "real_total": float(real_cap.get("total_usd") or 0.0),
        "clob_ready": clob_client.is_ready(),
    }


async def _ollama_stats_summary() -> dict:
    cfg = _ollama_cfg_for_dashboard()
    tiers: list[dict] = [
        {"tier": "fast", "model": cfg.get("fast_model") or cfg.get("deep_model") or cfg.get("model") or ""},
        {"tier": "deep", "model": cfg.get("deep_model") or cfg.get("model") or ""},
        {"tier": "validator", "model": cfg.get("validator_model") or cfg.get("deep_model") or ""},
    ]
    now = now_ts()
    one_hour = now - 3600
    one_day = now - 86400

    per_tier: list[dict] = []
    for entry in tiers:
        tier = entry["tier"]
        model = entry["model"]
        row_100 = await fetch_all(
            """SELECT AVG(latency_ms) AS avg_ms,
                      AVG(CASE WHEN success=1 THEN 0.0 ELSE 1.0 END) AS err_rate,
                      COUNT(*) AS n
               FROM (SELECT latency_ms, success FROM ollama_stats
                     WHERE call_type=? ORDER BY called_at DESC LIMIT 100)""",
            (tier,),
        )
        row_10 = await fetch_all(
            """SELECT AVG(latency_ms) AS avg_ms, COUNT(*) AS n
               FROM (SELECT latency_ms FROM ollama_stats
                     WHERE call_type=? AND success=1 ORDER BY called_at DESC LIMIT 10)""",
            (tier,),
        )
        vol_1h = await fetch_all(
            "SELECT COUNT(*) AS n FROM ollama_stats WHERE call_type=? AND called_at>=?",
            (tier, one_hour),
        )
        vol_24h = await fetch_all(
            "SELECT COUNT(*) AS n FROM ollama_stats WHERE call_type=? AND called_at>=?",
            (tier, one_day),
        )
        per_tier.append({
            "tier": tier,
            "model": model,
            "avg_latency_ms_10": float((row_10[0]["avg_ms"] if row_10 and row_10[0]["avg_ms"] is not None else 0.0)),
            "avg_latency_ms_100": float((row_100[0]["avg_ms"] if row_100 and row_100[0]["avg_ms"] is not None else 0.0)),
            "error_rate_100": float((row_100[0]["err_rate"] if row_100 and row_100[0]["err_rate"] is not None else 0.0)),
            "n_last_100": int((row_100[0]["n"] if row_100 else 0) or 0),
            "n_1h": int((vol_1h[0]["n"] if vol_1h else 0) or 0),
            "n_24h": int((vol_24h[0]["n"] if vol_24h else 0) or 0),
        })

    client = OllamaClient()
    running = await client.running_models()
    running_names = {m.get("name", "") for m in running if m.get("name")}
    for r in per_tier:
        name = r["model"]
        stem = name.split(":", 1)[0]
        r["loaded_in_vram"] = name in running_names or any(
            n == name or n.split(":", 1)[0] == stem for n in running_names
        )

    depths = OllamaClient.queue_depths()
    alert = int(_ollama_cfg_for_dashboard().get("queue_depth_alert", 5))
    validator_threshold = float(
        cfg.get("validator_high_stakes_usd", 10.0) or 10.0
    )
    return {
        "host": client.host,
        "tiers": per_tier,
        "queue_depths": depths,
        "queue_alert_threshold": alert,
        "fast_saturated": OllamaClient.fast_queue_saturated(),
        "running_models": running,
        "validator_threshold_usd": validator_threshold,
        "now": now,
    }


def _ollama_cfg_for_dashboard() -> dict:
    return get_config().get("ollama") or {}


async def _ollama_status() -> dict:
    host = env("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    configured = env("OLLAMA_MODEL", "") or str(
        (get_config().get("ollama") or {}).get("deep_model", "")
    )
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
        out["model_present"] = any(
            m == configured or m.split(":", 1)[0] == configured for m in models
        )
    except Exception as e:
        out["error"] = str(e)
    return out


async def _recent_trades(limit: int = 50, is_real: int | None = None) -> list[dict]:
    """Pull recent positions joined to market metadata for the Trades tab.

    Returns dicts with: lane, market_id, question, side, entry_price,
    last_price, size_usd, size_shares, status, mode, age_seconds,
    pnl_usd, pnl_pct, edge_entry, confidence_entry, entry_reason,
    close_reason, evidence_count, clob_order_id.
    """
    where = "1=1"
    params: list = []
    if is_real is not None:
        where = "COALESCE(p.is_real, 0) = ?"
        params.append(is_real)
    rows = await fetch_all(
        f"""SELECT p.*, m.question, m.category
            FROM shadow_positions p
            LEFT JOIN markets m ON m.market_id = p.market_id
            WHERE {where}
            ORDER BY p.entry_ts DESC LIMIT ?""",
        tuple(params + [limit]),
    )
    now = now_ts()
    out: list[dict] = []
    for r in rows:
        entry = float(r["entry_price"] or 0)
        last = float(r["last_price"] or entry)
        size_usd = float(r["size_usd"] or 0)
        side = r["side"] or ""
        if r["status"] == "CLOSED":
            pnl = float(r["realized_pnl_usd"] or 0)
        else:
            pnl = float(r["unrealized_pnl_usd"] or 0)
        if entry > 0:
            if side == "BUY":
                pnl_pct = (last - entry) / entry * 100.0
            else:
                pnl_pct = (entry - last) / entry * 100.0
        else:
            pnl_pct = 0.0
        try:
            is_real_row = int(r["is_real"] or 0)
        except (IndexError, KeyError):
            is_real_row = 0
        try:
            clob_id = r["clob_order_id"] or ""
        except (IndexError, KeyError):
            clob_id = ""
        # Evidence count from the JSON array.
        import json as _json
        try:
            ev = _json.loads(r["cited_evidence_ids"] or "[]")
            evidence_count = len(ev) if isinstance(ev, list) else 0
        except Exception:
            evidence_count = 0
        entry_ts = float(r["entry_ts"] or now)
        out.append({
            "id": r["id"],
            "lane": r["strategy"],
            "market_id": r["market_id"],
            "question": (r["question"] or "").strip() or "(market not in cache)",
            "category": r["category"] or "",
            "side": side,
            "entry_price": entry,
            "last_price": last,
            "size_usd": size_usd,
            "size_shares": float(r["size_shares"] or 0),
            "status": r["status"],
            "is_real": bool(is_real_row),
            "mode": "real" if is_real_row else "shadow",
            "age_seconds": now - entry_ts,
            "age_ago": _fmt_age_ago(now - entry_ts),
            "pnl_usd": pnl,
            "pnl_pct": pnl_pct,
            "true_prob_entry": float(r["true_prob_entry"] or 0),
            "confidence_entry": float(r["confidence_entry"] or 0),
            "edge_entry": float(r["true_prob_entry"] or 0) - entry if side == "BUY"
                          else entry - float(r["true_prob_entry"] or 0),
            "entry_reason": (r["entry_reason"] or "")[:200],
            "close_reason": r["close_reason"] or "",
            "entry_latency_ms": float(r["entry_latency_ms"] or 0),
            "evidence_count": evidence_count,
            "clob_order_id": clob_id,
        })
    return out


async def _dashboard_counts() -> dict:
    """Top-level counters for the header strip."""
    now = now_ts()
    markets = await fetch_one("SELECT COUNT(*) AS n FROM markets WHERE active=1")
    feed_1h = await fetch_one(
        "SELECT COUNT(*) AS n FROM feed_items WHERE ingested_at >= ?",
        (now - 3600,),
    )
    signals_1h = await fetch_one(
        "SELECT COUNT(*) AS n FROM signals WHERE created_at >= ?",
        (now - 3600,),
    )
    open_positions = await fetch_one(
        "SELECT COUNT(*) AS n FROM shadow_positions WHERE status IN ('OPEN','PENDING_FILL')"
    )
    return {
        "markets_active": int(markets["n"] if markets else 0),
        "feed_1h": int(feed_1h["n"] if feed_1h else 0),
        "signals_1h": int(signals_1h["n"] if signals_1h else 0),
        "open_positions": int(open_positions["n"] if open_positions else 0),
    }


# ================================================================
# Cinematic /markets helpers
# ================================================================
# The /markets page is server-rendered every 10s (meta-refresh), so
# N+1 queries across ~15 rows are fine: each hits an indexed lookup
# on (market_id) or (market_id, ts) and the total stays well under
# 50ms even on a cold SQLite cache. We avoid a single giant JOIN on
# purpose — it keeps each helper independently testable and easy to
# reason about when one bucket returns zero rows.


async def _price_ticks_for(
    market_id: str,
    minutes: int = 60,
    max_points: int = 30,
) -> list[tuple[float, float]]:
    """Return up to `max_points` (ts, mid) samples within the last
    `minutes`. Down-sampled by stride if we have more raw ticks than
    the sparkline needs — the eye can't see more than ~30 points in a
    120px-wide spark anyway."""
    since = now_ts() - minutes * 60
    rows = await fetch_all(
        """SELECT ts, bid, ask, last FROM price_ticks
           WHERE market_id = ? AND ts >= ?
           ORDER BY ts ASC""",
        (market_id, since),
    )
    if not rows:
        return []
    series: list[tuple[float, float]] = []
    for r in rows:
        bid = float(r["bid"] or 0)
        ask = float(r["ask"] or 0)
        last = float(r["last"] or 0)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        elif last > 0:
            mid = last
        else:
            continue
        series.append((float(r["ts"]), mid))
    if len(series) <= max_points:
        return series
    step = len(series) / max_points
    return [series[min(int(i * step), len(series) - 1)] for i in range(max_points)]


def _sparkline_points(
    points: list[tuple[float, float]],
    w: int = 120,
    h: int = 28,
) -> dict:
    """Project (ts, price) pairs into SVG polyline coordinates.
    Returns a dict with `line` (points string for <polyline>), `area`
    (closed polygon for fill), `pct` 1h change, and `direction` tag
    so the template can colour up/down/flat."""
    empty = {"line": "", "area": "", "pct": 0.0, "direction": "flat", "n": 0}
    if len(points) < 2:
        return empty
    ys = [y for _, y in points]
    y_min, y_max = min(ys), max(ys)
    span = max(y_max - y_min, 1e-6)
    n = len(points)
    pad_y = 2.0
    usable_h = h - 2 * pad_y

    def _proj(i: int, y: float) -> str:
        x = (i / (n - 1)) * w
        # SVG y grows downward — flip so "up in price" renders upward.
        yy = h - pad_y - ((y - y_min) / span) * usable_h
        return f"{x:.1f},{yy:.1f}"

    coords = [_proj(i, y) for i, (_, y) in enumerate(points)]
    line = " ".join(coords)
    # Closed polygon for the soft filled area below the line.
    area = f"0,{h} " + line + f" {w},{h}"
    first, last = ys[0], ys[-1]
    pct = ((last - first) / first * 100.0) if first > 0 else 0.0
    if pct > 0.3:
        direction = "up"
    elif pct < -0.3:
        direction = "down"
    else:
        direction = "flat"
    return {
        "line": line, "area": area, "pct": pct,
        "direction": direction, "n": n,
    }


async def _hydrate_featured(market_id: str, bucket: str) -> dict | None:
    """Pull every field the featured-row template needs in one shot.
    Correlated subqueries are fine at this scale — all indexed."""
    row = await fetch_one(
        """SELECT m.*,
             (SELECT MAX(created_at) FROM signals
                WHERE market_id = m.market_id) AS last_sig_ts,
             (SELECT status FROM signals
                WHERE market_id = m.market_id
                ORDER BY created_at DESC LIMIT 1) AS last_sig_status,
             (SELECT edge FROM signals
                WHERE market_id = m.market_id
                ORDER BY created_at DESC LIMIT 1) AS last_sig_edge,
             (SELECT side FROM signals
                WHERE market_id = m.market_id
                  AND side IN ('BUY','SELL')
                ORDER BY created_at DESC LIMIT 1) AS last_sig_side,
             (SELECT reasoning FROM signals
                WHERE market_id = m.market_id
                ORDER BY created_at DESC LIMIT 1) AS last_sig_reason,
             (SELECT COUNT(*) FROM shadow_positions
                WHERE market_id = m.market_id
                  AND status IN ('OPEN','PENDING_FILL')) AS open_positions,
             (SELECT COALESCE(SUM(unrealized_pnl_usd), 0.0) FROM shadow_positions
                WHERE market_id = m.market_id AND status='OPEN') AS open_pnl,
             (SELECT COALESCE(SUM(size_usd), 0.0) FROM shadow_positions
                WHERE market_id = m.market_id
                  AND status IN ('OPEN','PENDING_FILL')) AS open_notional,
             (SELECT strategy FROM shadow_positions
                WHERE market_id = m.market_id
                  AND status IN ('OPEN','PENDING_FILL')
                ORDER BY entry_ts DESC LIMIT 1) AS open_lane,
             (SELECT MAX(entry_ts) FROM shadow_positions
                WHERE market_id = m.market_id) AS last_trade_ts
           FROM markets m
           WHERE m.market_id = ?""",
        (market_id,),
    )
    if not row:
        return None
    spark_pts = await _price_ticks_for(market_id)
    out = dict(row)
    out["bucket"] = bucket
    out["spark"] = _sparkline_points(spark_pts)
    return out


async def _select_featured_rows(limit: int = 15) -> list[dict]:
    """Bucket-priority picker for the cinematic /markets grid.

    1. Open positions — never truncated; they're always on screen.
    2. Approved signals < 60m old, newest first.
    3. Rejected signals < 15m old, sorted by abs(edge) so the "close
       calls" show rather than a wall of obvious-no's.
    4. Cooldown/throttled markets, capped at 3 so one chatty market
       can't monopolise the watch slots.
    5. Padding with top-liquidity active markets so the page always
       fills its row budget even on a quiet day.
    """
    now = now_ts()
    hour_ago = now - 3600
    fifteen_min_ago = now - 900
    picked: list[dict] = []
    seen: set[str] = set()

    async def _take(market_id: str, bucket: str) -> None:
        if len(picked) >= limit or market_id in seen or not market_id:
            return
        hyd = await _hydrate_featured(market_id, bucket)
        if hyd:
            picked.append(hyd)
            seen.add(market_id)

    # Bucket 1 — open positions
    pos_rows = await fetch_all(
        """SELECT DISTINCT market_id, MAX(entry_ts) AS ts FROM shadow_positions
           WHERE status IN ('OPEN','PENDING_FILL')
           GROUP BY market_id
           ORDER BY ts DESC"""
    )
    for r in pos_rows:
        await _take(r["market_id"], "pos")

    # Bucket 2 — approved signals < 60m
    if len(picked) < limit:
        appr = await fetch_all(
            """SELECT market_id, MAX(created_at) AS ts FROM signals
               WHERE status='APPROVED' AND created_at >= ?
               GROUP BY market_id ORDER BY ts DESC LIMIT 30""",
            (hour_ago,),
        )
        for r in appr:
            await _take(r["market_id"], "sig")

    # Bucket 3 — rejected signals < 15m, sorted by |edge|
    if len(picked) < limit:
        rej = await fetch_all(
            """SELECT market_id,
                      MAX(created_at) AS ts,
                      MAX(ABS(COALESCE(edge, 0))) AS abs_edge
               FROM signals
               WHERE status IN ('REJECTED','implausible_edge','below_min_edge')
                 AND created_at >= ?
               GROUP BY market_id
               ORDER BY abs_edge DESC LIMIT 30""",
            (fifteen_min_ago,),
        )
        for r in rej:
            await _take(r["market_id"], "rej")

    # Bucket 4 — cooldown / throttled (cap 3)
    cd_taken = 0
    if len(picked) < limit:
        cd = await fetch_all(
            """SELECT market_id, MAX(created_at) AS ts FROM signals
               WHERE status IN ('market_throttled','duplicate','long_horizon',
                                'keyword_mismatch','stale_market')
               GROUP BY market_id ORDER BY ts DESC LIMIT 15"""
        )
        for r in cd:
            if cd_taken >= 3:
                break
            before = len(picked)
            await _take(r["market_id"], "watch")
            if len(picked) > before:
                cd_taken += 1

    # Bucket 5 — padding: top-liquidity active markets
    if len(picked) < limit:
        # SQLite has no tuple-NOT-IN shortcut; build a placeholder list.
        if seen:
            ph = ",".join("?" for _ in seen)
            q = (
                f"SELECT market_id FROM markets "
                f"WHERE active=1 AND market_id NOT IN ({ph}) "
                f"ORDER BY liquidity DESC LIMIT ?"
            )
            params: tuple = (*seen, limit - len(picked))
        else:
            q = (
                "SELECT market_id FROM markets "
                "WHERE active=1 ORDER BY liquidity DESC LIMIT ?"
            )
            params = (limit - len(picked),)
        pad = await fetch_all(q, params)
        for r in pad:
            await _take(r["market_id"], "bg")

    return picked


async def _ticker_events(limit: int = 24) -> list[dict]:
    """Build the marquee feed: recent signals + position opens + closes,
    newest first. One list, mixed kinds, so the scrolling banner reads
    like a trade blotter ('17:42 SIG GOOG BUY +8¢ · 17:41 OPEN electi…')."""
    window = now_ts() - 3 * 3600  # 3h keeps a busy bot's marquee fresh
    events: list[dict] = []

    sigs = await fetch_all(
        """SELECT s.created_at AS ts, s.status, s.side, s.edge, s.market_id,
                  m.question
           FROM signals s
           LEFT JOIN markets m ON m.market_id = s.market_id
           WHERE s.created_at >= ?
           ORDER BY s.created_at DESC LIMIT ?""",
        (window, limit),
    )
    for r in sigs:
        status = (r["status"] or "").lower()
        if status == "approved":
            kind = "approved"
        elif status in ("rejected", "implausible_edge", "below_min_edge"):
            kind = "rejected"
        else:
            kind = "scan"
        events.append({
            "kind": kind,
            "ts": float(r["ts"] or 0),
            "side": r["side"],
            "edge": float(r["edge"] or 0),
            "market_id": r["market_id"],
            "question": (r["question"] or "")[:55],
        })

    opens = await fetch_all(
        """SELECT entry_ts AS ts, side, market_id, strategy, size_usd
           FROM shadow_positions
           WHERE entry_ts >= ?
           ORDER BY entry_ts DESC LIMIT ?""",
        (window, limit),
    )
    for r in opens:
        events.append({
            "kind": "open",
            "ts": float(r["ts"] or 0),
            "side": r["side"],
            "market_id": r["market_id"],
            "lane": r["strategy"],
            "size": float(r["size_usd"] or 0),
        })

    closes = await fetch_all(
        """SELECT close_ts AS ts, side, market_id, strategy, realized_pnl_usd
           FROM shadow_positions
           WHERE close_ts IS NOT NULL AND close_ts >= ?
           ORDER BY close_ts DESC LIMIT ?""",
        (window, limit),
    )
    for r in closes:
        pnl = float(r["realized_pnl_usd"] or 0)
        events.append({
            "kind": "win" if pnl > 0 else ("loss" if pnl < 0 else "close"),
            "ts": float(r["ts"] or 0),
            "side": r["side"],
            "market_id": r["market_id"],
            "lane": r["strategy"],
            "pnl": pnl,
        })

    events.sort(key=lambda e: e["ts"], reverse=True)
    return events[:limit]


async def _compute_tape() -> dict:
    """Tape-strip aggregates reused by both /markets and /markets/all."""
    now = now_ts()
    hour_ago = now - 3600
    day_ago = now - 86400
    sig_1h = await fetch_one(
        "SELECT COUNT(*) AS n FROM signals WHERE created_at >= ?",
        (hour_ago,),
    )
    sig_24h = await fetch_one(
        "SELECT COUNT(*) AS n FROM signals WHERE created_at >= ?",
        (day_ago,),
    )
    approved_24h = await fetch_one(
        """SELECT COUNT(*) AS n FROM signals
           WHERE created_at >= ? AND status = 'APPROVED'""",
        (day_ago,),
    )
    open_positions = await fetch_one(
        """SELECT COUNT(*) AS n, COALESCE(SUM(size_usd), 0) AS sz
           FROM shadow_positions
           WHERE status IN ('OPEN','PENDING_FILL')"""
    )
    last_trade = await fetch_one(
        "SELECT MAX(entry_ts) AS ts FROM shadow_positions"
    )
    total_active = await fetch_one(
        "SELECT COUNT(*) AS n FROM markets WHERE active=1"
    )
    return {
        "now": now,
        "signals_1h": int(sig_1h["n"] if sig_1h else 0),
        "signals_24h": int(sig_24h["n"] if sig_24h else 0),
        "approved_24h": int(approved_24h["n"] if approved_24h else 0),
        "open_positions": int(open_positions["n"] if open_positions else 0),
        "open_exposure": float(open_positions["sz"] if open_positions else 0.0),
        "last_trade_ts": float(last_trade["ts"] or 0) if last_trade else 0.0,
        "total_active": int(total_active["n"] if total_active else 0),
    }


def create_app() -> FastAPI:
    app = FastAPI(title="NexoPolyBot")
    base = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base / "templates"))
    templates.env.cache = None
    # Humanise raw epochs / ISO timestamps so the UI stops showing
    # `1745318400` and friends.
    templates.env.filters["datetimefmt"] = _fmt_datetime
    templates.env.filters["signalstatus"] = _fmt_signal_status
    templates.env.filters["prescore_reject"] = _is_prescore_reject
    app.mount("/static", StaticFiles(directory=str(base / "static")), name="static")
    cfg = get_config()
    refresh = int(cfg.get("dashboard", "refresh_seconds", default=10))

    def _base_ctx(**extra) -> dict:
        from core.brand import brand_footer_html
        from core.i18n import current_lang, t
        ctx = {
            "refresh": refresh,
            "lang": current_lang(),
            "nexo_footer": brand_footer_html(),
            "t": t,
            **_mode_ctx(cfg),
        }
        ctx.update(extra)
        return ctx

    # ---------- Overview ----------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        health = await latest_health()
        if not health:
            health = {h.component: h for h in await run_all_health()}
        positions = await list_open()
        # Split PnL by mode so the overview shows both buckets.
        counts = await _dashboard_counts()
        ctx = _base_ctx(
            title="Overview",
            health=health,
            positions=positions,
            exposure_shadow=await total_open_exposure_usd(is_real=0),
            exposure_real=await total_open_exposure_usd(is_real=1),
            realized_pnl_shadow=await realized_pnl_usd(is_real=0),
            realized_pnl_real=await realized_pnl_usd(is_real=1),
            daily_pnl_shadow=await daily_pnl_usd(is_real=0),
            daily_pnl_real=await daily_pnl_usd(is_real=1),
            equity_points=await equity_curve_points(),
            counts=counts,
        )
        return templates.TemplateResponse(request, "index.html", ctx)

    # ---------- Markets ----------
    #
    # Two views share data plumbing:
    #   /markets      — cinematic "trader terminal" focus view.
    #                   Hero + ≤14 featured rows + scrolling ticker.
    #   /markets/all  — dense liquidity-sorted table (legacy look),
    #                   kept for fast scanning of the full watchlist.

    @app.get("/markets", response_class=HTMLResponse)
    async def markets(request: Request) -> HTMLResponse:
        tape = await _compute_tape()
        rows = await _select_featured_rows(limit=15)
        events = await _ticker_events(limit=24)
        hero = rows[0] if rows else None
        featured = rows[1:] if rows else []
        return templates.TemplateResponse(
            request, "markets.html",
            _base_ctx(
                title="Markets",
                hero=hero,
                rows=featured,
                events=events,
                tape=tape,
            ),
        )

    @app.get("/markets/all", response_class=HTMLResponse)
    async def markets_all(request: Request) -> HTMLResponse:
        # Legacy full-watchlist view: every active market, sorted by
        # liquidity, with the bot's footprint per row.
        rows = await fetch_all(
            """SELECT m.*,
                 (SELECT MAX(created_at) FROM signals
                    WHERE market_id = m.market_id) AS last_sig_ts,
                 (SELECT status FROM signals
                    WHERE market_id = m.market_id
                    ORDER BY created_at DESC LIMIT 1) AS last_sig_status,
                 (SELECT side FROM signals
                    WHERE market_id = m.market_id
                      AND side IN ('BUY','SELL')
                    ORDER BY created_at DESC LIMIT 1) AS last_sig_side,
                 (SELECT COUNT(*) FROM shadow_positions
                    WHERE market_id = m.market_id
                      AND status IN ('OPEN','PENDING_FILL')) AS open_positions,
                 (SELECT MAX(entry_ts) FROM shadow_positions
                    WHERE market_id = m.market_id) AS last_trade_ts
               FROM markets m
               WHERE m.active = 1
               ORDER BY m.liquidity DESC
               LIMIT 100"""
        )
        tape = await _compute_tape()
        return templates.TemplateResponse(
            request, "markets_all.html",
            _base_ctx(title="All markets", rows=rows, tape=tape),
        )

    # ---------- Signals ----------

    @app.get("/signals", response_class=HTMLResponse)
    async def signals(request: Request) -> HTMLResponse:
        rows = await fetch_all(
            """SELECT s.*, m.question FROM signals s
               LEFT JOIN markets m ON m.market_id = s.market_id
               ORDER BY s.created_at DESC LIMIT 100"""
        )
        return templates.TemplateResponse(
            request, "signals.html",
            _base_ctx(title="Signals", rows=rows),
        )

    # ---------- Trades (renamed from Orders) ----------

    @app.get("/trades", response_class=HTMLResponse)
    async def trades(request: Request, mode_filter: str = "") -> HTMLResponse:
        is_real_arg: int | None
        if mode_filter == "shadow":
            is_real_arg = 0
        elif mode_filter == "real":
            is_real_arg = 1
        else:
            is_real_arg = None
        rows = await _recent_trades(limit=120, is_real=is_real_arg)
        return templates.TemplateResponse(
            request, "trades.html",
            _base_ctx(
                title="Trades",
                trades=rows,
                mode_filter=mode_filter or "all",
            ),
        )

    # /orders kept as a redirect so old bookmarks still work.
    @app.get("/orders")
    async def orders_redirect() -> RedirectResponse:
        return RedirectResponse("/trades", status_code=307)

    # ---------- Sources ----------

    @app.get("/sources", response_class=HTMLResponse)
    async def sources(request: Request) -> HTMLResponse:
        agg = await fetch_all(
            """SELECT source,
                      COUNT(*) AS total,
                      MAX(ingested_at) AS last_fetch,
                      MAX(source_weight) AS weight,
                      SUM(CASE WHEN ingested_at >= ? THEN 1 ELSE 0 END) AS last_hour
               FROM feed_items
               GROUP BY source
               ORDER BY COALESCE(last_fetch, 0) DESC""",
            (now_ts() - 3600,),
        )
        now = now_ts()
        # Sources seen in the DB so far.
        seen: set[str] = set()
        sources_view = []
        for row in agg:
            name = row["source"] or "(unknown)"
            # Legacy rows: older versions of the RSS feed wrote the raw
            # URL into feed_items.source instead of the configured name
            # (e.g. "https://www.espn.com/..." vs "espn"). Those rows
            # still live in the DB but the named equivalent is now the
            # source of truth — hide the URL form so the sources page
            # doesn't double-list every feed.
            if name.startswith(("http://", "https://")):
                continue
            seen.add(name)
            last = row["last_fetch"] or 0
            age = now - last if last else None
            if age is None:
                status = "empty"
                human = "never"
            elif age < 1800:
                status = "active"
                human = _fmt_age_ago(age)
            else:
                status = "idle"
                human = _fmt_age_ago(age)
            sources_view.append({
                "source": name,
                "kind": "rss" if _looks_like_rss_name(name) else "feed",
                "total": row["total"] or 0,
                "last_hour": row["last_hour"] or 0,
                "last_fetch_human": human,
                "weight": row["weight"],
                "status": status,
            })
        # Every RSS feed the bot is configured to poll — even if no row has
        # landed yet, so a freshly-added source shows "never" rather than
        # being invisible.
        for spec in configured_feeds():
            if spec.name in seen:
                continue
            sources_view.append({
                "source": spec.name,
                "kind": "rss",
                "total": 0,
                "last_hour": 0,
                "last_fetch_human": "never",
                "weight": spec.weight,
                "status": "empty",
            })

        enrichment = await news_store.enrichment_summary_24h(now)

        xref_rows = await fetch_all(
            """SELECT cr.*, m.question
               FROM cross_references cr
               LEFT JOIN markets m ON m.market_id = cr.polymarket_id
               ORDER BY cr.divergence DESC, cr.fetched_at DESC
               LIMIT 50"""
        )
        xrefs = []
        for r in xref_rows:
            xrefs.append({
                "polymarket_id": r["polymarket_id"],
                "question": r["question"],
                "source_market_name": r["source_market_name"],
                "source_price": r["source_price"],
                "poly_price": r["poly_price"],
                "divergence": r["divergence"],
                "fetched_human": _fmt_age_ago(now - (r["fetched_at"] or now)),
            })
        return templates.TemplateResponse(
            request, "sources.html",
            _base_ctx(
                title="Sources",
                sources=sources_view,
                xrefs=xrefs,
                enrichment=enrichment,
            ),
        )

    # ---------- News JSON API ----------

    @app.get("/api/news/recent")
    async def api_news_recent(hours: float = 24.0, min_relevance: float = 0.3) -> dict:
        since = now_ts() - max(0.1, float(hours)) * 3600
        items = await news_store.recent_enriched(
            since_ts=since, min_relevance=float(min_relevance), limit=200,
        )
        return {"count": len(items), "items": items}

    @app.get("/api/news/sources")
    async def api_news_sources() -> dict:
        now = now_ts()
        rows = await news_store.source_stats_24h(now)
        by_name = {r["source"]: r for r in rows}
        out = []
        for spec in configured_feeds():
            r = by_name.get(spec.name) or {}
            last = r.get("last_fetch") or 0
            age = now - last if last else None
            healthy = age is not None and age < 1800
            out.append({
                "name": spec.name,
                "url": spec.url,
                "weight": spec.weight,
                "last_polled_ts": last or None,
                "age_seconds": age,
                "item_count_24h": int(r.get("total") or 0),
                "last_hour": int(r.get("last_hour") or 0),
                "ok": healthy,
            })
        return {"count": len(out), "feeds": out}

    @app.get("/api/news/stats")
    async def api_news_stats() -> dict:
        now = now_ts()
        return {
            "enrichment": await news_store.enrichment_summary_24h(now),
            "sources": await news_store.source_stats_24h(now),
        }

    # ---------- Logs ----------

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
            request, "logs.html",
            _base_ctx(title="Logs", rows=rows, tail=tail),
        )

    # ---------- Shadow / Real trading detail ----------

    @app.get("/shadow-trading", response_class=HTMLResponse)
    async def shadow_trading(request: Request, view: str = "") -> HTMLResponse:
        """Lane detail page. `?view=shadow` or `?view=real` picks the
        mode whose budgets + positions are shown. Default: the current
        active mode. Both are always computed so the user can compare."""
        active_mode = _current_mode()
        view_mode = view if view in MODE_CHOICES else active_mode
        lane_names = list(allocator.LANES)
        is_real_arg = 1 if view_mode == "real" else 0

        lanes = []
        total_realized = 0.0
        total_unrealized = 0.0
        for lane in lane_names:
            state = await allocator.get_state(lane, view_mode)
            metrics = await _lane_metrics_for_mode(lane, is_real_arg)
            paused = bool(state and state.is_paused)
            lanes.append({
                "lane": lane,
                "budget": state.total_budget if state else 0.0,
                "deployed": state.deployed if state else 0.0,
                "available": state.available if state else 0.0,
                "paused": paused,
                "paused_until": state.paused_until if state else 0.0,
                **metrics,
            })
            total_realized += metrics["realized_pnl"]
            total_unrealized += metrics["unrealized_pnl"]

        recs: list[str] = []
        for l in lanes:
            if l["budget"] > 0:
                roi = l["total_pnl"] / l["budget"] * 100.0
                if roi >= 2.0 and l["closed"] >= 10:
                    recs.append(f"{l['lane']}: +{roi:.1f}% — consider increasing budget")
                elif roi <= -2.0 and l["closed"] >= 10:
                    recs.append(f"{l['lane']}: {roi:.1f}% — underperforming, review")
        wr, wr_n = await _rolling_win_rate_for_mode(is_real_arg, 50)
        recent = await _recent_trades(limit=40, is_real=is_real_arg)

        # Summary for the OTHER mode so the dashboard can render the
        # compare-side-by-side strip.
        other = "real" if view_mode == "shadow" else "shadow"
        other_states = await allocator.all_states(other)
        other_summary = {
            "mode": other,
            "budget": sum(s.total_budget for s in other_states),
            "deployed": sum(s.deployed for s in other_states),
            "available": sum(s.available for s in other_states),
        }

        return templates.TemplateResponse(
            request, "shadow_trading.html",
            _base_ctx(
                title="Lanes",
                view_mode=view_mode,
                lanes=lanes,
                recent=recent,
                rolling_win_rate=wr,
                rolling_win_n=wr_n,
                total_realized=total_realized,
                total_unrealized=total_unrealized,
                recommendations=recs,
                other_summary=other_summary,
                now=now_ts(),
            ),
        )

    @app.post("/shadow-trading/pause")
    async def shadow_pause(
        lane: str = Form(""),
        action: str = Form("pause"),
        view: str = Form(""),
    ) -> RedirectResponse:
        target_mode = view if view in MODE_CHOICES else _current_mode()
        if lane == "all":
            if action == "pause":
                await allocator.pause_all(86400, "manual kill switch", mode=target_mode)
            else:
                for l in allocator.LANES:
                    await allocator.unpause(l, mode=target_mode)
        elif lane in allocator.LANES:
            if action == "pause":
                await allocator.pause(
                    lane, now_ts() + 86400, "manual kill switch",
                    mode=target_mode,
                )
            else:
                await allocator.unpause(lane, mode=target_mode)
        return RedirectResponse(
            f"/shadow-trading?view={target_mode}", status_code=303,
        )

    # ---------- Mode switch ----------

    @app.post("/mode")
    async def mode_switch(
        mode: str = Form(""),
        confirm: str = Form(""),
    ) -> RedirectResponse:
        target = mode.strip().lower()
        if target not in MODE_CHOICES:
            return RedirectResponse("/settings?saved=0&err=bad_mode", status_code=303)
        # Switching to real requires: (1) explicit confirm, (2) CLOB ready,
        # (3) real_capital.total_usd > 0.
        if target == "real":
            if confirm != "CONFIRM":
                return RedirectResponse(
                    "/settings?saved=0&err=real_needs_confirm", status_code=303,
                )
            if not clob_client.is_ready():
                return RedirectResponse(
                    "/settings?saved=0&err=clob_not_ready", status_code=303,
                )
            real_total = float((get_config().get("real_capital") or {}).get("total_usd") or 0)
            if real_total <= 0:
                return RedirectResponse(
                    "/settings?saved=0&err=real_budget_zero", status_code=303,
                )
        data = cfg.as_dict()
        data["mode"] = target
        cfg.save(data)
        cfg.reload()
        # Re-init lane_capital so any new real-mode rows appear with
        # fresh budgets pulled from the updated config.
        await allocator.init_lane_capital()
        return RedirectResponse(f"/settings?saved=1&mode={target}", status_code=303)

    @app.post("/budgets")
    async def budgets_save(
        shadow_total_usd: float = Form(0.0),
        real_total_usd: float = Form(0.0),
    ) -> RedirectResponse:
        """Save per-mode totals. Per-lane splits stay in the YAML for
        now (easier to version-control than a UI form)."""
        data = cfg.as_dict()
        sc = dict(data.get("shadow_capital") or {})
        rc = dict(data.get("real_capital") or {})
        sc["total_usd"] = max(0.0, float(shadow_total_usd))
        rc["total_usd"] = max(0.0, float(real_total_usd))
        data["shadow_capital"] = sc
        data["real_capital"] = rc
        cfg.save(data)
        cfg.reload()
        await allocator.init_lane_capital()
        return RedirectResponse("/settings?saved=1", status_code=303)

    @app.post("/reset-history")
    async def reset_history(confirm: str = Form("")) -> RedirectResponse:
        """Wipe SHADOW position history only.

        Real positions (``is_real=1``) live in the same ``shadow_positions``
        table and are left alone — the user is using this to start a fresh
        shadow test run while real trading may already be live. Every
        other table is preserved too: ``signals`` / ``price_ticks`` /
        ``ollama_stats`` / ``cross_references`` feed real-mode logic
        (stale-price detection, validator drift, concentration limits)
        and wiping them could bias the next real entry.

        Guarded by a literal ``RESET`` string in the form body — the UI
        sends it via a ``confirm()`` prompt, which also stops an errant
        double-click on the button from firing the DELETE.
        """
        if confirm.strip().upper() != "RESET":
            return RedirectResponse(
                "/settings?saved=0&err=reset_not_confirmed", status_code=303,
            )
        from core.utils.db import execute as db_execute
        # COALESCE(is_real, 0)=0 catches both explicit 0 rows and any
        # legacy rows predating the column where is_real may be NULL.
        await db_execute(
            "DELETE FROM shadow_positions WHERE COALESCE(is_real, 0)=0"
        )
        # Recompute lane_capital.deployed from the (now empty of shadow)
        # positions table. init_lane_capital's _deployed_for filters by
        # is_real per mode, so shadow lanes drop to deployed=0 while
        # real lanes keep their sum over the untouched is_real=1 rows.
        await allocator.init_lane_capital()
        return RedirectResponse("/settings?saved=1&mode=reset", status_code=303)

    # ---------- Settings ----------

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(
        request: Request,
        saved: int = 0,
        err: str = "",
        mode: str = "",
    ) -> HTMLResponse:
        shadow_cap = get_config().get("shadow_capital") or {}
        real_cap = get_config().get("real_capital") or {}
        return templates.TemplateResponse(
            request, "settings.html",
            _base_ctx(
                title="Settings",
                refresh=0,
                fields=fields_for_dashboard(),
                ollama_status=await _ollama_status(),
                saved=saved,
                error=err,
                just_switched_to=mode,
                shadow_capital=shadow_cap,
                real_capital=real_cap,
            ),
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
                continue
            if raw == "__clear__":
                raw = ""
            updates[key] = raw
        update_env(updates)
        return RedirectResponse(f"/settings?saved={len(updates)}", status_code=303)

    # ---------- Models ----------

    @app.get("/models", response_class=HTMLResponse)
    async def models(request: Request) -> HTMLResponse:
        summary = await _ollama_stats_summary()
        return templates.TemplateResponse(
            request, "models.html",
            _base_ctx(title="Models", summary=summary),
        )

    # ---------- APIs ----------

    @app.get("/api/health")
    async def api_health():
        from core.state.health import run_all
        reports = await run_all()
        return [{"component": r.component, "ok": r.ok, "detail": r.detail} for r in reports]

    @app.get("/api/equity")
    async def api_equity():
        return {"points": await equity_curve_points(), "now": now_ts()}

    @app.get("/api/ollama-stats")
    async def api_ollama_stats():
        return await _ollama_stats_summary()

    @app.get("/api/mode")
    async def api_mode():
        return {"mode": _current_mode(), "clob_ready": clob_client.is_ready()}

    return app


# ---- helpers ----


async def _lane_metrics_for_mode(strategy: str, is_real: int) -> dict:
    """Port of shadow_engine.lane_metrics but filtered to one mode."""
    rows = await fetch_all(
        """SELECT * FROM shadow_positions
           WHERE strategy=? AND COALESCE(is_real, 0)=?""",
        (strategy, is_real),
    )
    closed = [r for r in rows if r["status"] == "CLOSED"]
    open_rows = [r for r in rows if r["status"] in ("OPEN", "PENDING_FILL")]
    def _f(v) -> float:
        try:
            return float(v or 0.0)
        except (TypeError, ValueError):
            return 0.0
    wins = [r for r in closed if _f(r["realized_pnl_usd"]) > 0]
    losses = [r for r in closed if _f(r["realized_pnl_usd"]) < 0]
    realized = sum(_f(r["realized_pnl_usd"]) for r in closed)
    unrealized = sum(_f(r["unrealized_pnl_usd"]) for r in open_rows)
    hold_times = [
        _f(r["close_ts"]) - _f(r["entry_ts"])
        for r in closed if r["close_ts"] and r["entry_ts"]
    ]
    pnls = [_f(r["realized_pnl_usd"]) for r in closed]
    if len(pnls) >= 2:
        mean = sum(pnls) / len(pnls)
        variance = sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1)
        stdev = variance ** 0.5
        sharpe_like = mean / stdev if stdev > 0 else 0.0
    else:
        sharpe_like = 0.0
    avg_win = (sum(_f(r["realized_pnl_usd"]) for r in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(_f(r["realized_pnl_usd"]) for r in losses) / len(losses)) if losses else 0.0
    return {
        "strategy": strategy,
        "open": len(open_rows),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": realized + unrealized,
        "avg_hold_seconds": (sum(hold_times) / len(hold_times)) if hold_times else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe_like": sharpe_like,
    }


async def _rolling_win_rate_for_mode(is_real: int, limit: int) -> tuple[float, int]:
    rows = await fetch_all(
        """SELECT realized_pnl_usd FROM shadow_positions
           WHERE status='CLOSED' AND COALESCE(is_real, 0)=?
           ORDER BY close_ts DESC LIMIT ?""",
        (is_real, limit),
    )
    if not rows:
        return 0.0, 0
    def _f(v) -> float:
        try:
            return float(v or 0.0)
        except (TypeError, ValueError):
            return 0.0
    wins = sum(1 for r in rows if _f(r["realized_pnl_usd"]) > 0)
    return wins / len(rows), len(rows)
