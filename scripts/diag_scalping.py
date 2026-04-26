"""One-shot diagnostic: why is the scalping lane idle?

Walks each scalping entry gate against the live `markets` cache and
reports how many active markets survive each step. Also dumps
recent signal-pipeline outcomes and open shadow positions by lane.

Safe read-only — no writes, no API calls. Run with:
    python -m scripts.diag_scalping
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime


def _parse_close(s: str | None) -> float | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        try:
            return float(s)
        except Exception:
            return None


def main() -> None:
    con = sqlite3.connect("polybot.db")
    con.row_factory = sqlite3.Row
    c = con.cursor()
    now = time.time()

    c.execute("SELECT COUNT(*) AS n FROM markets WHERE active=1")
    total_active = c.fetchone()["n"]
    print(f"active markets total: {total_active}")

    c.execute(
        """SELECT market_id, question, best_bid, best_ask, close_time, liquidity
           FROM markets WHERE active=1 AND close_time IS NOT NULL"""
    )
    rows = c.fetchall()

    date_pass = []
    for r in rows:
        ct = _parse_close(r["close_time"])
        if ct is None:
            continue
        days = (ct - now) / 86400
        if 2 <= days <= 14:
            date_pass.append((r, days))
    print(f"markets in scalping date window (2-14d): {len(date_pass)}")

    spread_pass = []
    for r, days in date_pass:
        bid = r["best_bid"] or 0
        ask = r["best_ask"] or 0
        spread_c = (ask - bid) * 100
        if 0 < spread_c <= 3:
            spread_pass.append((r, days, spread_c))
    print(f"  + spread <=3c: {len(spread_pass)}")

    # Evidence check (last 7d feed_items tagging the market).
    mids = [r["market_id"] for r, *_ in spread_pass]
    per_market_sources: dict[str, set[str]] = {}
    if mids:
        c.execute(
            "SELECT meta, source FROM feed_items WHERE ingested_at > ? "
            "AND meta IS NOT NULL AND meta LIKE '%linked_market_id%'",
            (now - 7 * 86400,),
        )
        for h in c.fetchall():
            try:
                meta = json.loads(h["meta"])
            except Exception:
                continue
            lid = meta.get("linked_market_id")
            if lid in mids:
                per_market_sources.setdefault(lid, set()).add(h["source"] or "?")

    evidence_pass = []
    for r, days, spread_c in spread_pass:
        srcs = per_market_sources.get(r["market_id"], set())
        if len(srcs) >= 2:
            evidence_pass.append((r, days, spread_c, srcs))
    print(
        f"  + >=2 distinct evidence sources (last 7d): {len(evidence_pass)}"
    )

    print("\nfirst 10 candidates passing ALL scalping pre-Ollama gates:")
    for r, days, spread_c, srcs in evidence_pass[:10]:
        q = (r["question"] or "")[:60]
        print(
            f"  {r['market_id']} "
            f"days={days:.1f} spread={spread_c:.1f}c "
            f"srcs={len(srcs)} :: {q}"
        )

    # Recent signal pipeline breakdown (last 24h)
    c.execute(
        "SELECT status, COUNT(*) AS n FROM signals WHERE created_at > ? "
        "GROUP BY status",
        (now - 24 * 3600,),
    )
    print("\nsignals pipeline — last 24h status counts:")
    for r in c.fetchall():
        print(f"  {r['status']:>20}: {r['n']}")

    # Shadow positions snapshot
    c.execute(
        "SELECT strategy, status, COUNT(*) AS n, SUM(size_usd) AS sz "
        "FROM shadow_positions GROUP BY strategy, status "
        "ORDER BY strategy, status"
    )
    print("\nshadow positions by strategy + status:")
    for r in c.fetchall():
        sz = r["sz"] or 0
        print(
            f"  {r['strategy'] or '?':>16} {r['status'] or '?':>8} "
            f"n={r['n']:>3} total_size=${sz:.2f}"
        )


if __name__ == "__main__":
    main()
