"""Position bookkeeping. Open / close / netting from execution events."""

from __future__ import annotations

from dataclasses import dataclass

from core.utils.db import execute, fetch_all, fetch_one
from core.utils.helpers import now_ts, safe_float


@dataclass
class Position:
    id: int
    market_id: str
    token_id: str
    side: str
    size: float
    avg_price: float
    opened_at: float
    realized_pnl_usd: float
    status: str


async def open_position_for(market_id: str) -> int:
    row = await fetch_one(
        "SELECT COUNT(*) AS n FROM positions WHERE market_id=? AND status='OPEN'",
        (market_id,),
    )
    return int(row["n"] if row else 0)


async def upsert_from_fill(
    market_id: str,
    token_id: str,
    side: str,
    fill_price: float,
    fill_size: float,
) -> None:
    """Average up/down or close out position based on a fill."""
    row = await fetch_one(
        """SELECT * FROM positions
           WHERE market_id=? AND token_id=? AND status='OPEN'""",
        (market_id, token_id),
    )
    ts = now_ts()
    if row is None:
        await execute(
            """INSERT INTO positions
               (market_id, token_id, side, size, avg_price, opened_at,
                closed_at, realized_pnl_usd, status)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (market_id, token_id, side, fill_size, fill_price, ts, None, 0.0, "OPEN"),
        )
        return
    cur_side = row["side"]
    cur_size = safe_float(row["size"])
    cur_avg = safe_float(row["avg_price"])
    realized = safe_float(row["realized_pnl_usd"])
    if cur_side == side:
        new_size = cur_size + fill_size
        new_avg = ((cur_avg * cur_size) + (fill_price * fill_size)) / new_size if new_size else fill_price
        await execute(
            "UPDATE positions SET size=?, avg_price=? WHERE id=?",
            (new_size, new_avg, row["id"]),
        )
    else:
        # opposite side reduces or flips the position
        if fill_size < cur_size:
            pnl = (fill_price - cur_avg) * fill_size * (1 if cur_side == "BUY" else -1)
            await execute(
                "UPDATE positions SET size=?, realized_pnl_usd=? WHERE id=?",
                (cur_size - fill_size, realized + pnl, row["id"]),
            )
        elif abs(fill_size - cur_size) < 1e-9:
            pnl = (fill_price - cur_avg) * cur_size * (1 if cur_side == "BUY" else -1)
            await execute(
                """UPDATE positions SET size=0, closed_at=?, realized_pnl_usd=?,
                   status='CLOSED' WHERE id=?""",
                (ts, realized + pnl, row["id"]),
            )
        else:
            # close out and flip
            pnl = (fill_price - cur_avg) * cur_size * (1 if cur_side == "BUY" else -1)
            await execute(
                """UPDATE positions SET size=0, closed_at=?, realized_pnl_usd=?,
                   status='CLOSED' WHERE id=?""",
                (ts, realized + pnl, row["id"]),
            )
            remaining = fill_size - cur_size
            await execute(
                """INSERT INTO positions
                   (market_id, token_id, side, size, avg_price, opened_at,
                    closed_at, realized_pnl_usd, status)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (market_id, token_id, side, remaining, fill_price, ts, None, 0.0, "OPEN"),
            )


async def list_open() -> list[Position]:
    rows = await fetch_all(
        "SELECT * FROM positions WHERE status='OPEN' ORDER BY opened_at DESC"
    )
    return [
        Position(
            id=r["id"],
            market_id=r["market_id"],
            token_id=r["token_id"],
            side=r["side"],
            size=safe_float(r["size"]),
            avg_price=safe_float(r["avg_price"]),
            opened_at=safe_float(r["opened_at"]),
            realized_pnl_usd=safe_float(r["realized_pnl_usd"]),
            status=r["status"],
        )
        for r in rows
    ]
