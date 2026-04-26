"""Build the ``signal_outcomes`` table.

For each candidate row in ``event_market_candidates``, ``scan_skips``,
and ``shadow_positions``, extract the contributing source list from
the row's snapshot JSON, fan out one ``signal_outcomes`` row per
(candidate, source) pair, and JOIN against ``price_ticks`` to compute
price-after-N and max-favorable/adverse moves.

The bot never reads this table. It exists for the CLI's offline
analysis. Rebuilt idempotently — TRUNCATE + INSERT, NOT incremental.
That keeps the code simple and the rebuild safe to re-run any time
(e.g. after a config tweak that changes how `category` is derived).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

from core.utils.db import connect, fetch_all
from core.utils.helpers import now_ts, safe_float


# Time horizons we compute "price after N" for. Tuned to capture
# scout-relevant reaction windows: the signal might lead by 1-15 min,
# longer windows confirm whether the move stuck.
_PRICE_AFTER_HORIZONS_SECONDS = (60.0, 300.0, 900.0, 3600.0)
# 15-minute window for max-favorable / max-adverse computation.
_MAX_MOVE_WINDOW_SECONDS = 900.0
# Below this absolute mid-move (in YES probability units), we say
# the market "didn't move". 0.005 = 0.5 cent, ~ noise floor on
# liquid Polymarket books.
_NOISE_FLOOR_PROB = 0.005


@dataclass(frozen=True)
class _OutcomeRow:
    """One row to be INSERTed into signal_outcomes. Frozen so the
    builder pipeline can't accidentally mutate after construction."""
    source_table: str
    source_row_id: int
    detected_at: float
    source: str
    category: str
    market_id: str
    token_id: str
    snapshot_price: float | None
    snapshot_taken_at: float | None
    price_after_1m: float | None
    price_after_5m: float | None
    price_after_15m: float | None
    price_after_1h: float | None
    max_favorable_move_15m: float | None
    max_adverse_move_15m: float | None
    whether_market_moved: int | None
    direction_correct: int | None
    estimated_edge_captured: float | None
    estimated_edge_missed: float | None
    side: str
    final_status: str
    reject_reason: str
    polarity_source: str


# ============================================================
# Public entry point
# ============================================================


async def rebuild(*, since_seconds: float | None = None) -> int:
    """TRUNCATE + REBUILD ``signal_outcomes``. Returns the number of
    rows inserted. ``since_seconds`` filters source rows by
    detected_at; pass None to rebuild everything.

    No-throw: a partial failure on one source table still inserts the
    rows we successfully built. The CLI report handles missing
    horizons gracefully.
    """
    rebuilt_at = now_ts()
    cutoff = (rebuilt_at - since_seconds) if since_seconds else 0.0

    rows: list[_OutcomeRow] = []
    try:
        rows.extend(await _from_event_market_candidates(cutoff))
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("[analytics] event_market_candidates extract failed: {}", e)
    try:
        rows.extend(await _from_scan_skips(cutoff))
    except Exception as e:  # pragma: no cover
        logger.warning("[analytics] scan_skips extract failed: {}", e)
    try:
        rows.extend(await _from_shadow_positions(cutoff))
    except Exception as e:  # pragma: no cover
        logger.warning("[analytics] shadow_positions extract failed: {}", e)

    # Cache price_ticks lookups by market_id so a candidate with N
    # source-fan-out rows only queries the tick stream once. The
    # cache lives only for the duration of this rebuild — the source
    # tables don't change during it.
    #
    # _OutcomeRow is frozen, so _populate_price_outcomes returns a
    # NEW row with the price columns populated. Replace in-place so
    # the rebuild sees the populated rows.
    price_cache: dict[str, list[dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        rows[i] = await _populate_price_outcomes(r, price_cache)

    inserted = await _persist_rebuild(rows, rebuilt_at=rebuilt_at)
    logger.info(
        "[analytics] signal_outcomes rebuilt: {} rows from "
        "{} candidate sources",
        inserted, 3,
    )
    return inserted


# ============================================================
# Source extractors
# ============================================================


async def _from_event_market_candidates(cutoff_ts: float) -> list[_OutcomeRow]:
    """Pull every event_market_candidates row + fan out per source."""
    sql = (
        "SELECT id, event_id, market_id, considered_at, status, "
        "       reject_reason, side, true_prob, confidence, edge, "
        "       market_mid, impact_snapshot, shadow_position_id "
        "FROM event_market_candidates "
        "WHERE considered_at >= ? "
        "ORDER BY considered_at"
    )
    rows = await fetch_all(sql, (cutoff_ts,))
    out: list[_OutcomeRow] = []
    for r in rows:
        snap = _safe_json(r["impact_snapshot"])
        event_block = (snap.get("event") or {}) if isinstance(snap, dict) else {}
        market_block = (snap.get("market") or {}) if isinstance(snap, dict) else {}
        impact_block = (snap.get("impact") or {}) if isinstance(snap, dict) else {}
        sources = _coerce_source_list(event_block.get("sources"))
        category = str(event_block.get("category") or "unknown")
        # v3 fields preferred; fall back to mid / considered_at on main.
        snapshot_price = _safe_float_or_none(
            market_block.get("snapshot_price")
            if "snapshot_price" in market_block else market_block.get("mid")
        )
        snapshot_taken_at = _safe_float_or_none(
            market_block.get("snapshot_taken_at")
            if "snapshot_taken_at" in market_block else r["considered_at"]
        )
        token_id = str(market_block.get("yes_token") or "")
        polarity_source = str(impact_block.get("polarity_source") or "rules")
        detected_at = _safe_float_or_none(
            event_block.get("first_seen_timestamp")
            or event_block.get("timestamp_detected")
            or r["considered_at"]
        ) or 0.0
        side = str(r["side"] or "")
        status = str(r["status"] or "rejected")
        reject_reason = str(r["reject_reason"] or "")
        # Single-source events still produce one row.
        if not sources:
            sources = ["unknown"]
        for src in sources:
            out.append(_OutcomeRow(
                source_table="event_market_candidates",
                source_row_id=int(r["id"]),
                detected_at=detected_at,
                source=src,
                category=category,
                market_id=str(r["market_id"]),
                token_id=token_id,
                snapshot_price=snapshot_price,
                snapshot_taken_at=snapshot_taken_at,
                price_after_1m=None, price_after_5m=None,
                price_after_15m=None, price_after_1h=None,
                max_favorable_move_15m=None,
                max_adverse_move_15m=None,
                whether_market_moved=None,
                direction_correct=None,
                estimated_edge_captured=None,
                estimated_edge_missed=None,
                side=side,
                final_status=status,
                reject_reason=reject_reason,
                polarity_source=polarity_source,
            ))
    return out


async def _from_scan_skips(cutoff_ts: float) -> list[_OutcomeRow]:
    """Pull scan_skips rows. These are scalping-lane skips with their
    own evidence-source list embedded in score_snapshot."""
    sql = (
        "SELECT id, scan_ts, lane, market_id, tier_attempted, "
        "       reject_reason, evidence_tier, watchlist, score_snapshot "
        "FROM scan_skips "
        "WHERE scan_ts >= ? "
        "ORDER BY scan_ts"
    )
    rows = await fetch_all(sql, (cutoff_ts,))
    out: list[_OutcomeRow] = []
    for r in rows:
        snap = _safe_json(r["score_snapshot"])
        market_block = (snap.get("market") or {}) if isinstance(snap, dict) else {}
        evidence_block = (snap.get("evidence") or {}) if isinstance(snap, dict) else {}
        sources = _coerce_source_list(evidence_block.get("sources"))
        category = str(r["lane"] or "scalping")
        snapshot_price = _safe_float_or_none(market_block.get("mid"))
        token_id = str(market_block.get("yes_token") or "")
        scan_ts = float(r["scan_ts"])
        if not sources:
            sources = ["unknown"]
        # scan_skips never has a `side` (the lane skipped before
        # opening). The "final_status" is always rejected here —
        # scan_skips IS the reject log for scalping.
        for src in sources:
            out.append(_OutcomeRow(
                source_table="scan_skips",
                source_row_id=int(r["id"]),
                detected_at=scan_ts,
                source=src,
                category=category,
                market_id=str(r["market_id"]),
                token_id=token_id,
                snapshot_price=snapshot_price,
                snapshot_taken_at=scan_ts,
                price_after_1m=None, price_after_5m=None,
                price_after_15m=None, price_after_1h=None,
                max_favorable_move_15m=None,
                max_adverse_move_15m=None,
                whether_market_moved=None,
                direction_correct=None,
                estimated_edge_captured=None,
                estimated_edge_missed=None,
                side="",
                final_status="rejected",
                reject_reason=str(r["reject_reason"] or ""),
                polarity_source="rules",
            ))
    return out


async def _from_shadow_positions(cutoff_ts: float) -> list[_OutcomeRow]:
    """Shadow positions (accepted candidates that became live SHADOW
    bets). The `evidence_snapshot` JSON carries source attribution
    where available."""
    sql = (
        "SELECT id, strategy, market_id, token_id, side, entry_price, "
        "       entry_ts, true_prob_entry, confidence_entry, "
        "       close_price, close_ts, close_reason, "
        "       realized_pnl_usd, evidence_snapshot, status "
        "FROM shadow_positions "
        "WHERE entry_ts >= ? "
        "ORDER BY entry_ts"
    )
    rows = await fetch_all(sql, (cutoff_ts,))
    out: list[_OutcomeRow] = []
    for r in rows:
        snap = _safe_json(r["evidence_snapshot"])
        sources = _coerce_source_list(
            (snap.get("event_sources") if isinstance(snap, dict) else None)
            or (snap.get("sources") if isinstance(snap, dict) else None)
        )
        category = str(
            (snap.get("event_category") if isinstance(snap, dict) else None)
            or r["strategy"] or "unknown"
        )
        entry_ts = float(r["entry_ts"])
        if not sources:
            sources = ["unknown"]
        # Accepted candidates produce a position; outcome math uses
        # the entry price as the snapshot.
        for src in sources:
            out.append(_OutcomeRow(
                source_table="shadow_positions",
                source_row_id=int(r["id"]),
                detected_at=entry_ts,
                source=src,
                category=category,
                market_id=str(r["market_id"]),
                token_id=str(r["token_id"] or ""),
                snapshot_price=_safe_float_or_none(r["entry_price"]),
                snapshot_taken_at=entry_ts,
                price_after_1m=None, price_after_5m=None,
                price_after_15m=None, price_after_1h=None,
                max_favorable_move_15m=None,
                max_adverse_move_15m=None,
                whether_market_moved=None,
                direction_correct=None,
                # Realized PnL (USD) → percent of entry size, used as
                # estimated_edge_captured. Approximate; fine for v1.
                estimated_edge_captured=_safe_float_or_none(r["realized_pnl_usd"]),
                estimated_edge_missed=None,
                side=str(r["side"] or ""),
                final_status="accepted",
                reject_reason="",
                polarity_source="rules",
            ))
    return out


# ============================================================
# Price-after-N computation (price_ticks JOIN)
# ============================================================


async def _populate_price_outcomes(
    row: _OutcomeRow,
    cache: dict[str, list[dict[str, Any]]],
) -> _OutcomeRow:
    """Mutate (well, replace) ``row`` with computed price-after-N
    fields. The ticks for ``market_id`` are fetched once and cached
    for the whole rebuild.

    No-throw: if no ticks are available for the relevant window, all
    outcome fields stay None. Aggregations downstream skip those.
    """
    if row.snapshot_taken_at is None or not row.market_id:
        return row
    ticks = await _ticks_for_market(row.market_id, cache)
    if not ticks:
        return row
    base_ts = row.snapshot_taken_at
    base_price = row.snapshot_price

    # price_after_N
    horizons = _PRICE_AFTER_HORIZONS_SECONDS  # (60, 300, 900, 3600)
    after_prices = [
        _tick_price_at(ticks, base_ts + h, prefer="ge") for h in horizons
    ]

    # max favorable / adverse over 15 min (signed in YES-prob units;
    # positive == price went UP).
    max_fav, max_adv = _max_moves_over_window(
        ticks, base_ts, _MAX_MOVE_WINDOW_SECONDS, base_price,
    )
    moved = (
        None if base_price is None or max_fav is None or max_adv is None
        else (1 if max(abs(max_fav), abs(max_adv)) >= _NOISE_FLOOR_PROB else 0)
    )

    # direction_correct: only meaningful if we have a side AND the
    # market actually moved. BUY = correct iff price_after_5m > base.
    direction_correct: int | None = None
    if row.side in ("BUY", "SELL") and moved == 1 and after_prices[1] is not None and base_price is not None:
        delta_5m = after_prices[1] - base_price
        if row.side == "BUY":
            direction_correct = 1 if delta_5m > 0 else 0
        else:
            direction_correct = 1 if delta_5m < 0 else 0

    # estimated_edge_missed: for non-accepted rows, the magnitude
    # of the favorable move (assuming a hypothetical correct side).
    edge_missed = row.estimated_edge_missed
    if row.final_status != "accepted" and moved == 1 and base_price is not None:
        # Use whichever direction had the larger move — that's the
        # bound on the edge we COULD have captured if we'd traded.
        edge_missed = max(abs(max_fav or 0.0), abs(max_adv or 0.0))

    return _OutcomeRow(
        source_table=row.source_table,
        source_row_id=row.source_row_id,
        detected_at=row.detected_at,
        source=row.source,
        category=row.category,
        market_id=row.market_id,
        token_id=row.token_id,
        snapshot_price=row.snapshot_price,
        snapshot_taken_at=row.snapshot_taken_at,
        price_after_1m=after_prices[0],
        price_after_5m=after_prices[1],
        price_after_15m=after_prices[2],
        price_after_1h=after_prices[3],
        max_favorable_move_15m=max_fav,
        max_adverse_move_15m=max_adv,
        whether_market_moved=moved,
        direction_correct=direction_correct,
        estimated_edge_captured=row.estimated_edge_captured,
        estimated_edge_missed=edge_missed,
        side=row.side,
        final_status=row.final_status,
        reject_reason=row.reject_reason,
        polarity_source=row.polarity_source,
    )


async def _ticks_for_market(
    market_id: str, cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return ALL ticks for this market, ordered ascending by ts.
    Cached for the duration of one rebuild.

    For markets with many ticks (a hot scalping target may have
    thousands), this still fits in memory comfortably. If a future
    rebuild needs to scale, swap to a per-(market, base_ts) range
    query — the `idx_price_market_ts` index supports it directly.
    """
    if market_id in cache:
        return cache[market_id]
    rows = await fetch_all(
        "SELECT bid, ask, last, ts FROM price_ticks "
        "WHERE market_id=? ORDER BY ts ASC",
        (market_id,),
    )
    ticks = [
        {
            "bid": safe_float(r["bid"]),
            "ask": safe_float(r["ask"]),
            "last": safe_float(r["last"]),
            "ts": safe_float(r["ts"]),
            "mid": _mid_from_tick(r),
        }
        for r in rows
    ]
    cache[market_id] = ticks
    return ticks


def _mid_from_tick(row: Any) -> float:
    bid = safe_float(row["bid"])
    ask = safe_float(row["ask"])
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return safe_float(row["last"])


def _tick_price_at(
    ticks: list[dict[str, Any]],
    target_ts: float,
    *,
    prefer: str = "ge",
) -> float | None:
    """Return the mid price at the FIRST tick with ``ts >= target_ts``
    (when prefer="ge") or the LAST tick with ``ts <= target_ts`` (when
    prefer="le"). None if no tick satisfies.

    Linear scan — ticks are pre-sorted by ts. Acceptable for v1; if
    the rebuild ever grows to seconds, swap to bisect.
    """
    if not ticks:
        return None
    if prefer == "ge":
        for t in ticks:
            if t["ts"] >= target_ts and t["mid"] > 0:
                return t["mid"]
        return None
    # le
    found: float | None = None
    for t in ticks:
        if t["ts"] > target_ts:
            break
        if t["mid"] > 0:
            found = t["mid"]
    return found


def _max_moves_over_window(
    ticks: list[dict[str, Any]],
    base_ts: float,
    window_seconds: float,
    base_price: float | None,
) -> tuple[float | None, float | None]:
    """Return (max_favorable, max_adverse) where:
       max_favorable = max(mid - base) over [base_ts, base_ts+window]
       max_adverse   = min(mid - base) over the same window
    Both signed: positive means price went UP.

    Returns (None, None) if no ticks in the window OR base_price is
    None.
    """
    if base_price is None:
        return None, None
    end_ts = base_ts + window_seconds
    deltas: list[float] = []
    for t in ticks:
        if t["ts"] < base_ts:
            continue
        if t["ts"] > end_ts:
            break
        if t["mid"] > 0:
            deltas.append(t["mid"] - base_price)
    if not deltas:
        return None, None
    return max(deltas), min(deltas)


# ============================================================
# Rebuild persistence
# ============================================================


async def _persist_rebuild(
    rows: Iterable[_OutcomeRow], *, rebuilt_at: float,
) -> int:
    """TRUNCATE + INSERT in a single transaction. Returns rowcount."""
    rows_list = list(rows)
    async with connect() as conn:
        await conn.execute("DELETE FROM signal_outcomes")
        cur = await conn.executemany(
            """INSERT INTO signal_outcomes
               (rebuilt_at, source_table, source_row_id, detected_at,
                source, category, market_id, token_id, snapshot_price,
                snapshot_taken_at, price_after_1m, price_after_5m,
                price_after_15m, price_after_1h, max_favorable_move_15m,
                max_adverse_move_15m, whether_market_moved,
                direction_correct, estimated_edge_captured,
                estimated_edge_missed, side, final_status, reject_reason,
                polarity_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    rebuilt_at, r.source_table, r.source_row_id,
                    r.detected_at, r.source, r.category, r.market_id,
                    r.token_id, r.snapshot_price, r.snapshot_taken_at,
                    r.price_after_1m, r.price_after_5m,
                    r.price_after_15m, r.price_after_1h,
                    r.max_favorable_move_15m, r.max_adverse_move_15m,
                    r.whether_market_moved, r.direction_correct,
                    r.estimated_edge_captured, r.estimated_edge_missed,
                    r.side, r.final_status, r.reject_reason,
                    r.polarity_source,
                )
                for r in rows_list
            ],
        )
        await conn.commit()
        return len(rows_list)


# ============================================================
# Helpers
# ============================================================


def _safe_json(raw: Any) -> Any:
    if not raw:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_source_list(value: Any) -> list[str]:
    """Normalise the various source-list shapes we've seen into a
    sorted distinct list of source name strings."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return [value]
    if not isinstance(value, list):
        return []
    out: set[str] = set()
    for v in value:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.add(s)
    return sorted(out)
