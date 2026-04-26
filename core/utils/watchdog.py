"""Event-loop health watchdog + degraded-state signal.

Fires once every 30 seconds and reports three load indicators:

  - Number of pending asyncio tasks. Spiking task count is an early
    signal that something is blocking the loop.
  - Seconds since the last successful Ollama /api/generate response.
  - Seconds since the last Polymarket WebSocket message.

Escalation rules:
  - Ollama silent for >60s AND pending_tasks >30 -> clear the shared
    Ollama httpx pool (dump any stuck half-open sockets).
  - Ollama silent for >120s -> log ERROR (actionable — the tiered
    models should respond well inside this window even under load).
  - poly_ws silent for >120s -> call PolymarketWS.request_reconnect()
    to force the outer loop to re-establish the socket.
  - After 3 consecutive force-reconnects that don't recover, log ERROR
    and sit out the next 120s so we don't hammer a dead endpoint.

The watchdog also publishes a module-level ``DEGRADED`` flag that feed
coroutines read to stretch their poll intervals (see
core/feeds/manager.py). ``DEGRADED`` flips true on any WARN/ERROR and
flips back to false after ``HEALTHY_CONFIRM_TICKS`` clean ticks.

The watchdog is a strict observer — it does not touch lane budgets or
open positions. Failures here must never break trading.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from core.feeds import polymarket_ws as poly_ws_mod
from core.signals import ollama_client as ollama_mod
from core.utils.helpers import now_ts

INTERVAL_SECONDS = 30.0
OLLAMA_SILENT_ERROR_SECONDS = 120.0
OLLAMA_CONTENTION_SECONDS = 60.0
CONTENTION_TASK_THRESHOLD = 30
WS_SILENT_RECONNECT_SECONDS = 120.0
WS_SILENT_HARD_BACKOFF_SECONDS = 180.0
WS_RECONNECT_STRIKES_BEFORE_BACKOFF = 3
WS_HARD_BACKOFF_SECONDS = 120.0
# Number of consecutive healthy ticks before we flip DEGRADED back off.
# ~3 minutes at 30s ticks — matches the user spec "3min straight".
HEALTHY_CONFIRM_TICKS = 6

# Threshold for emitting the verbose "task dump" diagnostic: triggered
# when ollama is silent longer than this OR pending tasks exceed the
# warn-level task count (CONTENTION_TASK_THRESHOLD / 1.5). The cheap
# periodic tick line is always emitted; the task dump is only printed
# when something actually looks off, so log volume stays bounded.
DIAGNOSTIC_OLLAMA_SILENT_SECONDS = 30.0
DIAGNOSTIC_TASK_DUMP_LIMIT = 8
# Tasks whose creation time is older than this many seconds are treated
# as "long pending" in the dump (they're the ones worth looking at).
LONG_PENDING_AGE_SECONDS = 30.0

# Substring markers used to attribute suspected-cause. Matched against
# ``task.get_name()`` + the repr of ``task.get_coro()``. Order matters:
# the first rule that matches wins, so put the most specific at the top.
# Keep names lowercase — attribution does a case-insensitive substring
# test. The goal is NOT forensic-grade taxonomy; it's a single-word hint
# the human operator can use to know where to look first.
_CAUSE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("clob_build", ("clob_client", "py_clob_client", "create_or_derive", "ensure_ready")),
    ("ollama_queue", ("_generate", "ollama", "/api/generate", "fast_score", "deep_score", "validate")),
    ("db_locked", ("aiosqlite", "fetch_all", "fetch_one", "execute")),
    ("ws_reconnect", ("polymarket_ws", "poly_ws", "reconnect")),
    ("discovery", ("discovery", "refresh_once")),
    ("lane_entry", ("scalping", "event", "microscalp", "longshot", "resolution_day")),
)

# Module-level status flag read by FeedManager / feeds to throttle their
# poll loops when the event loop is under pressure.
DEGRADED: bool = False


def is_degraded() -> bool:
    return DEGRADED


class Watchdog:
    component = "utils.watchdog"

    def __init__(self, poly_ws: Any | None = None) -> None:
        self._stop = asyncio.Event()
        # Optional handle to the PolymarketWS instance so we can force a
        # reconnect when the channel goes silent. Passed in from run.py
        # (FeedManager.poly_ws).
        self._poly_ws = poly_ws
        # Record the watchdog's own start time so the first few ticks
        # don't falsely report "ollama silent" before any real call has
        # had a chance to run.
        self._started_at = now_ts()
        # WS force-reconnect tracking so we back off after repeated
        # strikes against a dead endpoint.
        self._ws_reconnect_strikes = 0
        self._ws_hard_backoff_until = 0.0
        # Count of consecutive healthy ticks; flips DEGRADED back off.
        self._healthy_ticks = 0

    async def run(self) -> None:
        logger.info("[watchdog] started (interval={:.0f}s)", INTERVAL_SECONDS)
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception as e:
                # A bug in the watchdog must not crash it — otherwise we
                # lose visibility into loop stalls.
                logger.exception("[watchdog] tick error: {}", e)
            await self._sleep(INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._stop.set()

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _task_signature(self, task: asyncio.Task) -> str:
        """Best-effort printable identifier for a task. Uses the task's
        ``name`` (set by ``asyncio.create_task(name=...)`` in run.py) and
        the coroutine's qualified name, both for matching + display.
        """
        name = task.get_name() or ""
        coro = task.get_coro()
        qualname = getattr(coro, "__qualname__", None) or getattr(
            coro, "__name__", "<anon>",
        )
        # repr(coro) reveals the call frame (frame filename:line) which
        # is the most useful thing for operator triage when a coro is
        # stuck inside a specific function.
        return f"{name}|{qualname}|{coro!r}"

    def _suspected_cause(self, long_pending: list[asyncio.Task]) -> str:
        """Attribute the loop stall to a single-word cause based on which
        module dominates the long-pending task list. ``unknown`` is the
        safe default when nothing matches — better to say "unknown"
        than mislead the operator on triage.
        """
        if not long_pending:
            return "event_loop_blocked"
        hits: dict[str, int] = {}
        for t in long_pending:
            sig = self._task_signature(t).lower()
            for cause, markers in _CAUSE_RULES:
                if any(mk in sig for mk in markers):
                    hits[cause] = hits.get(cause, 0) + 1
                    break
        if not hits:
            return "unknown"
        # Tie-break: most common, then _CAUSE_RULES declaration order
        # (which is roughly severity — CLOB build is the worst, lane
        # entries are the most expected/benign).
        declared_order = {cause: i for i, (cause, _) in enumerate(_CAUSE_RULES)}
        return min(hits.items(), key=lambda kv: (-kv[1], declared_order.get(kv[0], 99)))[0]

    def _emit_diagnostic_dump(
        self,
        all_tasks: list[asyncio.Task],
        ollama_age: float | None,
        queue_depths: dict[str, int],
    ) -> None:
        """Print the verbose diagnostic block. Called only on suspicious
        ticks so normal operation produces a single status line."""
        loop_now = time.monotonic()
        # Compute per-task age via ``_loop_time_at_creation`` which
        # CPython annotates on tasks started under the default loop.
        # Missing attribute -> age 0 (best-effort only).
        aged: list[tuple[float, asyncio.Task]] = []
        for t in all_tasks:
            created = getattr(t, "_loop_time_at_creation", None)
            if created is None:
                # Fallback: some tasks don't expose creation time, so
                # treat them as fresh. They'll rank low and won't
                # crowd out truly stuck ones.
                age_s = 0.0
            else:
                age_s = max(0.0, loop_now - float(created))
            aged.append((age_s, t))
        aged.sort(key=lambda p: p[0], reverse=True)

        long_pending = [t for age_s, t in aged if age_s >= LONG_PENDING_AGE_SECONDS]

        # "no requests made" vs "queued": if ollama has been silent but
        # the queue is empty, something upstream is blocking the lanes
        # from even calling Ollama. That's a loop-contention fingerprint,
        # not an Ollama-slowness fingerprint.
        total_q = sum(queue_depths.values())
        if ollama_age is not None and total_q == 0:
            traffic_note = "no-requests-in-flight (upstream blocked)"
        elif total_q > 0:
            traffic_note = f"{total_q}-in-flight (ollama processing)"
        else:
            traffic_note = "idle"
        logger.warning(
            "[watchdog] DIAG ollama_silent={} traffic={} long_pending={}",
            f"{ollama_age:.0f}s" if ollama_age is not None else "never",
            traffic_note,
            len(long_pending),
        )
        for age_s, t in aged[:DIAGNOSTIC_TASK_DUMP_LIMIT]:
            name = t.get_name() or "<unnamed>"
            coro = t.get_coro()
            qualname = getattr(coro, "__qualname__", None) or getattr(
                coro, "__name__", "<anon>",
            )
            logger.warning(
                "[watchdog] DIAG   task age={:.0f}s name={} coro={}",
                age_s, name, qualname,
            )
        cause = self._suspected_cause(long_pending)
        logger.warning("[watchdog] suspected cause: {}", cause)

    async def _tick(self) -> None:
        global DEGRADED
        now = now_ts()
        # Snapshot tasks once — ``asyncio.all_tasks()`` is cheap but
        # iterating it twice risks inconsistent views under churn.
        all_tasks = [t for t in asyncio.all_tasks() if not t.done()]
        pending = len(all_tasks)

        last_ollama = ollama_mod.LAST_SUCCESS_TS
        ollama_age = (now - last_ollama) if last_ollama else None

        last_ws = poly_ws_mod.LAST_MESSAGE_TS
        ws_age = (now - last_ws) if last_ws else None

        # Ollama queue depths by tier — distinguishes "no requests made"
        # (all zero + long ollama_age = loop blocked, NOT Ollama's fault)
        # from "requests queued" (non-zero = Ollama itself slow/stuck).
        q = ollama_mod.OllamaClient.queue_depths()

        logger.info(
            "[watchdog] pending_tasks={} ollama_silent={} ollama_q=fast:{}/deep:{}/validator:{} ws_silent={} degraded={}",
            pending,
            f"{ollama_age:.0f}s" if ollama_age is not None else "never",
            q.get("fast", 0), q.get("deep", 0), q.get("validator", 0),
            f"{ws_age:.0f}s" if ws_age is not None else "never",
            DEGRADED,
        )

        # Verbose diagnostic dump — only when things look off, to keep
        # log volume bounded during healthy operation. See the module
        # docstring for the escalation ladder.
        ollama_suspicious = (
            ollama_age is not None
            and ollama_age > DIAGNOSTIC_OLLAMA_SILENT_SECONDS
        )
        task_count_suspicious = pending > (CONTENTION_TASK_THRESHOLD // 2)
        if ollama_suspicious or task_count_suspicious:
            self._emit_diagnostic_dump(all_tasks, ollama_age, q)

        degraded_this_tick = False

        # ---- Ollama escalation ----
        # Pool-clear happens first: a locked-up pool is often what keeps
        # ollama_age climbing, so clear it before the hard error trigger.
        if (
            ollama_age is not None
            and ollama_age > OLLAMA_CONTENTION_SECONDS
            and pending > CONTENTION_TASK_THRESHOLD
        ):
            logger.warning(
                "[watchdog] cleared Ollama connection pool due to contention "
                "(ollama_silent={:.0f}s, pending_tasks={})",
                ollama_age, pending,
            )
            try:
                await ollama_mod.reset_shared_client()
            except Exception as e:
                logger.warning("[watchdog] reset_shared_client failed: {}", e)
            degraded_this_tick = True
        if ollama_age is not None and ollama_age > OLLAMA_SILENT_ERROR_SECONDS:
            logger.error(
                "[watchdog] Ollama silent for {:.0f}s — event loop contention"
                " or Ollama unresponsive", ollama_age,
            )
            degraded_this_tick = True

        # ---- poly_ws reconnect, with strike-count backoff ----
        ws_needs_reconnect = (
            ws_age is not None
            and ws_age > WS_SILENT_RECONNECT_SECONDS
            and self._poly_ws is not None
        )
        if ws_needs_reconnect:
            if now < self._ws_hard_backoff_until:
                # Honouring an in-progress hard backoff.
                remaining = self._ws_hard_backoff_until - now
                logger.warning(
                    "[watchdog] WS backoff active for {:.0f}s more, skipping reconnect",
                    remaining,
                )
                degraded_this_tick = True
            elif self._ws_reconnect_strikes >= WS_RECONNECT_STRIKES_BEFORE_BACKOFF:
                # 3rd+ strike with nothing to show — back off hard.
                logger.error(
                    "[watchdog] WS unreachable after {} reconnects — "
                    "backing off {:.0f}s",
                    self._ws_reconnect_strikes, WS_HARD_BACKOFF_SECONDS,
                )
                self._ws_hard_backoff_until = now + WS_HARD_BACKOFF_SECONDS
                self._ws_reconnect_strikes = 0
                degraded_this_tick = True
            else:
                logger.warning(
                    "[watchdog] poly_ws silent for {:.0f}s — forcing reconnect "
                    "(strike {}/{})",
                    ws_age,
                    self._ws_reconnect_strikes + 1,
                    WS_RECONNECT_STRIKES_BEFORE_BACKOFF,
                )
                try:
                    self._poly_ws.request_reconnect()
                    self._ws_reconnect_strikes += 1
                except Exception as e:
                    logger.warning("[watchdog] request_reconnect failed: {}", e)
                degraded_this_tick = True
        elif ws_age is not None and ws_age <= WS_SILENT_RECONNECT_SECONDS:
            # Fresh messages arriving — clear the strike counter.
            self._ws_reconnect_strikes = 0

        # Also treat a very long WS silence as degraded regardless of strikes.
        if ws_age is not None and ws_age > WS_SILENT_HARD_BACKOFF_SECONDS:
            degraded_this_tick = True

        # Contention pressure alone marks us degraded so feeds throttle
        # even before Ollama or the WS crosses their thresholds.
        if pending > CONTENTION_TASK_THRESHOLD:
            degraded_this_tick = True

        if degraded_this_tick:
            if not DEGRADED:
                logger.warning("[watchdog] entering DEGRADED state")
            DEGRADED = True
            self._healthy_ticks = 0
        else:
            self._healthy_ticks += 1
            if DEGRADED and self._healthy_ticks >= HEALTHY_CONFIRM_TICKS:
                logger.info(
                    "[watchdog] leaving DEGRADED after {} healthy ticks",
                    self._healthy_ticks,
                )
                DEGRADED = False
