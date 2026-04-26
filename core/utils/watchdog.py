"""Event-loop health watchdog — circuit-aware, lag-measured edition.

Three observations drive the diagnostic:

  - Real **loop lag**: a coroutine schedules itself ``call_later(N)``
    and measures ``actual_fire_time - expected_fire_time``. Above the
    configured threshold the loop is genuinely blocked. Below it,
    "pending tasks" or "Ollama silent" do not imply blocking — those
    are inferential signals that produced false positives during the
    April 2026 soak.

  - **Ollama circuit state** per tier (from
    ``core.signals.ollama_client.all_tier_circuit_states``). When any
    tier is OPEN, the bot is intentionally in heuristic-only mode and
    "Ollama silent" should NOT be reported as degradation.

  - **WebSocket liveness**: ``poly_ws_mod.LAST_MESSAGE_TS`` last-msg
    timestamp, with the same strike-counted reconnect logic as before.

Output shape every 30s:

    [watchdog] loop_lag_ms=2 ollama_state=fast:CLOSED/deep:OPEN(45s)/
        validator:CLOSED heuristic_only=True ws_silent=4s
        in_flight=fast:0/deep:0/validator:0 abandoned=deep:1
        pending_tasks=24 degraded=False

Escalation rules:

  - **loop_lag_ms > LOOP_LAG_BLOCKED_MS** -> WARN ``event_loop_blocked``;
    DEGRADED.
  - **Ollama silent > OLLAMA_SILENT_ERROR_SECONDS** AND no tier OPEN
    AND we expected traffic -> ERROR ``ollama_unresponsive``; DEGRADED.
  - **Ollama silent** while ANY tier is OPEN -> NOT degraded, log line
    says ``heuristic_only=True``.
  - **poly_ws silent** -> reconnect strikes (unchanged from prior).

The watchdog is a strict observer — it does not touch lane budgets or
open positions.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from core.feeds import polymarket_ws as poly_ws_mod
from core.signals import ollama_client as ollama_mod
from core.signals import ollama_executor
from core.utils.helpers import now_ts

INTERVAL_SECONDS = 30.0

# ---- Loop-lag thresholds ----
# How far into the future the lag probe schedules itself. The probe
# measures the delta between scheduled and actual fire time. A small
# delta means the loop ran when it said it would; a big delta means
# something held the loop's run-to-completion budget too long.
LOOP_LAG_PROBE_DELAY = 0.5
# Above this many milliseconds, we declare the loop genuinely blocked.
# Bumped 250 -> 500 after the April 2026 step-1.7 soak: a single 267ms
# blip during normal lane churn is not a degradation event, and
# trip-on-first-spike caused the watchdog to enter DEGRADED for a
# 17ms-over-threshold breach during a healthy minute.
LOOP_LAG_BLOCKED_MS = 500.0
# Number of consecutive lag breaches required before flipping
# DEGRADED. A trading bot should not degrade on a single blip — the
# loop is allowed one bad tick (e.g. boot-time discovery + WS
# subscribe burning CPU + DB writes) without crying wolf.
CONSECUTIVE_LAG_BREACHES_TO_DEGRADE = 2
# Seconds after watchdog start during which lag breaches log INFO but
# do NOT count toward DEGRADED. The very first tick after boot
# legitimately shows 1-2 seconds of lag because every subsystem (9
# feeds + WS + market discovery + signal pipeline + 3 lanes) is
# initialising at once. The 30s grace covers that ramp-up cleanly.
STARTUP_GRACE_SECONDS = 30.0

# ---- Ollama silence thresholds ----
# Both apply ONLY when no tier circuit is OPEN. When a circuit is open,
# silence is the design — the bot is in heuristic-only mode.
OLLAMA_SILENT_WARN_SECONDS = 60.0
OLLAMA_SILENT_ERROR_SECONDS = 120.0

# ---- WebSocket reconnect (unchanged) ----
WS_SILENT_RECONNECT_SECONDS = 120.0
WS_SILENT_HARD_BACKOFF_SECONDS = 180.0
WS_RECONNECT_STRIKES_BEFORE_BACKOFF = 3
WS_HARD_BACKOFF_SECONDS = 120.0

# Number of consecutive healthy ticks before we flip DEGRADED back off.
# Dropped 6 -> 3 (90s recovery) after the step-1.7 soak: a healthy
# heuristic_only mode with low loop_lag was sitting in DEGRADED for
# 3+ minutes after a single 267ms spike. 90s is enough to confirm
# recovery without trapping the bot in a stale state.
HEALTHY_CONFIRM_TICKS = 3

# Module-level status flag read by FeedManager / feeds to throttle their
# poll loops when the event loop is under pressure.
DEGRADED: bool = False


def is_degraded() -> bool:
    return DEGRADED


class Watchdog:
    component = "utils.watchdog"

    def __init__(self, poly_ws: Any | None = None) -> None:
        self._stop = asyncio.Event()
        self._poly_ws = poly_ws
        self._started_at = now_ts()
        self._ws_reconnect_strikes = 0
        self._ws_hard_backoff_until = 0.0
        self._healthy_ticks = 0
        # Counter for consecutive ticks where loop_lag_ms exceeded
        # ``LOOP_LAG_BLOCKED_MS``. Only flips DEGRADED once it hits
        # ``CONSECUTIVE_LAG_BREACHES_TO_DEGRADE`` so a single spike
        # (boot-time discovery, an RSS cycle, etc.) doesn't trigger.
        self._consecutive_lag_breaches = 0

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

    # ---- Real loop-lag probe ------------------------------------------

    async def _measure_loop_lag_ms(self) -> float:
        """Schedule a callback ``LOOP_LAG_PROBE_DELAY`` seconds in the
        future and measure ``actual - expected`` when it fires. Small
        values mean the loop ran when it said it would; large values
        mean something held the loop too long between scheduler ticks.

        This is the ONLY "is the loop blocked?" signal the watchdog
        trusts. Pending-task count and Ollama silence are inferential
        and produced false positives in the April 2026 soak."""
        loop = asyncio.get_running_loop()
        target = loop.time() + LOOP_LAG_PROBE_DELAY
        actual_holder: dict[str, float] = {}
        fired = asyncio.Event()

        def _on_fire() -> None:
            actual_holder["t"] = loop.time()
            fired.set()

        loop.call_at(target, _on_fire)
        # Cap the wait so a truly hung loop doesn't make us hang too —
        # if we don't fire within 2x the probe budget, we report the
        # actual elapsed wait as the lag.
        try:
            await asyncio.wait_for(fired.wait(), timeout=LOOP_LAG_PROBE_DELAY * 4)
        except asyncio.TimeoutError:
            return (LOOP_LAG_PROBE_DELAY * 4) * 1000.0
        actual = actual_holder.get("t", target)
        return max(0.0, (actual - target) * 1000.0)

    # ---- Per-tick collection helpers ----------------------------------

    @staticmethod
    def _format_circuit_summary(states: dict[str, dict[str, Any]]) -> str:
        parts: list[str] = []
        for tier, info in states.items():
            state = info["state"]
            if state == "OPEN":
                parts.append(f"{tier}:{state}({info['cooldown_remaining']:.0f}s)")
            elif state == "HALF_OPEN":
                parts.append(f"{tier}:{state}")
            else:
                parts.append(f"{tier}:{state}")
        return "/".join(parts)

    # ---- Tick ---------------------------------------------------------

    async def _tick(self) -> None:
        global DEGRADED
        now = now_ts()

        # ---- Real loop-lag measurement ----
        lag_ms = await self._measure_loop_lag_ms()
        lag_breach = lag_ms > LOOP_LAG_BLOCKED_MS
        # Grace window: very early after boot, lots of subsystems are
        # ramping up concurrently and a 1-2s lag spike is normal. Don't
        # let it trip DEGRADED — log it at INFO so the operator can
        # still see it.
        in_startup_grace = (now - self._started_at) < STARTUP_GRACE_SECONDS
        if lag_breach:
            if in_startup_grace:
                # Reset the counter — we don't want a startup spike to
                # leave us "1/2 breaches" pending into normal operation.
                self._consecutive_lag_breaches = 0
            else:
                self._consecutive_lag_breaches += 1
        else:
            self._consecutive_lag_breaches = 0
        loop_genuinely_blocked = (
            self._consecutive_lag_breaches >= CONSECUTIVE_LAG_BREACHES_TO_DEGRADE
        )

        # ---- Snapshot ----
        all_tasks = [t for t in asyncio.all_tasks() if not t.done()]
        pending = len(all_tasks)

        last_ollama = ollama_mod.LAST_SUCCESS_TS
        # Clamp to 0: ``now_ts`` (time.time) and ``LAST_SUCCESS_TS``
        # are both wall-clock seconds, but reads of one and writes of
        # the other can interleave by tens of microseconds — small
        # negatives surfaced as ``-0s`` in the soak. Same for ws below.
        ollama_age = max(0.0, now - last_ollama) if last_ollama else None

        last_ws = poly_ws_mod.LAST_MESSAGE_TS
        ws_age = max(0.0, now - last_ws) if last_ws else None

        circuit_states = ollama_mod.all_tier_circuit_states()
        any_open = ollama_mod.any_circuit_open()
        any_half_open = any(
            info["state"] == "HALF_OPEN" for info in circuit_states.values()
        )

        # In-flight + abandoned per tier (executor truth).
        executor_snap = ollama_executor.in_flight_snapshot()
        in_flight = executor_snap.per_tier_in_flight
        abandoned = executor_snap.per_tier_abandoned

        # Heuristic-only mode = at least one tier OPEN. Bot is still
        # functioning, just without that tier's LLM. Watchdog must not
        # interpret this as degradation.
        heuristic_only = any_open

        # ---- Status line ----
        in_flight_summary = "/".join(
            f"{t}:{in_flight.get(t, 0)}" for t in ("fast", "deep", "validator")
        )
        abandoned_summary = "/".join(
            f"{t}:{abandoned.get(t, 0)}" for t in ("fast", "deep", "validator")
            if abandoned.get(t, 0) > 0
        ) or "none"
        ollama_silent_str = (
            f"{ollama_age:.0f}s" if ollama_age is not None else "never"
        )
        ws_silent_str = f"{ws_age:.0f}s" if ws_age is not None else "never"
        breaches_str = (
            f" lag_breaches={self._consecutive_lag_breaches}/"
            f"{CONSECUTIVE_LAG_BREACHES_TO_DEGRADE}"
            if self._consecutive_lag_breaches > 0 else ""
        )
        logger.info(
            "[watchdog] loop_lag_ms={:.0f}{} ollama_state={} heuristic_only={} "
            "ollama_silent={} ws_silent={} in_flight={} abandoned={} "
            "pending_tasks={} degraded={}",
            lag_ms,
            breaches_str,
            self._format_circuit_summary(circuit_states),
            heuristic_only,
            ollama_silent_str,
            ws_silent_str,
            in_flight_summary,
            abandoned_summary,
            pending,
            DEGRADED,
        )

        degraded_this_tick = False

        # ---- Loop-lag escalation (the ONLY trustworthy signal) ----
        # Two-stage: first lag breach is a WARN; only after
        # ``CONSECUTIVE_LAG_BREACHES_TO_DEGRADE`` consecutive breaches
        # do we ERROR + flip DEGRADED. Inside the startup grace window
        # we suppress entirely (still INFO-logged via the structured
        # tick line above).
        if lag_breach and in_startup_grace:
            logger.info(
                "[watchdog] loop_lag_ms={:.0f} during startup grace "
                "({:.0f}s) — ignoring",
                lag_ms, STARTUP_GRACE_SECONDS,
            )
        elif loop_genuinely_blocked:
            logger.error(
                "[watchdog] event_loop_blocked: loop_lag_ms={:.0f} for "
                "{} consecutive ticks (threshold {:.0f}ms); pending_tasks={}",
                lag_ms, self._consecutive_lag_breaches,
                LOOP_LAG_BLOCKED_MS, pending,
            )
            self._emit_diagnostic_dump(all_tasks, lag_ms)
            degraded_this_tick = True
        elif lag_breach:
            logger.warning(
                "[watchdog] loop_lag_ms={:.0f} (single-tick breach "
                "{}/{}); waiting for confirmation before degrading",
                lag_ms,
                self._consecutive_lag_breaches,
                CONSECUTIVE_LAG_BREACHES_TO_DEGRADE,
            )

        # ---- Ollama silence escalation, GATED by circuit state ----
        # If any circuit is OPEN: silence is intentional. Don't log
        # ERROR, don't flip DEGRADED. Heuristic-only mode is fine.
        # If half-open: a probe is in flight; silence here is also OK
        # for one tick.
        if not any_open and not any_half_open and ollama_age is not None:
            if ollama_age > OLLAMA_SILENT_ERROR_SECONDS:
                # Only complain if we'd actually expect Ollama traffic.
                # "no in-flight + nothing queued" with a closed circuit
                # AND silence > error threshold means the lanes are
                # not even trying to call Ollama — usually because
                # they're upstream-blocked on something else.
                total_in_flight = sum(in_flight.values())
                if total_in_flight > 0:
                    logger.error(
                        "[watchdog] ollama_unresponsive: silent for {:.0f}s "
                        "with {} call(s) in flight",
                        ollama_age, total_in_flight,
                    )
                    degraded_this_tick = True
                # else: lanes haven't dispatched; this is benign during
                # the warm-up phase or on quiet nights. Silent silence.
            elif ollama_age > OLLAMA_SILENT_WARN_SECONDS:
                logger.warning(
                    "[watchdog] ollama_silent={:.0f}s (circuit closed) — "
                    "expecting a response soon",
                    ollama_age,
                )

        # ---- poly_ws reconnect, with strike-count backoff (unchanged) ----
        ws_needs_reconnect = (
            ws_age is not None
            and ws_age > WS_SILENT_RECONNECT_SECONDS
            and self._poly_ws is not None
        )
        if ws_needs_reconnect:
            if now < self._ws_hard_backoff_until:
                remaining = self._ws_hard_backoff_until - now
                logger.warning(
                    "[watchdog] WS backoff active for {:.0f}s more, skipping reconnect",
                    remaining,
                )
                degraded_this_tick = True
            elif self._ws_reconnect_strikes >= WS_RECONNECT_STRIKES_BEFORE_BACKOFF:
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
            self._ws_reconnect_strikes = 0

        if ws_age is not None and ws_age > WS_SILENT_HARD_BACKOFF_SECONDS:
            degraded_this_tick = True

        # NOTE: the previous "pending_tasks > 30 -> DEGRADED" rule has
        # been deleted. It was the source of the boot-time DEGRADED
        # false positive (the bot legitimately starts ~25 background
        # coroutines). Loop lag is the trustworthy signal now.

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

    # ---- Diagnostic dump (only on REAL loop-lag) ----------------------

    def _emit_diagnostic_dump(
        self, all_tasks: list[asyncio.Task], lag_ms: float,
    ) -> None:
        """Print task ages + suspected cause. Called only when
        loop_lag_ms is genuinely high — the previous version fired on
        a naive task-count threshold and produced thousands of false
        warnings during normal operation."""
        loop_now = time.monotonic()
        aged: list[tuple[float, asyncio.Task]] = []
        for t in all_tasks:
            created = getattr(t, "_loop_time_at_creation", None)
            age_s = max(0.0, loop_now - float(created)) if created is not None else 0.0
            aged.append((age_s, t))
        aged.sort(key=lambda p: p[0], reverse=True)
        logger.warning(
            "[watchdog] DIAG loop_lag_ms={:.0f} tasks={} top oldest:",
            lag_ms, len(aged),
        )
        for age_s, t in aged[:8]:
            name = t.get_name() or "<unnamed>"
            coro = t.get_coro()
            qualname = getattr(coro, "__qualname__", None) or getattr(
                coro, "__name__", "<anon>",
            )
            logger.warning(
                "[watchdog] DIAG   task age={:.0f}s name={} coro={}",
                age_s, name, qualname,
            )
