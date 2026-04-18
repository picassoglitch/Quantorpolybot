"""Tiny helpers shared across modules."""

from __future__ import annotations

import asyncio
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, TypeVar

T = TypeVar("T")

_WORD_RE = re.compile(r"[a-z0-9]+")


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def keywords(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def chunked(seq: list[T], size: int) -> Iterable[list[T]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@dataclass
class Backoff:
    """Exponential backoff with jitter for reconnect loops."""

    base: float = 1.0
    cap: float = 60.0
    factor: float = 2.0
    _attempt: int = 0

    def reset(self) -> None:
        self._attempt = 0

    def next_delay(self) -> float:
        delay = min(self.cap, self.base * (self.factor ** self._attempt))
        self._attempt += 1
        # mild jitter, deterministic enough for tests
        return delay * (0.75 + 0.5 * ((self._attempt * 2654435761) % 1000) / 1000)


@asynccontextmanager
async def suppress_cancel() -> AsyncIterator[None]:
    """Context that swallows CancelledError so cleanup can run."""
    try:
        yield
    except asyncio.CancelledError:
        return
