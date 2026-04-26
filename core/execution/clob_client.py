"""Thin wrapper around py-clob-client. Imports lazily so shadow mode works
without the wallet env vars being set.

py-clob-client is synchronous and uses ``requests`` internally; every
call that talks to the CLOB host must be dispatched via
``run_in_executor`` from async contexts. The historical bug fixed in
this module: ``is_ready()`` used to trigger ``_build_client()``, which
calls ``create_or_derive_api_creds()``. That's a synchronous HTTP round-
trip against clob.polymarket.com. On the first invocation from async
code (e.g. ``shadow.open_position`` on a real entry, or dashboard
render), it blocked the event loop for the full creds-derivation
duration — producing 100-150s "Ollama silent" windows in the watchdog
because every coroutine on the loop was starved. The fix: keep
``is_ready()`` as a cheap cached-state probe (no HTTP, no build), and
require async callers to ``await ensure_ready()`` which dispatches the
build to the default executor.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.utils.config import env

_client: Any | None = None
_last_fail_ts: float = 0.0
_last_fail_reason: str = ""
_FAIL_COOLDOWN_S: float = 60.0
_HEX_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


@dataclass
class OrderResult:
    ok: bool
    clob_order_id: str
    raw: dict[str, Any]
    error: str = ""


def last_init_error() -> str:
    return _last_fail_reason


def _record_failure(reason: str) -> None:
    global _last_fail_ts, _last_fail_reason
    _last_fail_ts = time.time()
    _last_fail_reason = reason


def _build_client() -> Any | None:
    """Construct py-clob-client lazily. Returns None on missing/invalid creds.

    Caches failure state for _FAIL_COOLDOWN_S seconds to avoid log spam when
    is_ready() is polled repeatedly.
    """
    global _client
    if _client is not None:
        return _client
    if _last_fail_ts and (time.time() - _last_fail_ts) < _FAIL_COOLDOWN_S:
        return None

    pk = env("POLY_PRIVATE_KEY")
    funder = env("POLY_FUNDER_ADDRESS")
    if not pk or not funder:
        _record_failure("POLY_PRIVATE_KEY/POLY_FUNDER_ADDRESS missing")
        logger.warning("[clob] credentials missing; live orders disabled (next retry in {}s)", int(_FAIL_COOLDOWN_S))
        return None
    if not _HEX_RE.match(pk.strip()):
        _record_failure("POLY_PRIVATE_KEY is not a valid 64-char hex string")
        logger.warning("[clob] POLY_PRIVATE_KEY invalid format (expect 64 hex chars, optional 0x); live orders disabled (next retry in {}s)", int(_FAIL_COOLDOWN_S))
        return None
    if not funder.strip().lower().startswith("0x") or len(funder.strip()) != 42:
        _record_failure("POLY_FUNDER_ADDRESS is not a valid 0x-prefixed address")
        logger.warning("[clob] POLY_FUNDER_ADDRESS invalid format; live orders disabled (next retry in {}s)", int(_FAIL_COOLDOWN_S))
        return None

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        from py_clob_client.constants import POLYGON
    except ImportError as e:
        _record_failure(f"py-clob-client not installed: {e}")
        logger.error("[clob] py-clob-client not installed: {}", e)
        return None
    try:
        host = env("POLY_HOST", "https://clob.polymarket.com")
        chain_id = int(env("POLY_CHAIN_ID", str(POLYGON)))
        client = ClobClient(host, key=pk, chain_id=chain_id, signature_type=2, funder=funder)
        api_key = env("POLY_API_KEY")
        api_secret = env("POLY_API_SECRET")
        passphrase = env("POLY_API_PASSPHRASE")
        if api_key and api_secret and passphrase:
            client.set_api_creds(ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=passphrase))
        else:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
        _client = client
        logger.info("[clob] py-clob-client ready (funder={}...)", funder[:8])
    except Exception as e:
        _record_failure(f"init failed: {e}")
        logger.error("[clob] init failed ({}); backing off for {}s", e, int(_FAIL_COOLDOWN_S))
        return None
    return _client


def is_ready() -> bool:
    """Non-blocking state probe: True iff py-clob-client has already been
    built successfully. Does NOT attempt a build — safe to call from any
    context (sync dashboard handler, async lane). Async callers that need
    the client to actually exist (e.g. before placing a real order)
    should ``await ensure_ready()`` first so the build runs off-loop.
    """
    return _client is not None


async def ensure_ready() -> bool:
    """Async-safe variant of ``is_ready()``. Dispatches the sync
    ``_build_client()`` call to the default executor so the first build —
    which makes an HTTP round-trip against clob.polymarket.com via
    ``create_or_derive_api_creds()`` — does not stall the event loop.

    Returns True when the client is usable, False on any init failure
    (missing/invalid creds, network, py-clob-client import error). Subsequent
    calls are essentially free once the client is cached.
    """
    if _client is not None:
        return True
    loop = asyncio.get_running_loop()
    # The build itself is CPU+IO mixed (env var checks + HTTP); run the
    # whole function off-loop. _build_client() is idempotent and guards
    # against double-build via the module-level _client cache.
    return await loop.run_in_executor(None, lambda: _build_client() is not None)


async def place_limit_order(token_id: str, side: str, price: float, size: float) -> OrderResult:
    """Synchronous py-clob-client call wrapped via run_in_executor."""
    if not await ensure_ready():
        return OrderResult(False, "", {}, "client not ready")
    client = _client  # guaranteed non-None after ensure_ready()

    def _do() -> OrderResult:
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            args = OrderArgs(token_id=token_id, price=price, size=size, side=side.upper())
            signed = client.create_order(args)
            resp = client.post_order(signed, OrderType.GTC)
            order_id = ""
            if isinstance(resp, dict):
                order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id") or ""
            return OrderResult(True, str(order_id), resp if isinstance(resp, dict) else {"raw": resp})
        except Exception as e:
            return OrderResult(False, "", {}, str(e))

    return await asyncio.get_running_loop().run_in_executor(None, _do)


async def cancel_order(clob_order_id: str) -> bool:
    if not clob_order_id:
        return False
    if not await ensure_ready():
        return False
    client = _client

    def _do() -> bool:
        try:
            client.cancel(order_id=clob_order_id)
            return True
        except Exception as e:
            logger.warning("[clob] cancel failed for {}: {}", clob_order_id, e)
            return False

    return await asyncio.get_running_loop().run_in_executor(None, _do)


async def order_status(clob_order_id: str) -> dict[str, Any]:
    if not clob_order_id:
        return {}
    if not await ensure_ready():
        return {}
    client = _client

    def _do() -> dict[str, Any]:
        try:
            return client.get_order(order_id=clob_order_id) or {}
        except Exception:
            return {}

    return await asyncio.get_running_loop().run_in_executor(None, _do)
