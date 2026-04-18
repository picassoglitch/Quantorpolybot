"""Thin wrapper around py-clob-client. Imports lazily so dry-run works
without the wallet env vars being set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.utils.config import env

_client: Any | None = None


@dataclass
class OrderResult:
    ok: bool
    clob_order_id: str
    raw: dict[str, Any]
    error: str = ""


def _build_client() -> Any | None:
    """Construct py-clob-client lazily. Returns None on missing creds."""
    global _client
    if _client is not None:
        return _client
    pk = env("POLY_PRIVATE_KEY")
    funder = env("POLY_FUNDER_ADDRESS")
    if not pk or not funder:
        logger.warning("[clob] POLY_PRIVATE_KEY/POLY_FUNDER_ADDRESS missing; live orders disabled")
        return None
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        from py_clob_client.constants import POLYGON
    except ImportError as e:
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
        logger.exception("[clob] init failed: {}", e)
        return None
    return _client


def is_ready() -> bool:
    return _build_client() is not None


async def place_limit_order(token_id: str, side: str, price: float, size: float) -> OrderResult:
    """Synchronous py-clob-client call wrapped via run_in_executor."""
    import asyncio

    client = _build_client()
    if client is None:
        return OrderResult(False, "", {}, "client not ready")

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
    import asyncio

    client = _build_client()
    if client is None or not clob_order_id:
        return False

    def _do() -> bool:
        try:
            client.cancel(order_id=clob_order_id)
            return True
        except Exception as e:
            logger.warning("[clob] cancel failed for {}: {}", clob_order_id, e)
            return False

    return await asyncio.get_running_loop().run_in_executor(None, _do)


async def order_status(clob_order_id: str) -> dict[str, Any]:
    import asyncio

    client = _build_client()
    if client is None or not clob_order_id:
        return {}

    def _do() -> dict[str, Any]:
        try:
            return client.get_order(order_id=clob_order_id) or {}
        except Exception:
            return {}

    return await asyncio.get_running_loop().run_in_executor(None, _do)
