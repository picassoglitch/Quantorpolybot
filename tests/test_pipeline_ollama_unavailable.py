"""SignalPipeline must persist a row when Ollama is unavailable.

Before this fix the pipeline silently ``return``ed when ``generate_json``
gave back ``None`` (cooldown / timeout / unparseable). That left feed
items processed but the signals table empty, so the dashboard couldn't
distinguish "no candidate matched the news" from "Ollama was down for
2 hours". A persisted row with ``status='ollama_unavailable'`` makes
the failure mode visible at a glance.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from core.risk.rules import RiskEngine
from core.signals import pipeline as pipeline_mod
from core.signals.pipeline import SignalPipeline
from core.utils import db as db_module
from core.utils.db import execute, fetch_all


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "pipeline_ollama_unavailable.db"
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    asyncio.run(db_module.init_db())
    yield
    monkeypatch.setattr(db_module, "_DB_PATH", None)


@pytest.mark.asyncio
async def test_pipeline_persists_row_when_ollama_returns_none(monkeypatch):
    # Seed a market the candidate scorer can find.
    market_id = "m-unavailable-1"
    await execute(
        """INSERT INTO markets
           (market_id, question, slug, category, active, close_time,
            token_ids, best_bid, best_ask, last_price, liquidity, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            market_id,
            "Will the candidate win the primary?",
            market_id,
            "politics",
            1,
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 5 * 86400),
            ),
            json.dumps(["yes-tok", "no-tok"]),
            0.39, 0.41, 0.40, 50_000.0, time.time(),
        ),
    )
    item_id = await execute(
        """INSERT INTO feed_items
           (url_hash, source, title, summary, url, ingested_at, meta)
           VALUES (?,?,?,?,?,?,?)""",
        (
            "ph-1", "reuters",
            "Candidate wins primary in shock result",
            "The frontrunner surges ahead and confirms victory.",
            "http://x/1", time.time() - 30,
            json.dumps({"linked_market_id": market_id}),
        ),
    )

    # Stub the candidate scorer so it returns our seeded market without
    # depending on the full keyword index — keeps the test focused on
    # the ollama_unavailable branch.
    from core.markets.cache import Market

    # Close time inside the long-horizon prefilter window
    # (max_candidate_days=365 in default config) so the item reaches
    # the Ollama call rather than being short-circuited by the
    # long_horizon prefilter.
    close_time = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30 * 86400),
    )
    seeded_market = Market(
        market_id=market_id,
        question="Will the candidate win the primary?",
        slug=market_id,
        category="politics",
        active=True,
        close_time=close_time,
        token_ids=["yes-tok", "no-tok"],
        best_bid=0.39, best_ask=0.41, last_price=0.40,
        liquidity=50_000.0, updated_at=time.time(),
    )

    async def fake_scored_candidates_for(text):
        return [(0.9, seeded_market)]

    monkeypatch.setattr(
        pipeline_mod, "scored_candidates_for", fake_scored_candidates_for,
    )

    # Stub build_market_context to avoid extra DB churn in this test.
    async def fake_context(market_id):
        return {}

    monkeypatch.setattr(
        pipeline_mod, "build_market_context", fake_context,
    )

    # Force generate_json to return None — i.e. Ollama unavailable.
    pipe = SignalPipeline(RiskEngine())

    async def fake_generate_json(prompt, *, tag=""):
        return None

    monkeypatch.setattr(pipe._ollama, "generate_json", fake_generate_json)

    # Drive a single item through the pipeline.
    item_row = {
        "id": int(item_id),
        "title": "Candidate wins primary in shock result",
        "summary": "The frontrunner surges ahead and confirms victory.",
        "source": "reuters",
        "url": "http://x/1",
        "ingested_at": time.time() - 30,
        "meta": json.dumps({"linked_market_id": market_id}),
    }
    await pipe._process_item(item_row)

    # Assert: a row exists with status='ollama_unavailable' against the
    # candidate market — this is the operator-visibility hook.
    rows = await fetch_all(
        "SELECT status, market_id, reasoning, feed_item_id "
        "FROM signals WHERE feed_item_id=?",
        (int(item_id),),
    )
    statuses = [r["status"] for r in rows]
    assert "ollama_unavailable" in statuses, (
        f"expected an ollama_unavailable row, got statuses={statuses}"
    )
    unavailable = next(r for r in rows if r["status"] == "ollama_unavailable")
    assert unavailable["market_id"] == market_id
    assert "ollama_unavailable" in (unavailable["reasoning"] or "")
