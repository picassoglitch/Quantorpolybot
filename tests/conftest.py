"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

# Make the project root importable when running `pytest` from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _test_sized_config():
    """Override capital/lane knobs on the shared config singleton so tests
    don't depend on the operator's production-tuned values (currently
    ``shadow_capital.total_usd: 15`` for their funded account).

    Tests were written against the canonical $10k shadow pool with a
    60/30/10 split and absolute per-lane sizes (scalping=$75,
    event_sniping heuristic fallback=$50). This fixture reinstates that
    baseline for the duration of each test and restores the original
    values on teardown so the file on disk is never touched.
    """
    from core.utils.config import get_config

    cfg = get_config()
    keys = ("shadow_capital", "real_capital", "scalping", "event_sniping", "ollama")
    saved = {k: cfg._data.get(k) for k in keys}

    cfg._data["shadow_capital"] = {
        "total_usd": 10000.0,
        "lane_allocations": {"scalping": 0.6, "event_sniping": 0.3, "longshot": 0.1},
        "min_lane_available_usd": 50.0,
    }
    cfg._data["real_capital"] = {
        "total_usd": 0.0,
        "lane_allocations": {"scalping": 0.6, "event_sniping": 0.3, "longshot": 0.1},
        "min_lane_available_usd": 50.0,
    }

    scalping = dict(saved["scalping"] or {})
    scalping.update({
        "base_position_pct": 0,
        "max_position_pct": 0,
        "base_position": 75,
        "max_position": 75,
    })
    cfg._data["scalping"] = scalping

    event = dict(saved["event_sniping"] or {})
    event.update({
        "heuristic_fallback_pct": 0,
        "heuristic_fallback_size_usd": 50.0,
    })
    cfg._data["event_sniping"] = event

    ollama = dict(saved["ollama"] or {})
    ollama["validator_high_stakes_usd"] = 200
    cfg._data["ollama"] = ollama

    yield

    for k, v in saved.items():
        if v is None:
            cfg._data.pop(k, None)
        else:
            cfg._data[k] = v
