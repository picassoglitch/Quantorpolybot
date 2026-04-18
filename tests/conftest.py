"""Shared pytest fixtures."""

import asyncio
import sys
from pathlib import Path

import pytest

# Make the project root importable when running `pytest` from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
