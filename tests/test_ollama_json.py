"""Tests for the Ollama JSON extractor."""

from core.signals.ollama_client import OllamaClient


def test_extracts_clean_json():
    out = OllamaClient._extract_json('{"market_id":"abc","implied_prob":0.7,"confidence":0.8,"reasoning":"x"}')
    assert out is not None
    assert out["market_id"] == "abc"


def test_extracts_json_inside_prose():
    text = "Sure, here is my answer: {\"market_id\": null, \"implied_prob\": 0.5, \"confidence\": 0.4, \"reasoning\": \"unclear\"}. done"
    out = OllamaClient._extract_json(text)
    assert out is not None
    assert out["confidence"] == 0.4


def test_returns_none_for_garbage():
    assert OllamaClient._extract_json("no json here") is None
    assert OllamaClient._extract_json("") is None
