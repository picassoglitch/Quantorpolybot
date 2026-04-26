"""Snapshot tests for the i18n layer.

Asserts: ES is default, EN is the fallback, missing keys return the key
itself, format placeholders work, LRU cache is bypassable by changing
the env var between calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.i18n import _load, current_lang, t


def test_es_is_default_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEXO_LANG", raising=False)
    _load.cache_clear()
    assert current_lang() == "es"


def test_known_key_renders_in_es(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXO_LANG", "es")
    _load.cache_clear()
    assert t("settings.mode.current") == "MODO ACTUAL"


def test_known_key_renders_in_en(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXO_LANG", "en")
    _load.cache_clear()
    assert t("settings.mode.current") == "CURRENT MODE"


def test_fallback_to_en_when_key_missing_in_primary(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a tiny language file with only one key, then lookup a key
    that exists in EN but not the primary — expect the EN fallback."""
    fake_lang = "tmpxx"
    locales_dir = Path(__file__).resolve().parents[1] / "core" / "i18n" / "locales"
    target = locales_dir / f"{fake_lang}.json"
    try:
        target.write_text(json.dumps({"only_key": "ONLY"}), encoding="utf-8")
        _load.cache_clear()
        monkeypatch.setenv("NEXO_LANG", fake_lang)
        assert t("only_key") == "ONLY"
        # missing locally, must fall back to EN
        assert t("settings.mode.current") == "CURRENT MODE"
    finally:
        if target.exists():
            target.unlink()
        _load.cache_clear()


def test_missing_everywhere_returns_key_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXO_LANG", "es")
    _load.cache_clear()
    assert t("does.not.exist") == "does.not.exist"


def test_format_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXO_LANG", "en")
    _load.cache_clear()
    out = t("news.lane.starting", n=12, model="llama3:8b")
    assert "12" in out and "llama3:8b" in out


def test_lang_override_param_wins() -> None:
    _load.cache_clear()
    assert t("settings.save", lang="en") == "SAVE"
    assert t("settings.save", lang="es") == "GUARDAR"
