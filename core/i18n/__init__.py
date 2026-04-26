"""Lightweight i18n for NexoPolyBot.

Lookup order: requested lang -> env NEXO_LANG -> "es" -> "en" fallback -> key.
Locale files live in core/i18n/locales/<lang>.json (flat key/value maps).

Kept deliberately small: no gettext plural forms, no ICU — the
dashboard has a couple dozen user-visible strings, not a whole
website.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_DEFAULT_LANG = "es"
_FALLBACK_LANG = "en"


def _locales_dir() -> Path:
    return Path(__file__).parent / "locales"


@lru_cache(maxsize=8)
def _load(lang: str) -> dict[str, str]:
    path = _locales_dir() / f"{lang}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def current_lang() -> str:
    return (os.environ.get("NEXO_LANG") or _DEFAULT_LANG).lower().strip() or _DEFAULT_LANG


def t(key: str, lang: str | None = None, **fmt: object) -> str:
    """Translate ``key`` into ``lang`` (or the configured default).

    - Missing in primary lang: falls back to EN.
    - Missing in both: returns the key itself (so the dev sees what's unbound).
    - ``**fmt`` is applied via ``str.format`` so callers can pass placeholders.
    """
    primary = (lang or current_lang()).lower().strip()
    value = _load(primary).get(key)
    if value is None and primary != _FALLBACK_LANG:
        value = _load(_FALLBACK_LANG).get(key)
    if value is None:
        value = key
    if fmt:
        try:
            return value.format(**fmt)
        except (KeyError, IndexError):
            return value
    return value


def available_langs() -> list[str]:
    return sorted(p.stem for p in _locales_dir().glob("*.json"))
