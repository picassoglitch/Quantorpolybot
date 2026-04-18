"""Config loader. YAML on disk + .env for secrets.

The config object is a plain dict-like wrapper so that the optimization
module can hot-reload it without restarting the process.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_CONFIG_LOCK = threading.RLock()
_PROMPTS_LOCK = threading.RLock()

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "config" / "config.yaml"
_PROMPTS_PATH = _ROOT / "config" / "prompts.yaml"


class _Config:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        with _CONFIG_LOCK:
            with _CONFIG_PATH.open("r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}

    def reload(self) -> None:
        self.load()

    def save(self, new_data: dict[str, Any]) -> None:
        """Atomic save with .bak rollback."""
        with _CONFIG_LOCK:
            backup = _CONFIG_PATH.with_suffix(".yaml.bak")
            if _CONFIG_PATH.exists():
                shutil.copy2(_CONFIG_PATH, backup)
            tmp = _CONFIG_PATH.with_suffix(".yaml.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                yaml.safe_dump(new_data, f, sort_keys=False)
            os.replace(tmp, _CONFIG_PATH)
            self._data = new_data

    def get(self, *path: str, default: Any = None) -> Any:
        cur: Any = self._data
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def as_dict(self) -> dict[str, Any]:
        with _CONFIG_LOCK:
            # shallow copy is fine for read; callers who mutate must save
            return dict(self._data)


class _Prompts:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        with _PROMPTS_LOCK:
            with _PROMPTS_PATH.open("r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}

    def reload(self) -> None:
        self.load()

    def save(self, new_data: dict[str, Any]) -> None:
        with _PROMPTS_LOCK:
            backup = _PROMPTS_PATH.with_suffix(".yaml.bak")
            if _PROMPTS_PATH.exists():
                shutil.copy2(_PROMPTS_PATH, backup)
            tmp = _PROMPTS_PATH.with_suffix(".yaml.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                yaml.safe_dump(new_data, f, sort_keys=False)
            os.replace(tmp, _PROMPTS_PATH)
            self._data = new_data

    def active(self) -> tuple[str, str]:
        with _PROMPTS_LOCK:
            version = self._data.get("active", "v1")
            template = (self._data.get("versions") or {}).get(version, "")
            return version, template

    def add_version(self, name: str, template: str, activate: bool) -> None:
        with _PROMPTS_LOCK:
            data = dict(self._data)
            versions = dict(data.get("versions") or {})
            versions[name] = template
            data["versions"] = versions
            if activate:
                data["active"] = name
            self.save(data)


# Lazy global so import order is forgiving.
_CONFIG_SINGLETON: _Config | None = None
_PROMPTS_SINGLETON: _Prompts | None = None


def load_env() -> None:
    load_dotenv(_ROOT / ".env", override=False)


def get_config() -> _Config:
    global _CONFIG_SINGLETON
    if _CONFIG_SINGLETON is None:
        _CONFIG_SINGLETON = _Config()
    return _CONFIG_SINGLETON


def get_prompts() -> _Prompts:
    global _PROMPTS_SINGLETON
    if _PROMPTS_SINGLETON is None:
        _PROMPTS_SINGLETON = _Prompts()
    return _PROMPTS_SINGLETON


def root_dir() -> Path:
    return _ROOT


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)
