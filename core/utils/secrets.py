"""Read / write the project's .env file safely from the dashboard.

Goals:
  - Round-trip preserve unknown keys, comments, and blank lines so a
    user editing one secret in the UI never wipes hand-edited config.
  - Mask sensitive values when surfacing them to the dashboard so the
    page source never leaks the raw key.
  - Re-export every change to ``os.environ`` so the running process
    picks it up without a restart (where the consumer reads env at
    call time, not at import time).

The .env format we accept is the dotenv subset:
  KEY=value
  # comment
  blank lines
Quoted values (KEY="..." / KEY='...') are preserved on read and
re-emitted with double quotes if the new value contains spaces or #.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from core.utils.config import root_dir

# Keys the dashboard's settings page exposes. The order here is the
# order they render on the page.
EDITABLE_KEYS: list[tuple[str, str, bool, str]] = [
    # (env_key, label, sensitive, help_text)
    ("OLLAMA_HOST", "Ollama host", False,
     "URL where the local Ollama server listens. Default: http://localhost:11434"),
    ("OLLAMA_MODEL", "Ollama model", False,
     "Model tag to use for signal generation. Run `ollama pull <name>` first."),
    ("FRED_API_KEY", "FRED API key", True,
     "Free key from https://fred.stlouisfed.org/docs/api/api_key.html — enables the FRED feed."),
    ("POLY_FUNDER_ADDRESS", "Polymarket funder address", False,
     "Public wallet address that funds CLOB orders. Required for live trading."),
    ("POLY_PRIVATE_KEY", "Polymarket private key", True,
     "Private key for the funder wallet. Use a dedicated wallet you can afford to lose."),
    ("POLY_API_KEY", "Polymarket API key", True,
     "Optional CLOB API key. Most users don't need this — the SDK derives credentials from the wallet."),
    ("POLY_API_SECRET", "Polymarket API secret", True, ""),
    ("POLY_API_PASSPHRASE", "Polymarket API passphrase", True, ""),
]

_LOCK = threading.RLock()
_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _env_path() -> Path:
    return root_dir() / ".env"


@dataclass
class EnvField:
    key: str
    label: str
    sensitive: bool
    help_text: str
    value: str            # raw value (only sent to the page when not sensitive)
    masked: str           # safe-to-display representation
    set: bool             # whether the value is non-empty


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}{'*' * (len(value) - 6)}{value[-3:]}"


def _strip_quotes(raw: str) -> str:
    raw = raw.strip()
    # Strip a trailing inline comment only when the value is unquoted.
    if raw and raw[0] in ('"', "'") and raw.endswith(raw[0]) and len(raw) >= 2:
        return raw[1:-1]
    # Unquoted: drop trailing # comment if any (preceded by whitespace).
    hash_idx = raw.find(" #")
    if hash_idx != -1:
        raw = raw[:hash_idx].rstrip()
    return raw


def _quote_for_write(value: str) -> str:
    """Quote the value if it contains characters that would confuse dotenv."""
    if value == "":
        return ""
    if any(c in value for c in (" ", "#", "\t", "\"", "'", "$")):
        # double-quote and escape embedded double quotes
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def read_env_values() -> dict[str, str]:
    """Return a {key: parsed_value} map of every assignment in .env."""
    path = _env_path()
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            key, raw_val = m.group(1), m.group(2)
            out[key] = _strip_quotes(raw_val)
    return out


def fields_for_dashboard() -> list[EnvField]:
    values = read_env_values()
    out: list[EnvField] = []
    for key, label, sensitive, help_text in EDITABLE_KEYS:
        raw = values.get(key, "")
        out.append(
            EnvField(
                key=key,
                label=label,
                sensitive=sensitive,
                help_text=help_text,
                # Never ship sensitive raw values back to the browser.
                value="" if sensitive else raw,
                masked=_mask(raw) if sensitive else raw,
                set=bool(raw),
            )
        )
    return out


def update_env(updates: dict[str, str]) -> None:
    """Write the supplied (key -> value) updates to .env in place.

    Behavior:
      - Only keys present in ``updates`` are touched. Other lines (incl.
        comments and blank lines) are preserved verbatim.
      - An empty string clears the value (KEY=).
      - Keys that don't yet exist in the file are appended at the end.
      - The previous file is backed up to .env.bak before writing.
      - ``os.environ`` is updated in-process so the running app picks
        up the new values immediately for code that reads env lazily.
    """
    if not updates:
        return
    path = _env_path()
    with _LOCK:
        existing_lines: list[str] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                existing_lines = f.readlines()

        seen: set[str] = set()
        new_lines: list[str] = []
        for line in existing_lines:
            m = _LINE_RE.match(line)
            if not m or m.group(1) not in updates:
                new_lines.append(line)
                continue
            key = m.group(1)
            seen.add(key)
            new_val = updates[key]
            new_lines.append(f"{key}={_quote_for_write(new_val)}\n")

        # Append keys that didn't exist in the file yet.
        appended_any = False
        for key, val in updates.items():
            if key in seen:
                continue
            if not appended_any and new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f"{key}={_quote_for_write(val)}\n")
            appended_any = True

        # Backup + atomic replace.
        if path.exists():
            shutil.copy2(path, path.with_suffix(".env.bak"))
        tmp = path.with_suffix(".env.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.writelines(new_lines)
        os.replace(tmp, path)

        # Reflect in-process so existing singletons don't have to restart.
        for key, val in updates.items():
            if val:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)
