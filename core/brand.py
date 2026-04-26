"""Nexo Academy brand surface for NexoPolyBot.

The bot itself is a flagship case-study for the Nexo Academy community,
so a small, tasteful brand layer shows through in the startup banner
and the dashboard footer. Everything lives behind functions so the
banner can be swapped or disabled without touching call sites.
"""

from __future__ import annotations

import os
import platform
import sys

from core.i18n import current_lang, t
from core.utils.config import env, get_config

NEXO_BRAND_NAME = "Nexo Academy"
NEXO_TAGLINE_ES = "Trading autónomo. Stack propio. Sin intermediarios."
NEXO_TAGLINE_EN = "Autonomous trading. Own stack. No middlemen."


def brand_url() -> str:
    return env("NEXO_BRAND_URL", "https://nexoacademy.com").strip() or "https://nexoacademy.com"


def tagline(lang: str | None = None) -> str:
    lang = (lang or current_lang()).lower()
    if lang.startswith("en"):
        return NEXO_TAGLINE_EN
    return NEXO_TAGLINE_ES


def generate_startup_banner(version: str = "v2") -> str:
    """ASCII banner printed once at boot. Swap the glyph block freely —
    the width is ~60 cols so it doesn't wrap on a standard terminal."""
    mode = str(get_config().get("mode", default="shadow")).upper()
    lang = current_lang().upper()
    py = ".".join(str(v) for v in sys.version_info[:3])
    art = r"""
 ███╗   ██╗███████╗██╗  ██╗ ██████╗      ██╗ ██╗
 ████╗  ██║██╔════╝╚██╗██╔╝██╔═══██╗    ██╔╝██╔╝
 ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║   ██╔╝██╔╝
 ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║  ██╔╝██╔╝
 ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝ ██╔╝██╔╝
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═╝ ╚═╝
         N E X O   / /   P O L Y B O T
""".rstrip("\n")
    tag = tagline()
    footer = (
        f"   {NEXO_BRAND_NAME} // {version}   mode={mode}   lang={lang}   "
        f"python={py}   os={platform.system().lower()}"
    )
    return f"{art}\n   {tag}\n{footer}\n"


def brand_footer_html() -> str:
    """Small attribution footer for dashboard templates. Keep inline so
    the templates don't need to know where the URL or label live."""
    label = t("brand.footer")
    return (
        f'<a class="nexo-footer" href="{brand_url()}" target="_blank" rel="noopener">'
        f"&#9830; {label}</a>"
    )


def print_startup_banner() -> None:
    # stderr so the banner sits with loguru output; no color codes so it
    # renders cleanly on livestream screen captures too.
    sys.stderr.write(generate_startup_banner() + "\n")
    sys.stderr.flush()
