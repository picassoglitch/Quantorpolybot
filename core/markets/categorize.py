"""Keyword-based market category inference.

Polymarket's Gamma API often returns a blank `category` field. Without
an inferred bucket, the risk engine's per-category correlation cap
either fires spuriously (grouping everything into `uncategorised`) or
short-circuits. We infer the category from the question text instead.
Returns an empty string when nothing matches so the risk engine can
skip the cap rather than misclassify.
"""

from __future__ import annotations

import re

# Order matters: first matching bucket wins. Put specific terms before
# generic ones (e.g. "fed" before "rate"). All patterns are lowercase
# and matched against a lowercased haystack.
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    (
        "sports",
        [
            r"\bnba\b", r"\bnfl\b", r"\bnhl\b", r"\bmlb\b", r"\bmls\b",
            r"\bfifa\b", r"\buefa\b", r"\bufc\b", r"\bpga\b", r"\batp\b",
            r"\bwta\b", r"\bf1\b", r"\bformula\s*1\b",
            r"\bsoccer\b", r"\bfootball\b", r"\bbasketball\b",
            r"\bbaseball\b", r"\bhockey\b", r"\btennis\b", r"\bgolf\b",
            r"\bcricket\b", r"\brugby\b", r"\bboxing\b",
            r"\bpremier\s+league\b", r"\bchampions\s+league\b",
            r"\bla\s+liga\b", r"\bbundesliga\b", r"\bserie\s+a\b",
            r"\bworld\s+cup\b", r"\bsuper\s+bowl\b", r"\bworld\s+series\b",
            r"\bstanley\s+cup\b", r"\bnba\s+finals\b",
            r"\bolympics?\b", r"\bwimbledon\b",
        ],
    ),
    (
        "politics",
        [
            r"\bpresident\b", r"\bpresidential\b", r"\bpresidency\b",
            r"\bvice\s+president\b", r"\bgovernor\b", r"\bsenator\b",
            r"\bcongress\b", r"\bhouse\s+of\s+representatives\b",
            r"\bnominee\b", r"\bnomination\b",
            r"\belection\b", r"\bprimary\b", r"\bcaucus\b", r"\bballot\b",
            r"\bparliament\b", r"\bprime\s+minister\b", r"\bchancellor\b",
            r"\bcabinet\b", r"\bimpeach(?:ed|ment)?\b",
            r"\bdemocrat\b", r"\brepublican\b", r"\btrump\b", r"\bbiden\b",
            r"\bharris\b", r"\bputin\b", r"\bzelensky\b",
            r"\bsupreme\s+court\b", r"\blegislation\b",
        ],
    ),
    (
        "crypto",
        [
            r"\bbitcoin\b", r"\bethereum\b", r"\bsolana\b", r"\bdoge(?:coin)?\b",
            r"\b(?:btc|eth|sol|xrp|ada|bnb|avax|matic|dot)\b",
            r"\bcrypto\b", r"\bblockchain\b", r"\bstablecoin\b",
            r"\bdefi\b", r"\bnft\b", r"\btether\b", r"\busdc\b", r"\busdt\b",
            r"\bcoinbase\b", r"\bbinance\b", r"\bsec\s+(?:vs|v)\b",
            r"\betf\s+approval\b", r"\bhalving\b", r"\bstaking\b",
        ],
    ),
    (
        "macro",
        [
            r"\bfed\b", r"\bfederal\s+reserve\b", r"\bfomc\b", r"\becb\b",
            r"\bboj\b", r"\bboe\b",
            r"\binterest\s+rate\b", r"\brate\s+(?:hike|cut|decision)\b",
            r"\bcpi\b", r"\binflation\b", r"\bdeflation\b",
            r"\bunemployment\b", r"\bjobs?\s+report\b", r"\bnonfarm\b",
            r"\bgdp\b", r"\brecession\b",
            r"\bs&p\s*500\b", r"\bnasdaq\b", r"\bdow\s+jones\b",
            r"\btreasur(?:y|ies)\b", r"\byield\b",
            r"\boil\s+price\b", r"\bgold\s+price\b",
        ],
    ),
]

_COMPILED: list[tuple[str, list[re.Pattern[str]]]] = [
    (cat, [re.compile(p, re.IGNORECASE) for p in patterns])
    for cat, patterns in _CATEGORY_RULES
]


def infer_category(text: str) -> str:
    """Return 'sports' | 'politics' | 'crypto' | 'macro' | 'other', or ''
    if the text is empty. Empty string means the risk engine should skip
    the correlation cap for this market rather than lump it into a fake
    bucket."""
    if not text:
        return ""
    haystack = text.lower()
    for category, patterns in _COMPILED:
        for p in patterns:
            if p.search(haystack):
                return category
    return "other"
