"""Stricter PredictIt ↔ Polymarket question matcher.

Jaccard keyword overlap was producing dangerous false positives — a
PredictIt Bulgaria-election contract would match a Polymarket Québec
question because "election" was common, driving a divergence signal on
two totally unrelated markets.

The replacement rule is deliberately conservative:

  1. Extract capitalized multi-word entities from each question (rough
     named-entity stand-in: "Donald Trump", "New York", "CA-11").
  2. Detect a country / region and an election type ("presidential",
     "senate", "primary", ...) via keyword lists.
  3. Detect the electoral *phase* ("round_1", "round_2", "runoff",
     "general", "primary", "nomination", "overall") — these are the
     concept that makes "will X win ROUND 1" a different question from
     "will X win the election OVERALL", even when every other token
     agrees. When both sides declare a phase and they disagree, reject.
  4. Accept a match iff EITHER:
       (a) the two texts share ≥ 2 named entities AND phases don't
           conflict, OR
       (b) election_type matches AND country matches AND phases don't
           conflict.
  5. Always reject when BOTH texts name a specific candidate (first
     match against a curated candidate list) and those names differ.

Returns (is_match: bool, reason: str) so the feed can log *why* a
candidate was accepted or thrown out.
"""

from __future__ import annotations

import re
from functools import lru_cache

# Election-type synonyms. The key is the canonical bucket.
_ELECTION_TYPES: dict[str, set[str]] = {
    "presidential": {"president", "presidential", "presidency"},
    "senate":       {"senate", "senator", "senatorial"},
    "house":        {"house", "representative", "congressional", "congress"},
    "governor":     {"governor", "gubernatorial"},
    "primary":      {"primary", "caucus"},
    "mayor":        {"mayor", "mayoral"},
    "general":      {"general election"},
}

# Country / region list. Plain lowercased strings; we match on substring
# of the lowercased haystack so "United States", "US", "u.s." all resolve.
_COUNTRIES: list[tuple[str, list[str]]] = [
    ("usa",       ["united states", "u.s.a", "u.s.", " usa ", " us ", "america"]),
    ("canada",    ["canada", "canadian"]),
    ("quebec",    ["quebec", "québec"]),
    ("mexico",    ["mexico", "mexican"]),
    ("uk",        ["united kingdom", "britain", "british", " uk "]),
    ("france",    ["france", "french"]),
    ("germany",   ["germany", "german"]),
    ("india",     ["india", "indian"]),
    ("brazil",    ["brazil", "brazilian"]),
    ("bulgaria",  ["bulgaria", "bulgarian"]),
    ("russia",    ["russia", "russian"]),
    ("ukraine",   ["ukraine", "ukrainian"]),
    ("china",     ["china", "chinese"]),
    ("japan",     ["japan", "japanese"]),
    ("argentina", ["argentina", "argentine", "argentinian"]),
    ("venezuela", ["venezuela", "venezuelan"]),
    ("iran",      ["iran", "iranian"]),
    ("israel",    ["israel", "israeli"]),
    ("poland",    ["poland", "polish"]),
    ("romania",   ["romania", "romanian"]),
    ("australia", ["australia", "australian"]),
    ("new_zealand", ["new zealand"]),
    ("south_korea", ["south korea", "korean"]),
    ("nigeria",   ["nigeria", "nigerian"]),
    # South America — the Colombian / Venezuelan / Argentine markets
    # are a common source of PredictIt xref traffic and were getting
    # misrouted because country detection fell through to None.
    ("colombia",  ["colombia", "colombian"]),
    ("peru",      ["peru", "peruvian"]),
    ("chile",     ["chile", "chilean"]),
    ("ecuador",   ["ecuador", "ecuadorian"]),
    ("bolivia",   ["bolivia", "bolivian"]),
    ("uruguay",   ["uruguay", "uruguayan"]),
    ("paraguay",  ["paraguay", "paraguayan"]),
    # Europe / Mediterranean gap fillers
    ("italy",     ["italy", "italian"]),
    ("spain",     ["spain", "spanish"]),
    ("portugal",  ["portugal", "portuguese"]),
    ("turkey",    ["turkey", "turkish"]),
    ("greece",    ["greece", "greek"]),
]

# Electoral phase. A Polymarket "will X win ROUND 1" contract should
# not match a PredictIt "wins election OVERALL" contract even when the
# candidate + country + etype all align. Patterns are literal substrings
# run against lower-cased text after whitespace normalisation. Order
# matters — we return the first hit, so the more specific phrases come
# first ("round 2", "second round" before bare "round").
_PHASE_PATTERNS: list[tuple[str, list[str]]] = [
    ("round_1", [
        "1st round", "first round", "round 1", "round one",
        "primera vuelta",  # Spanish-language Polymarket questions
    ]),
    ("round_2", [
        "2nd round", "second round", "round 2", "round two",
        "segunda vuelta",
    ]),
    ("runoff",   ["runoff", "run-off", "run off"]),
    ("primary",  ["primary", "caucus"]),
    ("nomination", ["nomination", "nominee"]),
    ("general",  ["general election"]),
    # "Overall / final winner" language — PredictIt contracts for
    # "who wins the election" default here when nothing more specific
    # is said.
    ("overall",  [
        "win the election", "wins the election",
        "win the 2024", "win the 2025", "win the 2026", "win the 2027",
        "win the 2028",
        "overall winner", "final winner",
    ]),
]

# Capitalized multi-word runs OR single-capitalized tokens of length ≥ 3.
# Pattern deliberately preserves hyphens and slashes so "CA-11" /
# "US-Russia" come through as one token.
_ENTITY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-/]*(?:\s+[A-Z][A-Za-z0-9\-/]*){0,3})\b"
)

# Generic capitalized words that are NOT informative on their own —
# showing up once adds zero signal.
_ENTITY_STOPS = {
    "The", "A", "An", "Will", "Is", "Are", "Was", "Were", "Be", "Who",
    "What", "When", "Where", "Why", "How", "Which", "This", "That",
    "Election", "Primary", "Race", "Winner", "Poll", "Market", "Yes", "No",
    "Vote", "Votes", "Voting", "President", "Senator", "Governor",
    "House", "Senate", "Congress", "Contract", "Question", "Over", "Under",
    "Above", "Below", "More", "Less", "Next", "First", "Last", "New",
    "Day", "Date", "Year", "Month", "Week", "Today",
}

# Minimal curated candidate list. These are names we don't want to
# accidentally conflate. Extend as the PredictIt / Polymarket universe
# actually presents us with misses.
_CANDIDATE_NAMES = {
    "donald trump", "trump", "kamala harris", "harris", "joe biden", "biden",
    "ron desantis", "desantis", "nikki haley", "haley", "mike pence", "pence",
    "vivek ramaswamy", "ramaswamy", "gavin newsom", "newsom",
    "pete buttigieg", "buttigieg", "bernie sanders", "sanders",
    "elizabeth warren", "warren", "marco rubio", "rubio", "ted cruz", "cruz",
    "tim scott", "scott", "chris christie", "christie",
    "josh shapiro", "shapiro", "glenn youngkin", "youngkin",
    "j.d. vance", "vance", "rfk jr", "kennedy",
    "rudy melendez", "michelle steel", "young kim",
    "justin trudeau", "trudeau", "pierre poilievre", "poilievre",
    "emmanuel macron", "macron", "marine le pen", "le pen",
    "keir starmer", "starmer", "rishi sunak", "sunak",
    "vladimir putin", "putin", "volodymyr zelensky", "zelensky",
    "benjamin netanyahu", "netanyahu",
    "javier milei", "milei", "lula", "bolsonaro",
    # Colombian 2026 cycle — without these, the cross-ref dashboard
    # mis-pairs "round 1" contracts against "overall winner" contracts
    # because neither candidate name was in the curated set, so the
    # candidate-mismatch reject couldn't fire.
    "claudia lópez", "claudia lopez", "sergio fajardo",
    "gustavo petro", "petro", "iván duque", "ivan duque",
    "rodolfo hernández", "rodolfo hernandez",
    # Other current-cycle names that show up often in PredictIt dumps
    "nicolás maduro", "nicolas maduro", "maduro",
    "javier mascherano", "xóchitl gálvez", "xochitl galvez",
    "claudia sheinbaum", "sheinbaum",
}


# The 5 helpers below are the per-cycle hotspot. ``match()`` is called
# ~750 PredictIt contracts × ~800 active Polymarket questions = ~600 000
# times per cycle, and each invocation calls all five helpers — but
# they're pure functions of a single string input. ``lru_cache``
# collapses the ~3M intra-cycle helper calls to ~1600 unique-input
# evaluations (one per distinct ``pi_text`` / ``poly_text``); cycle 2
# benefits cross-cycle too because active polymarket questions are
# stable. April 2026 soak: filter_thread cycle 240s → ~5s after
# caching, eliminating the loop_lag spikes that the worker thread had
# been producing through GIL contention.
#
# ``_extract_entities`` returns ``frozenset`` (not ``set``) so the
# shared cached instance can't be mutated by a caller. The only call
# site reads ``pi_ents & poly_ents`` which works identically on
# ``frozenset``.


@lru_cache(maxsize=4096)
def _extract_entities(text: str) -> frozenset[str]:
    if not text:
        return frozenset()
    ents: set[str] = set()
    for m in _ENTITY_RE.finditer(text):
        token = m.group(1).strip()
        if not token:
            continue
        # Strip any leading stop-words from multi-word runs so that
        # "Will Donald Trump" collapses to "donald trump" — otherwise the
        # stop prefix prevents entity overlap between sentences that
        # start with "Will ..." vs "... Donald Trump ...".
        parts = token.split()
        while len(parts) > 1 and parts[0] in _ENTITY_STOPS:
            parts.pop(0)
        if not parts:
            continue
        token = " ".join(parts)
        # Drop generic single-word stops; keep multi-word always.
        if " " not in token and token in _ENTITY_STOPS:
            continue
        ents.add(token.lower())
    return frozenset(ents)


@lru_cache(maxsize=4096)
def _detect_country(text: str) -> str | None:
    if not text:
        return None
    haystack = f" {text.lower()} "
    for country, aliases in _COUNTRIES:
        for alias in aliases:
            if alias in haystack:
                return country
    return None


@lru_cache(maxsize=4096)
def _detect_election_type(text: str) -> str | None:
    if not text:
        return None
    haystack = text.lower()
    for etype, synonyms in _ELECTION_TYPES.items():
        for syn in synonyms:
            # Word-boundary check so "president" doesn't collide with
            # "presidential" duplicates (we return the first hit bucket).
            if re.search(rf"\b{re.escape(syn)}\b", haystack):
                return etype
    return None


@lru_cache(maxsize=4096)
def _detect_candidate(text: str) -> str | None:
    if not text:
        return None
    haystack = text.lower()
    # Longest-first so "donald trump" is preferred over bare "trump".
    for name in sorted(_CANDIDATE_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", haystack):
            return name
    return None


@lru_cache(maxsize=4096)
def _detect_phase(text: str) -> str | None:
    """Return the electoral phase the text is asking about, or None.

    Normalises whitespace so "round   1" and "round 1" resolve the same,
    and runs substring checks against the lower-cased haystack since most
    phase phrases ("first round", "primera vuelta") span word boundaries
    we'd otherwise need a multi-token regex for.
    """
    if not text:
        return None
    haystack = re.sub(r"\s+", " ", text.lower())
    for phase, patterns in _PHASE_PATTERNS:
        for p in patterns:
            if p in haystack:
                return phase
    return None


def match(pi_text: str, poly_text: str) -> tuple[bool, str]:
    """Decide whether two questions refer to the same event.

    Returns (True, reason) on accept, (False, reason) on reject. The
    reason string is for audit logs so we can see *why* a borderline
    pair was allowed or dropped.
    """
    pi_cand = _detect_candidate(pi_text)
    poly_cand = _detect_candidate(poly_text)
    if pi_cand and poly_cand and pi_cand != poly_cand:
        return False, f"different candidates: {pi_cand!r} vs {poly_cand!r}"

    pi_country = _detect_country(pi_text)
    poly_country = _detect_country(poly_text)
    pi_etype = _detect_election_type(pi_text)
    poly_etype = _detect_election_type(poly_text)
    pi_phase = _detect_phase(pi_text)
    poly_phase = _detect_phase(poly_text)

    if pi_country and poly_country and pi_country != poly_country:
        return False, f"different countries: {pi_country} vs {poly_country}"

    # Phase mismatch — the reason this matcher added the concept.
    # Two cases to reject:
    #   (i) BOTH sides declare a phase and they disagree.
    #  (ii) ONE side names a specific round/runoff phase while the other
    #       is silent. PredictIt presidential-market contracts are
    #       semantically "overall winner" when no phase keyword is
    #       present — pairing one of those against a Polymarket "round
    #       1 winner" contract is exactly the false positive that
    #       produced the +0.989 Colombian-election divergence spam.
    _specific = {"round_1", "round_2", "runoff"}
    if pi_phase and poly_phase and pi_phase != poly_phase:
        return False, f"different phases: {pi_phase} vs {poly_phase}"
    if (pi_phase in _specific) != (poly_phase in _specific):
        # XOR: exactly one side asks about a specific round/runoff.
        declared = pi_phase if pi_phase in _specific else poly_phase
        return False, (
            f"one side asks about phase={declared}, the other is silent "
            f"(implicit overall) — questions aren't the same"
        )

    # Strong anchor: same country + same etype + no phase conflict.
    if (
        pi_etype and poly_etype and pi_etype == poly_etype
        and pi_country and poly_country and pi_country == poly_country
    ):
        reason = f"election_type={pi_etype} country={pi_country}"
        if pi_phase or poly_phase:
            reason += f" phase={pi_phase or poly_phase}"
        return True, reason

    # Fallback: ≥2 shared entities. Still strict because candidate-name
    # differences above already rejected the hard cases; the phase check
    # above keeps round-vs-overall pairs out.
    pi_ents = _extract_entities(pi_text)
    poly_ents = _extract_entities(poly_text)
    shared = pi_ents & poly_ents
    if len(shared) >= 2:
        # Extra guard: when only one side specifies a phase and that
        # side is a specific round (round_1 / round_2 / runoff), we
        # require the other side to ALSO name the candidate — otherwise
        # a generic PredictIt "wins election" contract still sneaks
        # through as a match for a specific-round Polymarket contract.
        specific_phases = {"round_1", "round_2", "runoff"}
        phase = pi_phase or poly_phase
        if phase in specific_phases and not (pi_cand and poly_cand):
            return False, (
                f"one side names phase={phase} but candidate not confirmed "
                f"on both sides (shared={sorted(shared)})"
            )
        return True, f"shared_entities={sorted(shared)}"

    return False, (
        f"insufficient overlap (shared={len(shared)} "
        f"etype={pi_etype}/{poly_etype} country={pi_country}/{poly_country} "
        f"phase={pi_phase}/{poly_phase})"
    )
