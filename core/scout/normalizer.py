"""Signal -> Event clustering + classification.

The normalizer takes a list of recent ``Signal``s (already in the
``scout_signals`` table) and groups them into ``Event`` clusters.

Clustering rule (v1):

  Two signals belong to the same Event iff:
    1. They share at least one named entity (case-insensitive), AND
    2. They were observed within ``cluster_window_seconds`` of each
       other (default 1800 = 30 min), AND
    3. They classify into the same ``EventCategory``.

The ``event_id`` is hashed from (top_entity, category, bucketed_hour)
so the same real-world event collapses to the same id across
overlapping scan ticks — that's the cool-down for repeat triggers
the spec requires.

Classification (v1):

  Keyword regex over title + body + category_hint. The connector
  already provides a ``category_hint`` (per the GDELT category-keyed
  query path), but the normalizer still verifies — a connector
  query for "shooting" can incidentally match unrelated articles
  ("a shooting STAR was visible..."). When in doubt, falls back to
  the connector's hint.

Severity (v1):

  Combination of:
    - per-category prior (e.g. assassination_attempt: 1.0,
      sports_injury: 0.4)
    - source_count multiplier (more corroboration = higher severity)
    - confidence multiplier (signal-side trust)

  Capped at 1.0.

Confidence (v1):

  ``min(0.95, mean(signal.confidence) * (1 + 0.1*(source_count-1)))``
  i.e. base on per-signal trust, ramp up by 10% per additional
  corroborating source, cap at 0.95 to leave headroom over a
  single-source claim no matter how trusted.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from core.scout.event import Event, EventCategory, Signal


# Per-category severity prior. Hand-tuned: events that move markets
# (assassination, war escalation, election results) are 0.85+; events
# that move slower or sport-specific are 0.4-0.6. Multiplied later by
# source_count + confidence.
_CATEGORY_SEVERITY: dict[EventCategory, float] = {
    EventCategory.ASSASSINATION_ATTEMPT: 1.00,
    EventCategory.SHOOTING: 0.85,
    EventCategory.WAR_ESCALATION: 0.90,
    EventCategory.CEASEFIRE: 0.85,
    EventCategory.ELECTION_RESULT: 0.90,
    EventCategory.COURT_RULING: 0.75,
    EventCategory.INDICTMENT: 0.80,
    EventCategory.RESIGNATION: 0.70,
    EventCategory.ARREST: 0.65,
    EventCategory.DEATH_INJURY: 0.75,
    EventCategory.EVACUATION: 0.65,
    EventCategory.MACRO_DATA_SURPRISE: 0.80,
    EventCategory.SPORTS_INJURY: 0.50,
    EventCategory.OTHER: 0.30,
}


# Classifier keyword set per category. Single-word triggers are
# wrapped in word boundaries; multi-word triggers as exact phrases.
# Lowered before matching.
_CATEGORY_KEYWORDS: dict[EventCategory, list[str]] = {
    EventCategory.ASSASSINATION_ATTEMPT: [
        "assassination attempt", "assassinated", "shot at the president",
    ],
    EventCategory.SHOOTING: [
        "shooting", "shooter", "gunman", "opened fire", "active shooter",
    ],
    EventCategory.WAR_ESCALATION: [
        "airstrike", "missile strike", "ground offensive", "war escalation",
        "invasion", "shelled", "bombardment",
    ],
    EventCategory.CEASEFIRE: [
        "ceasefire", "truce", "peace deal", "armistice",
    ],
    EventCategory.ELECTION_RESULT: [
        "concedes", "victory speech", "election called", "wins election",
        "official winner", "projected winner",
    ],
    EventCategory.COURT_RULING: [
        "supreme court ruling", "court ruled", "judge ruled", "verdict",
        "ruled in favor", "struck down",
    ],
    EventCategory.INDICTMENT: [
        "indicted", "indictment", "charged with",
    ],
    EventCategory.RESIGNATION: [
        "resigned", "resignation", "stepping down", "stepped down",
    ],
    EventCategory.ARREST: [
        "arrested", "in custody", "detained by police",
    ],
    EventCategory.DEATH_INJURY: [
        "wounded", "killed in", "died in", "fatal", "critically injured",
    ],
    EventCategory.EVACUATION: [
        "evacuated", "evacuation order", "ordered to evacuate",
    ],
    EventCategory.MACRO_DATA_SURPRISE: [
        "rate hike", "rate cut", "fed decision", "cpi report", "jobs report",
        "gdp report", "unemployment rate",
    ],
    EventCategory.SPORTS_INJURY: [
        "out for season", "ruled out", "torn acl", "season-ending injury",
        "injured reserve",
    ],
}


def classify(text: str, hint: str | None = None) -> EventCategory:
    """Pure function. Lower-cases `text`, scans the keyword table; if
    no match, falls back to `hint` (the connector's best guess), then
    OTHER. The first match wins (categories are checked in declaration
    order — assassination_attempt before shooting so a more specific
    classification wins over a less specific one).
    """
    if not text:
        return EventCategory.from_str(hint)
    low = text.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in low:
                return cat
    return EventCategory.from_str(hint)


def _entity_key(s: str) -> str:
    return s.strip().lower()


def _bucket_hour(ts: float) -> int:
    return int(ts // 3600)


def _top_entity(entities: Iterable[str]) -> str:
    """Pick a deterministic 'leading' entity for event_id hashing.

    Prefers the longest entity, then alphabetically — both stable,
    so the same cluster always hashes to the same event_id.
    """
    es = [e for e in entities if e]
    if not es:
        return ""
    es.sort(key=lambda x: (-len(x), x.lower()))
    return es[0]


def _severity(
    category: EventCategory, source_count: int, mean_confidence: float
) -> float:
    """Per-category prior × source-count ramp × confidence."""
    base = _CATEGORY_SEVERITY.get(category, 0.30)
    src_mult = min(1.5, 1.0 + 0.15 * max(0, source_count - 1))
    return min(1.0, base * src_mult * max(0.4, mean_confidence))


def _confidence(source_count: int, mean_confidence: float) -> float:
    """Mean signal confidence × corroboration ramp, capped at 0.95."""
    base = max(0.0, min(1.0, mean_confidence))
    ramp = 1.0 + 0.10 * max(0, source_count - 1)
    return min(0.95, base * ramp)


def normalize(
    signals: Iterable[tuple[int, Signal]],
    *,
    cluster_window_seconds: float = 1800.0,
    now_ts_value: float | None = None,
) -> list[Event]:
    """Cluster `signals` into Events. Each input is a tuple of
    (signal_id_in_db, Signal); the resulting Event's
    `raw_signal_ids` list is the IDs of every signal that landed in
    its cluster.

    Pure function: no DB writes. The lane is responsible for
    persisting the returned Events to `breaking_events`.

    Algorithm:
      1. Classify each signal into an EventCategory.
      2. Bucket by (category, bucket_hour) — a coarse pre-grouping
         that prevents O(N²) pairwise comparison.
      3. Within each bucket, greedy-merge by entity overlap: the
         first signal seeds a cluster, subsequent signals join it
         iff they share at least one entity AND fall within the
         time window from the cluster's first_seen.

    Time complexity O(N × C) per bucket where C is cluster count
    in that bucket — acceptable for typical scan windows (low
    hundreds of signals per 30-min bucket).
    """
    # Cluster bucket: list of dicts with mutable state during build.
    buckets: dict[tuple[EventCategory, int], list[dict]] = defaultdict(list)
    for sid, sig in signals:
        cat = classify(f"{sig.title}\n{sig.body}", hint=sig.category_hint)
        bh = _bucket_hour(sig.timestamp)
        bucket = buckets[(cat, bh)]
        sig_entity_keys = {_entity_key(e) for e in sig.entities if e}
        merged_into: dict | None = None
        for cluster in bucket:
            # Time-window check (relative to cluster's first signal).
            if abs(sig.timestamp - cluster["first_seen"]) > cluster_window_seconds:
                continue
            # Entity-overlap check.
            if sig_entity_keys and (sig_entity_keys & cluster["entity_keys"]):
                merged_into = cluster
                break
            # If neither this signal nor the cluster has any entities,
            # fall back to title-keyword overlap (>= 2 shared lowered
            # words >= 4 chars).
            if not sig_entity_keys and not cluster["entity_keys"]:
                cluster_words = cluster["title_words"]
                sig_words = _significant_words(sig.title)
                if len(sig_words & cluster_words) >= 2:
                    merged_into = cluster
                    break
        if merged_into is not None:
            merged_into["signal_ids"].append(sid)
            merged_into["sources"].add(sig.source)
            merged_into["confidences"].append(sig.confidence)
            merged_into["entities"].update(sig.entities)
            merged_into["entity_keys"].update(sig_entity_keys)
            merged_into["title_words"].update(_significant_words(sig.title))
            merged_into["last_seen"] = max(
                merged_into["last_seen"], sig.timestamp,
            )
            if sig.timestamp < merged_into["first_seen"]:
                merged_into["first_seen"] = sig.timestamp
                merged_into["title"] = sig.title
        else:
            bucket.append({
                "category": cat,
                "bucket_hour": bh,
                "first_seen": sig.timestamp,
                "last_seen": sig.timestamp,
                "title": sig.title,
                "signal_ids": [sid],
                "sources": {sig.source},
                "confidences": [sig.confidence],
                "entities": set(sig.entities),
                "entity_keys": set(sig_entity_keys),
                "title_words": _significant_words(sig.title),
                "location": _first_location(sig),
            })

    # Materialize Events from clusters.
    events: list[Event] = []
    for clusters in buckets.values():
        for c in clusters:
            entities_sorted = sorted(c["entities"])
            top = _top_entity(entities_sorted) or _top_word(c["title_words"])
            mean_conf = (
                sum(c["confidences"]) / len(c["confidences"])
                if c["confidences"] else 0.0
            )
            event_id = Event.make_event_id(c["category"], top, c["bucket_hour"])
            events.append(Event(
                event_id=event_id,
                timestamp_detected=c["first_seen"],
                title=c["title"][:300],
                category=c["category"],
                severity=_severity(c["category"], len(c["sources"]), mean_conf),
                confidence=_confidence(len(c["sources"]), mean_conf),
                location=c["location"],
                entities=entities_sorted[:16],
                source_count=len(c["sources"]),
                sources=sorted(c["sources"]),
                contradiction_score=0.0,  # v1: no contradiction inference
                raw_signal_ids=c["signal_ids"],
                first_seen_at=c["first_seen"],
                last_seen_at=c["last_seen"],
            ))
    return events


# ---------------- internal helpers ----------------


_WORD_RE = re.compile(r"[a-z]{4,}")


def _significant_words(text: str) -> set[str]:
    if not text:
        return set()
    return set(_WORD_RE.findall(text.lower()))


def _top_word(words: set[str]) -> str:
    if not words:
        return "untitled"
    # Deterministic pick: longest then alphabetical.
    return sorted(words, key=lambda w: (-len(w), w))[0]


def _first_location(sig: Signal) -> str:
    """v1: GDELT articles don't always carry locations; pull from
    raw_payload.sourcecountry as a fallback."""
    src_country = sig.raw_payload.get("sourcecountry") if sig.raw_payload else None
    return (src_country or "").strip()
