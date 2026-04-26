"""Event + Signal + EventCategory.

The two stable shapes the scout pipeline is built around:

  - ``Signal``: one normalized record from a connector (one news
    article, one GDELT row, one tweet, etc.). Connectors all return
    ``Signal`` so downstream code (normalizer, mapper) doesn't care
    where the data came from.
  - ``Event``: a cluster of ``Signal``s that together describe one
    real-world event (e.g. "shooting at White House Correspondents'
    Dinner, ~2026-04-26 02:00Z, entities=[Trump, WHCD, Secret Service]").
    Created by ``core.scout.normalizer``.

`EventCategory` is the v1 taxonomy from the user spec — 13 categories
that cover the breaking-event types we want to act on. The classifier
in ``normalizer`` maps signal text to one of these, defaulting to
``OTHER`` when nothing matches.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventCategory(str, Enum):
    """Spec'd 13-category taxonomy. String values are stored as-is in
    the ``breaking_events.category`` column for cheap WHERE/GROUP BY."""

    SHOOTING = "shooting"
    ASSASSINATION_ATTEMPT = "assassination_attempt"
    EVACUATION = "evacuation"
    DEATH_INJURY = "death_injury"
    RESIGNATION = "resignation"
    ARREST = "arrest"
    INDICTMENT = "indictment"
    WAR_ESCALATION = "war_escalation"
    CEASEFIRE = "ceasefire"
    ELECTION_RESULT = "election_result"
    COURT_RULING = "court_ruling"
    MACRO_DATA_SURPRISE = "macro_data_surprise"
    SPORTS_INJURY = "sports_injury"
    OTHER = "other"

    @classmethod
    def from_str(cls, value: str | None) -> "EventCategory":
        if not value:
            return cls.OTHER
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.OTHER


@dataclass
class Signal:
    """One normalized record from a single connector. Mirrors the
    user-spec'd Signal shape closely so future connectors can drop
    in without renaming fields.

    `confidence` is connector-side trust (e.g. GDELT articles from
    bbc.co.uk should score higher than a personal blog). Connectors
    that don't yet differentiate set 0.5.
    """

    source: str           # e.g. "gdelt", "newsapi"
    source_type: str      # e.g. "global_event_db", "news_api", "search_engine"
    timestamp: float      # ingest unix seconds
    title: str
    body: str
    entities: list[str]   # extracted named entities (people, orgs, places)
    category_hint: str    # connector's best-guess category; normalizer overrides
    url: str
    confidence: float     # 0..1 source-side trust
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def signal_hash(self) -> str:
        """Stable dedup key. Same article surfacing twice on the same
        connector collapses to one row (UNIQUE on `scout_signals.signal_hash`)."""
        h = hashlib.sha256()
        h.update(self.source.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update((self.url or self.title).encode("utf-8", errors="ignore"))
        return h.hexdigest()


@dataclass
class Event:
    """A normalized real-world event clustered from one or more Signals.

    `event_id` is a deterministic hash of (top_entity, category,
    bucketed_hour) so re-running the normalizer on the same evidence
    window produces the same id — that's how we dedupe across scan
    ticks (the spec requires a cool-down for repeat triggers).

    `severity` is in [0,1]: 1.0 = an event that should always be
    considered. Computed in the normalizer from category + source
    count + entity prominence. Lane-side gates use it together with
    `confidence` to decide whether to even map+score.
    """

    event_id: str
    timestamp_detected: float
    title: str
    category: EventCategory
    severity: float
    confidence: float
    location: str
    entities: list[str]
    source_count: int
    sources: list[str]                 # distinct source names
    contradiction_score: float         # 0..1; higher = more contradiction
    raw_signal_ids: list[int]          # IDs into scout_signals
    first_seen_at: float
    last_seen_at: float

    @staticmethod
    def make_event_id(
        category: EventCategory,
        top_entity: str,
        bucket_hour: int,
    ) -> str:
        """Deterministic id for cluster dedup. Same (category,
        top_entity, hour) bucket -> same event_id, regardless of how
        many times the normalizer is invoked.

        `bucket_hour` is unix timestamp // 3600 — events that span the
        bucket boundary will produce two ids, which is the
        conservative behaviour (the lane will see them as two events
        and decide independently; the cool-down rule then suppresses
        duplicate orders within the per-event window).
        """
        h = hashlib.sha256()
        payload = f"{category.value}|{top_entity.strip().lower()}|{bucket_hour}"
        h.update(payload.encode("utf-8"))
        return h.hexdigest()[:16]
