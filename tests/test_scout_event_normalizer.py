"""Event + classifier + normalizer tests.

Pure functions — no DB, no network. Covers:
  - EventCategory.from_str round-trip and OTHER fallback
  - classify() keyword matches + hint fallback
  - Event.make_event_id determinism (same inputs -> same id)
  - normalize() clustering: entity overlap, time window, category split
  - severity / confidence ramps
"""

from __future__ import annotations

import time

from core.scout.event import Event, EventCategory, Signal
from core.scout import normalizer


def _signal(
    title: str,
    *,
    body: str = "",
    entities: list[str] | None = None,
    timestamp: float | None = None,
    source: str = "gdelt",
    confidence: float = 0.7,
    category_hint: str = "shooting",
) -> Signal:
    return Signal(
        source=source,
        source_type="global_event_db",
        timestamp=timestamp if timestamp is not None else time.time(),
        title=title,
        body=body,
        entities=entities or [],
        category_hint=category_hint,
        url=f"https://example/{title[:20]}",
        confidence=confidence,
    )


# ---------------- EventCategory ----------------


def test_category_from_str_round_trips():
    for c in EventCategory:
        assert EventCategory.from_str(c.value) is c


def test_category_from_unknown_returns_other():
    assert EventCategory.from_str("not_a_real_category") is EventCategory.OTHER
    assert EventCategory.from_str(None) is EventCategory.OTHER
    assert EventCategory.from_str("") is EventCategory.OTHER


# ---------------- classify ----------------


def test_classify_shooting():
    assert normalizer.classify("Active shooter at school") is EventCategory.SHOOTING


def test_classify_assassination_beats_shooting():
    """Assassination is more specific; declaration order wins."""
    assert normalizer.classify(
        "Assassination attempt on senator; gunman opened fire"
    ) is EventCategory.ASSASSINATION_ATTEMPT


def test_classify_falls_back_to_hint():
    assert normalizer.classify(
        "Random news story about weather", hint="war_escalation",
    ) is EventCategory.WAR_ESCALATION


def test_classify_falls_back_to_other_when_no_hint():
    assert normalizer.classify("Weather is fine") is EventCategory.OTHER


def test_classify_ceasefire():
    assert normalizer.classify(
        "Ceasefire announced between warring parties"
    ) is EventCategory.CEASEFIRE


# ---------------- Event.make_event_id ----------------


def test_event_id_is_deterministic():
    a = Event.make_event_id(EventCategory.SHOOTING, "Trump", 12345)
    b = Event.make_event_id(EventCategory.SHOOTING, "trump", 12345)
    assert a == b  # case-insensitive


def test_event_id_changes_with_inputs():
    a = Event.make_event_id(EventCategory.SHOOTING, "Trump", 12345)
    b = Event.make_event_id(EventCategory.SHOOTING, "Trump", 12346)
    c = Event.make_event_id(EventCategory.WAR_ESCALATION, "Trump", 12345)
    assert a != b
    assert a != c


# ---------------- normalize() ----------------


def test_normalize_two_signals_same_entity_collapse_into_one_event():
    now = 1_700_000_000.0
    s1 = _signal(
        "White House Correspondents' Dinner shooting reported",
        entities=["White House", "WHCD"],
        timestamp=now,
        source="gdelt",
    )
    s2 = _signal(
        "Suspect detained after shots fired at WHCD venue",
        entities=["WHCD", "Secret Service"],
        timestamp=now + 60,
        source="gdelt",
    )
    events = normalizer.normalize([(1, s1), (2, s2)])
    assert len(events) == 1
    ev = events[0]
    assert ev.category is EventCategory.SHOOTING
    assert ev.source_count == 1  # both from gdelt -> one source
    assert {1, 2} == set(ev.raw_signal_ids)
    assert "WHCD" in ev.entities or "Whcd" in ev.entities


def test_normalize_distinct_entities_split_into_two_events():
    now = 1_700_000_000.0
    s1 = _signal(
        "Trump survives shooting at Pittsburgh rally",
        entities=["Trump", "Pittsburgh"],
        timestamp=now,
    )
    s2 = _signal(
        "Different shooting at unrelated mall in Houston",
        entities=["Houston Mall"],
        timestamp=now + 30,
    )
    events = normalizer.normalize([(1, s1), (2, s2)])
    assert len(events) == 2


def test_normalize_outside_time_window_splits_clusters():
    now = 1_700_000_000.0
    s1 = _signal(
        "Trump shooting", entities=["Trump"], timestamp=now,
    )
    s2 = _signal(
        "Trump shooting (delayed coverage)",
        entities=["Trump"],
        timestamp=now + 6 * 3600,  # 6h later, way outside default 30min window
    )
    events = normalizer.normalize(
        [(1, s1), (2, s2)], cluster_window_seconds=1800.0,
    )
    assert len(events) == 2


def test_normalize_severity_climbs_with_source_count():
    now = 1_700_000_000.0
    one_source = [(i, _signal("Assassination attempt", entities=["Trump"],
                                timestamp=now + i, source="gdelt"))
                  for i in range(1)]
    three_sources = [(i, _signal("Assassination attempt", entities=["Trump"],
                                  timestamp=now + i,
                                  source=f"src{i}"))
                     for i in range(3)]
    e1 = normalizer.normalize(one_source)[0]
    e3 = normalizer.normalize(three_sources)[0]
    assert e3.source_count == 3
    assert e1.source_count == 1
    assert e3.severity > e1.severity
    assert e3.confidence > e1.confidence


def test_normalize_confidence_capped_at_095():
    """Even with N=10 sources at 1.0 confidence each, event confidence
    must not exceed 0.95 — the spec leaves headroom over a maximally
    corroborated heuristic claim."""
    now = 1_700_000_000.0
    sigs = [
        (i, _signal(
            "Assassination attempt", entities=["Trump"],
            timestamp=now + i, source=f"src{i}", confidence=1.0,
        ))
        for i in range(10)
    ]
    e = normalizer.normalize(sigs)[0]
    assert e.confidence <= 0.95


def test_normalize_returns_empty_for_empty_input():
    assert normalizer.normalize([]) == []
