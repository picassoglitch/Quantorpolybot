"""Breaking Event Scout — Step #3.

Detects real-world breaking events from public news sources, normalizes
them into ``Event`` objects, maps them to relevant Polymarket markets,
estimates probability impact heuristically, and generates SHADOW trade
candidates.

PR #1 ships the GDELT-only end-to-end vertical slice. NewsAPI / SerpAPI
/ Kalshi / Trading Economics / Alpha Vantage / Google Trends / sports
APIs / Truth Social land in subsequent PRs against the same Signal /
Event / Mapper / Impact / Candidate interfaces.

SHADOW only — no real orders are produced by this lane.
"""
