"""Regression pins on `config/config.yaml` default values.

Some config keys aren't safe to silently revert during a rebase or
config rewrite — they encode a non-obvious operational decision and
the cost of regressing is a quiet runtime degradation rather than a
loud failure. Each pin here:

  - asserts the live default in ``config/config.yaml``
  - cites the soak-evidence comment block in the YAML for context
  - explains in the failure message why the value matters

Adding a new pin: keep tests narrow (one key per test), point at
the comment block in the YAML, and write the failure message so a
future operator who hits it knows whether their change is
intentional or a rebase mistake.
"""

from __future__ import annotations


def test_default_ollama_fast_timeout_seconds_is_twenty():
    """Pin ``ollama.fast_timeout_seconds`` at 20 so a future rebase
    can't silently revert to the 10s value that produced
    ``ollama timeout (>10.0s)`` warnings on the scalping rescore
    path during the April 2026 soak.

    20s matches deep/validator tiers and leaves ~10s of margin
    over qwen2.5:7b's healthy ~10s end-to-end latency
    (1-3s first-token + 300 tokens / 30-50 tok/s). The 10s value
    was structurally at the edge of the GPU's natural response
    time, so tail-variance overruns of 0.5-1s were producing
    intermittent timeouts on healthy calls — a config-shaped flap,
    not an Ollama hang. See the comment block above
    ``fast_timeout_seconds`` in ``config/config.yaml`` for the
    full history (45/60/60 → 10/20/20 → 20/20/20)."""
    from core.utils.config import get_config
    cfg = get_config().get("ollama") or {}
    assert cfg.get("fast_timeout_seconds") == 20, (
        f"ollama.fast_timeout_seconds defaulted to "
        f"{cfg.get('fast_timeout_seconds')!r}, expected 20. See the "
        "comment block above the key in config/config.yaml — values "
        "below 15 trigger intermittent timeout fallbacks on healthy "
        "Ollama calls because qwen2.5:7b's natural end-to-end "
        "latency is 9-10s with no tail-variance margin."
    )
