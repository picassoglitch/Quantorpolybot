"""Trust-tier classifier for Pattern Discovery PR #1.

Pure function over ``SourceMetrics``. No DB writes. No effects on
trading behavior — the bot doesn't read this. The CLI shows the
output for the operator to act on manually.

Tier definitions (v1):

  - ``NEW``                 sample_size below the minimum trust threshold;
                            we don't have enough data to score reliably.
  - ``BLACKLIST``           hit_rate is poor AND false_positive rate is
                            high AND sample_size is large enough to be
                            confident. Operator should consider
                            deprioritizing this source.
  - ``NOISY``               false_positive_rate is high but hit_rate is
                            inconclusive — the source produces signal
                            but most of it doesn't move the market.
  - ``LATE_CONFIRMATION``   hit_rate is decent but the market moved
                            BEFORE we logged the signal — i.e. the
                            source is reporting after-the-fact. We
                            approximate this when avg_move_15m_abs is
                            much larger than avg_move_5m_abs (most of
                            the move had already happened by minute 5).
                            Documented limitation: a true lead-time
                            calculation needs price-BEFORE comparison
                            which is a follow-up.
  - ``TRUSTED``             hit_rate is high AND sample_size is well
                            above the trust threshold AND there is
                            meaningful 5m movement after the signal.
  - ``WATCH``               everything else — sample is large enough
                            to score but the source hasn't earned
                            TRUSTED yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.analytics.aggregations import SourceMetrics


# Defaults — tuned conservatively. Operator can override via the CLI
# flag if they want to relax/tighten the bands.
_DEFAULT_MIN_SAMPLE_NEW = 20            # below this → NEW
_DEFAULT_MIN_SAMPLE_TRUSTED = 50        # at least this for TRUSTED
_DEFAULT_TRUSTED_HIT_RATE = 0.65        # at least this hit_rate for TRUSTED
_DEFAULT_TRUSTED_AVG_MOVE = 0.005       # at least this avg 5m move (0.5c)
_DEFAULT_BLACKLIST_HIT_RATE = 0.30      # below this → BLACKLIST candidate
_DEFAULT_BLACKLIST_FP_RATE = 0.60       # above this → BLACKLIST candidate
_DEFAULT_NOISY_FP_RATE = 0.50           # above this → NOISY candidate
_DEFAULT_LATE_CONFIRMATION_RATIO = 1.50  # avg_15m / avg_5m

# Fixed tier names — the CLI prints these literally.
TIER_NEW = "NEW"
TIER_WATCH = "WATCH"
TIER_TRUSTED = "TRUSTED"
TIER_LATE_CONFIRMATION = "LATE_CONFIRMATION"
TIER_NOISY = "NOISY"
TIER_BLACKLIST = "BLACKLIST"

# Order matters in `classify_tier`: more specific tiers come first.
_TIER_ORDER = (
    TIER_NEW, TIER_BLACKLIST, TIER_LATE_CONFIRMATION,
    TIER_NOISY, TIER_TRUSTED, TIER_WATCH,
)


@dataclass(frozen=True)
class TrustTierConfig:
    """Knobs for the classifier — exposed so the CLI can pass
    operator-provided overrides without modifying global state."""
    min_sample_new: int = _DEFAULT_MIN_SAMPLE_NEW
    min_sample_trusted: int = _DEFAULT_MIN_SAMPLE_TRUSTED
    trusted_hit_rate: float = _DEFAULT_TRUSTED_HIT_RATE
    trusted_avg_move: float = _DEFAULT_TRUSTED_AVG_MOVE
    blacklist_hit_rate: float = _DEFAULT_BLACKLIST_HIT_RATE
    blacklist_fp_rate: float = _DEFAULT_BLACKLIST_FP_RATE
    noisy_fp_rate: float = _DEFAULT_NOISY_FP_RATE
    late_confirmation_ratio: float = _DEFAULT_LATE_CONFIRMATION_RATIO


@dataclass(frozen=True)
class TrustTierAssignment:
    """One row's tier assignment + the metrics that drove it. The
    CLI prints both columns so the operator can audit."""
    source: str
    tier: str
    sample_size: int
    hit_rate: float | None
    false_positive_rate: float | None
    avg_move_5m_abs: float | None
    avg_move_15m_abs: float | None
    reasoning: str


def classify_tier(
    metrics: SourceMetrics,
    cfg: TrustTierConfig | None = None,
) -> TrustTierAssignment:
    """Pure classifier. Order:

      1. NEW — too-small sample
      2. BLACKLIST — clearly bad AND large enough sample
      3. LATE_CONFIRMATION — accurate but movement preceded the signal
      4. NOISY — high false-positive rate
      5. TRUSTED — high hit rate AND meaningful 5m move AND sample
      6. WATCH — fallback for "scoreable but unproven"

    Returns ``TrustTierAssignment`` with the reasoning string so the
    CLI can show WHY each source landed in its tier.
    """
    cfg = cfg or TrustTierConfig()

    # 1. NEW
    if metrics.sample_size < cfg.min_sample_new:
        return _assign(
            metrics, TIER_NEW,
            f"sample_size {metrics.sample_size} < min_sample_new "
            f"{cfg.min_sample_new}",
        )

    # 2. BLACKLIST — must have hit_rate AND false-positive data.
    if (
        metrics.hit_rate is not None
        and metrics.hit_rate < cfg.blacklist_hit_rate
        and metrics.false_positive_rate is not None
        and metrics.false_positive_rate >= cfg.blacklist_fp_rate
        and metrics.sample_size >= cfg.min_sample_trusted
    ):
        return _assign(
            metrics, TIER_BLACKLIST,
            f"hit_rate {metrics.hit_rate:.2f} < "
            f"{cfg.blacklist_hit_rate:.2f} AND "
            f"false_positive_rate {metrics.false_positive_rate:.2f} "
            f">= {cfg.blacklist_fp_rate:.2f}",
        )

    # 3. LATE_CONFIRMATION — approximated as avg_15m_move >> avg_5m_move,
    #    meaning most of the move had happened by minute 5 (i.e. the
    #    signal trailed the move). Only meaningful when both metrics
    #    are populated.
    if (
        metrics.avg_move_5m_abs is not None
        and metrics.avg_move_15m_abs is not None
        and metrics.avg_move_5m_abs > 0
        and metrics.avg_move_15m_abs / max(metrics.avg_move_5m_abs, 1e-6)
            >= cfg.late_confirmation_ratio
        and (metrics.hit_rate is None or metrics.hit_rate >= 0.50)
    ):
        return _assign(
            metrics, TIER_LATE_CONFIRMATION,
            f"avg_15m/avg_5m = "
            f"{metrics.avg_move_15m_abs / max(metrics.avg_move_5m_abs, 1e-6):.2f}"
            f" >= {cfg.late_confirmation_ratio:.2f} (signal may trail move)",
        )

    # 4. NOISY
    if (
        metrics.false_positive_rate is not None
        and metrics.false_positive_rate >= cfg.noisy_fp_rate
    ):
        return _assign(
            metrics, TIER_NOISY,
            f"false_positive_rate {metrics.false_positive_rate:.2f} "
            f">= {cfg.noisy_fp_rate:.2f}",
        )

    # 5. TRUSTED — needs all of: large sample, high hit_rate, real
    #    5m move.
    if (
        metrics.sample_size >= cfg.min_sample_trusted
        and metrics.hit_rate is not None
        and metrics.hit_rate >= cfg.trusted_hit_rate
        and metrics.avg_move_5m_abs is not None
        and metrics.avg_move_5m_abs >= cfg.trusted_avg_move
    ):
        return _assign(
            metrics, TIER_TRUSTED,
            f"sample_size {metrics.sample_size} >= "
            f"{cfg.min_sample_trusted} AND hit_rate "
            f"{metrics.hit_rate:.2f} >= {cfg.trusted_hit_rate:.2f} "
            f"AND avg_5m_move {metrics.avg_move_5m_abs:.4f} >= "
            f"{cfg.trusted_avg_move:.4f}",
        )

    # 6. WATCH — fallback
    return _assign(
        metrics, TIER_WATCH,
        f"sample_size {metrics.sample_size} >= {cfg.min_sample_new} "
        f"but missing TRUSTED criteria",
    )


def classify_all(
    metrics_list: list[SourceMetrics],
    cfg: TrustTierConfig | None = None,
) -> list[TrustTierAssignment]:
    """Classify every source in one call. Returns assignments
    ordered by tier severity (BLACKLIST first, NEW last)."""
    cfg = cfg or TrustTierConfig()
    assignments = [classify_tier(m, cfg) for m in metrics_list]
    severity = {t: i for i, t in enumerate((
        TIER_BLACKLIST, TIER_NOISY, TIER_LATE_CONFIRMATION,
        TIER_WATCH, TIER_TRUSTED, TIER_NEW,
    ))}
    assignments.sort(key=lambda a: (severity.get(a.tier, 99), -a.sample_size))
    return assignments


def _assign(m: SourceMetrics, tier: str, reasoning: str) -> TrustTierAssignment:
    return TrustTierAssignment(
        source=m.source, tier=tier, sample_size=m.sample_size,
        hit_rate=m.hit_rate, false_positive_rate=m.false_positive_rate,
        avg_move_5m_abs=m.avg_move_5m_abs,
        avg_move_15m_abs=m.avg_move_15m_abs,
        reasoning=reasoning,
    )
