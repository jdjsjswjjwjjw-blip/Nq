"""التمثيلات الكامنة / حالات السوق (Latent Market States / Regimes) — المحطة 5.

تُحوَّل التمثيلات الكامنة (مخرجات ``nq.models`` المشفّر) إلى حالات سوقية discrete
قابلة للتفسير (regimes)، مع:

* ``KMeansRegimes`` — تجميع حتمي (fit-on-train ثم توسيم للأمام، بلا تسريب).
* ``regime_labels_frame`` — توسيم بطوابع زمنية سليمة (point-in-time).
* ``transition_matrix`` / ``dwell_times`` — ديناميكية الحالات السببية.
* ``silhouette_score`` / ``regime_summary`` — الاستقرار/الجودة والتفسير.
"""

from __future__ import annotations

from nq.states.regimes import (
    MARKET_REGIME_FEATURE_NAMES,
    CausalRegimeTracker,
    KMeansRegimes,
    dwell_times,
    heuristic_market_phase,
    infer_market_phase_map,
    regime_labels_frame,
    regime_summary,
    silhouette_score,
    transition_matrix,
)

__all__ = [
    "MARKET_REGIME_FEATURE_NAMES",
    "CausalRegimeTracker",
    "KMeansRegimes",
    "dwell_times",
    "heuristic_market_phase",
    "infer_market_phase_map",
    "regime_labels_frame",
    "regime_summary",
    "silhouette_score",
    "transition_matrix",
]
