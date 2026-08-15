"""طبقة العلم 1–9: نموذج شرطي + معايرة + WF + drift + holdout مجمّد."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import (
    evaluate_calibration,
    evaluate_calibration_by_outcome,
)
from nq.auction_behavior.conditional import (
    ConditionalModel,
    fit_conditional_models,
    predict_probabilities_at_states,
    score_conditional_models,
    select_feature_names,
)
from nq.auction_behavior.drift import fold_stability, measure_drift
from nq.auction_behavior.holdout import (
    FrozenHoldout,
    HoldoutEvaluation,
    carve_frozen_holdout,
    evaluate_frozen_holdout_once,
)
from nq.auction_behavior.level_flow import LEVEL_FLOW_COLUMNS
from nq.auction_behavior.memory import SEQUENCE_MEMORY_COLUMNS, list_memory_columns
from nq.auction_behavior.outcomes import (
    OUTCOME_TARGETS,
    PRIMARY_OUTCOME_TARGETS,
    SETUP_AVAILABILITY_TS,
    attach_outcome_availability_guard,
    build_labeled_outcomes,
)
from nq.auction_behavior.projection import PROJECTION_NUMERIC_COLUMNS
from nq.auction_behavior.reliability import RELIABILITY_COLUMNS
from nq.auction_behavior.state import STATE_FEATURE_COLUMNS
from nq.auction_behavior.structure import STRUCTURE_FEATURE_COLUMNS
from nq.auction_behavior.walk_forward import (
    ScienceFold,
    build_contract_aware_folds,
    build_expanding_month_folds,
    folds_to_frame,
)
from nq.contracts.temporal import AVAILABILITY_TS

_PREFERRED_FEATURES: tuple[str, ...] = (
    *STATE_FEATURE_COLUMNS,
    *STRUCTURE_FEATURE_COLUMNS,
    *PROJECTION_NUMERIC_COLUMNS,
    *SEQUENCE_MEMORY_COLUMNS,
    *LEVEL_FLOW_COLUMNS,
    *RELIABILITY_COLUMNS,
    "signal_quality",
    "signal_evidence",
    "deceptive_score",
    "real_liquidity_ratio",
)


@dataclass(frozen=True, slots=True)
class ScienceConfig:
    """إعدادات طبقة العلم (بلا تداول)."""

    outcome_window: int = 8
    n_splits: int = 4
    embargo: int = 0
    purge_samples: int = 1
    min_train_size: int = 16
    holdout_frac: float = 0.2
    l2: float = 1.0
    max_features: int = 64
    use_month_folds: bool = True
    evaluate_holdout: bool = True
    calibration_bins: int = 10
    group_col: str | None = "_behavior_story_run"


@dataclass(frozen=True, slots=True)
class BehaviorScienceReport:
    """تقرير العلم الكامل."""

    labeled: pl.DataFrame
    feature_names: tuple[str, ...]
    fold_frame: pl.DataFrame
    fold_scores: pl.DataFrame
    calibration_by_outcome: pl.DataFrame
    drift_summary: dict[str, float]
    stability: dict[str, float]
    holdout: FrozenHoldout
    holdout_eval: HoldoutEvaluation | None
    final_model: ConditionalModel | None
    #: State(t)→probs من النموذج النهائي على إطار الميزات (بلا تسميات OOS داخل p).
    state_predictions: pl.DataFrame = field(default_factory=pl.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _preferred_with_memory(frame: pl.DataFrame) -> tuple[str, ...]:
    mem = tuple(list_memory_columns(frame))
    # فضّل حالة/بنية/إسقاط ثم ذاكرة
    return tuple(dict.fromkeys([*_PREFERRED_FEATURES, *mem]))


def run_behavior_science(  # noqa: PLR0915
    blended: pl.DataFrame,
    *,
    config: ScienceConfig | None = None,
) -> BehaviorScienceReport:
    """يشغّل الخطوات 1–9 على إطار سلوك مدمج (ميزات سببية فقط)."""
    cfg = config or ScienceConfig()
    empty_holdout = carve_frozen_holdout(pl.DataFrame(), holdout_frac=cfg.holdout_frac)
    if blended.height == 0 or AVAILABILITY_TS not in blended.columns:
        return BehaviorScienceReport(
            labeled=pl.DataFrame(),
            feature_names=(),
            fold_frame=pl.DataFrame(),
            fold_scores=pl.DataFrame(),
            calibration_by_outcome=pl.DataFrame(),
            drift_summary={},
            stability={},
            holdout=empty_holdout,
            holdout_eval=None,
            final_model=None,
            state_predictions=pl.DataFrame(),
            diagnostics={"empty": True},
        )

    # 2) Outcomes + outcome_available_ts
    outcomes = build_labeled_outcomes(
        blended,
        outcome_window=cfg.outcome_window,
        group_col=cfg.group_col if cfg.group_col in blended.columns else None,
    )
    labeled = attach_outcome_availability_guard(blended, outcomes)
    if "instrument_id" not in labeled.columns and "instrument_id" in blended.columns:
        # already joined via features if present
        pass

    feature_names = select_feature_names(
        labeled,
        preferred=_preferred_with_memory(blended),
        max_features=cfg.max_features,
    )

    # 9) carve frozen holdout first — development never sees it
    holdout_pack = carve_frozen_holdout(labeled, holdout_frac=cfg.holdout_frac)
    develop = holdout_pack.develop

    # 7) Walk-forward on develop only
    folds: list[ScienceFold] = []
    if cfg.use_month_folds:
        folds = build_expanding_month_folds(
            develop,
            ts_col=SETUP_AVAILABILITY_TS,
            min_train_months=1,
            embargo_ns=int(cfg.embargo),
        )
    if not folds:
        folds = build_contract_aware_folds(
            develop,
            ts_col=SETUP_AVAILABILITY_TS,
            n_splits=cfg.n_splits,
            embargo=cfg.embargo,
            purge_samples=cfg.purge_samples,
            min_train_size=cfg.min_train_size,
        )

    fold_score_rows: list[pl.DataFrame] = []
    fold_metric_rows: list[dict[str, float | int | str]] = []
    drift_psis: list[float] = []

    for sf in folds:
        train = develop[sf.train_idx]
        test = develop[sf.test_idx]
        model = fit_conditional_models(
            train,
            feature_names=feature_names,
            outcomes=OUTCOME_TARGETS,
            train_end_ts=sf.train_end_ts,
            l2=cfg.l2,
            min_train=max(8, cfg.min_train_size // 2),
        )
        scored = score_conditional_models(
            model,
            develop,
            test_start_ts=sf.test_start_ts,
            test_end_ts=sf.test_end_ts,
            embargo=float(cfg.embargo),
        )
        cal = evaluate_calibration(scored, n_bins=cfg.calibration_bins)
        cal_by = evaluate_calibration_by_outcome(scored, n_bins=cfg.calibration_bins)
        drift = measure_drift(
            train.select([c for c in feature_names if c in train.columns]),
            test.select([c for c in feature_names if c in test.columns]),
            feature_names=feature_names,
            ref_outcomes=train,
            cmp_outcomes=test,
            ref_calibration_ece=0.0,
            cmp_calibration_ece=cal.ece,
        )
        drift_psis.append(drift.mean_psi)
        fold_metric_rows.append(
            {
                "fold": sf.fold,
                "segment": sf.segment,
                "train_n": int(train.height),
                "test_n": int(test.height),
                "train_end_ts": sf.train_end_ts,
                "test_start_ts": sf.test_start_ts,
                "test_end_ts": sf.test_end_ts,
                "brier": cal.brier,
                "ece": cal.ece,
                "mae": cal.mae,
                "mean_psi": drift.mean_psi,
                "outcome_rate_l1": drift.outcome_rate_l1,
                "n_calibration_rows": int(cal_by.height),
            }
        )
        if scored.height:
            fold_score_rows.append(scored.with_columns(pl.lit(sf.fold).alias("fold")))

    fold_frame = pl.DataFrame(fold_metric_rows) if fold_metric_rows else folds_to_frame(folds)
    fold_scores = (
        pl.concat(fold_score_rows, how="diagonal_relaxed") if fold_score_rows else pl.DataFrame()
    )
    calibration_by_outcome = evaluate_calibration_by_outcome(
        fold_scores, n_bins=cfg.calibration_bins
    )
    stability = fold_stability(fold_frame, column="ece")
    drift_summary = {
        "mean_psi_across_folds": float(np.mean(drift_psis)) if drift_psis else 0.0,
        "max_psi_across_folds": float(np.max(drift_psis)) if drift_psis else 0.0,
        "n_folds": float(len(folds)),
    }

    # Final model on full develop (still excluding frozen holdout)
    final_model: ConditionalModel | None = None
    holdout_eval: HoldoutEvaluation | None = None
    holdout_state = holdout_pack
    state_predictions = pl.DataFrame()
    if develop.height >= cfg.min_train_size and feature_names:
        max_dev = develop[SETUP_AVAILABILITY_TS].max()
        assert max_dev is not None
        final_train_end = int(np.asarray(max_dev).item())
        final_model = fit_conditional_models(
            develop,
            feature_names=feature_names,
            outcomes=OUTCOME_TARGETS,
            train_end_ts=final_train_end,
            l2=cfg.l2,
            min_train=max(8, cfg.min_train_size // 2),
        )
        # تنبؤ على كل صف حالة: الميزات فقط — لا تُمرَّر تسميات y إلى predict
        state_predictions = predict_probabilities_at_states(
            final_model,
            blended,
            outcomes=PRIMARY_OUTCOME_TARGETS + tuple(
                o for o in OUTCOME_TARGETS if o not in PRIMARY_OUTCOME_TARGETS
            ),
        )
        if cfg.evaluate_holdout and holdout_pack.holdout.height > 0:
            ho_min = holdout_pack.holdout[SETUP_AVAILABILITY_TS].min()
            ho_max = holdout_pack.holdout[SETUP_AVAILABILITY_TS].max()
            assert ho_min is not None and ho_max is not None
            scored_ho = score_conditional_models(
                final_model,
                holdout_pack.holdout,
                test_start_ts=int(np.asarray(ho_min).item()),
                test_end_ts=int(np.asarray(ho_max).item()),
                embargo=0.0,
            )
            holdout_eval, holdout_state = evaluate_frozen_holdout_once(
                holdout_pack, scored_ho, allow_retouch=False
            )

    return BehaviorScienceReport(
        labeled=labeled,
        feature_names=feature_names,
        fold_frame=fold_frame,
        fold_scores=fold_scores,
        calibration_by_outcome=calibration_by_outcome,
        drift_summary=drift_summary,
        stability=stability,
        holdout=holdout_state,
        holdout_eval=holdout_eval,
        final_model=final_model,
        state_predictions=state_predictions,
        diagnostics={
            "n_labeled": int(labeled.height),
            "n_develop": int(develop.height),
            "n_holdout": int(holdout_pack.holdout.height),
            "holdout_cut_ts": int(holdout_pack.cut_ts),
            "holdout_touched": bool(holdout_state.touched),
            "n_features": len(feature_names),
            "n_folds": len(folds),
            "primary_outcomes": list(PRIMARY_OUTCOME_TARGETS),
            "signal_quality_is_calibrated_probability": False,
            "prediction_uses_oos_labels": False,
            "science_steps": (
                "conditional_model_state_to_probs",
                "outcomes_outcome_available_ts",
                "structure_features",
                "market_memory_sequence",
                "level_anchored_order_flow",
                "reliability_evidence_no_delete",
                "asia_london_projection_state",
                "calibration_ece_brier",
                "walk_forward_multi_segment",
                "drift_stability",
                "frozen_final_holdout",
            ),
        },
    )


__all__ = [
    "BehaviorScienceReport",
    "ScienceConfig",
    "run_behavior_science",
]
