"""طبقة العلم: نموذج شرطي + OOF تاريخي + معايرة سببية + holdout مجمّد."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import (
    PlattCalibrator,
    apply_calibrators_by_outcome,
    apply_calibrators_to_state_predictions,
    evaluate_calibration,
    evaluate_calibration_by_outcome,
    fit_platt_calibrators_by_outcome,
)
from nq.auction_behavior.conditional import (
    ConditionalModel,
    fit_conditional_models,
    predict_probabilities_at_states,
    score_conditional_models,
    select_feature_names_by_family,
)
from nq.auction_behavior.drift import fold_stability, measure_drift
from nq.auction_behavior.holdout import (
    FrozenHoldout,
    HoldoutEvaluation,
    carve_frozen_holdout,
    evaluate_frozen_holdout_once,
)
from nq.auction_behavior.outcomes import (
    OUTCOME_TARGETS,
    PRIMARY_OUTCOME_TARGETS,
    SETUP_AVAILABILITY_TS,
    attach_outcome_availability_guard,
    build_labeled_outcomes,
    filter_outcomes_known_by,
    filter_resolved_outcomes,
)
from nq.auction_behavior.walk_forward import (
    ScienceFold,
    build_contract_aware_folds,
    build_expanding_month_folds,
    folds_to_frame,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike

_MIN_TRAIN_FOR_CAL_SPLIT = 10
_MIN_CAL_ROWS = 5


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
    evaluate_holdout: bool = False
    calibration_bins: int = 10
    min_pos: int = 3
    min_neg: int = 3
    calibration_frac: float = 0.2
    group_col: str | None = "_behavior_story_run"


@dataclass(frozen=True, slots=True)
class BehaviorScienceReport:
    """تقرير العلم — عقود مخرجات منفصلة صراحة."""

    labeled: pl.DataFrame
    feature_names: tuple[str, ...]
    fold_frame: pl.DataFrame
    #: تنبؤات OOF على صفوف اختبار الطيّات فقط (صالحة للباك تست).
    fold_scores: pl.DataFrame
    conditional_oof_predictions: pl.DataFrame
    calibration_by_outcome: pl.DataFrame
    drift_summary: dict[str, float]
    stability: dict[str, float]
    holdout: FrozenHoldout
    holdout_eval: HoldoutEvaluation | None
    final_model: ConditionalModel | None
    final_calibrators: dict[str, PlattCalibrator] = field(default_factory=dict)
    #: تنبؤ حي من النموذج النهائي — ليس سلسلة باك تست.
    live_model_predictions: pl.DataFrame = field(default_factory=pl.DataFrame)
    #: توافق قديم = live (غير مؤهل للباك تست).
    state_predictions: pl.DataFrame = field(default_factory=pl.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _calibration_tail_split(
    train: pl.DataFrame,
    *,
    frac: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """يفصل ذيلًا زمنيًا من القطار للمعايرة — سببي داخل الطيّة."""
    if train.height < _MIN_TRAIN_FOR_CAL_SPLIT or frac <= 0.0:
        return train, train.head(0)
    work = train.sort(SETUP_AVAILABILITY_TS)
    setup_times = work[SETUP_AVAILABILITY_TS].unique(maintain_order=True).sort()
    n_cal_setups = max(1, round(setup_times.len() * float(frac)))
    n_cal_setups = min(n_cal_setups, max(0, setup_times.len() // 3))
    if n_cal_setups < 1:
        return work, work.head(0)
    cal_times = setup_times.tail(n_cal_setups).to_list()
    cal = work.filter(pl.col(SETUP_AVAILABILITY_TS).is_in(cal_times))
    fit = work.filter(~pl.col(SETUP_AVAILABILITY_TS).is_in(cal_times))
    if cal.height < _MIN_CAL_ROWS:
        return work, work.head(0)
    if set(fit[SETUP_AVAILABILITY_TS].to_list()) & set(cal[SETUP_AVAILABILITY_TS].to_list()):
        raise AssertionError("calibration split must not divide one setup timestamp")
    return fit, cal


def run_behavior_science(  # noqa: PLR0912, PLR0915
    blended: pl.DataFrame,
    *,
    config: ScienceConfig | None = None,
    progress: ProgressLike | None = None,
) -> BehaviorScienceReport:
    """يشغّل العلم مع OOF تاريخي منفصل عن التنبؤ الحي."""
    cfg = config or ScienceConfig()
    if progress is not None:
        progress.op(f"run_behavior_science bars={blended.height:,}")
    empty_holdout = carve_frozen_holdout(pl.DataFrame(), holdout_frac=cfg.holdout_frac)
    empty = BehaviorScienceReport(
        labeled=pl.DataFrame(),
        feature_names=(),
        fold_frame=pl.DataFrame(),
        fold_scores=pl.DataFrame(),
        conditional_oof_predictions=pl.DataFrame(),
        calibration_by_outcome=pl.DataFrame(),
        drift_summary={},
        stability={},
        holdout=empty_holdout,
        holdout_eval=None,
        final_model=None,
        final_calibrators={},
        live_model_predictions=pl.DataFrame(),
        state_predictions=pl.DataFrame(),
        diagnostics={"empty": True},
    )
    if blended.height == 0 or AVAILABILITY_TS not in blended.columns:
        if progress is not None:
            progress.op("science: empty blended")
        return empty

    if progress is not None:
        progress.op("science: labeled outcomes")
    outcomes = build_labeled_outcomes(
        blended,
        outcome_window=cfg.outcome_window,
        group_col=cfg.group_col if cfg.group_col in blended.columns else None,
        progress=progress,
    )
    labeled_all = attach_outcome_availability_guard(blended, outcomes)
    labeled = filter_resolved_outcomes(labeled_all)
    n_censored = int(labeled_all.height - labeled.height) if labeled_all.height else 0
    if progress is not None:
        progress.op(
            f"science: labeled={labeled_all.height:,} resolved={labeled.height:,} "
            f"censored={n_censored:,}"
        )

    # اختيار ميزات من صفوف التطوير المحسومة فقط (بعد carve لاحقًا نعيد على develop)
    if progress is not None:
        progress.op(f"science: carve holdout frac={cfg.holdout_frac}")
    holdout_pack = carve_frozen_holdout(labeled, holdout_frac=cfg.holdout_frac)
    develop = holdout_pack.develop
    if progress is not None:
        progress.op(f"science: develop={develop.height:,} holdout={holdout_pack.holdout.height:,}")

    if progress is not None:
        progress.op("science: family feature selection")
    candidate_feature_names = select_feature_names_by_family(
        develop if develop.height else blended,
        max_features=cfg.max_features,
    )
    if progress is not None:
        progress.op(f"science: features={len(candidate_feature_names)}")

    folds: list[ScienceFold] = []
    if cfg.use_month_folds:
        if progress is not None:
            progress.op("science: expanding month folds")
        folds = build_expanding_month_folds(
            develop,
            ts_col=SETUP_AVAILABILITY_TS,
            min_train_months=1,
            embargo_ns=int(cfg.embargo),
            purge_samples=cfg.purge_samples,
        )
    if not folds:
        if progress is not None:
            progress.op("science: contract-aware folds")
        folds = build_contract_aware_folds(
            develop,
            ts_col=SETUP_AVAILABILITY_TS,
            n_splits=cfg.n_splits,
            embargo=cfg.embargo,
            purge_samples=cfg.purge_samples,
            min_train_size=cfg.min_train_size,
        )
    if progress is not None:
        progress.op(f"science: n_folds={len(folds)}")

    fold_score_rows: list[pl.DataFrame] = []
    oof_state_rows: list[pl.DataFrame] = []
    fold_metric_rows: list[dict[str, float | int | str]] = []
    drift_psis: list[float] = []
    seen_setup_test: set[int] = set()
    n_folds = max(len(folds), 1)

    for fold_i, sf in enumerate(folds, start=1):
        if progress is not None:
            progress.heartbeat(fold_i, n_folds, label="science-folds", force=True)
            progress.op(
                f"science fold {fold_i}/{len(folds)} segment={sf.segment} "
                f"train_end={sf.train_end_ts}"
            )
        train = develop[sf.train_idx]
        test = develop[sf.test_idx]
        # امنع تكرار setup في OOS عبر الطيّات
        test_setups = set(int(x) for x in test[SETUP_AVAILABILITY_TS].unique().to_list())
        overlap = test_setups & seen_setup_test
        if overlap:
            # أسقط الصفوف المكررة من هذه الطيّة
            test = test.filter(~pl.col(SETUP_AVAILABILITY_TS).is_in(list(overlap)))
            if test.height == 0:
                continue
        seen_setup_test |= set(int(x) for x in test[SETUP_AVAILABILITY_TS].unique().to_list())

        known_train = filter_outcomes_known_by(train, asof_ts=sf.train_end_ts)
        fit_part, cal_part = _calibration_tail_split(known_train, frac=cfg.calibration_frac)
        fold_features = select_feature_names_by_family(
            fit_part if fit_part.height else known_train,
            max_features=cfg.max_features,
        )
        if progress is not None:
            progress.op(
                f"science fold {fold_i}: fit={fit_part.height:,} cal={cal_part.height:,} "
                f"test={test.height:,}"
            )
        model = fit_conditional_models(
            fit_part,
            feature_names=fold_features,
            outcomes=OUTCOME_TARGETS,
            train_end_ts=sf.train_end_ts,
            l2=cfg.l2,
            min_train=max(8, cfg.min_train_size // 2),
            min_pos=cfg.min_pos,
            min_neg=cfg.min_neg,
            progress=progress,
        )
        # معايرة Platt على ذيل القطار إن أمكن
        raw_cal = (
            score_conditional_models(
                model,
                cal_part,
                test_idx=np.arange(cal_part.height, dtype=np.intp),
                embargo=0.0,
                enforce_temporal_split=False,
            )
            if cal_part.height
            else pl.DataFrame()
        )
        calibrators: dict[str, PlattCalibrator] = {}
        if raw_cal.height and "p_hat" in raw_cal.columns and "y" in raw_cal.columns:
            calibrators = fit_platt_calibrators_by_outcome(
                raw_cal,
                min_samples=max(10, cfg.min_train_size // 2),
            )

        if progress is not None:
            progress.op(f"science fold {fold_i}: score + calibrate + drift")
        # سجّل الاختبار بمؤشرات الصفوف داخل develop
        scored = score_conditional_models(
            model,
            develop,
            test_idx=sf.test_idx,
            embargo=float(cfg.embargo),
        )
        # أعد فلترة إن حذفنا تكرارات
        if scored.height and overlap:
            scored = scored.filter(~pl.col(SETUP_AVAILABILITY_TS).is_in(list(overlap)))
        scored = apply_calibrators_by_outcome(scored, calibrators)
        if "p_cal" in scored.columns:
            scored = scored.with_columns(pl.col("p_cal").alias("p_hat"))

        cal = evaluate_calibration(scored, n_bins=cfg.calibration_bins)
        drift_features = model.feature_names
        drift = measure_drift(
            train.select([c for c in drift_features if c in train.columns]),
            test.select([c for c in drift_features if c in test.columns]),
            feature_names=drift_features,
            ref_outcomes=train,
            cmp_outcomes=test,
            ref_calibration_ece=evaluate_calibration(raw_cal).ece,
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
                "brier_skill": cal.brier_skill,
                "mean_psi": drift.mean_psi,
                "max_psi": drift.max_psi,
                "outcome_rate_l1": drift.outcome_rate_l1,
                "n_calibration_rows": int(scored.height),
            }
        )
        if scored.height:
            fold_score_rows.append(
                scored.with_columns(
                    pl.lit(sf.fold).alias("fold"),
                    pl.lit(True).alias("prediction_is_oof"),
                    pl.lit(True).alias("eligible_for_backtest"),
                    pl.lit(sf.train_end_ts).alias("model_train_end_ts"),
                )
            )

        # OOF على صفوف الحالة التي setup_ts ∈ اختبار هذه الطيّة
        test_ts_list = test[SETUP_AVAILABILITY_TS].unique().to_list()
        if test_ts_list and AVAILABILITY_TS in blended.columns:
            state_slice = blended.filter(pl.col(AVAILABILITY_TS).is_in(test_ts_list))
            if state_slice.height:
                state_predictions = predict_probabilities_at_states(
                    model,
                    state_slice,
                    outcomes=PRIMARY_OUTCOME_TARGETS
                    + tuple(o for o in OUTCOME_TARGETS if o not in PRIMARY_OUTCOME_TARGETS),
                    prediction_is_oof=True,
                    fold=sf.fold,
                    eligible_for_backtest=True,
                )
                oof_state_rows.append(
                    apply_calibrators_to_state_predictions(state_predictions, calibrators)
                )

    fold_frame = pl.DataFrame(fold_metric_rows) if fold_metric_rows else folds_to_frame(folds)
    fold_scores = (
        pl.concat(fold_score_rows, how="diagonal_relaxed") if fold_score_rows else pl.DataFrame()
    )
    conditional_oof_predictions = (
        pl.concat(oof_state_rows, how="diagonal_relaxed").unique(
            subset=[AVAILABILITY_TS], keep="first"
        )
        if oof_state_rows
        else pl.DataFrame()
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

    final_model: ConditionalModel | None = None
    final_calibrators: dict[str, PlattCalibrator] = {}
    holdout_eval: HoldoutEvaluation | None = None
    holdout_state = holdout_pack
    live_preds = pl.DataFrame()
    if develop.height >= cfg.min_train_size and candidate_feature_names:
        if progress is not None:
            progress.op("science: fit final live model")
        final_train_end = int(holdout_pack.cut_ts)
        final_known = filter_outcomes_known_by(develop, asof_ts=final_train_end)
        final_fit, final_cal = _calibration_tail_split(
            final_known,
            frac=cfg.calibration_frac,
        )
        final_features = select_feature_names_by_family(
            final_fit if final_fit.height else final_known,
            max_features=cfg.max_features,
        )
        final_model = fit_conditional_models(
            final_fit,
            feature_names=final_features,
            outcomes=OUTCOME_TARGETS,
            train_end_ts=final_train_end,
            l2=cfg.l2,
            min_train=max(8, cfg.min_train_size // 2),
            min_pos=cfg.min_pos,
            min_neg=cfg.min_neg,
            progress=progress,
        )
        raw_final_cal = (
            score_conditional_models(
                final_model,
                final_cal,
                test_idx=np.arange(final_cal.height, dtype=np.intp),
                embargo=0.0,
                enforce_temporal_split=False,
            )
            if final_cal.height
            else pl.DataFrame()
        )
        final_calibrators = fit_platt_calibrators_by_outcome(
            raw_final_cal,
            min_samples=max(10, cfg.min_train_size // 2),
        )
        # حي فقط — غير مؤهل للباك تست التاريخي
        if progress is not None:
            progress.op("science: live state predictions")
        live_preds = predict_probabilities_at_states(
            final_model,
            blended,
            outcomes=PRIMARY_OUTCOME_TARGETS
            + tuple(o for o in OUTCOME_TARGETS if o not in PRIMARY_OUTCOME_TARGETS),
            prediction_is_oof=False,
            fold=None,
            eligible_for_backtest=False,
        )
        live_preds = apply_calibrators_to_state_predictions(live_preds, final_calibrators)
        if cfg.evaluate_holdout and holdout_pack.holdout.height > 0:
            if progress is not None:
                progress.op(f"science: frozen holdout n={holdout_pack.holdout.height:,}")
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
            scored_ho = apply_calibrators_by_outcome(scored_ho, final_calibrators)
            holdout_eval, holdout_state = evaluate_frozen_holdout_once(
                holdout_pack, scored_ho, allow_retouch=False
            )

    feature_names = (
        final_model.feature_names if final_model is not None else candidate_feature_names
    )
    n_lf = sum(1 for c in feature_names if c.startswith("lf_"))
    n_rel = sum(1 for c in feature_names if c.startswith("rel_"))
    n_path = sum(1 for c in feature_names if c.startswith("path_"))
    n_mem = sum(
        1
        for c in feature_names
        if c.startswith("mem_") or "__lag" in c or "__rmean" in c or "__rsum" in c
    )

    if progress is not None:
        progress.op(
            f"science done folds={len(folds)} oof={conditional_oof_predictions.height:,} "
            f"live={live_preds.height:,}"
        )
    return BehaviorScienceReport(
        labeled=labeled_all,
        feature_names=feature_names,
        fold_frame=fold_frame,
        fold_scores=fold_scores,
        conditional_oof_predictions=conditional_oof_predictions,
        calibration_by_outcome=calibration_by_outcome,
        drift_summary=drift_summary,
        stability=stability,
        holdout=holdout_state,
        holdout_eval=holdout_eval,
        final_model=final_model,
        final_calibrators=final_calibrators,
        live_model_predictions=live_preds,
        state_predictions=live_preds,  # توافق: الحي فقط، موثّق كغير OOF
        diagnostics={
            "n_labeled": int(labeled_all.height),
            "n_resolved": int(labeled.height),
            "n_censored": n_censored,
            "n_develop": int(develop.height),
            "n_holdout": int(holdout_pack.holdout.height),
            "holdout_cut_ts": int(holdout_pack.cut_ts),
            "holdout_touched": bool(holdout_state.touched),
            "n_features": len(feature_names),
            "n_final_calibrators": len(final_calibrators),
            "n_level_flow_features": n_lf,
            "n_reliability_features": n_rel,
            "n_path_features": n_path,
            "n_memory_features": n_mem,
            "n_folds": len(folds),
            "n_oof_prediction_rows": int(conditional_oof_predictions.height),
            "n_live_prediction_rows": int(live_preds.height),
            "primary_outcomes": list(PRIMARY_OUTCOME_TARGETS),
            "conditional_probability_semantics": (
                "independent_binary_outcomes_not_a_joint_competing-risk_distribution"
            ),
            "signal_quality_is_calibrated_probability": False,
            "prediction_uses_oos_labels": False,
            "live_predictions_eligible_for_backtest": False,
            "oof_predictions_eligible_for_backtest": True,
            "science_steps": (
                "conditional_model_state_to_probs",
                "outcomes_censored_and_group_onset",
                "structure_features",
                "market_memory_sequence_group_safe",
                "level_anchored_order_flow",
                "reliability_evidence_no_delete",
                "family_aware_feature_selection",
                "asia_london_projection_state",
                "path_depth_confirmation_no_if",
                "platt_calibration_causal_tail",
                "calibration_ece_brier_bss",
                "walk_forward_unique_setup",
                "oof_vs_live_predictions",
                "drift_stability_open_psi",
                "frozen_final_holdout",
            ),
        },
    )


__all__ = [
    "BehaviorScienceReport",
    "ScienceConfig",
    "run_behavior_science",
]
