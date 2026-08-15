"""طبقة العلم: نموذج شرطي + OOF تاريخي + معايرة سببية + holdout مجمّد."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.ablation import AblationFoldSlice, run_feature_ablation
from nq.auction_behavior.calibration import (
    PlattCalibrator,
    apply_calibrators_by_outcome,
    apply_calibrators_to_state_predictions,
    evaluate_calibration,
    evaluate_calibration_by_outcome,
    fit_platt_calibrators_by_outcome,
)
from nq.auction_behavior.competing import (
    COMPETING_CLASS_NAMES,
    CompetingRiskModel,
    attach_competing_to_states,
    competing_known_by,
    evaluate_competing_scores,
    fit_competing_risk,
    pivot_competing_labels,
    score_competing_risk,
)
from nq.auction_behavior.conditional import (
    ConditionalModel,
    fit_conditional_models,
    group_feature_names_by_family,
    predict_probabilities_at_states,
    rank_feature_weights,
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
    run_ablation: bool = True
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
    competing_fold_scores: pl.DataFrame = field(default_factory=pl.DataFrame)
    ablation: pl.DataFrame = field(default_factory=pl.DataFrame)
    final_competing: CompetingRiskModel | None = None
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


def _log_feature_names(
    names: tuple[str, ...],
    *,
    progress: ProgressLike | None,
    label: str,
) -> dict[str, list[str]]:
    grouped = group_feature_names_by_family(names)
    if progress is not None:
        progress.op(f"{label}: n={len(names)}")
        for family, items in grouped.items():
            progress.op(f"{label} family {family} ({len(items)}): {', '.join(items)}")
    return grouped


def _log_feature_weights(
    ranked: dict[str, list[dict[str, float | str]]],
    *,
    progress: ProgressLike | None,
) -> None:
    if progress is None:
        return
    for outcome, rows in ranked.items():
        if not rows:
            progress.op(f"science knowledge {outcome}: no fitted weights")
            continue
        detail = ", ".join(f"{row['name']}={float(row['weight']):+.3f}" for row in rows[:8])
        progress.op(f"science knowledge {outcome}: {detail}")


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
    _log_feature_names(
        candidate_feature_names, progress=progress, label="science candidate features"
    )

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
    competing_score_rows: list[pl.DataFrame] = []
    ablation_slices: list[AblationFoldSlice] = []
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
                "log_loss": cal.log_loss,
                "auc": cal.auc,
                "mean_psi": drift.mean_psi,
                "max_psi": drift.max_psi,
                "outcome_rate_l1": drift.outcome_rate_l1,
                "n_calibration_rows": int(scored.height),
            }
        )
        if progress is not None:
            progress.op(f"science fold {fold_i}: competing-risk softmax")
        comp_train = competing_known_by(
            pivot_competing_labels(fit_part if fit_part.height else known_train),
            asof_ts=sf.train_end_ts,
        )
        competing_model = fit_competing_risk(
            comp_train,
            feature_names=fold_features,
            train_end_ts=sf.train_end_ts,
            l2=cfg.l2,
            min_train=max(8, cfg.min_train_size // 2),
            progress=progress,
        )
        comp_test = pivot_competing_labels(test)
        competing_metrics = evaluate_competing_scores(pl.DataFrame())
        if competing_model.is_usable() and comp_test.height:
            competing_scored = score_competing_risk(
                competing_model,
                comp_test,
                embargo=float(cfg.embargo),
            )
            competing_metrics = evaluate_competing_scores(competing_scored)
            competing_score_rows.append(
                competing_scored.with_columns(
                    pl.lit(sf.fold).alias("fold"),
                    pl.lit(True).alias("prediction_is_oof"),
                    pl.lit(True).alias("eligible_for_backtest"),
                    pl.lit(sf.train_end_ts).alias("model_train_end_ts"),
                )
            )
        fold_metric_rows[-1].update(
            {
                "competing_status": competing_model.status,
                "competing_n": competing_metrics["n"],
                "competing_log_loss": competing_metrics["log_loss"],
                "competing_brier": competing_metrics["brier"],
                "competing_brier_skill": competing_metrics["brier_skill"],
                "competing_ece": competing_metrics["ece"],
                "competing_auc_macro": competing_metrics["auc_macro"],
                "competing_accuracy": competing_metrics["accuracy"],
            }
        )
        ablation_slices.append(
            AblationFoldSlice(
                fold=int(sf.fold),
                segment=str(sf.segment),
                train_end_ts=int(sf.train_end_ts),
                train=fit_part if fit_part.height else known_train,
                test=test,
            )
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
                state_predictions = apply_calibrators_to_state_predictions(
                    state_predictions, calibrators
                )
                if competing_model.is_usable():
                    state_predictions = attach_competing_to_states(
                        competing_model,
                        state_slice,
                        state_predictions,
                        prediction_is_oof=True,
                        fold=sf.fold,
                        eligible_for_backtest=True,
                    )
                oof_state_rows.append(state_predictions)

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
    competing_fold_scores = (
        pl.concat(competing_score_rows, how="diagonal_relaxed")
        if competing_score_rows
        else pl.DataFrame()
    )
    competing_oof_metrics = evaluate_competing_scores(competing_fold_scores)
    competing_stability = (
        fold_stability(fold_frame, column="competing_brier")
        if fold_frame.height and "competing_brier" in fold_frame.columns
        else {"n": 0.0, "mean": 0.0, "std": 0.0, "cv": 0.0}
    )
    drift_summary = {
        "mean_psi_across_folds": float(np.mean(drift_psis)) if drift_psis else 0.0,
        "max_psi_across_folds": float(np.max(drift_psis)) if drift_psis else 0.0,
        "n_folds": float(len(folds)),
    }
    ablation = pl.DataFrame()
    if cfg.run_ablation and ablation_slices:
        if progress is not None:
            progress.op(f"science: feature ablation specs={len(ablation_slices)} folds")
        ablation = run_feature_ablation(
            ablation_slices,
            max_features=cfg.max_features,
            l2=cfg.l2,
            min_train=max(8, cfg.min_train_size // 2),
            embargo=float(cfg.embargo),
            progress=progress,
        )

    final_model: ConditionalModel | None = None
    final_competing: CompetingRiskModel | None = None
    final_calibrators: dict[str, PlattCalibrator] = {}
    holdout_eval: HoldoutEvaluation | None = None
    holdout_state = holdout_pack
    live_preds = pl.DataFrame()
    knowledge_weights: dict[str, list[dict[str, float | str]]] = {}
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
        _log_feature_names(final_features, progress=progress, label="science final features")
        knowledge_weights = rank_feature_weights(final_model)
        _log_feature_weights(knowledge_weights, progress=progress)
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
        if progress is not None:
            progress.op("science: fit final competing-risk model")
        final_competing = fit_competing_risk(
            competing_known_by(
                pivot_competing_labels(final_fit if final_fit.height else final_known),
                asof_ts=final_train_end,
            ),
            feature_names=final_features,
            train_end_ts=final_train_end,
            l2=cfg.l2,
            min_train=max(8, cfg.min_train_size // 2),
            progress=progress,
        )
        if final_competing.is_usable() and live_preds.height:
            live_preds = attach_competing_to_states(
                final_competing,
                blended,
                live_preds,
                prediction_is_oof=False,
                eligible_for_backtest=False,
            )
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

    n_unique_setups = (
        int(labeled[SETUP_AVAILABILITY_TS].n_unique())
        if labeled.height and SETUP_AVAILABILITY_TS in labeled.columns
        else 0
    )
    competing_develop = pivot_competing_labels(develop) if develop.height else pl.DataFrame()
    n_competing_setups = int(competing_develop.height)
    n_competing_conflicts = (
        int(competing_develop["conflict"].sum())
        if competing_develop.height and "conflict" in competing_develop.columns
        else 0
    )
    n_features = len(feature_names)
    joint_ok = bool(
        (final_competing is not None and final_competing.is_usable())
        or (
            conditional_oof_predictions.height
            and "probabilities_are_joint_distribution" in conditional_oof_predictions.columns
            and bool(conditional_oof_predictions["probabilities_are_joint_distribution"].any())
        )
    )
    sample_caution = bool(n_competing_setups > 0 and n_competing_setups < 10 * max(n_features, 1))

    if progress is not None:
        progress.op(
            f"science done folds={len(folds)} oof={conditional_oof_predictions.height:,} "
            f"live={live_preds.height:,} competing={n_competing_setups} "
            f"joint={joint_ok}"
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
        competing_fold_scores=competing_fold_scores,
        ablation=ablation,
        final_competing=final_competing,
        diagnostics={
            "n_labeled": int(labeled_all.height),
            "n_labeled_rows": int(labeled_all.height),
            "n_unique_setups": n_unique_setups,
            "n_resolved": int(labeled.height),
            "n_censored": n_censored,
            "n_develop": int(develop.height),
            "n_holdout": int(holdout_pack.holdout.height),
            "n_competing_setups": n_competing_setups,
            "n_competing_conflicts": n_competing_conflicts,
            "n_features_per_competing_setup": (
                float(n_features) / float(n_competing_setups) if n_competing_setups else None
            ),
            "sample_size_caution": sample_caution,
            "sample_size_note": (
                "labeled rows are setup×outcome, not MBO events; "
                "conditional model is identified on competing setups, not 5M MBO rows"
            ),
            "holdout_cut_ts": int(holdout_pack.cut_ts),
            "holdout_touched": bool(holdout_state.touched),
            "n_features": n_features,
            "feature_names": list(feature_names),
            "feature_names_by_family": group_feature_names_by_family(feature_names),
            "feature_weights_by_outcome": knowledge_weights,
            "n_final_calibrators": len(final_calibrators),
            "n_level_flow_features": n_lf,
            "n_reliability_features": n_rel,
            "n_path_features": n_path,
            "n_memory_features": n_mem,
            "n_folds": len(folds),
            "n_oof_prediction_rows": int(conditional_oof_predictions.height),
            "n_live_prediction_rows": int(live_preds.height),
            "n_competing_oof_rows": int(competing_fold_scores.height),
            "primary_outcomes": list(PRIMARY_OUTCOME_TARGETS),
            "competing_classes": list(COMPETING_CLASS_NAMES),
            "conditional_probability_semantics": (
                "joint_softmax_competing_risk_primary_outcomes"
                if joint_ok
                else "independent_binary_fallback_competing_insufficient"
            ),
            "probabilities_are_joint_distribution": joint_ok,
            "base_rate_is_bss_baseline_only": True,
            "competing_oof_metrics": {
                key: (None if isinstance(val, float) and not np.isfinite(val) else val)
                for key, val in competing_oof_metrics.items()
            },
            "competing_stability": competing_stability,
            "ablation": [
                {
                    key: (None if isinstance(val, float) and not np.isfinite(val) else val)
                    for key, val in row.items()
                }
                for row in (ablation.to_dicts() if ablation.height else [])
            ],
            "signal_quality_is_calibrated_probability": False,
            "prediction_uses_oos_labels": False,
            "live_predictions_eligible_for_backtest": False,
            "oof_predictions_eligible_for_backtest": True,
            "science_steps": (
                "conditional_model_state_to_probs",
                "competing_risk_softmax_joint",
                "oof_conditional_predictions",
                "feature_family_ablation",
                "outcomes_censored_and_group_onset",
                "structure_features",
                "market_memory_sequence_group_safe",
                "level_anchored_order_flow",
                "reliability_evidence_no_delete",
                "family_aware_feature_selection",
                "named_features_in_diagnostics",
                "asia_london_projection_state",
                "path_depth_confirmation_no_if",
                "platt_calibration_causal_tail",
                "calibration_ece_brier_bss_logloss",
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
