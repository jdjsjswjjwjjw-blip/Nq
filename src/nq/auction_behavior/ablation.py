"""دراسة ablation تراكمية: هل كل طبقة معلومات تضيف قيمة تنبؤية OOS فعلًا؟

السؤال العلمي: «هل MBO يضيف فوق Volume Profile؟ هل الإسقاط الديناميكي يضيف؟
هل الذاكرة تضيف؟ هل طبقة الموثوقية تضيف؟» — الإجابة الوحيدة المقبولة هي
مقارنة خارج العينة على **نفس** التسميات و**نفس** الطيّات و**نفس** المعالجة.

القواعد الصارمة:
  * تسميات واحدة وطيّات purged walk-forward واحدة تُبنى مرة ثم تُعاد لكل ستاك.
  * الـholdout المجمّد يُستبعد بالكامل — الدراسة على develop فقط.
  * كل ستاك يتدرّب خامًا بلا Platt (معالجة متطابقة → المقارنة عادلة).
  * ``signal_quality``/``signal_evidence`` مستبعدان من كل الستاكات: مركّبان
    يخلطان طبقات متعددة وسيلوّثان عزل المساهمة.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    log_loss_score,
    roc_auc,
)
from nq.auction_behavior.competing import (
    evaluate_competing_scores,
    fit_competing_risk_model,
    score_competing_risk_model,
)
from nq.auction_behavior.conditional import fit_conditional_models, score_conditional_models
from nq.auction_behavior.holdout import carve_frozen_holdout
from nq.auction_behavior.outcomes import (
    FIRST_TRANSITION_CLASSES,
    OUTCOME_AVAILABLE_TS,
    OUTCOME_TARGETS,
    SETUP_AVAILABILITY_TS,
    attach_outcome_availability_guard,
    build_first_transition_outcomes,
    build_labeled_outcomes,
    filter_outcomes_known_by,
)
from nq.auction_behavior.science import ScienceConfig
from nq.auction_behavior.walk_forward import (
    ScienceFold,
    build_contract_aware_folds,
    build_expanding_month_folds,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike

_EPS = 1e-12
_MIN_ROWS_FOR_VARIANCE = 2

#: ترتيب الستاكات التراكمية — كل ستاك يحتوي كل ما قبله.
ABLATION_STACKS = (
    "vp_price",
    "plus_projection",
    "plus_memory",
    "plus_mbo_flow",
    "plus_reliability",
)

_VP_STATE_COLUMNS = (
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_absorb",
    "vp_look_fail",
    "vp_fsm_break",
    "vp_fsm_retest",
    "vp_fsm_expand",
    "vp_early_imbalance",
    "vp_liquidity_session",
)
_BASE_STRUCT_COLUMNS = (
    "struct_dist_vah_ticks",
    "struct_dist_val_ticks",
    "struct_dist_poc_ticks",
    "struct_va_width_ticks",
    "struct_close_in_value",
    "struct_above_vah",
    "struct_below_val",
    "struct_near_vah",
    "struct_near_val",
    "struct_near_poc",
    "struct_break_pressure",
    "struct_retest_pressure",
)
_PROJECTION_STRUCT_COLUMNS = (
    "struct_dist_asia_vah_ticks",
    "struct_dist_asia_val_ticks",
    "struct_dist_asia_poc_ticks",
    "struct_dist_asia_hvn_ticks",
    "struct_dist_composite_hvn_ticks",
)
_MBO_FLOW_SCALARS = (
    "deceptive_score",
    "real_liquidity_ratio",
    "noise_instant",
    "noise_cum",
    "deceptive_volume_share",
    "deceptive_cancel_rate",
)


def _is_memory_column(name: str) -> bool:
    return (
        name.startswith("mem_")
        or "__lag" in name
        or "__rmean" in name
        or "__rsum" in name
        or "__ecount" in name
    )


def _stack_feature_groups(frame: pl.DataFrame, stack: str) -> tuple[tuple[str, ...], ...]:
    """عائلات الستاك بالترتيب التراكمي؛ كل عائلة مجموعة أعمدة موجودة فعلًا.

    ذاكرة الطبقات الأعلى لا تتسرب لستاك أدنى: ``lf_*__lag`` و``rel_*__rmean``
    تُنسب لعائلة مصدرها (MBO/الموثوقية)، لا لعائلة الذاكرة.
    """
    if stack not in ABLATION_STACKS:
        raise ValueError(f"unknown ablation stack: {stack}")
    level = ABLATION_STACKS.index(stack)
    base = tuple(c for c in (*_VP_STATE_COLUMNS, *_BASE_STRUCT_COLUMNS) if c in frame.columns)
    projection = tuple(
        c
        for c in frame.columns
        if not _is_memory_column(c)
        and (c.startswith("proj_") or c.startswith("path_") or c in _PROJECTION_STRUCT_COLUMNS)
    )
    memory = tuple(
        c for c in frame.columns if _is_memory_column(c) and not c.startswith(("lf_", "rel_"))
    )
    mbo_flow = tuple(
        c
        for c in frame.columns
        if (c.startswith("lf_") and not _is_memory_column(c)) or c in _MBO_FLOW_SCALARS
    ) + tuple(c for c in frame.columns if c.startswith("lf_") and _is_memory_column(c))
    reliability = tuple(c for c in frame.columns if c.startswith("rel_"))
    groups = (base, projection, memory, mbo_flow, reliability)
    return groups[: level + 1]


def stack_feature_columns(frame: pl.DataFrame, stack: str) -> tuple[str, ...]:
    """كل أعمدة الميزات المسموحة للستاك (تراكمي، بلا تكرار)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for group in _stack_feature_groups(frame, stack):
        for c in group:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
    return tuple(ordered)


def _variance_ok(frame: pl.DataFrame, name: str) -> bool:
    if name not in frame.columns or frame.height < _MIN_ROWS_FOR_VARIANCE:
        return False
    vals = frame[name].fill_null(0.0).to_numpy().astype(np.float64)
    return bool(np.nanstd(vals) > _EPS)


def _select_stack_features(
    frame: pl.DataFrame,
    groups: tuple[tuple[str, ...], ...],
    *,
    max_features: int,
) -> tuple[str, ...]:
    """اختيار round-robin عبر عائلات الستاك — يضمن تمثيل أحدث طبقة.

    اختيار «أول N بالترتيب» كان يشبع الميزانية بأعمدة الذاكرة فتتطابق
    الستاكات الأعلى مع الأدنى ويستحيل قياس مساهمة MBO/الموثوقية.
    """
    budget = max(0, int(max_features))
    pools = [[c for c in group if _variance_ok(frame, c)] for group in groups]
    chosen: list[str] = []
    seen: set[str] = set()
    while budget > 0 and any(pools):
        progressed = False
        for pool in pools:
            while pool:
                name = pool.pop(0)
                if name in seen:
                    continue
                chosen.append(name)
                seen.add(name)
                budget -= 1
                progressed = True
                break
            if budget <= 0:
                break
        if not progressed:
            break
    return tuple(chosen)


@dataclass(frozen=True, slots=True)
class AblationReport:
    """نتائج OOS لكل ستاك × هدف + رأس المخاطر المتنافسة لكل ستاك."""

    frame: pl.DataFrame
    competing_frame: pl.DataFrame
    stacks: tuple[str, ...] = ABLATION_STACKS
    n_folds: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _binary_metric_rows(
    stack: str,
    scored: pl.DataFrame,
    *,
    n_bins: int,
    n_features: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if scored.height == 0 or "outcome_name" not in scored.columns:
        return rows
    for key, part in scored.group_by("outcome_name", maintain_order=True):
        outcome = str(key[0] if isinstance(key, tuple) else key)
        y = part["y"].to_numpy().astype(np.float64)
        p = part["p_hat"].to_numpy().astype(np.float64)
        baseline = (
            part["baseline_p"].to_numpy().astype(np.float64)
            if "baseline_p" in part.columns
            else None
        )
        mask = np.isfinite(y) & np.isfinite(p)
        n = int(np.sum(mask))
        if n == 0:
            continue
        yy, pp = y[mask], p[mask]
        rows.append(
            {
                "stack": stack,
                "stack_rank": ABLATION_STACKS.index(stack),
                "outcome_name": outcome,
                "n_oof": n,
                "brier": brier_score(yy, pp),
                "log_loss": log_loss_score(yy, pp),
                "auc": roc_auc(yy, pp),
                "ece": expected_calibration_error(yy, pp, n_bins=n_bins),
                "brier_skill": brier_skill_score(
                    yy, pp, None if baseline is None else baseline[mask]
                ),
                "n_features": n_features,
            }
        )
    return rows


def run_behavior_ablation(  # noqa: PLR0912, PLR0915
    blended: pl.DataFrame,
    *,
    config: ScienceConfig | None = None,
    stacks: tuple[str, ...] | None = None,
    progress: ProgressLike | None = None,
) -> AblationReport:
    """يشغّل ablation تراكميًا على نفس التسميات/الطيّات — develop فقط.

    المخرج قابل للمقارنة مباشرة: إن لم يتحسن ستاك ``plus_mbo_flow`` على
    ``plus_memory`` خارج العينة، فلا دليل أن MBO يضيف معلومات لهذا الهدف.
    """
    cfg = config or ScienceConfig()
    use = stacks if stacks is not None else ABLATION_STACKS
    for s in use:
        if s not in ABLATION_STACKS:
            raise ValueError(f"unknown ablation stack: {s}")
    empty_frame = pl.DataFrame(
        schema={
            "stack": pl.Utf8(),
            "stack_rank": pl.Int64(),
            "outcome_name": pl.Utf8(),
            "n_oof": pl.Int64(),
            "brier": pl.Float64(),
            "log_loss": pl.Float64(),
            "auc": pl.Float64(),
            "ece": pl.Float64(),
            "brier_skill": pl.Float64(),
            "n_features": pl.Int64(),
        }
    )
    empty_competing = pl.DataFrame(
        schema={
            "stack": pl.Utf8(),
            "stack_rank": pl.Int64(),
            "n_oof": pl.Float64(),
            "brier": pl.Float64(),
            "log_loss": pl.Float64(),
            "accuracy": pl.Float64(),
            "n_features": pl.Int64(),
        }
    )
    if progress is not None:
        progress.op(f"run_behavior_ablation bars={blended.height:,} stacks={len(use)}")
    if blended.height == 0 or AVAILABILITY_TS not in blended.columns:
        return AblationReport(
            frame=empty_frame,
            competing_frame=empty_competing,
            stacks=tuple(use),
            n_folds=0,
            diagnostics={"empty": True},
        )

    group_col = cfg.group_col if cfg.group_col in blended.columns else None
    outcomes = build_labeled_outcomes(
        blended,
        outcome_window=cfg.outcome_window,
        group_col=group_col,
        progress=progress,
    )
    labeled_all = attach_outcome_availability_guard(blended, outcomes)
    labeled = (
        labeled_all.filter(pl.col("label_status") == "resolved")
        if labeled_all.height
        else labeled_all
    )
    holdout_pack = carve_frozen_holdout(labeled, holdout_frac=cfg.holdout_frac)
    develop = holdout_pack.develop

    ft_outcomes = build_first_transition_outcomes(
        blended,
        window=max(int(cfg.competing_window), int(cfg.outcome_window)),
        group_col=group_col,
        progress=progress,
    )
    ft_labeled = attach_outcome_availability_guard(blended, ft_outcomes)
    ft_resolved = (
        ft_labeled.filter(pl.col("label_status") == "resolved") if ft_labeled.height else ft_labeled
    )
    ft_develop = (
        ft_resolved.filter(pl.col(SETUP_AVAILABILITY_TS) <= holdout_pack.cut_ts)
        if ft_resolved.height and holdout_pack.cut_ts >= 0
        else ft_resolved
    )

    folds: list[ScienceFold] = []
    if cfg.use_month_folds:
        folds = build_expanding_month_folds(
            develop,
            ts_col=SETUP_AVAILABILITY_TS,
            min_train_months=1,
            embargo_ns=int(cfg.embargo),
            purge_samples=cfg.purge_samples,
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
    if progress is not None:
        progress.op(
            f"ablation labeled={labeled.height:,} develop={develop.height:,} folds={len(folds)}"
        )
    if not folds or develop.height == 0:
        return AblationReport(
            frame=empty_frame,
            competing_frame=empty_competing,
            stacks=tuple(use),
            n_folds=0,
            diagnostics={"empty": True, "n_labeled": int(labeled.height)},
        )

    rows: list[dict[str, Any]] = []
    competing_rows: list[dict[str, Any]] = []
    features_per_stack: dict[str, int] = {}
    for stack_i, stack in enumerate(use, start=1):
        if progress is not None:
            progress.heartbeat(stack_i, len(use), label="ablation-stacks", force=True)
            progress.op(f"ablation stack {stack} ({stack_i}/{len(use)})")
        groups = _stack_feature_groups(blended, stack)
        scored_parts: list[pl.DataFrame] = []
        competing_scored_parts: list[pl.DataFrame] = []
        n_features_used = 0
        for sf in folds:
            train = develop[sf.train_idx]
            known_train = filter_outcomes_known_by(train, asof_ts=sf.train_end_ts)
            stack_features = _select_stack_features(
                known_train if known_train.height else develop,
                groups,
                max_features=cfg.max_features,
            )
            n_features_used = max(n_features_used, len(stack_features))
            model = fit_conditional_models(
                known_train,
                feature_names=stack_features,
                outcomes=OUTCOME_TARGETS,
                train_end_ts=sf.train_end_ts,
                l2=cfg.l2,
                min_train=max(8, cfg.min_train_size // 2),
                min_pos=cfg.min_pos,
                min_neg=cfg.min_neg,
                progress=None,
            )
            scored = score_conditional_models(
                model,
                develop,
                test_idx=sf.test_idx,
                embargo=float(cfg.embargo),
            )
            if scored.height:
                scored_parts.append(scored)

            if ft_develop.height:
                ft_train = ft_develop.filter(
                    (pl.col(SETUP_AVAILABILITY_TS) <= sf.train_end_ts)
                    & (pl.col(OUTCOME_AVAILABLE_TS) <= sf.train_end_ts)
                )
                competing_model = fit_competing_risk_model(
                    ft_train,
                    feature_names=stack_features,
                    train_end_ts=sf.train_end_ts,
                    l2=cfg.l2,
                    min_train=cfg.competing_min_train,
                    min_class_count=cfg.competing_min_class,
                    progress=None,
                )
                if competing_model.is_usable():
                    ft_test = ft_develop.filter(
                        (pl.col(SETUP_AVAILABILITY_TS) >= sf.test_start_ts)
                        & (pl.col(SETUP_AVAILABILITY_TS) <= sf.test_end_ts)
                    )
                    if ft_test.height:
                        competing_scored_parts.append(
                            score_competing_risk_model(
                                competing_model, ft_test, embargo=float(cfg.embargo)
                            )
                        )

        pooled = pl.concat(scored_parts, how="diagonal_relaxed") if scored_parts else pl.DataFrame()
        rows.extend(
            _binary_metric_rows(
                stack,
                pooled,
                n_bins=cfg.calibration_bins,
                n_features=n_features_used,
            )
        )
        features_per_stack[stack] = n_features_used
        if competing_scored_parts:
            pooled_ft = pl.concat(competing_scored_parts, how="diagonal_relaxed")
            metrics = evaluate_competing_scores(pooled_ft, classes=FIRST_TRANSITION_CLASSES)
            competing_rows.append(
                {
                    "stack": stack,
                    "stack_rank": ABLATION_STACKS.index(stack),
                    "n_oof": metrics["n"],
                    "brier": metrics["brier"],
                    "log_loss": metrics["log_loss"],
                    "accuracy": metrics["accuracy"],
                    "n_features": n_features_used,
                }
            )

    frame = pl.DataFrame(rows) if rows else empty_frame
    competing_frame = pl.DataFrame(competing_rows) if competing_rows else empty_competing
    if progress is not None:
        progress.op(f"ablation done rows={frame.height} competing_rows={competing_frame.height}")
    return AblationReport(
        frame=frame.sort(["outcome_name", "stack_rank"]) if frame.height else frame,
        competing_frame=competing_frame,
        stacks=tuple(use),
        n_folds=len(folds),
        diagnostics={
            "n_labeled": int(labeled.height),
            "n_develop": int(develop.height),
            "n_holdout_excluded": int(holdout_pack.holdout.height),
            "holdout_cut_ts": int(holdout_pack.cut_ts),
            "holdout_untouched": True,
            "n_competing_develop": int(ft_develop.height),
            "features_per_stack": features_per_stack,
            "identical_folds_and_labels_across_stacks": True,
            "raw_uncalibrated_models_for_fair_comparison": True,
            "composite_quality_columns_excluded": True,
        },
    )


__all__ = [
    "ABLATION_STACKS",
    "AblationReport",
    "run_behavior_ablation",
    "stack_feature_columns",
]
