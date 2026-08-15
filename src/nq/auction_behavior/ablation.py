"""Ablation عائلات الميزات على نفس طيّات OOF competing-risk.

الهدف: هل الـ MBO أو الذاكرة أو الإسقاط الديناميكي يضيفون مهارة خارج العينة
فوق Volume Profile / السعر، وليس مجرد وجود الأعمدة.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.auction_behavior.competing import (
    competing_known_by,
    evaluate_competing_scores,
    fit_competing_risk,
    pivot_competing_labels,
    score_competing_risk,
)
from nq.auction_behavior.conditional import MODEL_STATUS_OK, select_feature_names_by_family
from nq.research.progress import ProgressLike

_A = ("state", "structure", "quality")
_B = (*_A, "projection", "path")
_C = (*_B, "memory_roll", "sequence")
_D = (*_C, "level_flow")
_E = (*_D, "reliability")


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """طبقة تراكمية من عائلات الميزات."""

    name: str
    families: tuple[str, ...]
    detail: str


ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        "A_volume_profile_price",
        _A,
        "Volume/Profile + price/state only",
    ),
    AblationSpec(
        "B_plus_dynamic_asia_london",
        _B,
        "+ Dynamic Asia→London projection and path-depth",
    ),
    AblationSpec(
        "C_plus_memory",
        _C,
        "+ market memory / sequence",
    ),
    AblationSpec(
        "D_plus_mbo",
        _D,
        "+ raw MBO level-flow",
    ),
    AblationSpec(
        "E_plus_reliability",
        _E,
        "+ reliability/intent evidence (no MBO deletion)",
    ),
)


@dataclass(frozen=True, slots=True)
class AblationFoldSlice:
    """قطار معروف + اختبار OOF من نفس طيّة العلم."""

    fold: int
    segment: str
    train_end_ts: int
    train: pl.DataFrame
    test: pl.DataFrame


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def run_feature_ablation(
    slices: list[AblationFoldSlice],
    *,
    max_features: int = 64,
    l2: float = 1.0,
    min_train: int = 12,
    min_class: int = 2,
    embargo: float = 0.0,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يعيد جدول OOS لكل طبقة — نفس الطيّات، ميزات أقل/أكثر."""
    schema = {
        "spec": pl.Utf8(),
        "detail": pl.Utf8(),
        "families": pl.Utf8(),
        "n_features_mean": pl.Float64(),
        "n_oof": pl.Int64(),
        "n_folds_used": pl.Int64(),
        "log_loss": pl.Float64(),
        "brier": pl.Float64(),
        "brier_skill": pl.Float64(),
        "ece": pl.Float64(),
        "auc_macro": pl.Float64(),
        "accuracy": pl.Float64(),
        "status": pl.Utf8(),
    }
    if not slices:
        return pl.DataFrame(schema=schema)
    rows: list[dict[str, float | int | str]] = []
    n_specs = len(ABLATION_SPECS)
    for i, spec in enumerate(ABLATION_SPECS, start=1):
        if progress is not None:
            progress.op(f"ablation {spec.name} ({i}/{n_specs})")
        fold_metrics: list[dict[str, float]] = []
        n_feat: list[int] = []
        n_oof = 0
        statuses: list[str] = []
        for sl in slices:
            train_comp = competing_known_by(
                pivot_competing_labels(sl.train),
                asof_ts=sl.train_end_ts,
            )
            test_comp = pivot_competing_labels(sl.test)
            feats = select_feature_names_by_family(
                train_comp if train_comp.height else sl.train,
                max_features=max_features,
                include_families=spec.families,
            )
            n_feat.append(len(feats))
            model = fit_competing_risk(
                train_comp,
                feature_names=feats,
                train_end_ts=sl.train_end_ts,
                l2=l2,
                min_train=min_train,
                min_class=min_class,
            )
            statuses.append(model.status)
            if not model.is_usable() or test_comp.height == 0:
                continue
            scored = score_competing_risk(
                model,
                test_comp,
                embargo=float(embargo),
                enforce_temporal_split=True,
            )
            metrics = evaluate_competing_scores(scored)
            n_oof += int(metrics["n"])
            fold_metrics.append(metrics)
        status = MODEL_STATUS_OK if fold_metrics else (statuses[0] if statuses else "skipped")
        rows.append(
            {
                "spec": spec.name,
                "detail": spec.detail,
                "families": ",".join(spec.families),
                "n_features_mean": float(np.mean(n_feat)) if n_feat else 0.0,
                "n_oof": int(n_oof),
                "n_folds_used": len(fold_metrics),
                "log_loss": _mean_metric(fold_metrics, "log_loss"),
                "brier": _mean_metric(fold_metrics, "brier"),
                "brier_skill": _mean_metric(fold_metrics, "brier_skill"),
                "ece": _mean_metric(fold_metrics, "ece"),
                "auc_macro": _mean_metric(fold_metrics, "auc_macro"),
                "accuracy": _mean_metric(fold_metrics, "accuracy"),
                "status": status,
            }
        )
    return pl.DataFrame(rows)


def ablation_nested_families() -> bool:
    """A ⊂ B ⊂ C ⊂ D ⊂ E."""
    prev: set[str] = set()
    for spec in ABLATION_SPECS:
        fams = set(spec.families)
        if not prev.issubset(fams):
            return False
        prev = fams
    return True


__all__ = [
    "ABLATION_SPECS",
    "AblationFoldSlice",
    "AblationSpec",
    "ablation_nested_families",
    "run_feature_ablation",
]
