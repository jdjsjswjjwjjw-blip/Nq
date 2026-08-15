"""كشف انجراف (drift) واستقرار عبر الطيّات والأنظمة."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

_EPS = 1e-12
_MIN_EDGES = 2


@dataclass(frozen=True, slots=True)
class DriftReport:
    """ملخص انجراف بين مرجع (قطار) ومقارنة (اختبار/لاحقة)."""

    n_ref: int
    n_cmp: int
    mean_psi: float
    max_psi: float
    outcome_rate_l1: float
    calibration_delta: float
    feature_mean_l1: float
    detail: str = ""


def _psi_1d(ref: np.ndarray, cmp: np.ndarray, *, n_bins: int = 10) -> float:
    """Population Stability Index لصفة واحدة."""
    ref = ref[np.isfinite(ref)]
    cmp = cmp[np.isfinite(cmp)]
    if ref.size == 0 or cmp.size == 0:
        return 0.0
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if edges.size < _MIN_EDGES:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cmp_hist, _ = np.histogram(cmp, bins=edges)
    ref_p = ref_hist.astype(np.float64) / max(ref.size, 1)
    cmp_p = cmp_hist.astype(np.float64) / max(cmp.size, 1)
    ref_p = np.clip(ref_p, _EPS, None)
    cmp_p = np.clip(cmp_p, _EPS, None)
    # أعد التطبيع بعد القص
    ref_p = ref_p / ref_p.sum()
    cmp_p = cmp_p / cmp_p.sum()
    return float(np.sum((cmp_p - ref_p) * np.log(cmp_p / ref_p)))


def feature_psi(
    ref: pl.DataFrame,
    cmp: pl.DataFrame,
    feature_names: tuple[str, ...] | list[str],
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name in feature_names:
        if name not in ref.columns or name not in cmp.columns:
            continue
        psi = _psi_1d(
            ref[name].fill_null(0.0).to_numpy().astype(np.float64),
            cmp[name].fill_null(0.0).to_numpy().astype(np.float64),
            n_bins=n_bins,
        )
        rows.append({"feature": name, "psi": psi})
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(schema={"feature": pl.Utf8(), "psi": pl.Float64()})
    )


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        return float(value.item())
    return float(str(value))


def outcome_rate_map(labeled: pl.DataFrame) -> dict[str, float]:
    if labeled.height == 0 or "outcome_name" not in labeled.columns:
        return {}
    out: dict[str, float] = {}
    for name, g in labeled.group_by("outcome_name", maintain_order=True):
        key = str(name[0] if isinstance(name, tuple) else name)
        out[key] = _as_float(g["y"].mean()) if "y" in g.columns and g.height else 0.0
    return out


def measure_drift(
    ref_features: pl.DataFrame,
    cmp_features: pl.DataFrame,
    *,
    feature_names: tuple[str, ...] | list[str],
    ref_outcomes: pl.DataFrame | None = None,
    cmp_outcomes: pl.DataFrame | None = None,
    ref_calibration_ece: float = 0.0,
    cmp_calibration_ece: float = 0.0,
) -> DriftReport:
    psi_df = feature_psi(ref_features, cmp_features, feature_names)
    psi_vals = psi_df["psi"].to_numpy().astype(np.float64) if psi_df.height else np.zeros(0)
    mean_psi = float(np.mean(psi_vals)) if psi_vals.size else 0.0
    max_psi = float(np.max(psi_vals)) if psi_vals.size else 0.0

    ref_rates = outcome_rate_map(ref_outcomes) if ref_outcomes is not None else {}
    cmp_rates = outcome_rate_map(cmp_outcomes) if cmp_outcomes is not None else {}
    keys = sorted(set(ref_rates) | set(cmp_rates))
    outcome_l1 = (
        float(
            sum(abs(ref_rates.get(k, 0.0) - cmp_rates.get(k, 0.0)) for k in keys)
            / max(len(keys), 1)
        )
        if keys
        else 0.0
    )

    feat_l1 = 0.0
    n_f = 0
    for name in feature_names:
        if name not in ref_features.columns or name not in cmp_features.columns:
            continue
        a = _as_float(ref_features[name].fill_null(0.0).mean())
        b = _as_float(cmp_features[name].fill_null(0.0).mean())
        feat_l1 += abs(a - b)
        n_f += 1
    feat_l1 = feat_l1 / max(n_f, 1)

    return DriftReport(
        n_ref=int(ref_features.height),
        n_cmp=int(cmp_features.height),
        mean_psi=mean_psi,
        max_psi=max_psi,
        outcome_rate_l1=outcome_l1,
        calibration_delta=float(abs(cmp_calibration_ece - ref_calibration_ece)),
        feature_mean_l1=feat_l1,
        detail=f"psi_features={psi_df.height}",
    )


def fold_stability(fold_metrics: pl.DataFrame, column: str = "ece") -> dict[str, float]:
    """تباين مقياس عبر الطيّات (استقرار)."""
    if fold_metrics.height == 0 or column not in fold_metrics.columns:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "cv": 0.0}
    vals = fold_metrics[column].fill_null(0.0).to_numpy().astype(np.float64)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    cv = float(std / max(abs(mean), _EPS))
    return {"n": float(vals.size), "mean": mean, "std": std, "cv": cv}


__all__ = [
    "DriftReport",
    "feature_psi",
    "fold_stability",
    "measure_drift",
    "outcome_rate_map",
]
