"""معايرة احتمالات: قياس (Brier/ECE/BSS) + Platt سببي بسيط."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

_EPS = 1e-12
_MAX_LOGIT = 20.0
_PLATT_MIN_CLASSES = 2
_PLATT_STEP_TOL = 1e-8


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """ملخص معايرة لهدف واحد أو مجمّع."""

    n: int
    brier: float
    ece: float
    mae: float
    base_rate: float
    brier_skill: float = 0.0
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """معاير لوجستي أحادي: p' = σ(a + b·logit(p))."""

    a: float
    b: float
    n_fit: int
    detail: str = "platt"

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
        z = np.log(p / (1.0 - p))
        z = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
        out = 1.0 / (1.0 + np.exp(-(self.a + self.b * z)))
        return np.asarray(out, dtype=np.float64)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.square(y[mask].astype(np.float64) - p[mask].astype(np.float64))))


def brier_skill_score(y: np.ndarray, p: np.ndarray) -> float:
    """BSS = 1 - Brier / Brier_base؛ base = معدل الفئة في العينة."""
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return 0.0
    yy = y[mask].astype(np.float64)
    pp = p[mask].astype(np.float64)
    br = brier_score(yy, pp)
    base = float(np.mean(yy))
    br_base = brier_score(yy, np.full_like(yy, base))
    if br_base < _EPS:
        return 0.0 if br < _EPS else -1.0
    return float(1.0 - br / br_base)


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """ECE على صناديق احتمالية؛ يتكيّف عدد الصناديق مع حجم العينة."""
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return 0.0
    y = y[mask].astype(np.float64)
    p = np.clip(p[mask].astype(np.float64), 0.0, 1.0)
    # صناديق أقل عند عينات صغيرة لثبات القياس
    adaptive = max(2, min(int(n_bins), max(2, int(y.size // 5))))
    if adaptive < 1:
        raise ValueError("n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, adaptive + 1)
    ece = 0.0
    n = y.size
    for i in range(adaptive):
        lo, hi = edges[i], edges[i + 1]
        mask_b = (p >= lo) & (p <= hi) if i == adaptive - 1 else (p >= lo) & (p < hi)
        if not np.any(mask_b):
            continue
        conf = float(np.mean(p[mask_b]))
        acc = float(np.mean(y[mask_b]))
        ece += (float(np.sum(mask_b)) / float(n)) * abs(acc - conf)
    return float(ece)


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """جدول موثوقية: لكل صندوق mean(p), mean(y), count."""
    schema = {
        "bin": pl.Int64(),
        "p_mean": pl.Float64(),
        "y_mean": pl.Float64(),
        "count": pl.Int64(),
    }
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return pl.DataFrame(schema=schema)
    p = np.clip(p[mask].astype(np.float64), 0.0, 1.0)
    y = y[mask].astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask_b = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if not np.any(mask_b):
            rows.append({"bin": i, "p_mean": 0.0, "y_mean": 0.0, "count": 0})
            continue
        rows.append(
            {
                "bin": i,
                "p_mean": float(np.mean(p[mask_b])),
                "y_mean": float(np.mean(y[mask_b])),
                "count": int(np.sum(mask_b)),
            }
        )
    return pl.DataFrame(rows)


def fit_platt_calibrator(
    y: np.ndarray,
    p: np.ndarray,
    *,
    min_samples: int = 20,
) -> PlattCalibrator | None:
    """يدرّب معاير Platt على ذيل معايرة سببي منفصل (ليس على test)."""
    mask = np.isfinite(y) & np.isfinite(p)
    if int(np.sum(mask)) < min_samples:
        return None
    yy = y[mask].astype(np.float64)
    pp = np.clip(p[mask].astype(np.float64), _EPS, 1.0 - _EPS)
    if np.unique(yy).size < _PLATT_MIN_CLASSES:
        return None
    z = np.log(pp / (1.0 - pp))
    z = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
    # نيوتن ثنائي الأبعاد لـ a,b
    a, b = 0.0, 1.0
    for _ in range(40):
        logits = a + b * z
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -_MAX_LOGIT, _MAX_LOGIT)))
        w = pred * (1.0 - pred)
        # تدرج/هيسّيان
        r = pred - yy
        g_a = float(np.sum(r))
        g_b = float(np.sum(r * z))
        h_aa = float(np.sum(w)) + 1e-6
        h_bb = float(np.sum(w * z * z)) + 1e-6
        h_ab = float(np.sum(w * z))
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < _EPS:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < _PLATT_STEP_TOL:
            break
    return PlattCalibrator(a=float(a), b=float(b), n_fit=int(yy.size))


def apply_calibrator(scored: pl.DataFrame, calibrator: PlattCalibrator | None) -> pl.DataFrame:
    """يطبق المعاير على ``p_hat`` → ``p_cal``؛ إن غاب يبقى ``p_cal=p_hat``."""
    if scored.height == 0 or "p_hat" not in scored.columns:
        return scored
    raw = scored["p_hat"].to_numpy().astype(np.float64)
    if calibrator is None:
        return scored.with_columns(pl.Series("p_cal", raw))
    return scored.with_columns(pl.Series("p_cal", calibrator.transform(raw)))


def evaluate_calibration(scored: pl.DataFrame, *, n_bins: int = 10) -> CalibrationReport:
    """معايرة من إطار فيه ``y`` و ``p_hat`` (يتجاهل NaN)."""
    if scored.height == 0 or "y" not in scored.columns or "p_hat" not in scored.columns:
        return CalibrationReport(
            n=0, brier=0.0, ece=0.0, mae=0.0, base_rate=0.0, brier_skill=0.0, detail="empty"
        )
    col = "p_cal" if "p_cal" in scored.columns else "p_hat"
    y = scored["y"].to_numpy().astype(np.float64)
    p = scored[col].to_numpy().astype(np.float64)
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return CalibrationReport(
            n=0, brier=0.0, ece=0.0, mae=0.0, base_rate=0.0, brier_skill=0.0, detail="all_nan"
        )
    yy, pp = y[mask], p[mask]
    return CalibrationReport(
        n=int(yy.size),
        brier=brier_score(yy, pp),
        ece=expected_calibration_error(yy, pp, n_bins=n_bins),
        mae=float(np.mean(np.abs(yy - pp))),
        base_rate=float(np.mean(yy)),
        brier_skill=brier_skill_score(yy, pp),
        detail=f"reliability_bins={n_bins}·col={col}",
    )


def evaluate_calibration_by_outcome(
    scored: pl.DataFrame,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    schema = {
        "outcome_name": pl.Utf8(),
        "n": pl.Int64(),
        "brier": pl.Float64(),
        "ece": pl.Float64(),
        "mae": pl.Float64(),
        "base_rate": pl.Float64(),
        "brier_skill": pl.Float64(),
    }
    if scored.height == 0 or "outcome_name" not in scored.columns:
        return pl.DataFrame(schema=schema)
    rows: list[dict[str, float | int | str]] = []
    for name, g in scored.group_by("outcome_name", maintain_order=True):
        outcome = name[0] if isinstance(name, tuple) else name
        rep = evaluate_calibration(g, n_bins=n_bins)
        rows.append(
            {
                "outcome_name": str(outcome),
                "n": rep.n,
                "brier": rep.brier,
                "ece": rep.ece,
                "mae": rep.mae,
                "base_rate": rep.base_rate,
                "brier_skill": rep.brier_skill,
            }
        )
    return pl.DataFrame(rows)


__all__ = [
    "CalibrationReport",
    "PlattCalibrator",
    "apply_calibrator",
    "brier_score",
    "brier_skill_score",
    "evaluate_calibration",
    "evaluate_calibration_by_outcome",
    "expected_calibration_error",
    "fit_platt_calibrator",
    "reliability_table",
]
