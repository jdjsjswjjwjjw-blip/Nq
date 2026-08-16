"""معايرة احتمالات: قياس (Brier/ECE/BSS) + Platt سببي بسيط."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

_EPS = 1e-12
_MAX_LOGIT = 20.0
_PLATT_MIN_CLASSES = 2
_PLATT_STEP_TOL = 1e-8
_PLATT_MIN_LINE_STEP = 1e-6
_POSITIVE_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """ملخص معايرة لهدف واحد أو مجمّع."""

    n: int
    brier: float
    ece: float
    mae: float
    base_rate: float
    brier_skill: float = 0.0
    log_loss: float = 0.0
    #: تمييز (Mann–Whitney AUC)؛ NaN عند صنف واحد فقط في العينة.
    auc: float = float("nan")
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
        logits = np.clip(self.a + self.b * z, -_MAX_LOGIT, _MAX_LOGIT)
        out = 1.0 / (1.0 + np.exp(-logits))
        return np.asarray(out, dtype=np.float64)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.square(y[mask].astype(np.float64) - p[mask].astype(np.float64))))


def brier_skill_score(
    y: np.ndarray,
    p: np.ndarray,
    baseline_p: np.ndarray | None = None,
) -> float:
    """BSS = 1 - Brier / Brier_base.

    يجب أن يأتي خط الأساس من التدريب الماضي. لا نستخدم معدل الاختبار إلا
    كمسار توافق عندما لا يمرر المستدعي baseline صريحًا.
    """
    mask = np.isfinite(y) & np.isfinite(p)
    if baseline_p is not None:
        baseline_arr = np.asarray(baseline_p, dtype=np.float64)
        if baseline_arr.shape != np.asarray(y).shape:
            raise ValueError("baseline_p must have the same shape as y")
        mask &= np.isfinite(baseline_arr)
    if not np.any(mask):
        return 0.0
    yy = y[mask].astype(np.float64)
    pp = p[mask].astype(np.float64)
    br = brier_score(yy, pp)
    if baseline_p is None:
        base_pred = np.full_like(yy, float(np.mean(yy)))
    else:
        base_pred = np.asarray(baseline_p, dtype=np.float64)[mask]
    br_base = brier_score(yy, base_pred)
    if br_base < _EPS:
        return 0.0 if br < _EPS else -1.0
    return float(1.0 - br / br_base)


def log_loss_score(y: np.ndarray, p: np.ndarray) -> float:
    """Negative log-likelihood متوسط مع قصّ رقمي (يتجاهل NaN)."""
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return 0.0
    yy = y[mask].astype(np.float64)
    pp = np.clip(p[mask].astype(np.float64), _EPS, 1.0 - _EPS)
    return float(-np.mean(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC عبر إحصاء الرتب (Mann–Whitney) مع تصحيح التعادلات.

    يعيد NaN عند غياب أحد الصنفين — لا يُخترع تمييز من عينة أحادية الصنف.
    """
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return float("nan")
    yy = (y[mask].astype(np.float64) >= _POSITIVE_THRESHOLD).astype(np.float64)
    pp = p[mask].astype(np.float64)
    n_pos = int(np.sum(yy))
    n_neg = int(yy.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(pp, kind="mergesort")
    sorted_p = pp[order]
    ranks = np.empty(pp.size, dtype=np.float64)
    i = 0
    while i < sorted_p.size:
        j = i
        while j + 1 < sorted_p.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = float(np.sum(ranks[yy > 0]))
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """ECE على صناديق احتمالية؛ يتكيّف عدد الصناديق مع حجم العينة."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return 0.0
    y = y[mask].astype(np.float64)
    p = np.clip(p[mask].astype(np.float64), 0.0, 1.0)
    # صناديق أقل عند عينات صغيرة لثبات القياس
    adaptive = max(1, min(int(n_bins), max(1, int(y.size // 5))))
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
    l2: float = 1e-2,
) -> PlattCalibrator | None:
    """يدرّب معاير Platt منظمًا على ذيل سببي منفصل (ليس على test).

    نستخدم تصحيح Platt للهدف الثنائي وridge نحو التحويل المحايد ``a=0,b=1``؛
    هذا يمنع معاملات لا نهائية وثقة 0/1 زائفة عند الفصل الكامل في عينات صغيرة.
    """
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    mask = np.isfinite(y) & np.isfinite(p)
    if int(np.sum(mask)) < min_samples:
        return None
    yy = y[mask].astype(np.float64)
    pp = np.clip(p[mask].astype(np.float64), _EPS, 1.0 - _EPS)
    if np.unique(yy).size < _PLATT_MIN_CLASSES:
        return None
    n_pos = int(np.sum(yy >= _POSITIVE_THRESHOLD))
    n_neg = int(yy.size - n_pos)
    hi_target = float((n_pos + 1.0) / (n_pos + 2.0))
    lo_target = float(1.0 / (n_neg + 2.0))
    target = np.where(yy >= _POSITIVE_THRESHOLD, hi_target, lo_target)
    z = np.log(pp / (1.0 - pp))
    z = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
    # نيوتن ثنائي الأبعاد لـ a,b
    a, b = 0.0, 1.0

    def _objective(a_value: float, b_value: float) -> float:
        logits = a_value + b_value * z
        loss = np.logaddexp(0.0, logits) - target * logits
        penalty = 0.5 * float(l2) * (a_value * a_value + (b_value - 1.0) ** 2)
        return float(np.sum(loss) + penalty)

    for _ in range(40):
        logits = a + b * z
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -_MAX_LOGIT, _MAX_LOGIT)))
        w = pred * (1.0 - pred)
        # تدرج/هيسّيان
        r = pred - target
        g_a = float(np.sum(r)) + float(l2) * a
        g_b = float(np.sum(r * z)) + float(l2) * (b - 1.0)
        h_aa = float(np.sum(w)) + float(l2) + 1e-6
        h_bb = float(np.sum(w * z * z)) + float(l2) + 1e-6
        h_ab = float(np.sum(w * z))
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < _EPS:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        old_objective = _objective(a, b)
        step_scale = 1.0
        accepted = False
        while step_scale >= _PLATT_MIN_LINE_STEP:
            candidate_a = a - step_scale * da
            candidate_b = b - step_scale * db
            if _objective(candidate_a, candidate_b) <= old_objective:
                a, b = candidate_a, candidate_b
                accepted = True
                break
            step_scale *= 0.5
        if not accepted or step_scale * (abs(da) + abs(db)) < _PLATT_STEP_TOL:
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


def fit_platt_calibrators_by_outcome(
    scored: pl.DataFrame,
    *,
    min_samples: int = 20,
) -> dict[str, PlattCalibrator]:
    """يدرّب معايرًا مستقلًا لكل outcome؛ يمنع خلط base rates المختلفة."""
    if scored.height == 0 or not {"outcome_name", "y", "p_hat"}.issubset(scored.columns):
        return {}
    result: dict[str, PlattCalibrator] = {}
    for key, part in scored.group_by("outcome_name", maintain_order=True):
        name = str(key[0] if isinstance(key, tuple) else key)
        calibrator = fit_platt_calibrator(
            part["y"].to_numpy().astype(np.float64),
            part["p_hat"].to_numpy().astype(np.float64),
            min_samples=min_samples,
        )
        if calibrator is not None:
            result[name] = calibrator
    return result


def apply_calibrators_by_outcome(
    scored: pl.DataFrame,
    calibrators: Mapping[str, PlattCalibrator],
) -> pl.DataFrame:
    """يطبق معاير الهدف الموافق ويحافظ على ترتيب الصفوف."""
    if scored.height == 0 or "p_hat" not in scored.columns:
        return scored
    if "outcome_name" not in scored.columns:
        raise ValueError("scored frame requires outcome_name for per-outcome calibration")
    indexed = scored.with_row_index("_cal_row")
    pieces: list[pl.DataFrame] = []
    for key, part in indexed.group_by("outcome_name", maintain_order=True):
        name = str(key[0] if isinstance(key, tuple) else key)
        pieces.append(apply_calibrator(part, calibrators.get(name)))
    return pl.concat(pieces, how="diagonal_relaxed").sort("_cal_row").drop("_cal_row")


def apply_calibrators_to_state_predictions(
    predictions: pl.DataFrame,
    calibrators: Mapping[str, PlattCalibrator],
) -> pl.DataFrame:
    """يعاير أعمدة ``p_y_*`` في إطار الحالة، كل هدف بمعايره المستقل."""
    if predictions.height == 0 or not calibrators:
        return predictions
    exprs: list[pl.Series] = []
    for outcome, calibrator in calibrators.items():
        name = f"p_{outcome}"
        if name not in predictions.columns:
            continue
        raw = predictions[name].to_numpy().astype(np.float64)
        finite = np.isfinite(raw)
        transformed = raw.copy()
        if np.any(finite):
            transformed[finite] = calibrator.transform(raw[finite])
        exprs.append(pl.Series(name, transformed))
    return predictions.with_columns(exprs) if exprs else predictions


def evaluate_calibration(scored: pl.DataFrame, *, n_bins: int = 10) -> CalibrationReport:
    """معايرة من إطار فيه ``y`` و ``p_hat`` (يتجاهل NaN)."""
    if scored.height == 0 or "y" not in scored.columns or "p_hat" not in scored.columns:
        return CalibrationReport(
            n=0, brier=0.0, ece=0.0, mae=0.0, base_rate=0.0, brier_skill=0.0, detail="empty"
        )
    col = "p_cal" if "p_cal" in scored.columns else "p_hat"
    y = scored["y"].to_numpy().astype(np.float64)
    p = scored[col].to_numpy().astype(np.float64)
    baseline = (
        scored["baseline_p"].to_numpy().astype(np.float64)
        if "baseline_p" in scored.columns
        else None
    )
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
        brier_skill=brier_skill_score(
            yy,
            pp,
            None if baseline is None else baseline[mask],
        ),
        log_loss=log_loss_score(yy, pp),
        auc=roc_auc(yy, pp),
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
        "log_loss": pl.Float64(),
        "auc": pl.Float64(),
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
                "log_loss": rep.log_loss,
                "auc": rep.auc,
            }
        )
    return pl.DataFrame(rows)


__all__ = [
    "CalibrationReport",
    "PlattCalibrator",
    "apply_calibrator",
    "apply_calibrators_by_outcome",
    "apply_calibrators_to_state_predictions",
    "brier_score",
    "brier_skill_score",
    "evaluate_calibration",
    "evaluate_calibration_by_outcome",
    "expected_calibration_error",
    "fit_platt_calibrator",
    "fit_platt_calibrators_by_outcome",
    "log_loss_score",
    "reliability_table",
    "roc_auc",
]
