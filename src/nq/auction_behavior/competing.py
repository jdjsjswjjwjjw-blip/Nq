"""نموذج مخاطر متنافسة: softmax متعدد الفئات لأول انتقال — مجموع الاحتمالات = 1.

يجيب عن السؤال المشترك مباشرة: «من هذه الحالة، أي انتقال يحدث **أولًا**؟»
بدل ثلاث ثنائيات مستقلة قد تتجاوز مجموعها 1. الفئة ``no_transition`` صريحة
(نافذة مكتملة بلا انتقال)، وright-censored لا يدخل التدريب إطلاقًا.

المعايرة عبر temperature واحد يُدرَّب على ذيل سببي من القطار (يقسم على
اللوجيت قبل softmax) — يحافظ على ترتيب الفئات ومجموع 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import (
    FIRST_TRANSITION_CLASS_COL,
    FIRST_TRANSITION_CLASSES,
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_temporal_split

_EPS = 1e-12
_MAX_LOGIT = 35.0
_NEWTON_TOL = 1e-7
_TEMPERATURE_MIN = 0.25
_TEMPERATURE_MAX = 4.0

COMPETING_STATUS_OK = "ok"
COMPETING_STATUS_INSUFFICIENT = "insufficient_support"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.clip(logits, -_MAX_LOGIT, _MAX_LOGIT)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / np.sum(e, axis=1, keepdims=True), dtype=np.float64)


def _design_matrix(frame: pl.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    n = frame.height
    cols: list[np.ndarray] = [np.ones(n, dtype=np.float64)]
    for name in feature_names:
        if name in frame.columns:
            cols.append(frame[name].fill_null(0.0).to_numpy().astype(np.float64))
        else:
            cols.append(np.zeros(n, dtype=np.float64))
    return np.column_stack(cols)


@dataclass(frozen=True, slots=True)
class CompetingRiskModel:
    """أوزان multinomial (K-1, d+1) — الفئة الأخيرة مرجعية (لوجيت صفري)."""

    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    weights: np.ndarray
    train_end_ts: int
    n_train: int
    class_counts: dict[str, int] = field(default_factory=dict)
    status: str = COMPETING_STATUS_INSUFFICIENT
    temperature: float = 1.0
    detail: str = ""

    def is_usable(self) -> bool:
        return self.status == COMPETING_STATUS_OK and self.weights.size > 0

    def _logits(self, frame: pl.DataFrame) -> np.ndarray:
        x = _design_matrix(frame, self.feature_names)
        raw = x @ self.weights.T  # (n, K-1)
        full = np.column_stack([raw, np.zeros(frame.height, dtype=np.float64)])
        return np.asarray(full / max(self.temperature, _EPS), dtype=np.float64)

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        """مصفوفة (n, K) صفوفها تجمع لـ1؛ NaN كاملة إن كان النموذج غير صالح."""
        n = frame.height
        k = len(self.classes)
        if not self.is_usable():
            return np.full((n, k), np.nan, dtype=np.float64)
        return _softmax(self._logits(frame))


def fit_temperature(logits: np.ndarray, y_index: np.ndarray) -> float:
    """temperature واحد يقلّل NLL على عينة معايرة سببية (بحث شبكي دقيق)."""
    if logits.shape[0] == 0:
        return 1.0
    grid = np.linspace(_TEMPERATURE_MIN, _TEMPERATURE_MAX, 76)
    best_t, best_nll = 1.0, float("inf")
    idx = np.arange(y_index.size)
    for t in grid:
        p = _softmax(logits / float(t))
        nll = float(-np.mean(np.log(np.clip(p[idx, y_index], _EPS, 1.0))))
        if nll < best_nll - 1e-12:
            best_nll, best_t = nll, float(t)
    return best_t


def fit_competing_risk_model(  # noqa: PLR0912, PLR0915
    labeled: pl.DataFrame,
    *,
    feature_names: tuple[str, ...],
    train_end_ts: int,
    l2: float = 1.0,
    min_train: int = 24,
    min_class_count: int = 3,
    max_iter: int = 60,
    classes: tuple[str, ...] = FIRST_TRANSITION_CLASSES,
    progress: ProgressLike | None = None,
) -> CompetingRiskModel:
    """يدرّب softmax على إعدادات محسومة معروفة حتى ``train_end_ts`` فقط.

    فئة بلا دعم كافٍ لا تُحذف من التوزيع (تبقى في softmax) لكن النموذج كله
    يُعلَّم ``insufficient_support`` إذا قلّ الدعم الكلي أو غاب صنفان.
    """
    if progress is not None:
        progress.op(f"fit_competing_risk features={len(feature_names)} rows={labeled.height:,}")
    work = labeled
    if "label_status" in work.columns:
        work = work.filter(pl.col("label_status") == "resolved")
    if OUTCOME_AVAILABLE_TS in work.columns:
        work = work.filter(pl.col(OUTCOME_AVAILABLE_TS) <= int(train_end_ts))
    if SETUP_AVAILABILITY_TS in work.columns:
        work = work.filter(pl.col(SETUP_AVAILABILITY_TS) <= int(train_end_ts))
    if work.height and FIRST_TRANSITION_CLASS_COL in work.columns:
        work = work.filter(pl.col(FIRST_TRANSITION_CLASS_COL).is_in(list(classes)))

    counts = {c: 0 for c in classes}
    if work.height and FIRST_TRANSITION_CLASS_COL in work.columns:
        for key, part in work.group_by(FIRST_TRANSITION_CLASS_COL):
            name = str(key[0] if isinstance(key, tuple) else key)
            if name in counts:
                counts[name] = int(part.height)
    n_supported = sum(1 for c in classes if counts[c] >= min_class_count)
    n_total = int(work.height)
    insufficient = n_total < min_train or n_supported < 2  # noqa: PLR2004

    if insufficient:
        return CompetingRiskModel(
            feature_names=feature_names,
            classes=classes,
            weights=np.zeros(0, dtype=np.float64),
            train_end_ts=int(train_end_ts),
            n_train=n_total,
            class_counts=counts,
            status=COMPETING_STATUS_INSUFFICIENT,
            detail=f"n={n_total} supported_classes={n_supported}",
        )

    class_to_idx = {c: k for k, c in enumerate(classes)}
    y_idx = np.asarray(
        [class_to_idx[str(v)] for v in work[FIRST_TRANSITION_CLASS_COL].to_list()],
        dtype=np.intp,
    )
    x = _design_matrix(work, feature_names)
    n, d = x.shape
    k = len(classes)
    scale = np.ones(d, dtype=np.float64)
    if d > 1:
        std = np.std(x[:, 1:], axis=0)
        scale[1:] = np.where(std < _EPS, 1.0, std)
        x = x / scale
    y_onehot = np.zeros((n, k), dtype=np.float64)
    y_onehot[np.arange(n), y_idx] = 1.0

    w = np.zeros((k - 1, d), dtype=np.float64)
    # intercept من معدلات الأساس (مقابل الفئة المرجعية) — smoothing لابلاس
    ref_rate = (counts[classes[-1]] + 1.0) / (n_total + k)
    for kk in range(k - 1):
        rate = (counts[classes[kk]] + 1.0) / (n_total + k)
        w[kk, 0] = float(np.log(rate / ref_rate))

    penalty = np.eye(d, dtype=np.float64)
    penalty[0, 0] = 0.0
    n_params = (k - 1) * d
    for it in range(max_iter):
        if progress is not None:
            progress.heartbeat(it + 1, max_iter, label="competing-newton")
        logits = np.column_stack([x @ w.T, np.zeros(n, dtype=np.float64)])
        p = _softmax(logits)
        grad = np.zeros(n_params, dtype=np.float64)
        hess = np.zeros((n_params, n_params), dtype=np.float64)
        for a in range(k - 1):
            ga = x.T @ (p[:, a] - y_onehot[:, a]) + l2 * (penalty @ w[a])
            grad[a * d : (a + 1) * d] = ga
            for b in range(k - 1):
                wt = p[:, a] * ((1.0 if a == b else 0.0) - p[:, b])
                block = (x.T * wt) @ x
                if a == b:
                    block = block + l2 * penalty + 1e-9 * np.eye(d)
                hess[a * d : (a + 1) * d, b * d : (b + 1) * d] = block
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step.reshape(k - 1, d)
        if float(np.linalg.norm(step)) < _NEWTON_TOL:
            break

    return CompetingRiskModel(
        feature_names=feature_names,
        classes=classes,
        weights=w / scale,
        train_end_ts=int(train_end_ts),
        n_train=n_total,
        class_counts=counts,
        status=COMPETING_STATUS_OK,
        detail=f"multinomial_l2 · n={n_total} · d={d - 1}",
    )


def calibrate_competing_temperature(
    model: CompetingRiskModel,
    calibration: pl.DataFrame,
) -> CompetingRiskModel:
    """يعاير temperature على ذيل قطار سببي؛ يعيد النموذج نفسه إن تعذّر."""
    if not model.is_usable() or calibration.height == 0:
        return model
    work = calibration
    if "label_status" in work.columns:
        work = work.filter(pl.col("label_status") == "resolved")
    if work.height == 0 or FIRST_TRANSITION_CLASS_COL not in work.columns:
        return model
    work = work.filter(pl.col(FIRST_TRANSITION_CLASS_COL).is_in(list(model.classes)))
    if work.height == 0:
        return model
    class_to_idx = {c: k for k, c in enumerate(model.classes)}
    y_idx = np.asarray(
        [class_to_idx[str(v)] for v in work[FIRST_TRANSITION_CLASS_COL].to_list()],
        dtype=np.intp,
    )
    x = _design_matrix(work, model.feature_names)
    raw = np.column_stack([x @ model.weights.T, np.zeros(work.height, dtype=np.float64)])
    temperature = fit_temperature(raw, y_idx)
    return CompetingRiskModel(
        feature_names=model.feature_names,
        classes=model.classes,
        weights=model.weights,
        train_end_ts=model.train_end_ts,
        n_train=model.n_train,
        class_counts=model.class_counts,
        status=model.status,
        temperature=temperature,
        detail=f"{model.detail} · temperature={temperature:.3f}",
    )


def score_competing_risk_model(
    model: CompetingRiskModel,
    labeled: pl.DataFrame,
    *,
    enforce_temporal_split: bool = True,
    embargo: float = 0.0,
) -> pl.DataFrame:
    """يتنبأ على صفوف اختبار محسومة ويعيد ``p_first_*`` + الفئة الحقيقية."""
    schema: dict[str, pl.DataType] = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        FIRST_TRANSITION_CLASS_COL: pl.Utf8(),
        "model_status": pl.Utf8(),
        "model_n_train": pl.Int64(),
    }
    for c in model.classes:
        schema[f"p_first_{c}"] = pl.Float64()
    work = labeled
    if "label_status" in work.columns and work.height:
        work = work.filter(pl.col("label_status") == "resolved")
    if work.height == 0:
        return pl.DataFrame(schema=schema)
    if enforce_temporal_split:
        assert_temporal_split(
            np.asarray([model.train_end_ts], dtype=np.float64),
            work[SETUP_AVAILABILITY_TS].to_numpy().astype(np.float64),
            embargo=float(embargo),
        )
    probs = model.predict_proba(work)
    out = work.select(
        SETUP_AVAILABILITY_TS,
        OUTCOME_AVAILABLE_TS,
        FIRST_TRANSITION_CLASS_COL,
    ).with_columns(
        pl.lit(model.status).alias("model_status"),
        pl.lit(model.n_train).alias("model_n_train"),
        *[pl.Series(f"p_first_{c}", probs[:, k]) for k, c in enumerate(model.classes)],
    )
    return out.sort(SETUP_AVAILABILITY_TS)


def predict_competing_at_states(
    model: CompetingRiskModel,
    states: pl.DataFrame,
    *,
    prediction_is_oof: bool = False,
    fold: int | None = None,
    eligible_for_backtest: bool | None = None,
) -> pl.DataFrame:
    """State(t) → توزيع أول انتقال (مجموعه 1) من ميزات الصف فقط."""
    eligible = (
        bool(eligible_for_backtest) if eligible_for_backtest is not None else prediction_is_oof
    )
    if states.height == 0:
        schema: dict[str, pl.DataType] = {
            AVAILABILITY_TS: pl.Int64(),
            "prediction_source": pl.Utf8(),
            "model_train_end_ts": pl.Int64(),
            "prediction_is_oof": pl.Boolean(),
            "eligible_for_backtest": pl.Boolean(),
            "fold": pl.Int64(),
        }
        for c in model.classes:
            schema[f"p_first_{c}"] = pl.Float64()
        return pl.DataFrame(schema=schema)
    work = states.sort(AVAILABILITY_TS)
    probs = model.predict_proba(work)
    return work.select(AVAILABILITY_TS).with_columns(
        *[pl.Series(f"p_first_{c}", probs[:, k]) for k, c in enumerate(model.classes)],
        pl.lit("competing_risk_multinomial_l2").alias("prediction_source"),
        pl.lit(int(model.train_end_ts)).alias("model_train_end_ts"),
        pl.lit(bool(prediction_is_oof)).alias("prediction_is_oof"),
        pl.lit(eligible).alias("eligible_for_backtest"),
        pl.lit(-1 if fold is None else int(fold)).alias("fold"),
    )


def evaluate_competing_scores(
    scored: pl.DataFrame, *, classes: tuple[str, ...]
) -> dict[str, float]:
    """Brier متعدد الفئات + log loss + دقة argmax على صفوف محسومة."""
    if scored.height == 0 or FIRST_TRANSITION_CLASS_COL not in scored.columns:
        return {"n": 0.0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0}
    prob_cols = [f"p_first_{c}" for c in classes]
    if any(c not in scored.columns for c in prob_cols):
        return {"n": 0.0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0}
    p = scored.select(prob_cols).to_numpy().astype(np.float64)
    finite = np.all(np.isfinite(p), axis=1)
    if not np.any(finite):
        return {"n": 0.0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0}
    labels = scored[FIRST_TRANSITION_CLASS_COL].to_list()
    class_to_idx = {c: k for k, c in enumerate(classes)}
    y_idx = np.asarray([class_to_idx.get(str(v), -1) for v in labels], dtype=np.intp)
    keep = finite & (y_idx >= 0)
    if not np.any(keep):
        return {"n": 0.0, "brier": 0.0, "log_loss": 0.0, "accuracy": 0.0}
    pp = p[keep]
    yy = y_idx[keep]
    onehot = np.zeros_like(pp)
    onehot[np.arange(yy.size), yy] = 1.0
    brier = float(np.mean(np.sum(np.square(pp - onehot), axis=1)))
    nll = float(-np.mean(np.log(np.clip(pp[np.arange(yy.size), yy], _EPS, 1.0))))
    acc = float(np.mean(np.argmax(pp, axis=1) == yy))
    return {"n": float(yy.size), "brier": brier, "log_loss": nll, "accuracy": acc}


__all__ = [
    "COMPETING_STATUS_INSUFFICIENT",
    "COMPETING_STATUS_OK",
    "CompetingRiskModel",
    "calibrate_competing_temperature",
    "evaluate_competing_scores",
    "fit_competing_risk_model",
    "fit_temperature",
    "predict_competing_at_states",
    "score_competing_risk_model",
]
