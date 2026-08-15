"""نموذج احتمالي شرطي (لوجستي L2 نقي numpy) — بلا sklearn / بلا PnL."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    filter_outcomes_known_by,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.validation.leakage import assert_temporal_split

_EPS = 1e-12
_NEWTON_TOL = 1e-7
_MAX_LOGIT = 35.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
    return np.asarray(1.0 / (1.0 + np.exp(-zc)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConditionalModel:
    """أوزان لوجستية لكل هدف + أسماء الميزات."""

    feature_names: tuple[str, ...]
    weights: dict[str, np.ndarray]  # outcome -> (d+1,) with intercept first
    train_end_ts: int
    n_train: dict[str, int]
    detail: str = ""

    def predict_proba(self, frame: pl.DataFrame, outcome: str) -> np.ndarray:
        if outcome not in self.weights:
            return np.full(frame.height, 0.5, dtype=np.float64)
        x = _design_matrix(frame, self.feature_names)
        return _sigmoid(x @ self.weights[outcome])


def _design_matrix(frame: pl.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    n = frame.height
    if not feature_names:
        return np.ones((n, 1), dtype=np.float64)
    cols: list[np.ndarray] = [np.ones(n, dtype=np.float64)]
    for name in feature_names:
        if name in frame.columns:
            cols.append(frame[name].fill_null(0.0).to_numpy().astype(np.float64))
        else:
            cols.append(np.zeros(n, dtype=np.float64))
    mat = np.column_stack(cols)
    # تطبيع أعمدة الميزات فقط (بدون الـintercept) لتثبيت نيوتن — إحصاء من المصفوفة نفسها
    # يُفضّل أن يُمرَّر train-fit scaler؛ هنا نستخدم z-score داخل الصفوف المتاحة.
    return mat


def _fit_logistic_l2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 40,
) -> np.ndarray:
    n, d = x.shape
    if n == 0:
        return np.zeros(d, dtype=np.float64)
    w = np.zeros(d, dtype=np.float64)
    # وحّد ميزات (كل الأعمدة ما عدا intercept)
    scale = np.ones(d, dtype=np.float64)
    if d > 1:
        std = np.std(x[:, 1:], axis=0)
        std = np.where(std < _EPS, 1.0, std)
        scale[1:] = std
        x = x / scale
    eye = np.eye(d, dtype=np.float64)
    eye[0, 0] = 0.0  # لا تنتظم الـintercept
    for _ in range(max_iter):
        p = _sigmoid(x @ w)
        grad = x.T @ (p - y) + l2 * (eye @ w)
        w_diag = p * (1.0 - p)
        hess = (x.T * w_diag) @ x + l2 * eye
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if float(np.linalg.norm(step)) < _NEWTON_TOL:
            break
    return w / scale  # أعد المقياس إلى وحدات الميزات الأصلية


def select_feature_names(
    frame: pl.DataFrame,
    *,
    preferred: tuple[str, ...] | list[str],
    max_features: int = 48,
) -> tuple[str, ...]:
    """يختار أعمدة رقمية موجودة بحد أقصى ثابت (حتمية بالترتيب)."""
    names = [c for c in preferred if c in frame.columns]
    return tuple(names[: max(0, int(max_features))])


def fit_conditional_models(
    labeled: pl.DataFrame,
    *,
    feature_names: tuple[str, ...],
    outcomes: tuple[str, ...],
    train_end_ts: int,
    l2: float = 1.0,
    min_train: int = 12,
) -> ConditionalModel:
    """يدرّب نموذجًا شرطيًا لكل هدف على نتائج معروفة حتى ``train_end_ts`` فقط."""
    known = filter_outcomes_known_by(labeled, asof_ts=train_end_ts)
    # إعدادات التدريب يجب أن تقع زمنيًا داخل القطار أيضًا.
    if SETUP_AVAILABILITY_TS in known.columns:
        known = known.filter(pl.col(SETUP_AVAILABILITY_TS) <= int(train_end_ts))
    weights: dict[str, np.ndarray] = {}
    n_train: dict[str, int] = {}
    for outcome in outcomes:
        part = known.filter(pl.col("outcome_name") == outcome)
        n_train[outcome] = int(part.height)
        if part.height < min_train:
            weights[outcome] = np.zeros(len(feature_names) + 1, dtype=np.float64)
            continue
        y = part["y"].to_numpy().astype(np.float64)
        x = _design_matrix(part, feature_names)
        weights[outcome] = _fit_logistic_l2(x, y, l2=l2)
    return ConditionalModel(
        feature_names=feature_names,
        weights=weights,
        train_end_ts=int(train_end_ts),
        n_train=n_train,
        detail=f"l2_logistic · train_end_ts={train_end_ts}",
    )


def score_conditional_models(
    model: ConditionalModel,
    labeled: pl.DataFrame,
    *,
    test_start_ts: int,
    test_end_ts: int,
    embargo: float = 0.0,
) -> pl.DataFrame:
    """يتنبأ على إعدادات الاختبار؛ لا يستخدم نتائج لم تُتح بعد لحظة الإعداد."""
    if labeled.height == 0:
        return pl.DataFrame(
            schema={
                SETUP_AVAILABILITY_TS: pl.Int64(),
                OUTCOME_AVAILABLE_TS: pl.Int64(),
                "outcome_name": pl.Utf8(),
                "y": pl.Float64(),
                "p_hat": pl.Float64(),
            }
        )
    test = labeled.filter(
        (pl.col(SETUP_AVAILABILITY_TS) >= int(test_start_ts))
        & (pl.col(SETUP_AVAILABILITY_TS) <= int(test_end_ts))
    )
    if test.height == 0:
        return test.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_hat"))

    # ضمان فصل زمني: إعدادات الاختبار بعد نهاية التدريب (+embargo إن لزم)
    train_ts = np.asarray([model.train_end_ts], dtype=np.float64)
    test_ts = test[SETUP_AVAILABILITY_TS].to_numpy().astype(np.float64)
    assert_temporal_split(train_ts, test_ts, embargo=float(embargo))

    rows: list[pl.DataFrame] = []
    for outcome, _w in model.weights.items():
        part = test.filter(pl.col("outcome_name") == outcome)
        if part.height == 0:
            continue
        p = model.predict_proba(part, outcome)
        rows.append(part.with_columns(pl.Series("p_hat", p)))
    if not rows:
        return test.with_columns(pl.lit(0.5).alias("p_hat"))
    return pl.concat(rows, how="diagonal_relaxed").sort(SETUP_AVAILABILITY_TS)


def predict_probabilities_at_states(
    model: ConditionalModel,
    states: pl.DataFrame,
    *,
    outcomes: tuple[str, ...] | None = None,
    ts_col: str = "availability_ts",
) -> pl.DataFrame:
    """State(t) → احتمالات شرطية — بلا استخدام تسميات النتائج أو المستقبل.

    الناتج إطار تنبؤ منفصل عن ``behavior_state_frame``.
    الأعمدة ``p_*`` احتمال النموذج الشرطي — ليست ``signal_quality``.
    """

    key = ts_col if ts_col in states.columns else AVAILABILITY_TS
    targets = outcomes if outcomes is not None else tuple(model.weights.keys())
    if states.height == 0:
        schema: dict[str, pl.DataType] = {key: pl.Int64()}
        for name in targets:
            schema[f"p_{name}"] = pl.Float64()
        schema["prediction_source"] = pl.Utf8()
        return pl.DataFrame(schema=schema)

    work = states.sort(key)
    out = work.select(key)
    for name in targets:
        p = model.predict_proba(work, name)
        out = out.with_columns(pl.Series(f"p_{name}", p))
    return out.with_columns(pl.lit("conditional_logistic_l2").alias("prediction_source"))


__all__ = [
    "ConditionalModel",
    "fit_conditional_models",
    "predict_probabilities_at_states",
    "score_conditional_models",
    "select_feature_names",
]
