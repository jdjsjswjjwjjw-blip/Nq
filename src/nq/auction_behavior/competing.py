"""توزيع competing-risk مشترك للأهداف الأساسية (softmax L2).

الثنائيات المستقلة تسمح بـ P(expansion)=0.8 و P(rejection)=0.7 معًا.
هذا النموذج يعطي صفًا واحدًا على simplex:

    P(expansion) + P(rejection) + P(repriced) + P(residual) = 1

الصفوف right-censored / ambiguous تُستبعد (لا تُفرَض فئة).
عند تعارض y=1 لأكثر من هدف تُؤخذ أولوية ثابتة ويُحسب التعارض في التشخيص.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import (
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
)
from nq.auction_behavior.conditional import (
    MODEL_STATUS_INSUFFICIENT,
    MODEL_STATUS_MISSING,
    MODEL_STATUS_OK,
    MODEL_STATUS_SINGLE_CLASS,
    design_matrix,
)
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    PRIMARY_OUTCOME_TARGETS,
    SETUP_AVAILABILITY_TS,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_temporal_split

_EPS = 1e-12
_MAX_LOGIT = 30.0
_Y_POS = 0.5
_NEWTON_TOL = 1e-7
_MIN_COMPETING_CLASSES = 2
_PROB_NDIM = 2
_FORBIDDEN_FEATURE_PREFIXES = ("y_", "p_y_", "p_hat", "p_cal")
_FORBIDDEN_FEATURE_NAMES = {
    "y",
    "outcome_name",
    "label_status",
    "horizon_bars",
    "class_id",
    "class_name",
}

RESIDUAL_CLASS = "residual"
COMPETING_CLASS_NAMES: tuple[str, ...] = (*PRIMARY_OUTCOME_TARGETS, RESIDUAL_CLASS)
PREDICTION_SOURCE = "state_conditional_competing_softmax"


@dataclass(frozen=True, slots=True)
class CompetingRiskModel:
    """Softmax L2 على فئات المزاد الأساسية + residual."""

    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    weights: np.ndarray  # (d+1, K)
    train_end_ts: int
    n_train: int
    n_per_class: dict[str, int] = field(default_factory=dict)
    n_conflicts: int = 0
    status: str = MODEL_STATUS_MISSING
    class_prior: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    def is_usable(self) -> bool:
        return self.status == MODEL_STATUS_OK and self.weights.size > 0

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        n = frame.height
        k = len(self.class_names)
        if n == 0:
            return np.zeros((0, k), dtype=np.float64)
        if not self.is_usable():
            return np.full((n, k), np.nan, dtype=np.float64)
        x = design_matrix(frame, self.feature_names)
        return _softmax(x @ self.weights)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if z.size == 0:
        return z
    z = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    denom = np.clip(exp.sum(axis=1, keepdims=True), _EPS, None)
    return np.asarray(exp / denom, dtype=np.float64)


def assert_features_are_state_only(feature_names: tuple[str, ...] | list[str]) -> None:
    """الميزات حالة عند الإعداد — ليست التسمية ولا احتمالات لاحقة."""
    for name in feature_names:
        key = str(name)
        if key in _FORBIDDEN_FEATURE_NAMES or key.startswith(_FORBIDDEN_FEATURE_PREFIXES):
            raise AssertionError(f"competing features must not include labels/probs: {key}")
        if key in {SETUP_AVAILABILITY_TS, OUTCOME_AVAILABLE_TS, AVAILABILITY_TS}:
            raise AssertionError(f"timestamp columns are not features: {key}")


def pivot_competing_labels(labeled: pl.DataFrame) -> pl.DataFrame:  # noqa: PLR0912
    """صف واحد لكل إعداد: فئة حصرية بعد حسم كل الأهداف الأساسية.

    يُستبعد الإعداد إذا بقي أي هدف أساسي censored/ambiguous/بدون y محدود.
    تعارض y=1 لأكثر من هدف → أولوية PRIMARY_OUTCOME_TARGETS مع ``n_conflicts``.
    """
    schema: dict[str, pl.DataType] = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        "class_id": pl.Int64(),
        "class_name": pl.Utf8(),
        "n_positive": pl.Int64(),
        "conflict": pl.Boolean(),
        "group_id": pl.Int64(),
    }
    if labeled.height == 0 or "outcome_name" not in labeled.columns:
        return pl.DataFrame(schema=schema)
    prim = labeled.filter(pl.col("outcome_name").is_in(list(PRIMARY_OUTCOME_TARGETS)))
    if prim.height == 0:
        return pl.DataFrame(schema=schema)

    class_index = {name: i for i, name in enumerate(COMPETING_CLASS_NAMES)}
    residual_id = class_index[RESIDUAL_CLASS]
    keep_cols = [c for c in prim.columns if c not in schema]
    rows: list[dict[str, object]] = []
    n_conflicts = 0
    for key, part in prim.group_by(SETUP_AVAILABILITY_TS, maintain_order=True):
        setup_ts = int(key[0] if isinstance(key, tuple) else key)
        if "label_status" in part.columns:
            statuses = set(str(s) for s in part["label_status"].to_list())
            if statuses - {"resolved"}:
                continue
        if "y" not in part.columns:
            continue
        y_map: dict[str, float] = {}
        avail = []
        skip = False
        for rec in part.iter_rows(named=True):
            name = str(rec["outcome_name"])
            y_val = rec["y"]
            if y_val is None or not np.isfinite(float(y_val)):
                skip = True
                break
            y_map[name] = float(y_val)
            avail.append(int(rec[OUTCOME_AVAILABLE_TS]))
        if skip:
            continue
        if any(name not in y_map for name in PRIMARY_OUTCOME_TARGETS):
            # لا نخمّن فئة من أهداف ناقصة
            continue
        positives = [name for name in PRIMARY_OUTCOME_TARGETS if y_map[name] >= _Y_POS]
        conflict = len(positives) > 1
        if conflict:
            n_conflicts += 1
        if not positives:
            class_name = RESIDUAL_CLASS
            class_id = residual_id
        else:
            class_name = positives[0]
            class_id = class_index[class_name]
        feat_row = part.head(1)
        row: dict[str, object] = {
            SETUP_AVAILABILITY_TS: setup_ts,
            OUTCOME_AVAILABLE_TS: int(max(avail)) if avail else setup_ts,
            "class_id": int(class_id),
            "class_name": class_name,
            "n_positive": len(positives),
            "conflict": bool(conflict),
            "group_id": int(feat_row["group_id"][0]) if "group_id" in feat_row.columns else 0,
        }
        for col in keep_cols:
            if col in {
                "outcome_name",
                "y",
                "label_status",
                SETUP_AVAILABILITY_TS,
                OUTCOME_AVAILABLE_TS,
            }:
                continue
            row[col] = feat_row[col][0]
        rows.append(row)
    out = pl.DataFrame(rows) if rows else pl.DataFrame(schema=schema)
    if out.height == 0:
        return out
    return out.with_columns(pl.lit(n_conflicts).alias("n_conflicts_in_batch"))


def competing_known_by(frame: pl.DataFrame, *, asof_ts: int) -> pl.DataFrame:
    """تدريب فقط على فئات عُرفت بالكامل عند/قبل ``asof_ts``."""
    if frame.height == 0 or OUTCOME_AVAILABLE_TS not in frame.columns:
        return frame
    known = frame.filter(pl.col(OUTCOME_AVAILABLE_TS) <= int(asof_ts))
    if SETUP_AVAILABILITY_TS in known.columns:
        known = known.filter(pl.col(SETUP_AVAILABILITY_TS) <= int(asof_ts))
    return known


def _scale_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    scale = np.ones(d, dtype=np.float64)
    if d <= 1 or n == 0:
        return x, scale
    std = np.std(x[:, 1:], axis=0)
    std = np.where(std < _EPS, 1.0, std)
    scale[1:] = std
    return x / scale, scale


def _fit_softmax_l2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    l2: float,
    max_iter: int = 80,
) -> np.ndarray:
    """تدرج مع بحث خطي — بدون scipy."""
    n, d = x.shape
    k = int(n_classes)
    w = np.zeros((d, k), dtype=np.float64)
    if n == 0:
        return w
    counts = np.bincount(y.astype(np.int64), minlength=k).astype(np.float64) + 1.0
    w[0] = np.log(np.clip(counts / counts.sum(), _EPS, 1.0))
    x_s, scale = _scale_features(x)
    y_oh = np.eye(k, dtype=np.float64)[y.astype(np.int64)]
    step = 1.0
    prev = np.inf
    for _ in range(max_iter):
        probs = _softmax(x_s @ w)
        p_true = np.clip(probs[np.arange(n), y.astype(np.int64)], _EPS, 1.0)
        nll = float(-np.mean(np.log(p_true)))
        pen = 0.5 * float(l2) * float(np.sum(w[1:] ** 2)) / float(n)
        loss = nll + pen
        grad = (x_s.T @ (probs - y_oh)) / float(n)
        grad[1:] += (float(l2) / float(n)) * w[1:]
        if float(np.linalg.norm(grad)) < _NEWTON_TOL:
            break
        accepted = False
        trial = step
        for _ls in range(24):
            candidate = w - trial * grad
            probs_c = _softmax(x_s @ candidate)
            p_c = np.clip(probs_c[np.arange(n), y.astype(np.int64)], _EPS, 1.0)
            nll_c = float(-np.mean(np.log(p_c)))
            pen_c = 0.5 * float(l2) * float(np.sum(candidate[1:] ** 2)) / float(n)
            if nll_c + pen_c <= loss + 1e-12:
                w = candidate
                step = min(trial * 1.15, 8.0)
                accepted = True
                loss = nll_c + pen_c
                break
            trial *= 0.5
        if not accepted or abs(prev - loss) < _EPS:
            break
        prev = loss
    return np.asarray(w / scale[:, None], dtype=np.float64)


def fit_competing_risk(
    labeled: pl.DataFrame,
    *,
    feature_names: tuple[str, ...],
    train_end_ts: int,
    l2: float = 1.0,
    min_train: int = 12,
    min_class: int = 2,
    progress: ProgressLike | None = None,
) -> CompetingRiskModel:
    """يدرّب softmax على إعدادات competing محسومة ومعروفة حتى ``train_end_ts``."""
    if progress is not None:
        progress.op(f"fit_competing_risk features={len(feature_names)}")
    assert_features_are_state_only(feature_names)
    work = labeled if "class_id" in labeled.columns else pivot_competing_labels(labeled)
    known = competing_known_by(work, asof_ts=train_end_ts)
    empty_prior = {name: float("nan") for name in COMPETING_CLASS_NAMES}
    empty_counts = {name: 0 for name in COMPETING_CLASS_NAMES}
    n_conflicts = 0
    if known.height and "conflict" in known.columns:
        n_conflicts = int(known["conflict"].sum())

    def _blank(status: str, detail: str) -> CompetingRiskModel:
        return CompetingRiskModel(
            feature_names=feature_names,
            class_names=COMPETING_CLASS_NAMES,
            weights=np.zeros((0, 0), dtype=np.float64),
            train_end_ts=int(train_end_ts),
            n_train=int(known.height),
            n_per_class=empty_counts,
            n_conflicts=n_conflicts,
            status=status,
            class_prior=empty_prior,
            detail=detail,
        )

    if known.height < min_train or not feature_names:
        return _blank(MODEL_STATUS_INSUFFICIENT, "insufficient competing setups")
    y = known["class_id"].to_numpy().astype(np.int64)
    k = len(COMPETING_CLASS_NAMES)
    if y.min() < 0 or y.max() >= k:
        return _blank(MODEL_STATUS_MISSING, "class_id out of range")
    counts = np.bincount(y, minlength=k)
    n_present = int(np.sum(counts >= int(min_class)))
    n_per_class = {name: int(counts[i]) for i, name in enumerate(COMPETING_CLASS_NAMES)}
    prior = {
        name: float((counts[i] + 1.0) / (known.height + k))
        for i, name in enumerate(COMPETING_CLASS_NAMES)
    }
    if n_present < _MIN_COMPETING_CLASSES:
        return CompetingRiskModel(
            feature_names=feature_names,
            class_names=COMPETING_CLASS_NAMES,
            weights=np.zeros((0, 0), dtype=np.float64),
            train_end_ts=int(train_end_ts),
            n_train=int(known.height),
            n_per_class=n_per_class,
            n_conflicts=n_conflicts,
            status=MODEL_STATUS_SINGLE_CLASS,
            class_prior=prior,
            detail="need >=2 classes with support",
        )
    x = design_matrix(known, feature_names)
    weights = _fit_softmax_l2(x, y, n_classes=k, l2=float(l2))
    return CompetingRiskModel(
        feature_names=feature_names,
        class_names=COMPETING_CLASS_NAMES,
        weights=weights,
        train_end_ts=int(train_end_ts),
        n_train=int(known.height),
        n_per_class=n_per_class,
        n_conflicts=n_conflicts,
        status=MODEL_STATUS_OK,
        class_prior=prior,
        detail=f"softmax_l2 · n={known.height} · K={k} · features={len(feature_names)}",
    )


def score_competing_risk(
    model: CompetingRiskModel,
    labeled: pl.DataFrame,
    *,
    embargo: float = 0.0,
    enforce_temporal_split: bool = True,
) -> pl.DataFrame:
    """يسجّل إعدادات اختبار ذات فئة معروفة — بلا استخدام y داخل p."""
    work = labeled if "class_id" in labeled.columns else pivot_competing_labels(labeled)
    empty = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        "class_id": pl.Int64(),
        "class_name": pl.Utf8(),
        "p_true": pl.Float64(),
        "y_hit": pl.Float64(),
        "baseline_p": pl.Float64(),
        "model_status": pl.Utf8(),
    }
    for name in COMPETING_CLASS_NAMES:
        empty[f"p_{name}"] = pl.Float64()
    if work.height == 0:
        return pl.DataFrame(schema=empty)
    if enforce_temporal_split:
        train_ts = np.asarray([model.train_end_ts], dtype=np.float64)
        test_ts = work[SETUP_AVAILABILITY_TS].to_numpy().astype(np.float64)
        assert_temporal_split(train_ts, test_ts, embargo=float(embargo))
    probs = model.predict_proba(work)
    y = work["class_id"].to_numpy().astype(np.int64)
    p_true = np.full(work.height, np.nan, dtype=np.float64)
    if probs.shape[0] == work.height:
        ok = (y >= 0) & (y < probs.shape[1])
        p_true[ok] = probs[np.arange(work.height)[ok], y[ok]]
    baseline = np.array(
        [model.class_prior.get(COMPETING_CLASS_NAMES[int(i)], np.nan) for i in y],
        dtype=np.float64,
    )
    y_hit = (np.argmax(np.nan_to_num(probs, nan=-1.0), axis=1) == y).astype(np.float64)
    cols = [
        pl.Series("p_true", p_true),
        pl.Series("y_hit", y_hit),
        pl.Series("baseline_p", baseline),
        pl.lit(model.status).alias("model_status"),
    ]
    for i, name in enumerate(COMPETING_CLASS_NAMES):
        p_col = probs[:, i] if probs.size else np.zeros(work.height, dtype=np.float64)
        cols.append(pl.Series(f"p_{name}", p_col))
        cols.append(pl.lit(float(model.class_prior.get(name, np.nan))).alias(f"prior_{name}"))
    return work.with_columns(cols)


def attach_competing_to_states(
    model: CompetingRiskModel,
    states: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    ts_col: str = AVAILABILITY_TS,
    prediction_is_oof: bool = False,
    eligible_for_backtest: bool | None = None,
    fold: int | None = None,
) -> pl.DataFrame:
    """يلصق احتمالات simplex على إطار الحالة/التنبؤ (نفس ترتيب ts)."""
    key = ts_col if ts_col in states.columns else AVAILABILITY_TS
    eligible = (
        bool(eligible_for_backtest)
        if eligible_for_backtest is not None
        else bool(prediction_is_oof)
    )
    if states.height == 0:
        return predictions
    work = states.sort(key)
    probs = model.predict_proba(work)
    out = predictions.sort(key) if key in predictions.columns else predictions
    drop = [c for c in (f"p_{name}" for name in COMPETING_CLASS_NAMES) if c in out.columns]
    drop.extend(
        c
        for c in (
            "competing_mass",
            "competing_status",
            "probabilities_are_joint_distribution",
        )
        if c in out.columns
    )
    if drop:
        out = out.drop(drop)
    attached = work.select(key)
    mass = np.nansum(probs, axis=1) if probs.size else np.zeros(work.height)
    exprs = [pl.Series("competing_mass", mass)]
    for i, name in enumerate(COMPETING_CLASS_NAMES):
        exprs.append(pl.Series(f"p_{name}", probs[:, i]))
    attached = attached.with_columns(exprs)
    merged = out.join(attached, on=key, how="left")
    return merged.with_columns(
        pl.lit(PREDICTION_SOURCE).alias("prediction_source"),
        pl.lit(True).alias("probabilities_are_joint_distribution"),
        pl.lit(model.status).alias("competing_status"),
        pl.lit(int(model.train_end_ts)).alias("model_train_end_ts"),
        pl.lit(bool(prediction_is_oof)).alias("prediction_is_oof"),
        pl.lit(eligible).alias("eligible_for_backtest"),
        pl.lit(-1 if fold is None else int(fold)).alias("fold"),
    )


def predict_competing_at_states(
    model: CompetingRiskModel,
    states: pl.DataFrame,
    *,
    ts_col: str = AVAILABILITY_TS,
    prediction_is_oof: bool = False,
    fold: int | None = None,
    eligible_for_backtest: bool | None = None,
) -> pl.DataFrame:
    key = ts_col if ts_col in states.columns else AVAILABILITY_TS
    eligible = (
        bool(eligible_for_backtest)
        if eligible_for_backtest is not None
        else bool(prediction_is_oof)
    )
    schema: dict[str, pl.DataType] = {
        key: pl.Int64(),
        "prediction_source": pl.Utf8(),
        "probabilities_are_joint_distribution": pl.Boolean(),
        "competing_status": pl.Utf8(),
        "model_train_end_ts": pl.Int64(),
        "prediction_is_oof": pl.Boolean(),
        "eligible_for_backtest": pl.Boolean(),
        "fold": pl.Int64(),
        "competing_mass": pl.Float64(),
    }
    for name in COMPETING_CLASS_NAMES:
        schema[f"p_{name}"] = pl.Float64()
    if states.height == 0:
        return pl.DataFrame(schema=schema)
    work = states.sort(key)
    base = work.select(key)
    return attach_competing_to_states(
        model,
        work,
        base,
        ts_col=key,
        prediction_is_oof=prediction_is_oof,
        eligible_for_backtest=eligible,
        fold=fold,
    )


def multiclass_log_loss(y: np.ndarray, probs: np.ndarray) -> float:
    if y.size == 0 or probs.size == 0:
        return 0.0
    ok = np.isfinite(probs).all(axis=1) if probs.ndim == _PROB_NDIM else np.isfinite(probs)
    if not np.any(ok):
        return 0.0
    yy = y[ok].astype(np.int64)
    pp = np.clip(probs[ok], _EPS, 1.0)
    return float(-np.mean(np.log(pp[np.arange(yy.size), yy])))


def multiclass_brier(y: np.ndarray, probs: np.ndarray) -> float:
    if y.size == 0 or probs.size == 0:
        return 0.0
    ok = np.isfinite(probs).all(axis=1)
    if not np.any(ok):
        return 0.0
    yy = y[ok].astype(np.int64)
    pp = probs[ok]
    k = pp.shape[1]
    onehot = np.eye(k, dtype=np.float64)[yy]
    return float(np.mean(np.sum(np.square(pp - onehot), axis=1)))


def evaluate_competing_scores(scored: pl.DataFrame) -> dict[str, float]:
    """Brier / logloss / ECE / AUC / BSS مقابل prior التدريب."""
    empty = {
        "n": 0.0,
        "log_loss": 0.0,
        "brier": 0.0,
        "brier_skill": 0.0,
        "ece": 0.0,
        "accuracy": 0.0,
        "auc_macro": float("nan"),
        "mean_p_true": 0.0,
    }
    if scored.height == 0 or "class_id" not in scored.columns:
        return empty
    y = scored["class_id"].to_numpy().astype(np.int64)
    cols = [f"p_{name}" for name in COMPETING_CLASS_NAMES]
    if any(c not in scored.columns for c in cols):
        return empty
    probs = np.column_stack([scored[c].to_numpy().astype(np.float64) for c in cols])
    ok = np.isfinite(probs).all(axis=1) & np.isfinite(y)
    if not np.any(ok):
        return empty
    yy, pp = y[ok], probs[ok]
    p_true = pp[np.arange(yy.size), yy]
    prior_cols = [f"prior_{name}" for name in COMPETING_CLASS_NAMES]
    if all(c in scored.columns for c in prior_cols):
        prior = np.column_stack([scored[c].to_numpy().astype(np.float64)[ok] for c in prior_cols])
        brier_base = multiclass_brier(yy, prior)
        brier_model = multiclass_brier(yy, pp)
        bss = 0.0 if brier_base < _EPS else float(1.0 - brier_model / brier_base)
    else:
        baseline = (
            scored["baseline_p"].to_numpy().astype(np.float64)[ok]
            if "baseline_p" in scored.columns
            else None
        )
        bss = brier_skill_score(
            np.ones(yy.size, dtype=np.float64),
            p_true,
            None if baseline is None else baseline,
        )
    aucs: list[float] = []
    for k, _name in enumerate(COMPETING_CLASS_NAMES):
        yk = (yy == k).astype(np.float64)
        if yk.min() == yk.max():
            continue
        aucs.append(roc_auc(yk, pp[:, k]))
    auc_macro = float(np.nanmean(aucs)) if aucs else float("nan")
    return {
        "n": float(yy.size),
        "log_loss": multiclass_log_loss(yy, pp),
        "brier": multiclass_brier(yy, pp),
        "brier_skill": bss,
        "ece": expected_calibration_error(
            (np.argmax(pp, axis=1) == yy).astype(np.float64),
            np.max(pp, axis=1),
        ),
        "accuracy": float(np.mean(np.argmax(pp, axis=1) == yy)),
        "auc_macro": auc_macro,
        "mean_p_true": float(np.mean(p_true)),
        "binary_log_loss_true_class": log_loss(
            np.ones(yy.size, dtype=np.float64),
            p_true,
        ),
        "true_class_brier": brier_score(np.ones(yy.size, dtype=np.float64), p_true),
    }


def competing_rows_sum_to_one(probs: np.ndarray, *, atol: float = 1e-6) -> bool:
    if probs.size == 0:
        return True
    mass = np.nansum(probs, axis=1)
    finite = np.isfinite(mass)
    if not np.any(finite):
        return False
    return bool(np.all(np.abs(mass[finite] - 1.0) <= atol))


__all__ = [
    "COMPETING_CLASS_NAMES",
    "PREDICTION_SOURCE",
    "RESIDUAL_CLASS",
    "CompetingRiskModel",
    "assert_features_are_state_only",
    "attach_competing_to_states",
    "competing_known_by",
    "competing_rows_sum_to_one",
    "evaluate_competing_scores",
    "fit_competing_risk",
    "multiclass_brier",
    "multiclass_log_loss",
    "pivot_competing_labels",
    "predict_competing_at_states",
    "score_competing_risk",
]
