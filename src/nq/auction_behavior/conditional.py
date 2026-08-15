"""نموذج احتمالي شرطي (لوجستي L2) — دعم ناقص → null، اختيار عائلات ميزات."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nq.auction_behavior.level_flow import LEVEL_FLOW_COLUMNS
from nq.auction_behavior.memory import SEQUENCE_MEMORY_COLUMNS, list_memory_columns
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    filter_outcomes_known_by,
    filter_resolved_outcomes,
)
from nq.auction_behavior.projection import PROJECTION_NUMERIC_COLUMNS
from nq.auction_behavior.reliability import RELIABILITY_COLUMNS
from nq.auction_behavior.state import STATE_FEATURE_COLUMNS
from nq.auction_behavior.structure import STRUCTURE_FEATURE_COLUMNS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_temporal_split

_EPS = 1e-12
_NEWTON_TOL = 1e-7
_MAX_LOGIT = 35.0
_MIN_ROWS_FOR_VARIANCE = 2
_Y_POS_THRESHOLD = 0.5

MODEL_STATUS_OK = "ok"
MODEL_STATUS_INSUFFICIENT = "insufficient_support"
MODEL_STATUS_SINGLE_CLASS = "single_class"
MODEL_STATUS_MISSING = "missing"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
    return np.asarray(1.0 / (1.0 + np.exp(-zc)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConditionalModel:
    """أوزان لوجستية لكل هدف + حالة الدعم + أسماء الميزات."""

    feature_names: tuple[str, ...]
    weights: dict[str, np.ndarray]  # outcome -> (d+1,) intercept first; empty if unusable
    train_end_ts: int
    n_train: dict[str, int]
    n_pos: dict[str, int] = field(default_factory=dict)
    n_neg: dict[str, int] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)
    feature_names_by_outcome: dict[str, tuple[str, ...]] = field(default_factory=dict)
    base_rate: dict[str, float] = field(default_factory=dict)
    detail: str = ""

    def is_usable(self, outcome: str) -> bool:
        return self.status.get(outcome, MODEL_STATUS_MISSING) == MODEL_STATUS_OK

    def predict_proba(self, frame: pl.DataFrame, outcome: str) -> np.ndarray:
        """احتمال ∈ (0,1) إن كان النموذج صالحًا؛ وإلا NaN (ليس 0.5 صامت)."""
        n = frame.height
        if not self.is_usable(outcome) or outcome not in self.weights:
            return np.full(n, np.nan, dtype=np.float64)
        w = self.weights[outcome]
        if w.size == 0:
            return np.full(n, np.nan, dtype=np.float64)
        names = self.feature_names_by_outcome.get(outcome, self.feature_names)
        x = _design_matrix(frame, names)
        return _sigmoid(x @ w)


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
    return np.column_stack(cols)


def _fit_logistic_l2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 40,
    intercept_prior: float = 0.0,
) -> np.ndarray:
    n, d = x.shape
    if n == 0:
        return np.zeros(d, dtype=np.float64)
    w = np.zeros(d, dtype=np.float64)
    w[0] = float(intercept_prior)
    scale = np.ones(d, dtype=np.float64)
    if d > 1:
        std = np.std(x[:, 1:], axis=0)
        std = np.where(std < _EPS, 1.0, std)
        scale[1:] = std
        x = x / scale
    eye = np.eye(d, dtype=np.float64)
    eye[0, 0] = 0.0
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
    return w / scale


def _variance_ok(frame: pl.DataFrame, name: str) -> bool:
    if name not in frame.columns or frame.height < _MIN_ROWS_FOR_VARIANCE:
        return False
    vals = frame[name].fill_null(0.0).to_numpy().astype(np.float64)
    return bool(np.nanstd(vals) > _EPS)


def select_feature_names(
    frame: pl.DataFrame,
    *,
    preferred: tuple[str, ...] | list[str],
    max_features: int = 64,
) -> tuple[str, ...]:
    """اختيار بسيط (توافق) — يفضّل :func:`select_feature_names_by_family`."""
    names = [c for c in preferred if c in frame.columns and _variance_ok(frame, c)]
    return tuple(names[: max(0, int(max_features))])


def select_feature_names_by_family(
    frame: pl.DataFrame,
    *,
    max_features: int = 64,
    quotas: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """حصص إلزامية لكل عائلة ثم ملء الباقي — يزيل الثابت.

    الافتراضي يضمن دخول projection / structure / sequence / level_flow / reliability
    قبل امتلاء السقف بأعمدة الحالة فقط.
    """
    mem_roll = tuple(
        c
        for c in list_memory_columns(frame)
        if c.startswith("mem_")
        or "__lag" in c
        or "__rmean" in c
        or "__rsum" in c
        or "__ecount" in c
    )
    families: dict[str, tuple[str, ...]] = {
        "state": tuple(c for c in STATE_FEATURE_COLUMNS if c in frame.columns),
        "structure": tuple(c for c in STRUCTURE_FEATURE_COLUMNS if c in frame.columns),
        "projection": tuple(c for c in PROJECTION_NUMERIC_COLUMNS if c in frame.columns),
        "sequence": tuple(c for c in SEQUENCE_MEMORY_COLUMNS if c in frame.columns),
        "level_flow": tuple(c for c in LEVEL_FLOW_COLUMNS if c in frame.columns),
        "reliability": tuple(c for c in RELIABILITY_COLUMNS if c in frame.columns),
        "memory_roll": mem_roll,
        "quality": tuple(
            c
            for c in (
                "signal_quality",
                "signal_evidence",
                "deceptive_score",
                "real_liquidity_ratio",
            )
            if c in frame.columns
        ),
    }
    default_q = {
        "projection": 10,
        "structure": 8,
        "sequence": 8,
        "level_flow": 10,
        "reliability": 6,
        "memory_roll": 8,
        "state": 8,
        "quality": 2,
    }
    q = {**default_q, **(quotas or {})}
    budget = max(0, int(max_features))
    chosen: list[str] = []
    seen: set[str] = set()

    def _take(family: str, limit: int) -> None:
        nonlocal budget
        if budget <= 0 or limit <= 0:
            return
        taken = 0
        for name in families.get(family, ()):
            if taken >= limit or budget <= 0:
                break
            if name in seen or not _variance_ok(frame, name):
                continue
            chosen.append(name)
            seen.add(name)
            taken += 1
            budget -= 1

    family_order = (
        "projection",
        "structure",
        "sequence",
        "level_flow",
        "reliability",
        "memory_roll",
        "state",
        "quality",
    )
    # round-robin: حتى عند ميزانية صغيرة لا تستحوذ عائلة الإسقاط على كل السعة.
    remaining = {fam: max(0, int(q.get(fam, 0))) for fam in family_order}
    while budget > 0 and any(value > 0 for value in remaining.values()):
        before = budget
        for fam in family_order:
            if remaining[fam] <= 0 or budget <= 0:
                continue
            old_n = len(chosen)
            _take(fam, 1)
            if len(chosen) > old_n:
                remaining[fam] -= 1
            else:
                remaining[fam] = 0
        if budget == before:
            break

    # املأ الباقي بأي متغير ذي تباين من العائلات
    if budget > 0:
        for fam in families:
            _take(fam, budget)

    return tuple(chosen)


def fit_conditional_models(
    labeled: pl.DataFrame,
    *,
    feature_names: tuple[str, ...],
    outcomes: tuple[str, ...],
    train_end_ts: int,
    l2: float = 1.0,
    min_train: int = 12,
    min_pos: int = 3,
    min_neg: int = 3,
    min_samples_per_feature: int = 2,
    progress: ProgressLike | None = None,
) -> ConditionalModel:
    """يدرّب نموذجًا شرطيًا لكل هدف على نتائج **محسومة** معروفة حتى ``train_end_ts``."""
    if progress is not None:
        progress.op(
            f"fit_conditional_models outcomes={len(outcomes)} features={len(feature_names)}"
        )
    known = filter_outcomes_known_by(labeled, asof_ts=train_end_ts)
    known = filter_resolved_outcomes(known)
    if SETUP_AVAILABILITY_TS in known.columns:
        known = known.filter(pl.col(SETUP_AVAILABILITY_TS) <= int(train_end_ts))
    if min_samples_per_feature < 1:
        raise ValueError("min_samples_per_feature must be >= 1")

    weights: dict[str, np.ndarray] = {}
    n_train: dict[str, int] = {}
    n_pos: dict[str, int] = {}
    n_neg: dict[str, int] = {}
    status: dict[str, str] = {}
    features_by_outcome: dict[str, tuple[str, ...]] = {}
    base_rate: dict[str, float] = {}

    for i, outcome in enumerate(outcomes, start=1):
        if progress is not None:
            progress.heartbeat(i, len(outcomes), label="fit-outcomes", force=True)
            progress.op(f"fit {outcome} ({i}/{len(outcomes)})")
        part = known.filter(pl.col("outcome_name") == outcome)
        # اسقط NaN في y إن وُجد
        if "y" in part.columns:
            part = part.filter(pl.col("y").is_not_null() & pl.col("y").is_finite())
        n_train[outcome] = int(part.height)
        if part.height == 0:
            weights[outcome] = np.zeros(0, dtype=np.float64)
            n_pos[outcome] = 0
            n_neg[outcome] = 0
            status[outcome] = MODEL_STATUS_INSUFFICIENT
            features_by_outcome[outcome] = ()
            base_rate[outcome] = float("nan")
            continue
        y = part["y"].to_numpy().astype(np.float64)
        pos = int(np.sum(y >= _Y_POS_THRESHOLD))
        neg = int(np.sum(y < _Y_POS_THRESHOLD))
        n_pos[outcome] = pos
        n_neg[outcome] = neg
        base_rate[outcome] = float((pos + 1.0) / (part.height + 2.0))
        if part.height < min_train or pos < min_pos or neg < min_neg:
            weights[outcome] = np.zeros(0, dtype=np.float64)
            status[outcome] = (
                MODEL_STATUS_SINGLE_CLASS if pos == 0 or neg == 0 else MODEL_STATUS_INSUFFICIENT
            )
            features_by_outcome[outcome] = ()
            continue
        max_features_for_outcome = max(1, part.height // int(min_samples_per_feature))
        feats = feature_names[: min(len(feature_names), max_features_for_outcome)]
        features_by_outcome[outcome] = feats
        # smoothing خفيف للـintercept من base rate
        p0 = base_rate[outcome]
        intercept_prior = float(np.log(p0 / max(1.0 - p0, _EPS)))
        x = _design_matrix(part, feats)
        weights[outcome] = _fit_logistic_l2(x, y, l2=l2, intercept_prior=intercept_prior)
        status[outcome] = MODEL_STATUS_OK

    return ConditionalModel(
        feature_names=feature_names,
        weights=weights,
        train_end_ts=int(train_end_ts),
        n_train=n_train,
        n_pos=n_pos,
        n_neg=n_neg,
        status=status,
        feature_names_by_outcome=features_by_outcome,
        base_rate=base_rate,
        detail=(
            f"l2_logistic · train_end_ts={train_end_ts} · candidate_features={len(feature_names)}"
        ),
    )


def score_conditional_models(
    model: ConditionalModel,
    labeled: pl.DataFrame,
    *,
    test_idx: np.ndarray | None = None,
    test_start_ts: int | None = None,
    test_end_ts: int | None = None,
    embargo: float = 0.0,
    enforce_temporal_split: bool = True,
) -> pl.DataFrame:
    """يتنبأ على صفوف اختبار صريحة (مفضّل) أو نطاق زمني.

    لا يستخدم نتائج لم تُتح بعد؛ يُفضَّل ``test_idx`` لمنع تكرار setup عبر الطيّات.
    ``enforce_temporal_split=False`` لمعايرة ذيل داخل القطار فقط.
    """
    empty_schema = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        "outcome_name": pl.Utf8(),
        "y": pl.Float64(),
        "p_hat": pl.Float64(),
        "model_status": pl.Utf8(),
        "baseline_p": pl.Float64(),
        "model_n_train": pl.Int64(),
        "model_n_pos": pl.Int64(),
        "model_n_neg": pl.Int64(),
    }
    work = filter_resolved_outcomes(labeled) if labeled.height else labeled
    if work.height == 0:
        return pl.DataFrame(schema=empty_schema)

    if test_idx is not None:
        idx = np.asarray(test_idx, dtype=np.intp)
        idx = idx[(idx >= 0) & (idx < work.height)]
        test = work[idx]
    else:
        if test_start_ts is None or test_end_ts is None:
            raise ValueError("provide test_idx or both test_start_ts and test_end_ts")
        test = work.filter(
            (pl.col(SETUP_AVAILABILITY_TS) >= int(test_start_ts))
            & (pl.col(SETUP_AVAILABILITY_TS) <= int(test_end_ts))
        )
    if test.height == 0:
        return test.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("p_hat"),
            pl.lit(MODEL_STATUS_MISSING).alias("model_status"),
        )

    if enforce_temporal_split:
        train_ts = np.asarray([model.train_end_ts], dtype=np.float64)
        test_ts = test[SETUP_AVAILABILITY_TS].to_numpy().astype(np.float64)
        assert_temporal_split(train_ts, test_ts, embargo=float(embargo))

    rows: list[pl.DataFrame] = []
    for outcome in model.weights:
        part = test.filter(pl.col("outcome_name") == outcome)
        if part.height == 0:
            continue
        status = model.status.get(outcome, MODEL_STATUS_MISSING)
        p = model.predict_proba(part, outcome)
        rows.append(
            part.with_columns(
                pl.Series("p_hat", p),
                pl.lit(status).alias("model_status"),
                pl.lit(model.base_rate.get(outcome, np.nan)).alias("baseline_p"),
                pl.lit(model.n_train.get(outcome, 0)).alias("model_n_train"),
                pl.lit(model.n_pos.get(outcome, 0)).alias("model_n_pos"),
                pl.lit(model.n_neg.get(outcome, 0)).alias("model_n_neg"),
            )
        )
    if not rows:
        return test.with_columns(
            pl.lit(np.nan).alias("p_hat"),
            pl.lit(MODEL_STATUS_MISSING).alias("model_status"),
        )
    return pl.concat(rows, how="diagonal_relaxed").sort(SETUP_AVAILABILITY_TS)


def predict_probabilities_at_states(
    model: ConditionalModel,
    states: pl.DataFrame,
    *,
    outcomes: tuple[str, ...] | None = None,
    ts_col: str = "availability_ts",
    prediction_is_oof: bool = False,
    fold: int | None = None,
    eligible_for_backtest: bool | None = None,
) -> pl.DataFrame:
    """State(t) → احتمالات شرطية من ميزات الصف فقط (بلا تسميات).

    للتنبؤ الحي: ``prediction_is_oof=False`` و``eligible_for_backtest=False``.
    لسلسلة OOF التاريخية: ابنِ من طيّات الاختبار فقط مع ``prediction_is_oof=True``.
    """
    key = ts_col if ts_col in states.columns else AVAILABILITY_TS
    targets = outcomes if outcomes is not None else tuple(model.weights.keys())
    eligible = (
        bool(eligible_for_backtest)
        if eligible_for_backtest is not None
        else bool(prediction_is_oof)
    )
    meta = {
        "prediction_source": "conditional_logistic_l2",
        "model_train_end_ts": int(model.train_end_ts),
        "prediction_is_oof": bool(prediction_is_oof),
        "eligible_for_backtest": eligible,
        "fold": -1 if fold is None else int(fold),
    }
    if states.height == 0:
        schema: dict[str, pl.DataType] = {
            key: pl.Int64(),
            "prediction_source": pl.Utf8(),
            "model_train_end_ts": pl.Int64(),
            "prediction_is_oof": pl.Boolean(),
            "eligible_for_backtest": pl.Boolean(),
            "fold": pl.Int64(),
        }
        for name in targets:
            schema[f"p_{name}"] = pl.Float64()
            schema[f"status_{name}"] = pl.Utf8()
            schema[f"n_train_{name}"] = pl.Int64()
            schema[f"n_pos_{name}"] = pl.Int64()
            schema[f"n_neg_{name}"] = pl.Int64()
        return pl.DataFrame(schema=schema)

    work = states.sort(key)
    out = work.select(key)
    for name in targets:
        p = model.predict_proba(work, name)
        out = out.with_columns(
            pl.Series(f"p_{name}", p),
            pl.lit(model.status.get(name, MODEL_STATUS_MISSING)).alias(f"status_{name}"),
            pl.lit(model.n_train.get(name, 0)).alias(f"n_train_{name}"),
            pl.lit(model.n_pos.get(name, 0)).alias(f"n_pos_{name}"),
            pl.lit(model.n_neg.get(name, 0)).alias(f"n_neg_{name}"),
        )
    return out.with_columns(
        pl.lit(meta["prediction_source"]).alias("prediction_source"),
        pl.lit(meta["model_train_end_ts"]).alias("model_train_end_ts"),
        pl.lit(meta["prediction_is_oof"]).alias("prediction_is_oof"),
        pl.lit(meta["eligible_for_backtest"]).alias("eligible_for_backtest"),
        pl.lit(meta["fold"]).alias("fold"),
    )


__all__ = [
    "MODEL_STATUS_INSUFFICIENT",
    "MODEL_STATUS_MISSING",
    "MODEL_STATUS_OK",
    "MODEL_STATUS_SINGLE_CLASS",
    "ConditionalModel",
    "fit_conditional_models",
    "predict_probabilities_at_states",
    "score_conditional_models",
    "select_feature_names",
    "select_feature_names_by_family",
]
