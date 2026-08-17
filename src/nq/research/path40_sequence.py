"""مسار 40 برميلًا سببيًا: لقطة البرميل مقابل مسار خطي مقابل لفّ سببي.

ليس إعادة تدريب للرؤوس الثلاثة. ليس live. الـholdout (2025-09+) يُستبعد.
التعلّم العميق هنا = شبكة تسلسل صغيرة على الماضي حتى ``t`` فقط.
احذف هذا الملف + السكربت + الاختبار للإزالة. لا يُصدَّر من ``nq.research``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import brier_skill_score, roc_auc
from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.auction_behavior.phase_extend import (
    PHASE_EXPAND_ATR_FRAC,
    PHASE_GIVEBACK_ATR_FRAC,
    build_phase_extend_outcomes,
    prior_london_atr14,
)
from nq.auction_behavior.walk_forward import build_time_folds_for_frame
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.core.session import session_date_from_ns
from nq.research.behavior_period import assert_not_raw_mbo_stream
from nq.validation.leakage import assert_availability_not_before_event, assert_causal_order

LAYER_ID = "path40_sequence"
Y_PATH40 = "y_path40"
LOOKBACK_BARS = 40
HORIZON_BARS = 40
HOLDOUT_START_DATE = "2025-09-01"
N_CHANNELS = 5  # close, high, low, ret, mask
_GROUP = "_behavior_story_run"
_FIXED_POINT_FLOOR = 1.0 / float(PRICE_SCALE)
_EPS = 1e-12
_MAX_LOGIT = 35.0
_NEWTON_TOL = 1e-7
_CONV_OUT = 8
_CONV_K = 5
_CONV_EPOCHS = 25
_CONV_LR = 0.05
_CONV_L2 = 0.01
_CONV_BATCH = 64


def _price_to_points(px: float) -> float:
    value = float(px)
    if abs(value) >= _FIXED_POINT_FLOOR:
        return value * float(PRICE_SCALE)
    return value


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
    return np.asarray(1.0 / (1.0 + np.exp(-zc)), dtype=np.float64)


def _fit_logistic_l2(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0) -> np.ndarray:
    n, d = x.shape
    if n == 0:
        return np.zeros(d, dtype=np.float64)
    w = np.zeros(d, dtype=np.float64)
    scale = np.ones(d, dtype=np.float64)
    if d > 1:
        std = np.std(x[:, 1:], axis=0)
        std = np.where(std < _EPS, 1.0, std)
        scale[1:] = std
        x = x / scale
    eye = np.eye(d, dtype=np.float64)
    eye[0, 0] = 0.0
    for _ in range(40):
        p = _sigmoid(x @ w)
        grad = x.T @ (p - y) + l2 * (eye @ w)
        hess = (x.T * (p * (1.0 - p))) @ x + l2 * eye
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if float(np.linalg.norm(step)) < _NEWTON_TOL:
            break
    return w / scale


def build_path40_labels(
    blended: pl.DataFrame,
    *,
    window: int = HORIZON_BARS,
) -> pl.DataFrame:
    """وسم 40 برميلًا أمام الإعداد — نفس هندسة الطور، أفق أطول."""
    labeled = build_phase_extend_outcomes(
        blended,
        window=int(window),
        expand_atr_frac=PHASE_EXPAND_ATR_FRAC,
        giveback_atr_frac=PHASE_GIVEBACK_ATR_FRAC,
        group_col=_GROUP if _GROUP in blended.columns else None,
    )
    if labeled.height == 0:
        return labeled
    out = labeled.with_columns(pl.lit(Y_PATH40).alias("outcome_name"))
    if out.height:
        assert_availability_not_before_event(
            out[SETUP_AVAILABILITY_TS].to_numpy().astype(np.int64),
            out[OUTCOME_AVAILABLE_TS].to_numpy().astype(np.int64),
        )
    return out


def _drop_holdout(labels: pl.DataFrame, *, holdout_start: str) -> pl.DataFrame:
    if labels.height == 0:
        return labels
    dates = [session_date_from_ns(int(t)) for t in labels[SETUP_AVAILABILITY_TS].to_list()]
    keep = [d < holdout_start for d in dates]
    return labels.filter(pl.Series("keep", keep))


def build_lookback_tensor(
    blended: pl.DataFrame,
    setup_ts: np.ndarray,
) -> np.ndarray:
    """``(n, 40, 5)`` حتى ``t`` ضمن نفس القصة. القناة الأخيرة قناع."""
    setups = np.asarray(setup_ts, dtype=np.int64)
    n_setup = int(setups.size)
    x = np.zeros((n_setup, LOOKBACK_BARS, N_CHANNELS), dtype=np.float64)
    if n_setup == 0 or blended.height == 0:
        return x
    work = blended.sort(AVAILABILITY_TS)
    ts = work[AVAILABILITY_TS].to_numpy().astype(np.int64)
    assert_causal_order(ts)
    index = {int(t): i for i, t in enumerate(ts.tolist())}
    n = work.height
    groups = (
        work[_GROUP].fill_null(-1).to_numpy().astype(np.int64)
        if _GROUP in work.columns
        else np.zeros(n, dtype=np.int64)
    )
    close = np.array(
        [_price_to_points(float(v)) for v in work["close"].fill_null(0.0).to_list()],
        dtype=np.float64,
    )
    high = np.array(
        [_price_to_points(float(v)) for v in work["high"].fill_null(0.0).to_list()],
        dtype=np.float64,
    )
    low = np.array(
        [_price_to_points(float(v)) for v in work["low"].fill_null(0.0).to_list()],
        dtype=np.float64,
    )
    atr_pts = np.array(
        [_price_to_points(float(v)) for v in prior_london_atr14(work).tolist()],
        dtype=np.float64,
    )
    for s, setup in enumerate(setups.tolist()):
        i = index.get(int(setup))
        if i is None:
            continue
        atr = float(atr_pts[i]) if float(atr_pts[i]) > _EPS else 1.0
        entry = float(close[i])
        for k in range(LOOKBACK_BARS):
            j = i - (LOOKBACK_BARS - 1 - k)
            if j < 0 or groups[j] != groups[i]:
                continue
            if int(ts[j]) > int(ts[i]):
                raise AssertionError("lookback used a bar after setup t")
            prev = float(close[j - 1]) if j > 0 and groups[j - 1] == groups[i] else float(close[j])
            x[s, k, 0] = (float(close[j]) - entry) / atr
            x[s, k, 1] = (float(high[j]) - entry) / atr
            x[s, k, 2] = (float(low[j]) - entry) / atr
            x[s, k, 3] = (float(close[j]) - prev) / atr
            x[s, k, 4] = 1.0
    return x


def last_bar_matrix(x: np.ndarray) -> np.ndarray:
    """لقطة البرميل الأخير فقط (طبقة البرميل)."""
    last = x[:, -1, :4]
    return np.column_stack([np.ones(x.shape[0], dtype=np.float64), last])


def flatten_path_matrix(x: np.ndarray) -> np.ndarray:
    """المسار كاملًا كمتجه خطي — يرى 40 برميلًا بلا لاخطية زمنية."""
    flat = x[:, :, :4].reshape(x.shape[0], LOOKBACK_BARS * 4)
    return np.column_stack([np.ones(x.shape[0], dtype=np.float64), flat])


def _conv1d_causal(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    n, t_len, _cin = x.shape
    c_out, _ci, k_size = weight.shape
    out = np.zeros((n, t_len, c_out), dtype=np.float64)
    out += bias.reshape(1, 1, c_out)
    for k in range(k_size):
        delay = k_size - 1 - k
        if delay == 0:
            src = x
        else:
            pad = np.zeros((n, delay, x.shape[2]), dtype=np.float64)
            src = np.concatenate([pad, x[:, :-delay, :]], axis=1)
        out += src @ weight[:, :, k].T
    return out


def _conv1d_causal_backward(
    x: np.ndarray,
    weight: np.ndarray,
    dout: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, t_len, cin = x.shape
    c_out, _ci, k_size = weight.shape
    d_w = np.zeros_like(weight)
    d_b = dout.sum(axis=(0, 1))
    d_x = np.zeros_like(x)
    for k in range(k_size):
        delay = k_size - 1 - k
        if delay == 0:
            src = x
        else:
            pad = np.zeros((n, delay, cin), dtype=np.float64)
            src = np.concatenate([pad, x[:, :-delay, :]], axis=1)
        d_w[:, :, k] = dout.reshape(n * t_len, c_out).T @ src.reshape(n * t_len, cin)
        dx_src = dout @ weight[:, :, k]
        if delay == 0:
            d_x += dx_src
        else:
            d_x[:, :-delay, :] += dx_src[:, delay:, :]
    return d_x, d_w, d_b


def fit_predict_logistic(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> np.ndarray:
    w = _fit_logistic_l2(x[train], y[train])
    return _sigmoid(x @ w)


def fit_predict_conv(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    rng: np.random.Generator,
    epochs: int = _CONV_EPOCHS,
) -> np.ndarray:
    """لفّ سببي واحد + تجميع مقنّع + رأس لوجستي. الماضي حتى ``t`` فقط."""
    cin = int(x.shape[2])
    weight = rng.normal(0.0, 0.1, size=(_CONV_OUT, cin, _CONV_K))
    bias = np.zeros(_CONV_OUT, dtype=np.float64)
    head = rng.normal(0.0, 0.1, size=_CONV_OUT)
    head_b = 0.0
    mask = x[:, :, 4:5]
    denom = np.clip(mask.sum(axis=1), 1.0, None)
    idx = train.copy()
    for _ in range(int(epochs)):
        rng.shuffle(idx)
        for start in range(0, idx.size, _CONV_BATCH):
            batch = idx[start : start + _CONV_BATCH]
            xb = x[batch]
            yb = y[batch]
            mb = mask[batch]
            db = np.clip(mb.sum(axis=1), 1.0, None)
            pre = _conv1d_causal(xb, weight, bias)
            hid = np.maximum(pre, 0.0)
            pooled = (hid * mb).sum(axis=1) / db
            logits = pooled @ head + head_b
            p = _sigmoid(logits)
            dlogit = p - yb
            d_head = pooled.T @ dlogit + _CONV_L2 * head
            d_head_b = float(dlogit.sum())
            d_pooled = dlogit[:, None] * head.reshape(1, -1)
            d_hid = (d_pooled[:, None, :] * mb) / db[:, None, :]
            d_pre = d_hid * (pre > 0.0)
            _, d_w, d_b = _conv1d_causal_backward(xb, weight, d_pre)
            d_w = d_w + _CONV_L2 * weight
            weight = weight - _CONV_LR * d_w / max(1, batch.size)
            bias = bias - _CONV_LR * d_b / max(1, batch.size)
            head = head - _CONV_LR * d_head / max(1, batch.size)
            head_b = head_b - _CONV_LR * d_head_b / max(1, batch.size)
    pre = _conv1d_causal(x, weight, bias)
    hid = np.maximum(pre, 0.0)
    pooled = (hid * mask).sum(axis=1) / denom
    return _sigmoid(pooled @ head + head_b)


def _score(y: np.ndarray, p: np.ndarray, *, train_y: np.ndarray) -> dict[str, float]:
    base = float(np.mean(train_y)) if train_y.size else 0.5
    baseline = np.full(y.shape, base, dtype=np.float64)
    return {
        "n": float(y.size),
        "base_rate": float(np.mean(y)) if y.size else float("nan"),
        "auc": float(roc_auc(y, p)),
        "brier_skill": float(brier_skill_score(y, p, baseline_p=baseline)),
    }


def run_path40_sequence(
    blended: pl.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
    seed: int = 0,
    conv_epochs: int = _CONV_EPOCHS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """OOF: لقطة برميل مقابل مسار خطي مقابل لفّ سببي على نفس الـ40 برميلًا."""
    assert_not_raw_mbo_stream(blended, source="path40 blended")
    labels = build_path40_labels(blended)
    labels = labels.filter(pl.col("label_status") == "resolved")
    labels = _drop_holdout(labels, holdout_start=holdout_start)
    empty = pl.DataFrame(
        schema={
            SETUP_AVAILABILITY_TS: pl.Int64(),
            "y": pl.Float64(),
            "p_last_bar": pl.Float64(),
            "p_path_linear": pl.Float64(),
            "p_path_conv": pl.Float64(),
            "fold": pl.Int64(),
        }
    )
    diagnostics: dict[str, Any] = {
        "layer_id": LAYER_ID,
        "retrained_existing_heads": False,
        "holdout_touched": False,
        "holdout_start": holdout_start,
        "lookback_bars": LOOKBACK_BARS,
        "horizon_bars": HORIZON_BARS,
        "is_live_overlay": False,
        "uses_future_bars_as_features": False,
        "n_labeled": int(labels.height),
        "last_bar": {"n": 0.0, "auc": float("nan"), "brier_skill": float("nan")},
        "path_linear": {"n": 0.0, "auc": float("nan"), "brier_skill": float("nan")},
        "path_conv": {"n": 0.0, "auc": float("nan"), "brier_skill": float("nan")},
    }
    if labels.height == 0:
        return empty, diagnostics
    setup = labels[SETUP_AVAILABILITY_TS].to_numpy().astype(np.int64)
    y = labels["y"].to_numpy().astype(np.float64)
    tensor = build_lookback_tensor(blended, setup)
    x_last = last_bar_matrix(tensor)
    x_flat = flatten_path_matrix(tensor)
    work = labels.select(SETUP_AVAILABILITY_TS, "y")
    folds = build_time_folds_for_frame(work, n_splits=min(4, max(1, labels.height // 8)))
    if not folds:
        return empty, diagnostics
    rng = make_generator(seed)
    p_last = np.full(y.size, np.nan)
    p_flat = np.full(y.size, np.nan)
    p_conv = np.full(y.size, np.nan)
    fold_id = np.full(y.size, -1, dtype=np.int64)
    y_oof: list[np.ndarray] = []
    last_oof: list[np.ndarray] = []
    flat_oof: list[np.ndarray] = []
    conv_oof: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    for sf in folds:
        test_dates = [session_date_from_ns(int(t)) for t in setup[sf.test_idx].tolist()]
        if any(d >= holdout_start for d in test_dates):
            raise AssertionError("holdout month entered path40 OOF")
        p_last[sf.test_idx] = fit_predict_logistic(x_last, y, sf.train_idx)[sf.test_idx]
        p_flat[sf.test_idx] = fit_predict_logistic(x_flat, y, sf.train_idx)[sf.test_idx]
        p_conv[sf.test_idx] = fit_predict_conv(
            tensor, y, sf.train_idx, rng=rng, epochs=conv_epochs
        )[sf.test_idx]
        fold_id[sf.test_idx] = int(sf.fold)
        y_oof.append(y[sf.test_idx])
        last_oof.append(p_last[sf.test_idx])
        flat_oof.append(p_flat[sf.test_idx])
        conv_oof.append(p_conv[sf.test_idx])
        train_y_parts.append(y[sf.train_idx])
    if not y_oof:
        return empty, diagnostics
    y_cat = np.concatenate(y_oof)
    train_cat = np.concatenate(train_y_parts)
    diagnostics["last_bar"] = _score(y_cat, np.concatenate(last_oof), train_y=train_cat)
    diagnostics["path_linear"] = _score(y_cat, np.concatenate(flat_oof), train_y=train_cat)
    diagnostics["path_conv"] = _score(y_cat, np.concatenate(conv_oof), train_y=train_cat)
    scored = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: setup,
            "y": y,
            "p_last_bar": p_last,
            "p_path_linear": p_flat,
            "p_path_conv": p_conv,
            "fold": fold_id,
        }
    ).filter(pl.col("fold") >= 0)
    return scored, diagnostics


def write_path40_report(
    scored: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scored.height:
        scored.write_parquet(out / "path40_oof.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    last_s = diagnostics.get("last_bar", {})
    lin_s = diagnostics.get("path_linear", {})
    conv_s = diagnostics.get("path_conv", {})

    def _fmt(block: object, key: str) -> str:
        if not isinstance(block, Mapping):
            return "nan"
        raw = block[key] if key in block else float("nan")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return "nan"
        if not np.isfinite(number):
            return "nan"
        return f"{number:.3f}"

    lines = [
        "# path40 sequence (bar vs linear path vs causal conv)",
        "",
        "Lookback 40 bars ending at t. Label is the next 40 bars.",
        "No future bars as features. Existing heads not retrained. Holdout excluded.",
        "",
        "| model | n | AUC | Brier skill |",
        "|---|---:|---:|---:|",
        (
            f"| last_bar | {_fmt(last_s, 'n')} | {_fmt(last_s, 'auc')} | "
            f"{_fmt(last_s, 'brier_skill')} |"
        ),
        (
            f"| path_linear | {_fmt(lin_s, 'n')} | {_fmt(lin_s, 'auc')} | "
            f"{_fmt(lin_s, 'brier_skill')} |"
        ),
        (
            f"| path_conv | {_fmt(conv_s, 'n')} | {_fmt(conv_s, 'auc')} | "
            f"{_fmt(conv_s, 'brier_skill')} |"
        ),
        "",
        "If last_bar fails and path_linear works, the gap was the bar snapshot.",
        "If path_linear fails and path_conv works, the gap is sequence nonlinearity.",
        "If both path models fail, the blended 30s path does not contain the signal.",
        "",
    ]
    (out / "PATH40.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "HOLDOUT_START_DATE",
    "HORIZON_BARS",
    "LAYER_ID",
    "LOOKBACK_BARS",
    "Y_PATH40",
    "build_lookback_tensor",
    "build_path40_labels",
    "flatten_path_matrix",
    "last_bar_matrix",
    "run_path40_sequence",
    "write_path40_report",
]
