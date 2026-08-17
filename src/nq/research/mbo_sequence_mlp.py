"""تسلسل أوامر داخل البرميل (1ث) → MLP وLSTM صغير. بلا دفتر وبلا concat MBO.

قفل: لا تُكدَّس طبقات أعمق على هذا المدخل. OOF السنة (May–Aug 2025،
``y_phase_extend``): aggregate AUC 0.608، MLP 0.529، LSTM-32 0.616.
مسار 40 برميل OHLC كان 0.60–0.62. التيك/المسار ``y_path_further_beyond``
يبقى AUC 0.94. الإشارة الناقصة ليست في البرميل ولا في صناديق 1ث ولا في
شريط OHLC. التشغيل العملي: رأس المسار + خروج يدوي. ليست overlay حيّة.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import brier_skill_score, roc_auc
from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.auction_behavior.phase_extend import build_phase_extend_outcomes
from nq.auction_behavior.walk_forward import build_time_folds_for_frame
from nq.contracts.mbo import MboAction, MboSide
from nq.contracts.temporal import EVENT_TS, INGEST_TS
from nq.core.determinism import make_generator
from nq.core.session import session_date_from_ns
from nq.core.time import sort_causal
from nq.validation.leakage import assert_causal_order

LAYER_ID = "mbo_sequence_mlp"
Y_TARGET = "y_phase_extend"
BIN_NS = 1_000_000_000
N_BINS = 30
N_CHANNELS = 8
HOLDOUT_START_DATE = "2025-09-01"
_GROUP = "_behavior_story_run"
_EPS = 1e-12
_MAX_LOGIT = 35.0
_NEWTON_TOL = 1e-7
_HIDDEN = (32, 16)
_EPOCHS = 40
_LR = 0.05
_L2 = 0.01
_BATCH = 64
_LSTM_HIDDEN = 32
_LSTM_EPOCHS = 25
_LSTM_LR = 0.02
_LSTM_L2 = 0.001
_LSTM_CLIP = 5.0
_MAX_SESSION_SPAN_NS = 36 * 3600 * 1_000_000_000
_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_MODIFY = MboAction.MODIFY.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_BID = MboSide.BID.value
_ASK = MboSide.ASK.value


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -_MAX_LOGIT, _MAX_LOGIT)
    return np.asarray(1.0 / (1.0 + np.exp(-zc)), dtype=np.float64)


def _relu(z: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(z, 0.0), dtype=np.float64)


def assert_single_day_mbo(mbo: pl.DataFrame) -> None:
    """يرفض لصق تدفق MBO عبر الأيام. يقاس على ``ingest_ts`` (نافذة الاستلام).

    ``event_ts`` في ملف جلسة Databento قد يحمل طوابع أوامر أقدم؛ ذلك لا يعني لصق أيام.
    """
    if mbo.height == 0:
        return
    col = INGEST_TS if INGEST_TS in mbo.columns else EVENT_TS
    if col not in mbo.columns:
        return
    lo = int(mbo.select(pl.col(col).min()).item())
    hi = int(mbo.select(pl.col(col).max()).item())
    if hi - lo > _MAX_SESSION_SPAN_NS:
        raise ValueError(
            "refuse concatenated multi-day MBO; extract sequences one session day at a time"
        )


def assert_no_future_events(mbo: pl.DataFrame, setup_ts: int) -> pl.DataFrame:
    """أحداث معروفة عند ``t`` فقط: event_ts و ingest_ts ≤ الإعداد."""
    work = sort_causal(mbo)
    t = int(setup_ts)
    if INGEST_TS in work.columns:
        work = work.filter((pl.col(EVENT_TS) <= t) & (pl.col(INGEST_TS) <= t))
    else:
        work = work.filter(pl.col(EVENT_TS) <= t)
    return work


def _mbo_event_arrays(
    mbo: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """مصفوفات مرتّبة بـ ``event_ts`` لنافذة البحث. بلا دفتر."""
    if mbo.height == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        empty_u = np.zeros(0, dtype=np.str_)
        return empty_i, empty_i, empty_u, empty_u, empty_f
    work = sort_causal(mbo)
    times = work[EVENT_TS].to_numpy().astype(np.int64, copy=False)
    ingest = (
        work[INGEST_TS].to_numpy().astype(np.int64, copy=False)
        if INGEST_TS in work.columns
        else times
    )
    actions = work["action"].cast(pl.Utf8).fill_null("").to_numpy()
    sides = (
        work["side"].cast(pl.Utf8).fill_null("N").to_numpy()
        if "side" in work.columns
        else np.full(work.height, "N", dtype=np.str_)
    )
    sizes = (
        work["size"].cast(pl.Float64).fill_null(0.0).to_numpy()
        if "size" in work.columns
        else np.zeros(work.height, dtype=np.float64)
    )
    order = np.argsort(times, kind="mergesort")
    return times[order], ingest[order], actions[order], sides[order], sizes[order]


def _fill_window(
    seq: np.ndarray,
    *,
    times: np.ndarray,
    ingest: np.ndarray,
    actions: np.ndarray,
    sides: np.ndarray,
    sizes: np.ndarray,
    t_end: int,
    n_bins: int,
    bin_ns: int,
) -> None:
    t_start = int(t_end) - int(n_bins) * int(bin_ns)
    lo = int(np.searchsorted(times, t_start, side="right"))
    hi = int(np.searchsorted(times, t_end, side="right"))
    if hi <= lo:
        return
    sl = slice(lo, hi)
    known = ingest[sl] <= int(t_end)
    if not np.any(known):
        return
    win_t = times[sl][known]
    win_a = actions[sl][known]
    win_s = sides[sl][known]
    win_z = sizes[sl][known]
    bin_end = ((win_t - 1) // int(bin_ns) + 1) * int(bin_ns)
    idx = int(n_bins) - 1 - ((int(t_end) - bin_end) // int(bin_ns))
    valid = (idx >= 0) & (idx < int(n_bins)) & (bin_end <= int(t_end))
    if not np.any(valid):
        return
    idx_v = idx[valid]
    act_v = win_a[valid]
    side_v = win_s[valid]
    sz_v = win_z[valid]
    is_add = act_v == _ADD
    is_cancel = act_v == _CANCEL
    is_trade = (act_v == _TRADE) | (act_v == _FILL)
    is_mod = act_v == _MODIFY
    if np.any(is_add):
        np.add.at(seq[:, 0], idx_v[is_add], 1.0)
        np.add.at(seq[:, 4], idx_v[is_add], sz_v[is_add])
        bid = is_add & (side_v == _BID)
        ask = is_add & (side_v == _ASK)
        if np.any(bid):
            np.add.at(seq[:, 6], idx_v[bid], 1.0)
        if np.any(ask):
            np.add.at(seq[:, 7], idx_v[ask], 1.0)
    if np.any(is_cancel):
        np.add.at(seq[:, 1], idx_v[is_cancel], 1.0)
        np.add.at(seq[:, 5], idx_v[is_cancel], sz_v[is_cancel])
    if np.any(is_trade):
        np.add.at(seq[:, 2], idx_v[is_trade], 1.0)
    if np.any(is_mod):
        np.add.at(seq[:, 3], idx_v[is_mod], 1.0)


def build_intra_bar_sequence(
    mbo: pl.DataFrame,
    setup_ts: int,
    *,
    n_bins: int = N_BINS,
    bin_ns: int = BIN_NS,
) -> np.ndarray:
    """``(30, 8)`` براميل 1ث مكتملة تنتهي عند ``t``. لا دفتر."""
    seq = build_sequences_for_setups(
        mbo, np.asarray([int(setup_ts)], dtype=np.int64), n_bins=n_bins, bin_ns=bin_ns
    )
    return np.asarray(seq[0], dtype=np.float64)


def build_sequences_for_setups(
    mbo: pl.DataFrame,
    setup_ts: np.ndarray,
    *,
    n_bins: int = N_BINS,
    bin_ns: int = BIN_NS,
) -> np.ndarray:
    assert_single_day_mbo(mbo)
    setups = np.asarray(setup_ts, dtype=np.int64)
    out = np.zeros((setups.size, int(n_bins), N_CHANNELS), dtype=np.float64)
    if mbo.height == 0 or setups.size == 0:
        return out
    times, ingest, actions, sides, sizes = _mbo_event_arrays(mbo)
    for i, t in enumerate(setups.tolist()):
        _fill_window(
            out[i],
            times=times,
            ingest=ingest,
            actions=actions,
            sides=sides,
            sizes=sizes,
            t_end=int(t),
            n_bins=int(n_bins),
            bin_ns=int(bin_ns),
        )
    return out


def collapse_sequence(x: np.ndarray) -> np.ndarray:
    """مجاميع 30ث — مكافئ ``lf_*``: نفس الأحداث بلا ترتيب."""
    collapsed = x.sum(axis=1)
    return np.column_stack([np.ones(x.shape[0], dtype=np.float64), collapsed])


def flatten_sequence(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], N_BINS * N_CHANNELS)
    return np.column_stack([np.ones(x.shape[0], dtype=np.float64), flat])


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


def fit_predict_logistic(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> np.ndarray:
    w = _fit_logistic_l2(x[train], y[train])
    return _sigmoid(x @ w)


def _forward_mlp(
    x: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    acts = [x]
    h = x
    for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
        z = h @ w + b
        h = _sigmoid(z) if i == len(weights) - 1 else _relu(z)
        acts.append(h)
    return acts, h.reshape(-1)


def fit_predict_mlp(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    rng: np.random.Generator,
    epochs: int = _EPOCHS,
) -> np.ndarray:
    """MLP طبقتان خفيتان على التسلسل المسطّح. حتمي من ``rng``."""
    feat = flatten_sequence(x)
    d_in = feat.shape[1]
    sizes = [d_in, *_HIDDEN, 1]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for a, b in pairwise(sizes):
        weights.append(rng.normal(0.0, 0.1, size=(a, b)))
        biases.append(np.zeros(b, dtype=np.float64))
    idx = train.copy()
    for _ in range(int(epochs)):
        rng.shuffle(idx)
        for start in range(0, idx.size, _BATCH):
            batch = idx[start : start + _BATCH]
            xb = feat[batch]
            yb = y[batch]
            acts, p = _forward_mlp(xb, weights, biases)
            d = (p - yb).reshape(-1, 1)
            for layer in range(len(weights) - 1, -1, -1):
                a_prev = acts[layer]
                w = weights[layer]
                grad_w = a_prev.T @ d / max(1, batch.size) + _L2 * w
                grad_b = d.mean(axis=0)
                d_prev = d @ w.T
                if layer > 0:
                    d_prev = d_prev * (a_prev > 0.0)
                weights[layer] = w - _LR * grad_w
                biases[layer] = biases[layer] - _LR * grad_b
                d = d_prev
    _, out = _forward_mlp(feat, weights, biases)
    return out


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.asarray(np.tanh(np.clip(z, -20.0, 20.0)), dtype=np.float64)


def _standardize_seq(x: np.ndarray, train: np.ndarray) -> np.ndarray:
    """متوسط/انحراف القنوات من القطار فقط."""
    src = x[train]
    mu = src.mean(axis=(0, 1), keepdims=True)
    sd = src.std(axis=(0, 1), keepdims=True)
    sd = np.where(sd < _EPS, 1.0, sd)
    return np.asarray((x - mu) / sd, dtype=np.float64)


def _lstm_forward(
    xb: np.ndarray,
    wx: np.ndarray,
    wh: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, list[tuple[np.ndarray, ...]]]:
    """``xb`` شكل ``(B, T, F)``. يعيد آخر ``h`` والكاش للخلف."""
    batch, steps, _feat = xb.shape
    hidden = wh.shape[0]
    h = np.zeros((batch, hidden), dtype=np.float64)
    cell = np.zeros((batch, hidden), dtype=np.float64)
    cache: list[tuple[np.ndarray, ...]] = []
    for step in range(steps):
        x_t = xb[:, step]
        h_prev = h
        c_prev = cell
        pre = x_t @ wx + h_prev @ wh + bias
        f_g = _sigmoid(pre[:, :hidden])
        i_g = _sigmoid(pre[:, hidden : 2 * hidden])
        o_g = _sigmoid(pre[:, 2 * hidden : 3 * hidden])
        g_g = _tanh(pre[:, 3 * hidden :])
        cell = f_g * c_prev + i_g * g_g
        h = o_g * _tanh(cell)
        cache.append((x_t, h_prev, c_prev, f_g, i_g, o_g, g_g, cell, h))
    return h, cache


def _lstm_backward(
    d_h_last: np.ndarray,
    cache: list[tuple[np.ndarray, ...]],
    wx: np.ndarray,
    wh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = wh.shape[0]
    d_wx = np.zeros_like(wx)
    d_wh = np.zeros_like(wh)
    d_b = np.zeros(4 * hidden, dtype=np.float64)
    d_h = d_h_last
    d_c = np.zeros_like(d_h)
    for x_t, h_prev, c_prev, f_g, i_g, o_g, g_g, c_t, _h_t in reversed(cache):
        tanh_c = _tanh(c_t)
        d_o = d_h * tanh_c
        d_c = d_c + d_h * o_g * (1.0 - tanh_c**2)
        d_f = d_c * c_prev
        d_i = d_c * g_g
        d_g = d_c * i_g
        d_c = d_c * f_g
        d_pre = np.concatenate(
            [
                d_f * f_g * (1.0 - f_g),
                d_i * i_g * (1.0 - i_g),
                d_o * o_g * (1.0 - o_g),
                d_g * (1.0 - g_g**2),
            ],
            axis=1,
        )
        d_wx = d_wx + x_t.T @ d_pre
        d_wh = d_wh + h_prev.T @ d_pre
        d_b = d_b + d_pre.sum(axis=0)
        d_h = d_pre @ wh.T
    return d_wx, d_wh, d_b


def fit_predict_lstm(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    rng: np.random.Generator,
    hidden: int = _LSTM_HIDDEN,
    epochs: int = _LSTM_EPOCHS,
) -> np.ndarray:
    """LSTM واحد 32 وحدة على ``(T, F)``. حتمي من ``rng``. بلا Torch."""
    if train.size == 0:
        return np.full(x.shape[0], 0.5, dtype=np.float64)
    seq = _standardize_seq(x, train)
    n_feat = int(seq.shape[2])
    n_hidden = int(hidden)
    scale = 0.1
    wx = rng.normal(0.0, scale, size=(n_feat, 4 * n_hidden))
    wh = rng.normal(0.0, scale, size=(n_hidden, 4 * n_hidden))
    bias = np.zeros(4 * n_hidden, dtype=np.float64)
    bias[:n_hidden] = 1.0
    w_out = rng.normal(0.0, scale, size=n_hidden)
    b_out = 0.0
    idx = train.copy()
    for _ in range(int(epochs)):
        rng.shuffle(idx)
        for start in range(0, idx.size, _BATCH):
            batch = idx[start : start + _BATCH]
            xb = seq[batch]
            yb = y[batch]
            h_last, cache = _lstm_forward(xb, wx, wh, bias)
            logit = h_last @ w_out + b_out
            pred = _sigmoid(logit)
            d_logit = (pred - yb) / max(1, batch.size)
            d_w_out = h_last.T @ d_logit + _LSTM_L2 * w_out
            d_b_out = float(d_logit.sum())
            d_h = d_logit[:, None] * w_out[None, :]
            d_wx, d_wh, d_b = _lstm_backward(d_h, cache, wx, wh)
            d_wx = _clip_grad(d_wx + _LSTM_L2 * wx)
            d_wh = _clip_grad(d_wh + _LSTM_L2 * wh)
            d_w_out = _clip_grad(d_w_out)
            wx = wx - _LSTM_LR * d_wx
            wh = wh - _LSTM_LR * d_wh
            bias = bias - _LSTM_LR * d_b
            w_out = w_out - _LSTM_LR * d_w_out
            b_out = b_out - _LSTM_LR * d_b_out
    out = np.empty(seq.shape[0], dtype=np.float64)
    for start in range(0, seq.shape[0], _BATCH):
        sl = slice(start, start + _BATCH)
        h_last, _cache = _lstm_forward(seq[sl], wx, wh, bias)
        out[sl] = _sigmoid(h_last @ w_out + b_out)
    return out


def _clip_grad(grad: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(grad))
    if norm > _LSTM_CLIP:
        return grad * (_LSTM_CLIP / norm)
    return grad


def _score(y: np.ndarray, p: np.ndarray, *, train_y: np.ndarray) -> dict[str, float]:
    base = float(np.mean(train_y)) if train_y.size else 0.5
    baseline = np.full(y.shape, base, dtype=np.float64)
    return {
        "n": float(y.size),
        "base_rate": float(np.mean(y)) if y.size else float("nan"),
        "auc": float(roc_auc(y, p)),
        "brier_skill": float(brier_skill_score(y, p, baseline_p=baseline)),
    }


def resolve_idrive_mbo(mbo_root: Path | str, day: str) -> Path | None:
    """``2025-05-01`` → ملف جلسة IDrive لذلك اليوم فقط. لا لصق عبر الأيام."""
    root = Path(mbo_root)
    yyyy, mm, dd = day.split("-")
    stem = f"glbx-mdp3-{yyyy}{mm}{dd}"
    month_dir = root / f"MES_MBO_{yyyy}_{mm}"
    candidates = (
        month_dir / f"{stem}.continuous.clean.parquet",
        month_dir / f"{stem}.mbo.continuous.clean.parquet",
        root / day / "mbo.parquet",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def labels_from_blended(blended: pl.DataFrame) -> pl.DataFrame:
    labeled = build_phase_extend_outcomes(
        blended,
        group_col=_GROUP if _GROUP in blended.columns else None,
    )
    if labeled.height == 0:
        return labeled
    return labeled.filter(pl.col("label_status") == "resolved")


def prepare_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """تسميات جاهزة من fold_scores/period، أو إعادة حساب من blended الفترة — ليس من يوم واحد."""
    work = frame
    if "outcome_name" in work.columns:
        work = work.filter(pl.col("outcome_name") == Y_TARGET)
    if "label_status" in work.columns:
        work = work.filter(pl.col("label_status") == "resolved")
    if "y" in work.columns and SETUP_AVAILABILITY_TS in work.columns and work.height:
        return work
    return labels_from_blended(frame)


def sequences_from_labels(
    mbo: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """يوم واحد من MBO + تسميات ذلك اليوم فقط. لا لصق تدفق خام."""
    assert_single_day_mbo(mbo)
    empty = np.zeros((0, N_BINS, N_CHANNELS), dtype=np.float64)
    if labels.height == 0:
        return empty, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    dates = [session_date_from_ns(int(t)) for t in labels[SETUP_AVAILABILITY_TS].to_list()]
    keep = [d < holdout_start for d in dates]
    labels = labels.filter(pl.Series("_keep", keep))
    if labels.height == 0:
        return empty, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    setup = labels[SETUP_AVAILABILITY_TS].to_numpy().astype(np.int64)
    y = labels["y"].to_numpy().astype(np.float64)
    x = build_sequences_for_setups(mbo, setup)
    return x, y, setup


def sequences_from_day(
    mbo: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """يوم واحد: MBO ذلك اليوم فقط + تسميات الطور."""
    return sequences_from_labels(mbo, prepare_labels(blended), holdout_start=holdout_start)


def run_mbo_sequence_mlp(
    days: Iterable[tuple[pl.DataFrame, pl.DataFrame]],
    *,
    seed: int = 0,
    epochs: int = _EPOCHS,
    holdout_start: str = HOLDOUT_START_DATE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """يجمع *مصفوفات* الأيام بعد الاستخراج. لا يلصق MBO الخام."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for mbo, blended in days:
        x, y, setup = sequences_from_day(mbo, blended, holdout_start=holdout_start)
        if x.shape[0] == 0:
            continue
        xs.append(x)
        ys.append(y)
        ts.append(setup)
    empty = pl.DataFrame(
        schema={
            SETUP_AVAILABILITY_TS: pl.Int64(),
            "y": pl.Float64(),
            "p_aggregate": pl.Float64(),
            "p_mlp": pl.Float64(),
            "p_lstm": pl.Float64(),
            "fold": pl.Int64(),
        }
    )
    diagnostics: dict[str, Any] = {
        "layer_id": LAYER_ID,
        "retrained_existing_heads": False,
        "holdout_touched": False,
        "holdout_start": holdout_start,
        "reconstructed_order_book": False,
        "concatenated_raw_mbo": False,
        "bin_ns": BIN_NS,
        "n_bins": N_BINS,
        "target": Y_TARGET,
        "n_days": len(xs),
        "lstm_hidden": _LSTM_HIDDEN,
        "aggregate": {"n": 0.0, "auc": float("nan"), "brier_skill": float("nan")},
        "mlp": {"n": 0.0, "auc": float("nan"), "brier_skill": float("nan")},
        "lstm": {"n": 0.0, "auc": float("nan"), "brier_skill": float("nan")},
    }
    if not xs:
        return empty, diagnostics
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    setup = np.concatenate(ts, axis=0)
    assert_causal_order(np.sort(setup))
    work = pl.DataFrame({SETUP_AVAILABILITY_TS: setup, "y": y})
    folds = build_time_folds_for_frame(work, n_splits=min(4, max(1, y.size // 8)))
    if not folds:
        return empty, diagnostics
    rng = make_generator(seed)
    p_agg = np.full(y.size, np.nan)
    p_mlp = np.full(y.size, np.nan)
    p_lstm = np.full(y.size, np.nan)
    fold_id = np.full(y.size, -1, dtype=np.int64)
    y_oof: list[np.ndarray] = []
    agg_oof: list[np.ndarray] = []
    mlp_oof: list[np.ndarray] = []
    lstm_oof: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    x_agg = collapse_sequence(x)
    for sf in folds:
        test_dates = [session_date_from_ns(int(t)) for t in setup[sf.test_idx].tolist()]
        if any(d >= holdout_start for d in test_dates):
            raise AssertionError("holdout month entered mbo-sequence OOF")
        p_agg[sf.test_idx] = fit_predict_logistic(x_agg, y, sf.train_idx)[sf.test_idx]
        p_mlp[sf.test_idx] = fit_predict_mlp(x, y, sf.train_idx, rng=rng, epochs=epochs)[
            sf.test_idx
        ]
        p_lstm[sf.test_idx] = fit_predict_lstm(x, y, sf.train_idx, rng=rng)[sf.test_idx]
        fold_id[sf.test_idx] = int(sf.fold)
        y_oof.append(y[sf.test_idx])
        agg_oof.append(p_agg[sf.test_idx])
        mlp_oof.append(p_mlp[sf.test_idx])
        lstm_oof.append(p_lstm[sf.test_idx])
        train_parts.append(y[sf.train_idx])
    y_cat = np.concatenate(y_oof)
    train_cat = np.concatenate(train_parts)
    diagnostics["aggregate"] = _score(y_cat, np.concatenate(agg_oof), train_y=train_cat)
    diagnostics["mlp"] = _score(y_cat, np.concatenate(mlp_oof), train_y=train_cat)
    diagnostics["lstm"] = _score(y_cat, np.concatenate(lstm_oof), train_y=train_cat)
    scored = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: setup,
            "y": y,
            "p_aggregate": p_agg,
            "p_mlp": p_mlp,
            "p_lstm": p_lstm,
            "fold": fold_id,
        }
    ).filter(pl.col("fold") >= 0)
    return scored, diagnostics


def write_mbo_sequence_report(
    scored: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scored.height:
        scored.write_parquet(out / "mbo_sequence_oof.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    agg_s = diagnostics.get("aggregate", {})
    mlp_s = diagnostics.get("mlp", {})
    lstm_s = diagnostics.get("lstm", {})

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
        "# MBO intra-bar sequence MLP / LSTM",
        "",
        "1-second bins inside the completed 30s bar at t. No book reconstruction.",
        "Raw MBO is never concatenated across days. Holdout excluded.",
        "Aggregate = same events summed (lf_* analogue).",
        "MLP sees flattened order. LSTM (32) sees (T, F) order.",
        "",
        "| model | n | AUC | Brier skill |",
        "|---|---:|---:|---:|",
        (
            f"| aggregate 30s | {_fmt(agg_s, 'n')} | {_fmt(agg_s, 'auc')} | "
            f"{_fmt(agg_s, 'brier_skill')} |"
        ),
        (
            f"| mlp sequence | {_fmt(mlp_s, 'n')} | {_fmt(mlp_s, 'auc')} | "
            f"{_fmt(mlp_s, 'brier_skill')} |"
        ),
        (
            f"| lstm sequence | {_fmt(lstm_s, 'n')} | {_fmt(lstm_s, 'auc')} | "
            f"{_fmt(lstm_s, 'brier_skill')} |"
        ),
        "",
        "If LSTM ≈ aggregate, 1s add/cancel order is not the missing signal.",
        "If LSTM beats aggregate and MLP, the choke was flattened time.",
        "",
        "Freeze: do not stack deeper nets on 30s bars, 1s MBO bins, or 40-bar OHLC.",
        "Path/tick head (y_path_further_beyond) stays the working model; exits stay manual.",
        "",
    ]
    (out / "MBO_SEQUENCE.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "BIN_NS",
    "HOLDOUT_START_DATE",
    "LAYER_ID",
    "N_BINS",
    "Y_TARGET",
    "assert_single_day_mbo",
    "build_intra_bar_sequence",
    "build_sequences_for_setups",
    "collapse_sequence",
    "fit_predict_logistic",
    "fit_predict_lstm",
    "fit_predict_mlp",
    "labels_from_blended",
    "prepare_labels",
    "resolve_idrive_mbo",
    "run_mbo_sequence_mlp",
    "sequences_from_day",
    "sequences_from_labels",
    "write_mbo_sequence_report",
]
