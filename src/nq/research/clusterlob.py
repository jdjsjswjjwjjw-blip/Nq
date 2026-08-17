"""ClusterLOB: ست سمات للأمر → k-means++ على القطار فقط → OFI لكل عنقود.

طبقة أساس جديدة (ليست شبكة أعمق على براميل 30ث). الدفتر يُعاد يومًا بيوم؛
لا لصق MBO خام عبر الأيام؛ holdout أيلول–كانون مجمّد. العناقيد تُلائَم على
قطار الطيّة فقط. المقارنة: OFI مُجمَّع بلا عنقدة مقابل OFI ثلاثي العناقيد
على ``y_phase_extend``. لا ادّعاء AUC قبل القياس.

Zhang et al., arXiv:2504.20349.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import brier_skill_score, roc_auc
from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.auction_behavior.walk_forward import build_time_folds_for_frame
from nq.contracts.mbo import MboAction, MboSide
from nq.contracts.temporal import EVENT_TS, INGEST_TS
from nq.core.determinism import make_generator
from nq.core.session import session_date_from_ns
from nq.core.time import sort_causal
from nq.orderbook.book import OrderBook
from nq.research.mbo_sequence_mlp import (
    HOLDOUT_START_DATE,
    Y_TARGET,
    assert_single_day_mbo,
    fit_predict_logistic,
    prepare_labels,
)
from nq.states.regimes import KMeansRegimes
from nq.validation.leakage import assert_causal_order

LAYER_ID = "clusterlob_ofi"
WINDOW_NS: Final = 30 * 1_000_000_000
Z_WINDOW: Final = 100
N_CLUSTERS: Final = 3
KMEANS_SUBSAMPLE: Final = 50_000
N_FEATURES: Final = 6
_MATRIX_NDIM: Final = 2
_MIN_Z_SAMPLES: Final = 2
_EPS = 1e-12
_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_CLEAR = MboAction.CLEAR.value
_BID = MboSide.BID.value
_ASK = MboSide.ASK.value
_FEATURE_NAMES: Final = ("v", "t_m", "t_1", "t_prev", "sbs", "obs")


@dataclass(frozen=True, slots=True)
class ClusterDayExtract:
    """أحداث OFI عند الأفضل + سمات الإضافة ليوم جلسة واحد. بلا أيام أخرى."""

    setup_ts: np.ndarray
    y: np.ndarray
    event_ts: np.ndarray
    ingest_ts: np.ndarray
    contrib_s: np.ndarray
    contrib_c: np.ndarray
    order_id: np.ndarray
    add_oid: np.ndarray
    add_z: np.ndarray
    reservoir_z: np.ndarray


def _sum_inclusive(levels: dict[int, int], left: int, right: int) -> int:
    lo = left if left <= right else right
    hi = right if left <= right else left
    total = 0
    for price, size in levels.items():
        if lo <= price <= hi:
            total += size
    return total


def same_side_depth(book: OrderBook, side: str, price: int) -> int:
    """SBS: الحجم من ``p_x`` حتى أفضل عرض على نفس الجانب (قبل إضافة الأمر)."""
    if side == _BID:
        best = book.best_bid()
        if best is None:
            return 0
        return _sum_inclusive(book.bids, price, best[0])
    best = book.best_ask()
    if best is None:
        return 0
    return _sum_inclusive(book.asks, best[0], price)


def opposite_side_depth(book: OrderBook, side: str, price: int) -> int:
    """OBS: الحجم على الجانب المقابل من السعر المتناظر حول الوسط حتى الأفضل."""
    bid = book.best_bid()
    ask = book.best_ask()
    if bid is None or ask is None:
        return 0
    px_sym = bid[0] + ask[0] - price
    if side == _BID:
        return _sum_inclusive(book.asks, ask[0], px_sym)
    return _sum_inclusive(book.bids, px_sym, bid[0])


def _at_or_inside_best(book: OrderBook, side: str, price: int) -> bool:
    if side == _BID:
        best = book.best_bid()
        return best is None or price >= best[0]
    best = book.best_ask()
    return best is None or price <= best[0]


def _trade_signed_size(book: OrderBook, price: int, size: int) -> int | None:
    """ضغط الشراء عند رفع العرض، البيع عند ضرب الطلب. ``None`` إن لم يكن عند الأفضل."""
    ask = book.best_ask()
    if ask is not None and price == ask[0]:
        return int(size)
    bid = book.best_bid()
    if bid is not None and price == bid[0]:
        return -int(size)
    return None


def _add_signed(side: str, size: int) -> tuple[int, int]:
    if side == _BID:
        return int(size), 1
    return -int(size), -1


def _cancel_signed(side: str, size: int) -> tuple[int, int]:
    signed_s, signed_c = _add_signed(side, size)
    return -signed_s, -signed_c


def causal_zscore(raw: np.ndarray, *, window: int = Z_WINDOW) -> np.ndarray:
    """z-score من الـ ``window`` أوامر السابقة فقط. بلا مستقبل."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != _MATRIX_NDIM:
        raise ValueError(f"causal_zscore expects (n, {N_FEATURES}), got {arr.shape}")
    n, dim = arr.shape
    out = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        lo = 0 if i < window else i - window
        prev = arr[lo:i]
        if prev.shape[0] < _MIN_Z_SAMPLES:
            continue
        mu = prev.mean(axis=0)
        sd = prev.std(axis=0)
        sd = np.where(sd < _EPS, 1.0, sd)
        out[i] = (arr[i] - mu) / sd
    return out


def _z_from_prior(raw: np.ndarray, buf: np.ndarray, filled: int) -> np.ndarray:
    if filled < _MIN_Z_SAMPLES:
        return np.zeros(N_FEATURES, dtype=np.float64)
    sl = buf[:filled]
    mu = sl.mean(axis=0)
    sd = sl.std(axis=0)
    sd = np.where(sd < _EPS, 1.0, sd)
    return (raw - mu) / sd


def _push_raw_window(buf: np.ndarray, filled: int, raw: np.ndarray) -> int:
    if filled < buf.shape[0]:
        buf[filled] = raw
        return filled + 1
    buf[:-1] = buf[1:]
    buf[-1] = raw
    return filled


def _reservoir_push(
    reservoir: list[np.ndarray],
    z: np.ndarray,
    *,
    seen: int,
    cap: int,
    rng: np.random.Generator,
) -> int:
    nxt = seen + 1
    if len(reservoir) < cap:
        reservoir.append(z.copy())
        return nxt
    j = int(rng.integers(nxt))
    if j < cap:
        reservoir[j] = z.copy()
    return nxt


def extract_add_features(mbo: pl.DataFrame) -> pl.DataFrame:
    """ست السمات الخام + z السببي لكل إضافة في يوم واحد."""
    stream = extract_day_stream(mbo)
    if stream.add_oid.size == 0:
        return pl.DataFrame(
            schema={
                "order_id": pl.Int64(),
                EVENT_TS: pl.Int64(),
                **{name: pl.Float64() for name in _FEATURE_NAMES},
                **{f"z_{name}": pl.Float64() for name in _FEATURE_NAMES},
            }
        )
    raw = stream.add_raw
    z = stream.add_z
    data: dict[str, list[object]] = {
        "order_id": stream.add_oid.tolist(),
        EVENT_TS: stream.add_ts.tolist(),
    }
    for i, name in enumerate(_FEATURE_NAMES):
        data[name] = raw[:, i].tolist()
        data[f"z_{name}"] = z[:, i].tolist()
    return pl.DataFrame(data)


@dataclass(frozen=True, slots=True)
class _DayStream:
    event_ts: np.ndarray
    ingest_ts: np.ndarray
    contrib_s: np.ndarray
    contrib_c: np.ndarray
    order_id: np.ndarray
    add_oid: np.ndarray
    add_ts: np.ndarray
    add_raw: np.ndarray
    add_z: np.ndarray
    reservoir_z: np.ndarray


class _BookWalker:
    """حالة دفتر يوم واحد: سمات الإضافة + OFI عند الأفضل."""

    def __init__(self, *, seed: int, reservoir_cap: int, z_window: int) -> None:
        self.book = OrderBook()
        self.first_at_price: dict[int, int] = {}
        self.last_at_price: dict[int, int] = {}
        self.last_mid: int | None = None
        self.last_mid_ts: int | None = None
        self.raw_rows: list[list[float]] = []
        self.z_buf = np.zeros((int(z_window), N_FEATURES), dtype=np.float64)
        self.z_filled = 0
        self.add_oid: list[int] = []
        self.add_ts: list[int] = []
        self.add_z: list[np.ndarray] = []
        self.ev_ts: list[int] = []
        self.ev_ing: list[int] = []
        self.ev_s: list[int] = []
        self.ev_c: list[int] = []
        self.ev_oid: list[int] = []
        self.rng = make_generator(seed)
        self.reservoir: list[np.ndarray] = []
        self.seen_z = 0
        self.cap = max(int(reservoir_cap), 1)

    def _push_ofi(self, ts: int, ing: int, signed_s: int, signed_c: int, oid: int) -> None:
        self.ev_ts.append(int(ts))
        self.ev_ing.append(int(ing))
        self.ev_s.append(int(signed_s))
        self.ev_c.append(int(signed_c))
        self.ev_oid.append(int(oid))

    def _note_mid(self, ts: int) -> None:
        bid = self.book.best_bid()
        ask = self.book.best_ask()
        if bid is None or ask is None:
            return
        mid = bid[0] + ask[0]
        if mid != self.last_mid:
            self.last_mid = mid
            self.last_mid_ts = ts

    def _on_add(self, ts: int, ing: int, side: str, price: int, size: int, oid: int) -> None:
        book = self.book
        v = float(book.size_at(side, price)) if side in {_BID, _ASK} else 0.0
        t_m = 0.0 if self.last_mid_ts is None else float(ts - self.last_mid_ts)
        first_ts = self.first_at_price.get(price)
        last_ts = self.last_at_price.get(price)
        t_1 = 0.0 if first_ts is None else float(ts - first_ts)
        t_prev = 0.0 if last_ts is None else float(ts - last_ts)
        sbs = float(same_side_depth(book, side, price)) if side in {_BID, _ASK} else 0.0
        obs = float(opposite_side_depth(book, side, price)) if side in {_BID, _ASK} else 0.0
        raw = np.asarray([v, t_m, t_1, t_prev, sbs, obs], dtype=np.float64)
        z = _z_from_prior(raw, self.z_buf, self.z_filled)
        self.z_filled = _push_raw_window(self.z_buf, self.z_filled, raw)
        self.raw_rows.append(raw.tolist())
        self.add_oid.append(oid)
        self.add_ts.append(ts)
        self.add_z.append(z)
        self.seen_z = _reservoir_push(
            self.reservoir, z, seen=self.seen_z, cap=self.cap, rng=self.rng
        )
        if side in {_BID, _ASK} and _at_or_inside_best(book, side, price):
            signed_s, signed_c = _add_signed(side, size)
            self._push_ofi(ts, ing, signed_s, signed_c, oid)
        if first_ts is None:
            self.first_at_price[price] = ts
        self.last_at_price[price] = ts
        book.apply(_ADD, side, price, size, oid)

    def _on_cancel(self, ts: int, ing: int, side: str, price: int, size: int, oid: int) -> None:
        rec = self.book.orders.get(oid)
        if rec is not None:
            is_bid, rest_price, rest_size = rec
            rest_side = _BID if is_bid else _ASK
            cancel_qty = size if size > 0 else int(rest_size)
            best = self.book.best_bid() if is_bid else self.book.best_ask()
            if best is not None and rest_price == best[0]:
                signed_s, signed_c = _cancel_signed(rest_side, cancel_qty)
                self._push_ofi(ts, ing, signed_s, signed_c, oid)
        self.book.apply(_CANCEL, side, price, size, oid)

    def _on_trade(self, ts: int, ing: int, side: str, price: int, size: int, oid: int) -> None:
        signed = _trade_signed_size(self.book, price, size)
        if signed is not None:
            self._push_ofi(ts, ing, signed, 1 if signed > 0 else -1, oid)
        self.book.apply(_TRADE, side, price, size, oid)

    def on_event(
        self, action: str, ts: int, ing: int, side: str, price: int, size: int, oid: int
    ) -> None:
        if action == _CLEAR:
            self.book.apply(action, side, price, size, oid)
            self.first_at_price.clear()
            self.last_at_price.clear()
            self.last_mid = None
            self.last_mid_ts = None
            return
        if action == _ADD:
            self._on_add(ts, ing, side, price, size, oid)
        elif action == _CANCEL:
            self._on_cancel(ts, ing, side, price, size, oid)
        elif action in {_TRADE, _FILL}:
            self._on_trade(ts, ing, side, price, size, oid)
        else:
            self.book.apply(action, side, price, size, oid)
        self._note_mid(ts)

    def to_stream(self) -> _DayStream:
        n_add = len(self.add_oid)
        empty_f = np.zeros((0, N_FEATURES), dtype=np.float64)
        add_raw = np.asarray(self.raw_rows, dtype=np.float64) if n_add else empty_f
        z_mat = np.vstack(self.add_z) if n_add else empty_f
        res_z = np.vstack(self.reservoir) if self.reservoir else empty_f
        return _DayStream(
            event_ts=np.asarray(self.ev_ts, dtype=np.int64),
            ingest_ts=np.asarray(self.ev_ing, dtype=np.int64),
            contrib_s=np.asarray(self.ev_s, dtype=np.int64),
            contrib_c=np.asarray(self.ev_c, dtype=np.int64),
            order_id=np.asarray(self.ev_oid, dtype=np.int64),
            add_oid=np.asarray(self.add_oid, dtype=np.int64),
            add_ts=np.asarray(self.add_ts, dtype=np.int64),
            add_raw=add_raw,
            add_z=z_mat,
            reservoir_z=res_z,
        )


def _empty_stream() -> _DayStream:
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros((0, N_FEATURES), dtype=np.float64)
    return _DayStream(
        event_ts=empty_i,
        ingest_ts=empty_i,
        contrib_s=empty_i,
        contrib_c=empty_i,
        order_id=empty_i,
        add_oid=empty_i,
        add_ts=empty_i,
        add_raw=empty_f,
        add_z=empty_f,
        reservoir_z=empty_f,
    )


def extract_day_stream(
    mbo: pl.DataFrame,
    *,
    seed: int = 0,
    reservoir_cap: int = KMEANS_SUBSAMPLE,
    z_window: int = Z_WINDOW,
) -> _DayStream:
    """يمشي الدفتر حدثًا بحدث ليوم واحد: سمات الإضافة + مساهمات OFI عند الأفضل."""
    assert_single_day_mbo(mbo)
    if mbo.height == 0:
        return _empty_stream()
    work = sort_causal(mbo)
    times = work[EVENT_TS].to_list()
    ingest = work[INGEST_TS].to_list() if INGEST_TS in work.columns else list(times)
    actions = work["action"].cast(pl.Utf8).fill_null("").to_list()
    sides = (
        work["side"].cast(pl.Utf8).fill_null("N").to_list()
        if "side" in work.columns
        else ["N"] * work.height
    )
    prices = work["price"].cast(pl.Int64).fill_null(0).to_list()
    sizes = work["size"].cast(pl.Int64).fill_null(0).to_list()
    oids = (
        work["order_id"].cast(pl.Int64).fill_null(0).to_list()
        if "order_id" in work.columns
        else [0] * work.height
    )
    walker = _BookWalker(seed=seed, reservoir_cap=reservoir_cap, z_window=z_window)
    for i, action in enumerate(actions):
        walker.on_event(
            str(action),
            int(times[i]),
            int(ingest[i]),
            str(sides[i]),
            int(prices[i]),
            int(sizes[i]),
            int(oids[i]),
        )
    return walker.to_stream()


def ofi_feature_matrix(
    event_ts: np.ndarray,
    ingest_ts: np.ndarray,
    contrib_s: np.ndarray,
    contrib_c: np.ndarray,
    order_id: np.ndarray,
    oid_cluster: Mapping[int, int],
    setup_ts: np.ndarray,
    *,
    window_ns: int = WINDOW_NS,
    n_clusters: int = N_CLUSTERS,
) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 3)`` OFI مُجمَّع و ``(n, 7)`` OFI ثلاثي العناقيد. عمود 0 = اعتراض."""
    setups = np.asarray(setup_ts, dtype=np.int64)
    times = np.asarray(event_ts, dtype=np.int64)
    ingest = np.asarray(ingest_ts, dtype=np.int64)
    size_c = np.asarray(contrib_s, dtype=np.float64)
    count_c = np.asarray(contrib_c, dtype=np.float64)
    oids = np.asarray(order_id, dtype=np.int64)
    pooled = np.zeros((setups.size, 3), dtype=np.float64)
    clustered = np.zeros((setups.size, 1 + 2 * int(n_clusters)), dtype=np.float64)
    pooled[:, 0] = 1.0
    clustered[:, 0] = 1.0
    if times.size == 0 or setups.size == 0:
        return pooled, clustered
    clusters = np.full(oids.size, -1, dtype=np.int64)
    for i, oid in enumerate(oids.tolist()):
        clusters[i] = int(oid_cluster.get(int(oid), -1))
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    ingest = ingest[order]
    size_c = size_c[order]
    count_c = count_c[order]
    clusters = clusters[order]
    for j, t_end in enumerate(setups.tolist()):
        t = int(t_end)
        lo = int(np.searchsorted(times, t - int(window_ns), side="right"))
        hi = int(np.searchsorted(times, t, side="right"))
        if hi <= lo:
            continue
        sl = slice(lo, hi)
        known = ingest[sl] <= t
        if not np.any(known):
            continue
        s_win = size_c[sl][known]
        c_win = count_c[sl][known]
        k_win = clusters[sl][known]
        pooled[j, 1] = float(s_win.sum())
        pooled[j, 2] = float(c_win.sum())
        for k in range(int(n_clusters)):
            mask = k_win == k
            clustered[j, 1 + 2 * k] = float(s_win[mask].sum())
            clustered[j, 2 + 2 * k] = float(c_win[mask].sum())
    return pooled, clustered


def fit_kmeans_train_only(
    train_z: np.ndarray,
    *,
    seed: int = 0,
    n_clusters: int = N_CLUSTERS,
    subsample: int = KMEANS_SUBSAMPLE,
) -> KMeansRegimes:
    """يلائم المراكز على عيّنة القطار فقط. الاختبار يُوسَم بـ ``predict``."""
    arr = np.asarray(train_z, dtype=np.float64)
    if arr.ndim != _MATRIX_NDIM or arr.shape[1] != N_FEATURES:
        raise ValueError(f"train_z must be (n, {N_FEATURES}), got {arr.shape}")
    if arr.shape[0] > int(subsample):
        rng = make_generator(seed)
        idx = rng.choice(arr.shape[0], size=int(subsample), replace=False)
        arr = arr[idx]
    km = KMeansRegimes(int(n_clusters), seed=seed, n_init=5, max_iter=40)
    km.fit(arr)
    return km


def assign_clusters(z: np.ndarray, km: KMeansRegimes) -> np.ndarray:
    if z.shape[0] == 0:
        return np.zeros(0, dtype=np.intp)
    return km.predict(z)


def extract_day(
    mbo: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
    seed: int = 0,
) -> ClusterDayExtract | None:
    """يوم واحد: دفتر ذلك اليوم + تسميات الطور. لا holdout."""
    assert_single_day_mbo(mbo)
    ready = prepare_labels(labels)
    if ready.height == 0:
        return None
    dates = [session_date_from_ns(int(t)) for t in ready[SETUP_AVAILABILITY_TS].to_list()]
    keep = [d < holdout_start for d in dates]
    ready = ready.filter(pl.Series("_keep", keep))
    if ready.height == 0:
        return None
    stream = extract_day_stream(mbo, seed=seed)
    setup = ready[SETUP_AVAILABILITY_TS].to_numpy().astype(np.int64)
    y = ready["y"].to_numpy().astype(np.float64)
    return ClusterDayExtract(
        setup_ts=setup,
        y=y,
        event_ts=stream.event_ts,
        ingest_ts=stream.ingest_ts,
        contrib_s=stream.contrib_s,
        contrib_c=stream.contrib_c,
        order_id=stream.order_id,
        add_oid=stream.add_oid,
        add_z=stream.add_z,
        reservoir_z=stream.reservoir_z,
    )


def _score(y: np.ndarray, p: np.ndarray, *, train_y: np.ndarray) -> dict[str, float]:
    base = float(np.mean(train_y)) if train_y.size else 0.5
    baseline = np.full(y.shape, base, dtype=np.float64)
    return {
        "n": float(y.size),
        "base_rate": float(np.mean(y)) if y.size else float("nan"),
        "auc": float(roc_auc(y, p)),
        "brier_skill": float(brier_skill_score(y, p, baseline_p=baseline)),
    }


def _concat_reservoirs(extracts: list[ClusterDayExtract], idx: np.ndarray) -> np.ndarray:
    parts = [
        extracts[int(i)].reservoir_z for i in idx.tolist() if extracts[int(i)].reservoir_z.size
    ]
    if not parts:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _day_index_for_rows(day_of_row: np.ndarray, row_idx: np.ndarray) -> np.ndarray:
    days = np.unique(day_of_row[row_idx])
    return days.astype(np.int64, copy=False)


def _empty_scored_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            SETUP_AVAILABILITY_TS: pl.Int64(),
            "y": pl.Float64(),
            "p_pooled": pl.Float64(),
            "p_clustered": pl.Float64(),
            "fold": pl.Int64(),
        }
    )


def _base_diagnostics(*, holdout_start: str, window_ns: int, n_days: int) -> dict[str, Any]:
    nan_score = {"n": 0.0, "auc": float("nan"), "brier_skill": float("nan")}
    return {
        "layer_id": LAYER_ID,
        "retrained_existing_heads": False,
        "holdout_touched": False,
        "holdout_start": holdout_start,
        "reconstructed_order_book": True,
        "concatenated_raw_mbo": False,
        "window_ns": int(window_ns),
        "z_window": Z_WINDOW,
        "n_clusters": N_CLUSTERS,
        "target": Y_TARGET,
        "n_days": n_days,
        "kmeans_train_only": True,
        "pooled": dict(nan_score),
        "clustered": dict(nan_score),
    }


def _ofi_design(
    extracts: list[ClusterDayExtract],
    day_of_row: np.ndarray,
    km: KMeansRegimes,
    *,
    n_rows: int,
    window_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_p = np.zeros((n_rows, 3), dtype=np.float64)
    x_c = np.zeros((n_rows, 1 + 2 * N_CLUSTERS), dtype=np.float64)
    for d, ext in enumerate(extracts):
        labels = assign_clusters(ext.add_z, km)
        oid_cluster = {
            int(oid): int(lab)
            for oid, lab in zip(ext.add_oid.tolist(), labels.tolist(), strict=True)
        }
        pooled, clustered = ofi_feature_matrix(
            ext.event_ts,
            ext.ingest_ts,
            ext.contrib_s,
            ext.contrib_c,
            ext.order_id,
            oid_cluster,
            ext.setup_ts,
            window_ns=window_ns,
        )
        mask = day_of_row == d
        x_p[mask] = pooled
        x_c[mask] = clustered
    return x_p, x_c


def _train_z_for_fold(
    extracts: list[ClusterDayExtract],
    day_of_row: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, bool]:
    train_day_set = set(_day_index_for_rows(day_of_row, train_idx).tolist())
    test_day_set = set(_day_index_for_rows(day_of_row, test_idx).tolist())
    pure_train = np.asarray(sorted(train_day_set - test_day_set), dtype=np.int64)
    used_test_day = False
    if pure_train.size == 0:
        pure_train = np.asarray(sorted(train_day_set), dtype=np.int64)
        used_test_day = True
    return _concat_reservoirs(extracts, pure_train), used_test_day


def run_clusterlob(
    days: Iterable[tuple[pl.DataFrame, pl.DataFrame]],
    *,
    seed: int = 0,
    holdout_start: str = HOLDOUT_START_DATE,
    window_ns: int = WINDOW_NS,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """يجمع استخراج الأيام بعد دفتر يومي. لا يلصق MBO الخام. k-means على القطار فقط."""
    extracts: list[ClusterDayExtract] = []
    for i, (mbo, blended) in enumerate(days):
        got = extract_day(mbo, blended, holdout_start=holdout_start, seed=seed + i)
        if got is None:
            continue
        extracts.append(got)
    empty = _empty_scored_frame()
    diagnostics = _base_diagnostics(
        holdout_start=holdout_start, window_ns=window_ns, n_days=len(extracts)
    )
    if not extracts:
        return empty, diagnostics
    setup = np.concatenate([e.setup_ts for e in extracts], axis=0)
    y = np.concatenate([e.y for e in extracts], axis=0)
    day_of_row = np.concatenate(
        [np.full(e.setup_ts.size, i, dtype=np.int64) for i, e in enumerate(extracts)],
        axis=0,
    )
    assert_causal_order(np.sort(setup))
    work = pl.DataFrame({SETUP_AVAILABILITY_TS: setup, "y": y})
    folds = build_time_folds_for_frame(work, n_splits=min(4, max(1, y.size // 8)))
    if not folds:
        return empty, diagnostics
    p_pooled = np.full(y.size, np.nan)
    p_clustered = np.full(y.size, np.nan)
    fold_id = np.full(y.size, -1, dtype=np.int64)
    y_oof: list[np.ndarray] = []
    pooled_oof: list[np.ndarray] = []
    clustered_oof: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    kmeans_used_test_z = False
    for sf in folds:
        test_dates = [session_date_from_ns(int(t)) for t in setup[sf.test_idx].tolist()]
        if any(d >= holdout_start for d in test_dates):
            raise AssertionError("holdout month entered clusterlob OOF")
        train_z, used_test = _train_z_for_fold(extracts, day_of_row, sf.train_idx, sf.test_idx)
        kmeans_used_test_z = kmeans_used_test_z or used_test
        if train_z.shape[0] < N_CLUSTERS:
            continue
        km = fit_kmeans_train_only(train_z, seed=seed + int(sf.fold))
        x_p, x_c = _ofi_design(extracts, day_of_row, km, n_rows=y.size, window_ns=window_ns)
        p_pooled[sf.test_idx] = fit_predict_logistic(x_p, y, sf.train_idx)[sf.test_idx]
        p_clustered[sf.test_idx] = fit_predict_logistic(x_c, y, sf.train_idx)[sf.test_idx]
        fold_id[sf.test_idx] = int(sf.fold)
        y_oof.append(y[sf.test_idx])
        pooled_oof.append(p_pooled[sf.test_idx])
        clustered_oof.append(p_clustered[sf.test_idx])
        train_parts.append(y[sf.train_idx])
    diagnostics["kmeans_fit_includes_test_day_rows"] = kmeans_used_test_z
    if not y_oof:
        return empty, diagnostics
    y_cat = np.concatenate(y_oof)
    train_cat = np.concatenate(train_parts)
    diagnostics["pooled"] = _score(y_cat, np.concatenate(pooled_oof), train_y=train_cat)
    diagnostics["clustered"] = _score(y_cat, np.concatenate(clustered_oof), train_y=train_cat)
    scored = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: setup,
            "y": y,
            "p_pooled": p_pooled,
            "p_clustered": p_clustered,
            "fold": fold_id,
        }
    ).filter(pl.col("fold") >= 0)
    return scored, diagnostics


def write_clusterlob_report(
    scored: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scored.height:
        scored.write_parquet(out / "clusterlob_oof.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    pooled_s = diagnostics.get("pooled", {})
    clustered_s = diagnostics.get("clustered", {})

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
        "# ClusterLOB order-flow imbalance",
        "",
        "Per-day book reconstruction for V / SBS / OBS. Raw MBO is never concatenated.",
        "K-means++ (K=3) is fit on the walk-forward train subsample only.",
        "Primary window is the completed 30s bar at setup_availability_ts.",
        "Control: pooled best-level OFI (no clustering) vs 3-cluster OFI.",
        "Target: y_phase_extend. Holdout Sep–Dec 2025 excluded.",
        "",
        "| model | n | AUC | Brier skill |",
        "|---|---:|---:|---:|",
        (
            f"| pooled OFI | {_fmt(pooled_s, 'n')} | {_fmt(pooled_s, 'auc')} | "
            f"{_fmt(pooled_s, 'brier_skill')} |"
        ),
        (
            f"| clustered OFI | {_fmt(clustered_s, 'n')} | {_fmt(clustered_s, 'auc')} | "
            f"{_fmt(clustered_s, 'brier_skill')} |"
        ),
        "",
        "If clustered ≈ pooled, cluster membership is not orthogonal for this target.",
        "Do not claim 0.65–0.75 until these numbers are measured on May–Aug OOF.",
        "",
    ]
    (out / "CLUSTERLOB.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "HOLDOUT_START_DATE",
    "LAYER_ID",
    "N_CLUSTERS",
    "WINDOW_NS",
    "Y_TARGET",
    "Z_WINDOW",
    "assign_clusters",
    "causal_zscore",
    "extract_add_features",
    "extract_day",
    "extract_day_stream",
    "fit_kmeans_train_only",
    "ofi_feature_matrix",
    "opposite_side_depth",
    "run_clusterlob",
    "same_side_depth",
    "write_clusterlob_report",
]
