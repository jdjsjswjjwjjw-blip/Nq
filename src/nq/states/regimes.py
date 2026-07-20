"""حالات السوق عبر تجميع حتمي (Deterministic Regime Clustering).

``KMeansRegimes`` تنفيذ حتمي لـ k-means (تهيئة k-means++ + تكرارات Lloyd) على
numpy، يُلائَم على التدريب فقط ويُوسِّم الاختبار للأمام (منع التسريب). تُشتق منه
حالات سوقية discrete قابلة للتفسير عبر مراكزها وإحصاءاتها، وديناميكيتها عبر
مصفوفة الانتقالات السببية وأزمنة المكوث.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

_MATRIX_NDIM = 2
_MIN_CLUSTERS = 2

# ترتيب ميزات التجميع في tick stream (يجب أن يطابق ``tick_stream``)
MARKET_REGIME_FEATURE_NAMES: Final = (
    "near_vah",
    "near_val",
    "in_value_area",
    "poc_dist_norm",
    "vah_dist_norm",
    "val_dist_norm",
    "nq_trail_bid_liq_log",
    "nq_trail_ask_liq_log",
)

PHASE_BALANCE = 0
PHASE_EXPANSION = 1
PHASE_NEUTRAL = 2

_IN_VALUE_ACTIVE = 0.5
_MIN_PHASE_FEATURES = 3
_REGIME_FEATURE_DIM = len(MARKET_REGIME_FEATURE_NAMES)


def _pairwise_sqdist(x: FloatArray, centers: FloatArray) -> FloatArray:
    """مصفوفة مربّعات المسافات ``(n, k)`` بين النقاط والمراكز."""
    x_sq = np.sum(x**2, axis=1, keepdims=True)
    c_sq = np.sum(centers**2, axis=1)
    cross = x @ centers.T
    return np.asarray(np.maximum(x_sq + c_sq - 2.0 * cross, 0.0), dtype=np.float64)


class KMeansRegimes:
    """مُجمِّع حالات سوقية حتمي (k-means)."""

    __slots__ = (
        "_fitted",
        "centroids_",
        "inertia_",
        "max_iter",
        "n_init",
        "n_regimes",
        "seed",
        "tol",
    )

    def __init__(
        self,
        n_regimes: int,
        *,
        seed: int = 0,
        max_iter: int = 100,
        n_init: int = 10,
        tol: float = 1e-8,
    ) -> None:
        if n_regimes < 1:
            raise ValueError(f"n_regimes must be >= 1, got {n_regimes}")
        self.n_regimes = n_regimes
        self.seed = seed
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.centroids_: FloatArray | None = None
        self.inertia_: float = np.inf
        self._fitted = False

    def _kmeanspp_init(self, x: FloatArray, rng: np.random.Generator) -> FloatArray:
        n = x.shape[0]
        first = int(rng.integers(n))
        centers = [x[first]]
        for _ in range(1, self.n_regimes):
            dist2 = _pairwise_sqdist(x, np.asarray(centers)).min(axis=1)
            total = float(dist2.sum())
            if total <= 0:
                centers.append(x[int(rng.integers(n))])
                continue
            probs = dist2 / total
            centers.append(x[int(rng.choice(n, p=probs))])
        return np.asarray(centers, dtype=np.float64)

    def _lloyd(self, x: FloatArray, centers: FloatArray) -> tuple[FloatArray, float]:
        for _ in range(self.max_iter):
            labels = _pairwise_sqdist(x, centers).argmin(axis=1)
            new_centers = centers.copy()
            for k in range(self.n_regimes):
                members = x[labels == k]
                if members.shape[0] > 0:
                    new_centers[k] = members.mean(axis=0)
            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            if shift <= self.tol:
                break
        inertia = float(_pairwise_sqdist(x, centers).min(axis=1).sum())
        return centers, inertia

    def fit(self, x: FloatArray) -> KMeansRegimes:
        """يلائم المراكز على بيانات التدريب فقط (أفضل ``n_init`` تهيئات)."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != _MATRIX_NDIM:
            raise ValueError(f"KMeansRegimes expects a 2-D matrix, got shape {arr.shape}")
        if arr.shape[0] < self.n_regimes:
            raise ValueError(
                f"need at least n_regimes={self.n_regimes} samples, got {arr.shape[0]}"
            )
        rng = np.random.default_rng(self.seed)
        best_centers: FloatArray | None = None
        best_inertia = np.inf
        for _ in range(self.n_init):
            centers = self._kmeanspp_init(arr, rng)
            centers, inertia = self._lloyd(arr, centers)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
        self.centroids_ = best_centers
        self.inertia_ = best_inertia
        self._fitted = True
        return self

    def predict(self, x: FloatArray) -> IntArray:
        """يُوسِّم النقاط بأقرب مركز (توسيم للأمام على الاختبار)."""
        if not self._fitted or self.centroids_ is None:
            raise RuntimeError("KMeansRegimes must be fitted before predict().")
        arr = np.asarray(x, dtype=np.float64)
        labels = _pairwise_sqdist(arr, self.centroids_).argmin(axis=1)
        return np.asarray(labels, dtype=np.intp)

    def fit_predict(self, x: FloatArray) -> IntArray:
        return self.fit(x).predict(x)


def regime_labels_frame(
    labels: IntArray | Sequence[int],
    times: npt.NDArray[np.integer] | Sequence[int],
) -> pl.DataFrame:
    """يبني إطار توسيم بطوابع زمنية سليمة: ``availability_ts`` + ``regime``."""
    lab = np.asarray(labels, dtype=np.int64)
    ts = np.asarray(times, dtype=np.int64)
    if lab.shape[0] != ts.shape[0]:
        raise ValueError(f"labels and times must align, got {lab.shape} vs {ts.shape}")
    return pl.DataFrame({AVAILABILITY_TS: ts, "regime": lab}).sort(AVAILABILITY_TS)


def transition_matrix(labels: IntArray | Sequence[int], *, n_regimes: int) -> FloatArray:
    """مصفوفة انتقالات سببية row-stochastic من تتابع الحالات (t → t+1)."""
    if n_regimes < 1:
        raise ValueError(f"n_regimes must be >= 1, got {n_regimes}")
    lab = np.asarray(labels, dtype=np.intp)
    counts = np.zeros((n_regimes, n_regimes), dtype=np.float64)
    for src, dst in pairwise(lab):
        counts[src, dst] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return np.asarray(probs, dtype=np.float64)


def dwell_times(labels: IntArray | Sequence[int]) -> dict[int, float]:
    """متوسّط طول فترة المكوث المتّصلة (persistence) لكل حالة."""
    lab = np.asarray(labels, dtype=np.intp)
    runs: dict[int, list[int]] = {}
    if lab.shape[0] == 0:
        return {}
    current = int(lab[0])
    length = 1
    for value in lab[1:]:
        v = int(value)
        if v == current:
            length += 1
        else:
            runs.setdefault(current, []).append(length)
            current = v
            length = 1
    runs.setdefault(current, []).append(length)
    return {regime: float(np.mean(lengths)) for regime, lengths in runs.items()}


def silhouette_score(x: FloatArray, labels: IntArray | Sequence[int]) -> float:
    """معامل الظلّ (silhouette) لجودة/استقرار التجميع، ∈ [-1, 1]."""
    arr = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.intp)
    n = arr.shape[0]
    unique = np.unique(lab)
    if unique.shape[0] < _MIN_CLUSTERS or n <= unique.shape[0]:
        return 0.0
    dist = np.sqrt(_pairwise_sqdist(arr, arr))
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = lab == lab[i]
        same[i] = False
        a = float(dist[i, same].mean()) if bool(np.any(same)) else 0.0
        b = np.inf
        for other in unique:
            if other == lab[i]:
                continue
            mask = lab == other
            b = min(b, float(dist[i, mask].mean()))
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def regime_summary(
    x: FloatArray,
    labels: IntArray | Sequence[int],
    *,
    feature_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """تفسير الحالات: العدد ومتوسّط كل ميزة لكل حالة (interpretability)."""
    arr = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.intp)
    names = (
        list(feature_names) if feature_names is not None else [f"f{i}" for i in range(arr.shape[1])]
    )
    rows: list[dict[str, float]] = []
    for regime in np.unique(lab):
        members = arr[lab == regime]
        row: dict[str, float] = {"regime": float(regime), "count": float(members.shape[0])}
        means = members.mean(axis=0)
        for name, value in zip(names, means, strict=True):
            row[name] = float(value)
        rows.append(row)
    return pl.DataFrame(rows)


def infer_market_phase_map(
    centroids: FloatArray,
    *,
    in_value_idx: int = 2,
    trail_bid_idx: int = 6,
    trail_ask_idx: int = 7,
) -> dict[int, int]:
    """يربط كل عنقود KMeans بمرحلة سوق (balance/expansion/neutral)."""
    k = centroids.shape[0]
    phases = dict.fromkeys(range(k), PHASE_NEUTRAL)
    balance_cluster = int(np.argmax(centroids[:, in_value_idx]))
    phases[balance_cluster] = PHASE_BALANCE
    trail_scores = centroids[:, trail_bid_idx] + centroids[:, trail_ask_idx]
    expansion_candidates = [i for i in range(k) if i != balance_cluster]
    if expansion_candidates:
        expansion_cluster = max(expansion_candidates, key=lambda i: float(trail_scores[i]))
        phases[expansion_cluster] = PHASE_EXPANSION
    return phases


def heuristic_market_phase(features: Sequence[float]) -> int:
    """مرحلة سوق سببية قبل توفر نموذج KMeans كافٍ."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.shape[0] < _MIN_PHASE_FEATURES:
        return PHASE_NEUTRAL
    near_vah, near_val, in_va = float(arr[0]), float(arr[1]), float(arr[2])
    if near_vah > 0 or near_val > 0 or in_va > _IN_VALUE_ACTIVE:
        return PHASE_BALANCE
    if arr.shape[0] >= _REGIME_FEATURE_DIM:
        trail = float(arr[6]) + float(arr[7])
        if trail > 0:
            return PHASE_EXPANSION
    return PHASE_NEUTRAL


@dataclass
class CausalRegimeTracker:
    """تتبّع سببي لحالات السوق عبر KMeansRegimes على الماضي فقط."""

    n_regimes: int = 3
    seed: int = 0
    refit_interval: int = 50
    min_samples: int = 12
    _history: list[FloatArray] = field(default_factory=list, repr=False)
    _model: KMeansRegimes | None = field(default=None, repr=False)
    _phase_map: dict[int, int] = field(default_factory=dict, repr=False)

    def update(self, features: Sequence[float]) -> int:
        vec = np.asarray(features, dtype=np.float64)
        self._history.append(vec)
        if (
            len(self._history) >= self.min_samples
            and len(self._history) % self.refit_interval == 0
        ):
            mat = np.stack(self._history)
            self._model = KMeansRegimes(self.n_regimes, seed=self.seed).fit(mat)
            if self._model.centroids_ is not None:
                self._phase_map = infer_market_phase_map(self._model.centroids_)
        if self._model is None or self._model.centroids_ is None:
            return heuristic_market_phase(vec.tolist())
        regime = int(self._model.predict(vec.reshape(1, -1))[0])
        return self._phase_map.get(regime, PHASE_NEUTRAL)
