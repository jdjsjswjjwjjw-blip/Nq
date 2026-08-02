"""مقاييس المسافة والاعتماد (Distance-Based Dependence).

تُستخدم لتقدير فجوة المعلومات الشرطية (MFIG) دون افتراض خطية أو توزيع طبيعي.

الأداء: تخفيف حتمي، حد أقصى لمحاور dCor، تخزين مسبق + ``einsum`` لاختبارات التبديل.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nq.research.progress import ProgressLike

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

_MIN_SAMPLES = 3
_MATRIX_NDIM = 2
# dCor ثنائي O(n²)؛ سقف حتمي يبقي MFIG عمليًا على ~5k شمعة
_DCOR_MAX_SAMPLES = 256
_MAX_AXES = 16
_PRECOMPUTE_BUDGET_BYTES = 512 * 1024 * 1024


def _thin_indices(n: int, max_samples: int = _DCOR_MAX_SAMPLES) -> IntArray:
    """عيّنة حتمية متباعدة — قابلة لإعادة الإنتاج دون عشوائية."""
    if n <= max_samples:
        return np.arange(n, dtype=np.intp)
    return np.linspace(0, n - 1, max_samples).astype(np.intp)


def _select_axes(matrix: FloatArray, *, max_axes: int = _MAX_AXES) -> FloatArray:
    """أبقِ أعلى المحاور تباينًا (حتمي عبر argsort)."""
    if matrix.shape[1] <= max_axes:
        return matrix
    std = np.std(matrix, axis=0)
    order = np.argsort(std)[::-1][:max_axes]
    return matrix[:, np.sort(order)]


def _centered_abs_distance(x: FloatArray) -> FloatArray:
    """مصفوفة مسافة مطلقة مزدوجة التمركز لمتجه أحادي."""
    a = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    dist = np.abs(a - a.T)
    return dist - dist.mean(axis=0) - dist.mean(axis=1, keepdims=True) + dist.mean()


def _dcor_from_centered(centered_a: FloatArray, centered_b: FloatArray) -> float:
    n = centered_a.shape[0]
    if n < _MIN_SAMPLES:
        return 0.0
    inv = 1.0 / float(n * n)
    dcov2 = float((centered_a * centered_b).sum() * inv)
    dvar_a = float((centered_a * centered_a).sum() * inv)
    dvar_b = float((centered_b * centered_b).sum() * inv)
    denom = np.sqrt(dvar_a * dvar_b)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denom))


def distance_correlation(x: FloatArray, y: FloatArray) -> float:
    """معامل الارتباط بالمسافة (Székely & Rizzo) بين متجهين أحاديَي البعد."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("x and y must have the same length")
    n = a.shape[0]
    if n < _MIN_SAMPLES:
        return 0.0
    return _dcor_from_centered(_centered_abs_distance(a), _centered_abs_distance(b))


def max_axis_dependence(
    matrix: FloatArray,
    target: FloatArray,
    *,
    max_samples: int = _DCOR_MAX_SAMPLES,
    max_axes: int = _MAX_AXES,
) -> float:
    """أقصى ارتباط بالمسافة عبر أعمدة مصفوفة مع هدف أحادي البعد."""
    arr = np.asarray(matrix, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1)
    if arr.ndim != _MATRIX_NDIM or arr.shape[0] != tgt.shape[0]:
        raise ValueError("matrix and target must align row-wise")
    if arr.shape[0] < _MIN_SAMPLES:
        return 0.0
    idx = _thin_indices(arr.shape[0], max_samples=max_samples)
    sub = _select_axes(arr[idx], max_axes=max_axes)
    y = tgt[idx]
    centered_b = _centered_abs_distance(y)
    best = 0.0
    for j in range(sub.shape[1]):
        col = sub[:, j]
        if not np.isfinite(col).all() or float(np.std(col)) == 0.0:
            continue
        best = max(best, _dcor_from_centered(_centered_abs_distance(col), centered_b))
    return best


@dataclass(slots=True)
class _AxisDcorCache:
    """تخزين مسبق: مكدّس centered (c,n,n) أو قائمة أعمدة خام."""

    stacked: npt.NDArray[np.float32] | None
    raw_columns: list[npt.NDArray[np.float32]]
    dvar_a: FloatArray | None
    n: int
    precomputed: bool


def _build_axis_cache(
    matrix: FloatArray,
    *,
    max_samples: int = _DCOR_MAX_SAMPLES,
    max_axes: int = _MAX_AXES,
) -> _AxisDcorCache:
    arr = np.asarray(matrix, dtype=np.float64)
    idx = _thin_indices(arr.shape[0], max_samples=max_samples)
    sub = _select_axes(arr[idx], max_axes=max_axes)
    n = int(sub.shape[0])
    # أسقط الأعمدة الثابتة
    keep: list[int] = []
    for j in range(sub.shape[1]):
        col = sub[:, j]
        if np.isfinite(col).all() and float(np.std(col)) > 0.0:
            keep.append(j)
    if not keep:
        return _AxisDcorCache(stacked=None, raw_columns=[], dvar_a=None, n=n, precomputed=True)
    sub = sub[:, keep]
    bytes_needed = n * n * 4 * sub.shape[1]
    can_precompute = n >= _MIN_SAMPLES and bytes_needed <= _PRECOMPUTE_BUDGET_BYTES
    if can_precompute:
        layers = np.empty((sub.shape[1], n, n), dtype=np.float32)
        dvar_a = np.empty(sub.shape[1], dtype=np.float64)
        inv = 1.0 / float(n * n)
        for j in range(sub.shape[1]):
            centered = _centered_abs_distance(sub[:, j]).astype(np.float32, copy=False)
            layers[j] = centered
            dvar_a[j] = float((centered * centered).sum() * inv)
        return _AxisDcorCache(
            stacked=layers, raw_columns=[], dvar_a=dvar_a, n=n, precomputed=True
        )
    raw = [sub[:, j].astype(np.float32, copy=False) for j in range(sub.shape[1])]
    return _AxisDcorCache(stacked=None, raw_columns=raw, dvar_a=None, n=n, precomputed=False)


def _max_dep_cached(cache: _AxisDcorCache, target_sub: FloatArray) -> float:
    if cache.n < _MIN_SAMPLES:
        return 0.0
    y = np.asarray(target_sub, dtype=np.float64).reshape(-1)
    if y.shape[0] != cache.n:
        raise ValueError("target length mismatch with dCor cache")
    if cache.precomputed and cache.stacked is not None and cache.dvar_a is not None:
        if cache.stacked.shape[0] == 0:
            return 0.0
        centered_b = _centered_abs_distance(y).astype(np.float32, copy=False)
        inv = 1.0 / float(cache.n * cache.n)
        dcov2 = np.einsum("cij,ij->c", cache.stacked, centered_b, dtype=np.float64) * inv
        dvar_b = float((centered_b * centered_b).sum() * inv)
        if dvar_b > 0:
            denom = np.sqrt(cache.dvar_a * dvar_b)
            valid = denom > 0
            if bool(np.any(valid)):
                scores = np.zeros_like(dcov2)
                scores[valid] = np.sqrt(np.maximum(dcov2[valid], 0.0) / denom[valid])
                return float(np.max(scores))
        return 0.0
    if not cache.raw_columns:
        return 0.0
    centered_b = _centered_abs_distance(y)
    best = 0.0
    for col in cache.raw_columns:
        best = max(best, _dcor_from_centered(_centered_abs_distance(col), centered_b))
    return best


def information_gap_perm_null(
    descriptors: FloatArray,
    features: FloatArray,
    returns: FloatArray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
    max_samples: int = _DCOR_MAX_SAMPLES,
    progress: ProgressLike | None = None,
    progress_label: str = "mfig-perm",
) -> tuple[float, FloatArray]:
    """فجوة معلومات + توزيع عدم بتبديل العوائد — بنفس التخفيف الحتمي."""
    desc = np.asarray(descriptors, dtype=np.float64)
    feat = np.asarray(features, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64).reshape(-1)
    if desc.shape[0] != feat.shape[0] or desc.shape[0] != ret.shape[0]:
        raise ValueError("descriptors, features, and returns must align row-wise")
    idx = _thin_indices(ret.shape[0], max_samples=max_samples)
    y = ret[idx]
    desc_cache = _build_axis_cache(desc[idx], max_samples=idx.shape[0])
    feat_cache = _build_axis_cache(feat[idx], max_samples=idx.shape[0])
    observed = _max_dep_cached(desc_cache, y) - _max_dep_cached(feat_cache, y)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(y)
        null[i] = _max_dep_cached(desc_cache, perm) - _max_dep_cached(feat_cache, perm)
        if progress is not None:
            progress.heartbeat(i + 1, n_permutations, label=progress_label)
    return float(observed), null


def fold_information_gap_perm_null(
    descriptors: FloatArray,
    features: FloatArray,
    returns: FloatArray,
    fold_test_indices: list[IntArray],
    *,
    n_permutations: int,
    rng: np.random.Generator,
    max_samples: int = _DCOR_MAX_SAMPLES,
    progress: ProgressLike | None = None,
    progress_label: str = "mfig-perm",
) -> tuple[float, FloatArray]:
    """فجوة معلومات عبر طيّات اختبار + عدم بتبديل العوائد داخل كل طيّة."""
    desc = np.asarray(descriptors, dtype=np.float64)
    feat = np.asarray(features, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64).reshape(-1)
    caches: list[tuple[_AxisDcorCache, _AxisDcorCache, FloatArray]] = []
    gaps: list[float] = []
    for test_idx in fold_test_indices:
        if test_idx.shape[0] < _MIN_SAMPLES:
            continue
        local = _thin_indices(int(test_idx.shape[0]), max_samples=max_samples)
        idx = np.asarray(test_idx, dtype=np.intp)[local]
        y = ret[idx]
        desc_cache = _build_axis_cache(desc[idx], max_samples=int(idx.shape[0]))
        feat_cache = _build_axis_cache(feat[idx], max_samples=int(idx.shape[0]))
        gaps.append(_max_dep_cached(desc_cache, y) - _max_dep_cached(feat_cache, y))
        caches.append((desc_cache, feat_cache, y))
    if not gaps:
        return 0.0, np.empty(0, dtype=np.float64)
    observed = float(np.mean(gaps))
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm_gaps: list[float] = []
        for desc_cache, feat_cache, y in caches:
            perm = rng.permutation(y)
            perm_gaps.append(
                _max_dep_cached(desc_cache, perm) - _max_dep_cached(feat_cache, perm)
            )
        null[i] = float(np.mean(perm_gaps))
        if progress is not None:
            progress.heartbeat(i + 1, n_permutations, label=progress_label)
    return observed, null


__all__ = [
    "distance_correlation",
    "fold_information_gap_perm_null",
    "information_gap_perm_null",
    "max_axis_dependence",
]
