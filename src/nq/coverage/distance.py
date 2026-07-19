"""مقاييس المسافة والاعتماد (Distance-Based Dependence).

تُستخدم لتقدير فجوة المعلومات الشرطية (MFIG) دون افتراض خطية أو توزيع طبيعي.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_MIN_SAMPLES = 3
_MATRIX_NDIM = 2


def distance_correlation(x: FloatArray, y: FloatArray) -> float:
    """معامل الارتباط بالمسافة (Székely & Rizzo) بين متجهين أحاديَي البعد."""
    a = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    b = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n = a.shape[0]
    if n < _MIN_SAMPLES:
        return 0.0

    dist_a = np.abs(a - a.T)
    dist_b = np.abs(b - b.T)
    mean_a = dist_a.mean()
    mean_b = dist_b.mean()
    centered_a = dist_a - dist_a.mean(axis=0) - dist_a.mean(axis=1, keepdims=True) + mean_a
    centered_b = dist_b - dist_b.mean(axis=0) - dist_b.mean(axis=1, keepdims=True) + mean_b

    dcov2 = float((centered_a * centered_b).sum() / (n * n))
    dvar_a = float((centered_a * centered_a).sum() / (n * n))
    dvar_b = float((centered_b * centered_b).sum() / (n * n))
    denom = np.sqrt(dvar_a * dvar_b)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denom))


def max_axis_dependence(matrix: FloatArray, target: FloatArray) -> float:
    """أقصى ارتباط بالمسافة عبر أعمدة مصفوفة مع هدف أحادي البعد."""
    arr = np.asarray(matrix, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if arr.ndim != _MATRIX_NDIM or arr.shape[0] != tgt.shape[0]:
        raise ValueError("matrix and target must align row-wise")
    if arr.shape[0] < _MIN_SAMPLES:
        return 0.0
    return max((distance_correlation(arr[:, j], tgt) for j in range(arr.shape[1])), default=0.0)
