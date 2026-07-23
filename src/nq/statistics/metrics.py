"""مقاييس الأداء والتنبّؤ (Performance & Predictive Metrics).

* ``sharpe_ratio``            — نسبة شارب (عائد فائض معدّل بالمخاطرة، مُقيّسة زمنيًا).
* ``information_coefficient`` — ارتباط التنبّؤ بالنتيجة (Pearson أو Spearman).
* ``t_statistic``            — إحصائية t لمتوسّط عيّنة (اختلافه عن الصفر).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def sharpe_ratio(
    returns: npt.NDArray[np.floating] | list[float],
    *,
    risk_free: float = 0.0,
    periods_per_year: float = 1.0,
) -> float:
    """نسبة شارب: ``mean(excess)/std(excess) * sqrt(periods_per_year)``."""
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    arr = np.asarray(returns, dtype=np.float64) - risk_free
    std = float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0
    if std == 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(periods_per_year))


def _rankdata(x: FloatArray) -> FloatArray:
    """رُتب متوسّطة للتعادلات (average ranks)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    # معالجة التعادلات بمتوسّط الرتب
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.shape[0], dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    mean_ranks = sums / counts
    return mean_ranks[inverse]


def information_coefficient(
    prediction: npt.NDArray[np.floating] | list[float],
    outcome: npt.NDArray[np.floating] | list[float],
    *,
    method: Literal["pearson", "spearman"] = "spearman",
) -> float:
    """معامل المعلومات: ارتباط التنبّؤ بالنتيجة الفعلية."""
    pred = np.asarray(prediction, dtype=np.float64)
    out = np.asarray(outcome, dtype=np.float64)
    if pred.shape != out.shape:
        raise ValueError(f"prediction and outcome must align, got {pred.shape} vs {out.shape}")
    if pred.shape[0] < 2:  # noqa: PLR2004 - correlation needs >= 2 points
        return 0.0
    if method == "spearman":
        pred = _rankdata(pred)
        out = _rankdata(out)
    if np.std(pred) == 0 or np.std(out) == 0:
        return 0.0
    return float(np.corrcoef(pred, out)[0, 1])


def t_statistic(x: npt.NDArray[np.floating] | list[float]) -> float:
    """إحصائية t لاختلاف متوسّط العيّنة عن الصفر."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[0]
    if n < 2:  # noqa: PLR2004 - t needs >= 2 points
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(arr) / (std / np.sqrt(n)))
