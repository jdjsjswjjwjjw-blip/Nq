"""إعادة المعاينة: دلالة وفترات ثقة (Resampling-Based Inference).

* ``permutation_test``          — اختبار تبديل لا معلمي لفرق بين مجموعتين.
* ``bootstrap_ci``              — فترة ثقة bootstrap (عيّنات مستقلة).
* ``moving_block_bootstrap_ci`` — فترة ثقة block-bootstrap للسلاسل الزمنية
  المترابطة (يحافظ على البنية الزمنية للارتباط الذاتي).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True, slots=True)
class TestResult:
    """نتيجة اختبار إحصائي."""

    statistic: float
    pvalue: float
    n_resamples: int
    alternative: str


def _mean_difference(a: FloatArray, b: FloatArray) -> float:
    return float(np.mean(a) - np.mean(b))


def _pvalue_from_null(observed: float, null: FloatArray, alternative: Alternative) -> float:
    n = null.shape[0]
    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    elif alternative == "less":
        hits = int(np.sum(null <= observed))
    else:
        hits = int(np.sum(np.abs(null) >= abs(observed)))
    return float((hits + 1) / (n + 1))


def permutation_test(
    a: npt.NDArray[np.floating] | list[float],
    b: npt.NDArray[np.floating] | list[float],
    *,
    statistic: Callable[[FloatArray, FloatArray], float] = _mean_difference,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
    alternative: Alternative = "two-sided",
) -> TestResult:
    """اختبار تبديل لفرضية عدم وجود فرق بين ``a`` و ``b``.

    يخلط التسميات ``n_permutations`` مرّة لبناء التوزيع الصفري، ويحسب p-value
    بتصحيح ``(hits + 1)/(n + 1)`` (غير متحيّز ولا يعطي صفرًا).
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    generator = rng if rng is not None else np.random.default_rng(0)
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    observed = statistic(arr_a, arr_b)

    pooled = np.concatenate([arr_a, arr_b])
    n_a = arr_a.shape[0]
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = generator.permutation(pooled)
        null[i] = statistic(perm[:n_a], perm[n_a:])

    return TestResult(
        statistic=observed,
        pvalue=_pvalue_from_null(observed, null, alternative),
        n_resamples=n_permutations,
        alternative=alternative,
    )


def block_permutation(
    series: npt.NDArray[np.floating] | list[float],
    *,
    block_size: int,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Permute contiguous time blocks while preserving within-block order."""
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    generator = rng if rng is not None else np.random.default_rng(0)
    arr = np.asarray(series, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return arr.copy()
    if block_size == 1:
        return generator.permutation(arr)

    blocks = [arr[start : start + block_size] for start in range(0, n, block_size)]
    order = generator.permutation(len(blocks))
    return np.concatenate([blocks[int(i)] for i in order])[:n].astype(np.float64, copy=False)


def bootstrap_ci(
    data: npt.NDArray[np.floating] | list[float],
    *,
    statistic: Callable[[FloatArray], float] = lambda x: float(np.mean(x)),
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """فترة ثقة bootstrap بالنِّسب المئوية. يُعيد ``(low, point, high)``."""
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    generator = rng if rng is not None else np.random.default_rng(0)
    arr = np.asarray(data, dtype=np.float64)
    n = arr.shape[0]
    point = statistic(arr)
    estimates = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = arr[generator.integers(0, n, size=n)]
        estimates[i] = statistic(sample)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    low, high = np.percentile(estimates, [lo, hi])
    return float(low), float(point), float(high)


def moving_block_bootstrap_ci(
    series: npt.NDArray[np.floating] | list[float],
    *,
    statistic: Callable[[FloatArray], float] = lambda x: float(np.mean(x)),
    block_size: int,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """فترة ثقة block-bootstrap متحرّك لسلسلة زمنية مترابطة.

    يعيد بناء سلسلة بطول قريب من الأصل عبر كتل متتالية بطول ``block_size``،
    محافظًا على الارتباط الذاتي قصير المدى.
    """
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    generator = rng if rng is not None else np.random.default_rng(0)
    arr = np.asarray(series, dtype=np.float64)
    n = arr.shape[0]
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) exceeds series length ({n}).")

    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size + 1
    point = statistic(arr)
    estimates = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        starts = generator.integers(0, max_start, size=n_blocks)
        blocks = [arr[s : s + block_size] for s in starts]
        resampled = np.concatenate(blocks)[:n]
        estimates[i] = statistic(resampled)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    low, high = np.percentile(estimates, [lo, hi])
    return float(low), float(point), float(high)
