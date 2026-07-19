"""تحقّق الحالات (Regime Validation).

يختبر ما إذا كان مقياسٌ ما (مثل العائد أو الدلتا) يختلف اختلافًا ذا دلالة عبر
حالات السوق (regimes)، عبر إحصائية F بين المجموعات مع p-value بالتبديل
(permutation) — لا معلمي ولا يفترض توزيعًا طبيعيًا.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nq.statistics.resampling import TestResult

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


def _f_statistic(values: FloatArray, labels: IntArray) -> float:
    groups = np.unique(labels)
    grand_mean = float(np.mean(values))
    n = values.shape[0]
    k = groups.shape[0]
    if k < 2 or n <= k:  # noqa: PLR2004 - F needs >= 2 groups
        return 0.0
    ss_between = 0.0
    ss_within = 0.0
    for g in groups:
        member = values[labels == g]
        mean_g = float(np.mean(member))
        ss_between += member.shape[0] * (mean_g - grand_mean) ** 2
        ss_within += float(np.sum((member - mean_g) ** 2))
    df_between = k - 1
    df_within = n - k
    if ss_within == 0:
        return 0.0
    return float((ss_between / df_between) / (ss_within / df_within))


def regime_difference_test(
    values: npt.NDArray[np.floating] | list[float],
    labels: npt.NDArray[np.integer] | list[int],
    *,
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> TestResult:
    """اختبار تبديل لفرضية تساوي متوسّط ``values`` عبر الحالات ``labels``.

    يبني التوزيع الصفري بخلط تسميات الحالات، ويحسب p-value أحادي الجانب
    (F أكبر يعني اختلافًا أقوى).
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    generator = rng if rng is not None else np.random.default_rng(0)
    vals = np.asarray(values, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.intp)
    if vals.shape[0] != labs.shape[0]:
        raise ValueError(f"values and labels must align, got {vals.shape} vs {labs.shape}")

    observed = _f_statistic(vals, labs)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        null[i] = _f_statistic(vals, generator.permutation(labs))
    pvalue = (int(np.sum(null >= observed)) + 1) / (n_permutations + 1)
    return TestResult(
        statistic=observed, pvalue=pvalue, n_resamples=n_permutations, alternative="greater"
    )
