"""تقسيم زمني walk-forward مع purge/embargo (Purged Walk-Forward Split).

التقسيم الزمني الصارم شرطٌ أساسي لمنع التسريب: التدريب دائمًا **قبل** الاختبار
زمنيًا، وتُحذف (purge) عيّنات التدريب الملاصقة لبداية الاختبار ضمن فترة حظر
(``embargo``) لعزل أي أثر متبقٍّ عند الحدود (مثل تراكب النوافذ أو الأهداف).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """طيّة زمنية واحدة: مؤشّرات التدريب والاختبار."""

    train_idx: IntArray
    test_idx: IntArray


def purged_walk_forward_split(
    times: npt.NDArray[np.integer] | list[int],
    *,
    n_splits: int,
    embargo: int = 0,
    min_train_size: int = 1,
) -> list[WalkForwardFold]:
    """يُنتج طيّات walk-forward متوسّعة مع فترة حظر زمنية.

    المعاملات:
        times: طوابع زمنية غير متناقصة (سببية) لكل عيّنة.
        n_splits: عدد كتل الاختبار المتتالية.
        embargo: فترة الحظر الزمنية؛ يُستبعَد من التدريب كل ما يقع زمنه ضمن
            ``embargo`` قبل بداية كتلة الاختبار.
        min_train_size: أدنى حجم تدريب لقبول الطيّة.

    يرفع ``ValueError`` عند طوابع متناقصة أو معاملات غير صالحة.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if embargo < 0:
        raise ValueError(f"embargo must be non-negative, got {embargo}")

    ts = np.asarray(times)
    n = ts.shape[0]
    if n == 0:
        return []
    if bool(np.any(np.diff(ts) < 0)):
        raise ValueError("times must be non-decreasing (causal order) for walk-forward split.")

    fold_size = n // (n_splits + 1)
    if fold_size < 1:
        raise ValueError(f"not enough samples ({n}) for n_splits={n_splits}.")

    folds: list[WalkForwardFold] = []
    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = n if k == n_splits else (k + 1) * fold_size
        test_idx = np.arange(test_start, test_end, dtype=np.intp)

        cutoff = ts[test_start] - embargo
        train_mask = np.arange(test_start, dtype=np.intp)
        train_idx = train_mask[ts[:test_start] <= cutoff]

        if train_idx.shape[0] < min_train_size:
            continue
        folds.append(WalkForwardFold(train_idx=train_idx, test_idx=test_idx))

    return folds
