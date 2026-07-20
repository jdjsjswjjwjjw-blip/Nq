"""اختبارات التقسيم الزمني walk-forward."""

from __future__ import annotations

import numpy as np
import pytest

from nq.models.splitting import purged_walk_forward_split


def test_walk_forward_train_before_test() -> None:
    times = np.arange(100, dtype=np.int64)
    folds = purged_walk_forward_split(times, n_splits=4)
    assert len(folds) == 4
    for fold in folds:
        # كل طابع تدريب يسبق كل طابع اختبار (لا تداخل)
        assert times[fold.train_idx].max() < times[fold.test_idx].min()


def test_embargo_purges_adjacent_train() -> None:
    times = np.arange(100, dtype=np.int64)
    no_embargo = purged_walk_forward_split(times, n_splits=4, embargo=0)
    with_embargo = purged_walk_forward_split(times, n_splits=4, embargo=5)
    # الحظر يقلّص التدريب الملاصق لبداية الاختبار
    assert with_embargo[0].train_idx.shape[0] < no_embargo[0].train_idx.shape[0]
    # فجوة زمنية >= embargo بين نهاية التدريب وبداية الاختبار
    gap = times[with_embargo[0].test_idx.min()] - times[with_embargo[0].train_idx.max()]
    assert gap >= 5


def test_purge_samples_removes_overlapping_train() -> None:
    times = np.arange(100, dtype=np.int64)
    no_purge = purged_walk_forward_split(times, n_splits=4, embargo=0, purge_samples=0)
    with_purge = purged_walk_forward_split(times, n_splits=4, embargo=0, purge_samples=4)
    assert with_purge[0].train_idx.shape[0] < no_purge[0].train_idx.shape[0]
    assert with_purge[0].train_idx.max() < no_purge[0].train_idx.max()


def test_non_decreasing_required() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        purged_walk_forward_split(np.array([0, 2, 1], dtype=np.int64), n_splits=1)


def test_invalid_params() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        purged_walk_forward_split([0, 1, 2], n_splits=0)
    with pytest.raises(ValueError, match="embargo"):
        purged_walk_forward_split([0, 1, 2], n_splits=1, embargo=-1)
    with pytest.raises(ValueError, match="purge_samples"):
        purged_walk_forward_split([0, 1, 2], n_splits=1, purge_samples=-1)


def test_expanding_train_grows() -> None:
    times = np.arange(100, dtype=np.int64)
    folds = purged_walk_forward_split(times, n_splits=4)
    sizes = [f.train_idx.shape[0] for f in folds]
    assert sizes == sorted(sizes)  # توسّع تدريجي
