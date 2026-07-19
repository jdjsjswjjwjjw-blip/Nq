"""اختبارات تقطيع التسلسلات والتطبيع السببي."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.models.preprocessing import CausalStandardScaler
from nq.models.windowing import build_sequences


def _frame(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "availability_ts": list(range(n)),
            "a": [float(i) for i in range(n)],
            "b": [float(2 * i) for i in range(n)],
        }
    )


def test_build_sequences_shapes_and_causality() -> None:
    ds = build_sequences(_frame(10), feature_columns=["a", "b"], window=3)
    assert ds.x.shape == (8, 3, 2)
    assert ds.times.tolist() == list(range(2, 10))
    # النافذة الأولى: صفوف 0..2 من العمود a
    np.testing.assert_array_equal(ds.x[0, :, 0], [0.0, 1.0, 2.0])
    assert ds.flatten().shape == (8, 6)


def test_build_sequences_stride() -> None:
    ds = build_sequences(_frame(10), feature_columns=["a"], window=2, stride=2)
    assert ds.times.tolist() == [1, 3, 5, 7, 9]


def test_build_sequences_requires_sorted_time() -> None:
    frame = pl.DataFrame({"availability_ts": [0, 2, 1], "a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="non-decreasing"):
        build_sequences(frame, feature_columns=["a"], window=2)


def test_causal_scaler_fit_on_train_only() -> None:
    train = np.array([[0.0, 0.0], [2.0, 4.0]])
    scaler = CausalStandardScaler().fit(train)
    assert scaler.mean_ is not None
    # المتوسّط من التدريب فقط
    np.testing.assert_allclose(scaler.mean_, [1.0, 2.0])
    transformed = scaler.transform(np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(transformed, [[0.0, 0.0]])


def test_scaler_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        CausalStandardScaler().transform(np.array([[1.0]]))
