"""اختبارات أداة اختبار التسريب الزمني — قلب الحوكمة."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from nq.validation import (
    LeakageError,
    assert_availability_not_before_event,
    assert_causal_order,
    assert_temporal_split,
    detect_leakage_by_perturbation,
)

FloatArray = npt.NDArray[np.floating]

#: تفاوت عددي دقيق لتأكيد الخلوّ التام من التسريب في الدوال السببية.
CAUSAL_TOL = 1e-9


def _causal_cumsum(x: FloatArray) -> FloatArray:
    """دالة سببية: المخرَج عند t يعتمد على المدخلات حتى t فقط."""
    return np.cumsum(x)


def _leaky_global_mean(x: FloatArray) -> FloatArray:
    """دالة متسرّبة: تستخدم متوسط كل البيانات (يشمل المستقبل)."""
    return np.full_like(x, float(np.mean(x)))


def _leaky_suffix_sum(x: FloatArray) -> FloatArray:
    """دالة متسرّبة: المخرَج عند t يجمع القيم المستقبلية."""
    return np.cumsum(x[::-1])[::-1]


# --- assert_causal_order ----------------------------------------------------


def test_causal_order_accepts_non_decreasing() -> None:
    assert_causal_order([1, 1, 2, 3, 3, 4])


def test_causal_order_rejects_decrease() -> None:
    with pytest.raises(LeakageError, match="non-decreasing"):
        assert_causal_order([1, 2, 1])


def test_causal_order_strict_rejects_duplicates() -> None:
    with pytest.raises(LeakageError, match="strictly increasing"):
        assert_causal_order([1, 1, 2], strict=True)


# --- assert_availability_not_before_event -----------------------------------


def test_availability_ok() -> None:
    assert_availability_not_before_event([10, 20, 30], [10, 25, 40])


def test_availability_violation() -> None:
    with pytest.raises(LeakageError, match="point-in-time"):
        assert_availability_not_before_event([10, 20, 30], [10, 19, 40])


# --- assert_temporal_split --------------------------------------------------


def test_temporal_split_ok() -> None:
    assert_temporal_split([1, 2, 3], [5, 6, 7], embargo=1)


def test_temporal_split_overlap_rejected() -> None:
    with pytest.raises(LeakageError, match="temporal-split violation"):
        assert_temporal_split([1, 2, 5], [4, 6], embargo=0)


def test_temporal_split_embargo_enforced() -> None:
    with pytest.raises(LeakageError, match="temporal-split violation"):
        assert_temporal_split([1, 2, 3], [4], embargo=5)


# --- detect_leakage_by_perturbation -----------------------------------------


def test_causal_function_reports_no_leakage() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(200)
    report = detect_leakage_by_perturbation(_causal_cumsum, data, rng=rng, n_trials=8)
    assert not report.leaked
    assert report.max_abs_diff < CAUSAL_TOL
    report.raise_for_leakage()  # must not raise


def test_global_mean_function_detected_as_leaky() -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal(200)
    report = detect_leakage_by_perturbation(_leaky_global_mean, data, rng=rng, n_trials=8)
    assert report.leaked
    assert report.first_violation_cut is not None
    with pytest.raises(LeakageError, match="temporal leakage detected"):
        report.raise_for_leakage()


def test_suffix_sum_function_detected_as_leaky() -> None:
    rng = np.random.default_rng(2)
    data = rng.standard_normal(150)
    report = detect_leakage_by_perturbation(_leaky_suffix_sum, data, rng=rng, n_trials=8)
    assert report.leaked


def test_output_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="output length"):
        detect_leakage_by_perturbation(lambda x: x[:-1], np.arange(10.0))


def test_short_input_is_noop() -> None:
    report = detect_leakage_by_perturbation(_causal_cumsum, [1.0])
    assert not report.leaked
    assert report.cuts_tested == ()
