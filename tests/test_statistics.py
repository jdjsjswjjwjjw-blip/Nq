"""اختبارات المحطة 6: الاختبار الإحصائي."""

from __future__ import annotations

import numpy as np
import pytest

from nq.core.determinism import make_generator
from nq.statistics import (
    benjamini_hochberg,
    bonferroni,
    bootstrap_ci,
    holm,
    information_coefficient,
    moving_block_bootstrap_ci,
    permutation_test,
    regime_difference_test,
    sharpe_ratio,
    t_statistic,
    verify_hypotheses,
)

# --- resampling -------------------------------------------------------------


def test_permutation_detects_true_difference() -> None:
    rng = make_generator(0)
    a = rng.normal(1.0, 0.5, 200)
    b = rng.normal(0.0, 0.5, 200)
    res = permutation_test(a, b, n_permutations=2000, rng=rng)
    assert res.pvalue < 0.01
    assert res.statistic > 0


def test_permutation_no_difference_is_not_significant() -> None:
    rng = make_generator(1)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.0, 1.0, 200)
    res = permutation_test(a, b, n_permutations=2000, rng=rng)
    assert res.pvalue > 0.05


def test_bootstrap_ci_contains_true_mean() -> None:
    rng = make_generator(2)
    data = rng.normal(5.0, 1.0, 500)
    low, point, high = bootstrap_ci(data, n_boot=2000, rng=rng)
    assert low < 5.0 < high
    assert low < point < high


def test_block_bootstrap_ci() -> None:
    rng = make_generator(3)
    series = rng.normal(0.2, 1.0, 400)
    low, point, high = moving_block_bootstrap_ci(series, block_size=10, n_boot=1000, rng=rng)
    assert low < point < high


def test_block_size_validation() -> None:
    with pytest.raises(ValueError, match="exceeds series length"):
        moving_block_bootstrap_ci([1.0, 2.0], block_size=5)


# --- multiple testing -------------------------------------------------------


def test_bh_monotone_and_rejects() -> None:
    p = [0.001, 0.008, 0.039, 0.5, 0.9]
    res = benjamini_hochberg(p, alpha=0.05)
    assert res.reject[0] and res.reject[1]
    assert not res.reject[-1]
    # القيم المُعدّلة رتيبة بعد الترتيب
    assert np.all(np.diff(np.sort(res.adjusted)) >= -1e-12)


def test_bonferroni_more_conservative_than_holm() -> None:
    p = [0.01, 0.02, 0.03, 0.04]
    bonf = bonferroni(p, alpha=0.05)
    holm_res = holm(p, alpha=0.05)
    assert int(bonf.reject.sum()) <= int(holm_res.reject.sum())


def test_invalid_pvalues_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.1, 1.5])


# --- metrics ----------------------------------------------------------------


def test_sharpe_and_tstat_signs() -> None:
    rng = make_generator(4)
    pos = rng.normal(0.1, 0.05, 300)
    assert sharpe_ratio(pos) > 0
    assert t_statistic(pos) > 0


def test_information_coefficient_perfect() -> None:
    x = np.arange(50, dtype=np.float64)
    assert information_coefficient(x, 2 * x, method="spearman") == pytest.approx(1.0)
    assert information_coefficient(x, -x, method="pearson") == pytest.approx(-1.0)


# --- regime tests -----------------------------------------------------------


def test_regime_difference_detected() -> None:
    rng = make_generator(5)
    values = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    labels = np.repeat([0, 1], 100)
    res = regime_difference_test(values, labels, n_permutations=1000, rng=rng)
    assert res.pvalue < 0.01


def test_regime_no_difference() -> None:
    rng = make_generator(6)
    values = rng.normal(0, 1, 200)
    labels = np.repeat([0, 1], 100)
    res = regime_difference_test(values, labels, n_permutations=1000, rng=rng)
    assert res.pvalue > 0.05


# --- hypothesis verification ------------------------------------------------


def test_verify_hypotheses_report() -> None:
    report = verify_hypotheses(
        {"h_strong": 0.0001, "h_weak": 0.6, "h_mid": 0.02},
        alpha=0.05,
        method="benjamini_hochberg",
    )
    assert report.columns == ["hypothesis", "pvalue", "adjusted_pvalue", "reject"]
    # مرتّب تصاعديًا بالقيمة المُعدّلة
    assert report["hypothesis"].to_list()[0] == "h_strong"
    strong = report.filter(report["hypothesis"] == "h_strong")
    assert strong["reject"].to_list()[0] is True


def test_verify_hypotheses_empty_and_invalid() -> None:
    assert verify_hypotheses({}).height == 0
    with pytest.raises(ValueError, match="unknown method"):
        verify_hypotheses({"a": 0.1}, method="nope")  # type: ignore[arg-type]
