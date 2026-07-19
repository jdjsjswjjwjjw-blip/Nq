"""اختبارات حالات السوق (Regimes)."""

from __future__ import annotations

import numpy as np
import pytest

from nq.core.determinism import make_generator
from nq.states import (
    KMeansRegimes,
    dwell_times,
    regime_labels_frame,
    regime_summary,
    silhouette_score,
    transition_matrix,
)


def _three_clusters(rng: np.random.Generator, per: int = 60) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    parts = [c + rng.standard_normal((per, 2)) * 0.3 for c in centers]
    x = np.vstack(parts)
    truth = np.repeat([0, 1, 2], per)
    return x, truth


def test_kmeans_recovers_separated_clusters() -> None:
    rng = make_generator(0)
    x, truth = _three_clusters(rng)
    labels = KMeansRegimes(3, seed=0).fit_predict(x)
    # كل عنقود حقيقي يُوسَّم بحالة واحدة متّسقة
    for regime in np.unique(truth):
        assigned = labels[truth == regime]
        assert len(np.unique(assigned)) == 1


def test_kmeans_is_deterministic() -> None:
    rng = make_generator(1)
    x, _ = _three_clusters(rng)
    a = KMeansRegimes(3, seed=42).fit_predict(x)
    b = KMeansRegimes(3, seed=42).fit_predict(x)
    np.testing.assert_array_equal(a, b)


def test_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        KMeansRegimes(2).predict(np.zeros((2, 2)))


def test_too_few_samples_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        KMeansRegimes(5).fit(np.zeros((3, 2)))


def test_fit_on_train_predict_test_no_leakage() -> None:
    rng = make_generator(2)
    x, _ = _three_clusters(rng)
    model = KMeansRegimes(3, seed=0).fit(x[:120])  # train only
    test_labels = model.predict(x[120:])
    assert test_labels.shape[0] == x.shape[0] - 120


def test_regime_labels_frame_timestamps() -> None:
    frame = regime_labels_frame([0, 1, 0], [30, 10, 20])
    assert frame["availability_ts"].to_list() == [10, 20, 30]
    assert frame["regime"].to_list() == [1, 0, 0]


def test_transition_matrix_row_stochastic() -> None:
    labels = [0, 1, 1, 0, 1]
    mat = transition_matrix(labels, n_regimes=2)
    row_sums = mat.sum(axis=1)
    np.testing.assert_allclose(row_sums[row_sums > 0], 1.0)
    assert mat[1, 1] == pytest.approx(0.5)  # 1->1 once, 1->0 once


def test_dwell_times() -> None:
    dwell = dwell_times([0, 0, 1, 1, 1, 0])
    assert dwell[1] == pytest.approx(3.0)
    assert dwell[0] == pytest.approx(1.5)  # runs of length 2 and 1


def test_silhouette_high_for_separated() -> None:
    rng = make_generator(3)
    x, _ = _three_clusters(rng)
    labels = KMeansRegimes(3, seed=0).fit_predict(x)
    assert silhouette_score(x, labels) > 0.8


def test_regime_summary_interpretability() -> None:
    rng = make_generator(4)
    x, _ = _three_clusters(rng)
    labels = KMeansRegimes(3, seed=0).fit_predict(x)
    summary = regime_summary(x, labels, feature_names=["px", "vol"])
    assert summary.height == 3
    assert set(summary.columns) == {"regime", "count", "px", "vol"}
    assert summary["count"].sum() == x.shape[0]
