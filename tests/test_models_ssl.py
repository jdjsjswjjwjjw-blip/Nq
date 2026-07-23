"""اختبارات التعلّم ذاتي الإشراف: المشفّر، الإخفاء، نموذج العالم، والتباين."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.models.contrastive import augment_windows, info_nce_loss
from nq.models.encoder import Encoder, PCAEncoder
from nq.models.masking import mask_matrix, masked_reconstruction_error
from nq.models.ssl_pipeline import _feature_columns, run_ssl_tick_pipeline
from nq.models.world_model import NextStatePredictor, r2_score
from tests.test_tick_stream import _paired_mbo


def _low_rank_data(n: int, d: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    latent = rng.standard_normal((n, rank))
    loading = rng.standard_normal((rank, d))
    return latent @ loading


def test_pca_encoder_is_encoder_protocol() -> None:
    assert isinstance(PCAEncoder(2), Encoder)


def test_pca_reconstructs_low_rank_data() -> None:
    rng = make_generator(0)
    x = _low_rank_data(200, 8, rank=3, rng=rng)
    enc = PCAEncoder(n_components=3).fit(x)
    recon = enc.reconstruct(x)
    # رتبة 3 تُستعاد شبه تام بثلاث مكوّنات
    assert np.mean((recon - x) ** 2) < 1e-6
    assert enc.transform(x).shape == (200, 3)


def test_pca_requires_fit_and_2d() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        PCAEncoder(2).transform(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="2-D"):
        PCAEncoder(2).fit(np.zeros((2, 2, 2)))


def test_masked_modeling_beats_zero_fill_baseline() -> None:
    rng = make_generator(1)
    x = _low_rank_data(300, 10, rank=3, rng=rng)
    enc = PCAEncoder(n_components=3).fit(x)
    masked = mask_matrix(x, mask_ratio=0.2, rng=rng)
    recon = enc.reconstruct(masked.masked)
    model_err = masked_reconstruction_error(recon, masked)
    baseline_err = masked_reconstruction_error(np.zeros_like(x), masked)
    assert model_err < baseline_err


def test_next_state_predictor_learns_linear_map() -> None:
    rng = make_generator(2)
    x = rng.standard_normal((300, 4))
    true_map = rng.standard_normal((4, 2))
    y = x @ true_map
    model = NextStatePredictor(alpha=1e-6).fit(x[:200], y[:200])
    pred = model.predict(x[200:])
    train_mean = y[:200].mean(axis=0)
    assert r2_score(y[200:], pred, baseline_mean=train_mean) > 0.99


def test_oos_r2_uses_train_baseline_not_test_mean() -> None:
    """Campbell OOS R²: ss_tot مقابل متوسط التدريب؛ متوسط الاختبار يحرّف المقياس."""
    rng = make_generator(11)
    # أهداف اختبار مزاحَة عن متوسط التدريب → ss_tot أكبر مع baseline التدريب
    y_train = rng.normal(0.0, 1.0, size=(100, 2))
    y_test = rng.normal(5.0, 1.0, size=(50, 2))
    # تنبّؤ ضعيف = متوسط التدريب (خط الأساس نفسه)
    y_pred = np.broadcast_to(y_train.mean(axis=0), y_test.shape).copy()
    r2_oos = r2_score(y_test, y_pred, baseline_mean=y_train.mean(axis=0))
    # تنبّؤ بنفس متوسط الاختبار يُظهر R²≈0 بصيغة خاطئة؛ هنا نثبت أن الصيغة
    # الصحيحة تعطي R² قريبًا من 0 عندما التنبّؤ = متوسط التدريب فقط
    assert r2_oos < 0.05
    with pytest.raises(TypeError):
        r2_score(y_test, y_pred)  # type: ignore[call-arg]


def test_next_state_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        NextStatePredictor().predict(np.zeros((2, 2)))


def test_info_nce_lower_for_aligned_views() -> None:
    rng = make_generator(3)
    z = rng.standard_normal((16, 8))
    # مناظير متطابقة -> فقدان منخفض؛ مناظير عشوائية مستقلة -> أعلى
    aligned = info_nce_loss(z, z.copy(), temperature=0.1)
    random_pos = info_nce_loss(z, rng.standard_normal((16, 8)), temperature=0.1)
    assert aligned < random_pos


def test_augment_is_deterministic_and_shaped() -> None:
    x = np.ones((5, 4))
    a = augment_windows(x, rng=make_generator(7))
    b = augment_windows(x, rng=make_generator(7))
    np.testing.assert_array_equal(a, b)
    assert a.shape == x.shape


def test_info_nce_needs_batch() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        info_nce_loss(np.zeros((1, 3)), np.zeros((1, 3)))


def test_run_ssl_tick_pipeline_produces_report() -> None:
    nq, mnq = _paired_mbo(30)
    result = run_ssl_tick_pipeline(
        nq,
        mnq,
        window=3,
        n_components=3,
        n_splits=2,
        rng=make_generator(0),
    )
    assert "SSL Tick/Event" in result.report.to_markdown()


def test_ssl_feature_selection_is_invariant_to_future_missingness() -> None:
    n = 40
    base = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "nq_close": np.linspace(100.0, 101.0, n),
            "stable": np.linspace(0.0, 1.0, n),
            "late": pl.Series("late", [None] * n, dtype=pl.Float64),
        }
    )
    perturbed = base.with_columns(
        pl.when(pl.col(AVAILABILITY_TS) < 2)
        .then(None)
        .otherwise(1.0)
        .cast(pl.Float64)
        .alias("late")
    )

    assert _feature_columns(base) == _feature_columns(perturbed)
    assert "late" in _feature_columns(base)
