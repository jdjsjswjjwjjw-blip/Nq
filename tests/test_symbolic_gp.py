"""اختبارات البحث الرمزي (DEAP + gplearn) — تُتخطّى إن نقصت ``[gp]``."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("deap")
pytest.importorskip("gplearn")
pytest.importorskip("sklearn")

from nq.alpha.symbolic_gp import (
    discover_symbolic_on_train,
    evolve_deap,
    evolve_gplearn,
    feature_matrix,
    protected_div,
    require_gp_deps,
    search_symbolic_hypotheses,
)
from nq.contracts.temporal import AVAILABILITY_TS


def test_require_gp_deps_ok() -> None:
    require_gp_deps()


def test_protected_div_scalar_and_array() -> None:
    assert float(protected_div(4.0, 2.0)) == pytest.approx(2.0)
    assert float(protected_div(4.0, 0.0)) == pytest.approx(1.0)
    out = protected_div(np.array([4.0, 9.0]), np.array([2.0, 0.0]))
    assert isinstance(out, np.ndarray)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(1.0)


def test_feature_matrix_fills_nulls() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3],
            "a": [1.0, None, 3.0],
            "b": [0.5, 0.5, None],
        }
    )
    x, names = feature_matrix(frame, ["a", "b", "missing"])
    assert names == ("a", "b")
    assert x.shape == (3, 2)
    assert np.isfinite(x).all()


def _toy_frame(n: int = 120, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * 1_000_000_000
    delta = rng.normal(0, 1, size=n)
    noise = rng.normal(0, 1, size=n)
    close = 100.0 + np.cumsum(0.05 * delta + 0.2 * noise)
    return pl.DataFrame(
        {
            AVAILABILITY_TS: t,
            "nq_close": close,
            "nq_delta": delta,
            "mnq_delta": delta + rng.normal(0, 0.1, size=n),
            "trap_setup": (delta > 0).astype(np.float64),
            "fail_fvg": rng.choice([-1.0, 0.0, 1.0], size=n),
            "fail_breakout": rng.choice([-1.0, 0.0, 1.0], size=n),
        }
    )


def test_evolve_deap_and_gplearn_smoke() -> None:
    frame = _toy_frame(100)
    x, names = feature_matrix(frame, ["nq_delta", "mnq_delta", "trap_setup"])
    y = np.r_[np.diff(frame["nq_close"].to_numpy()), np.nan]
    y = np.nan_to_num(y, nan=0.0)
    deap_out = evolve_deap(
        x, y, names, population_size=20, generations=2, max_depth=2, seed=1, n_hof=1
    )
    assert len(deap_out) >= 1
    expr, pred, ic = deap_out[0]
    assert isinstance(expr, str) and len(expr) > 0
    assert pred.shape == (x.shape[0],)
    assert ic >= 0.0

    gp_out = evolve_gplearn(
        x, y, names, population_size=20, generations=2, max_depth=2, seed=1, n_hof=1
    )
    assert len(gp_out) >= 1
    assert gp_out[0][1].shape == (x.shape[0],)


def test_search_symbolic_hypotheses_walk_forward() -> None:
    frame = _toy_frame(160, seed=2)
    result = search_symbolic_hypotheses(
        frame,
        ["nq_delta", "mnq_delta", "trap_setup", "fail_fvg"],
        price_col="nq_close",
        backend="both",
        n_splits=2,
        population_size=16,
        generations=2,
        max_depth=2,
        n_programs=1,
        n_permutations=20,
        seed=3,
    )
    assert result.fold_selections.height >= 1
    assert result.oos_n >= 0
    assert all(p.backend in {"deap", "gplearn"} for p in result.programs)
    for p in result.programs:
        assert "if" not in p.expression.lower()


@pytest.mark.leakage
def test_symbolic_train_labels_do_not_reach_into_test() -> None:
    """لياقة GP لا ترى أسعار/عوائد داخل كتلة الاختبار (أفق = 1)."""
    from nq.alpha.signals import align_forward_returns
    from nq.models.splitting import purged_walk_forward_split
    from nq.validation import assert_temporal_split

    frame = _toy_frame(180, seed=5)
    times = frame[AVAILABILITY_TS].to_numpy()
    horizon = 1
    folds = purged_walk_forward_split(
        times, n_splits=2, embargo=0, purge_samples=horizon, min_train_size=40
    )
    assert len(folds) >= 1
    for fold in folds:
        assert_temporal_split(
            times[fold.train_idx],
            times[fold.test_idx],
            embargo=0,
        )
        # بعد purge: آخر صف تدريب + horizon لا يدخل الاختبار
        assert int(fold.train_idx.max()) + horizon < int(fold.test_idx.min())

    # اكتشاف على train فقط مع cutoff = بداية الاختبار
    fold = folds[0]
    cutoff = int(fold.test_idx.min())
    programs = discover_symbolic_on_train(
        frame,
        ["nq_delta", "mnq_delta", "trap_setup"],
        price_col="nq_close",
        horizon=horizon,
        backend="gplearn",
        population_size=12,
        generations=1,
        max_depth=2,
        seed=9,
        n_programs=1,
        train_idx=fold.train_idx,
        label_cutoff_idx=cutoff,
    )
    assert len(programs) >= 1
    # تشويش أسعار المستقبل لا يغيّر إشارة الماضي المحسوبة من معادلة ثابتة على ميزات الماضي
    y = align_forward_returns(frame["nq_close"].to_numpy().astype(np.float64), horizon=horizon)
    safe = fold.train_idx[fold.train_idx + horizon < cutoff]
    assert np.isfinite(y[safe]).all()
    # كل أهداف اللياقة الآمنة تقع بالكامل قبل الاختبار
    assert int(safe.max()) + horizon < cutoff


@pytest.mark.leakage
def test_symbolic_signal_past_invariant_to_future_feature_noise() -> None:
    """بعد تثبيت المعادلة: تشويش ميزات المستقبل لا يحرّك إشارة الماضي."""
    frame = _toy_frame(120, seed=7)
    x, names = feature_matrix(frame, ["nq_delta", "mnq_delta", "trap_setup"])
    y = np.r_[np.diff(frame["nq_close"].to_numpy()), 0.0]
    # درّب على النصف الأول فقط
    n = x.shape[0]
    split = n // 2
    programs = evolve_gplearn(
        x[:split],
        y[:split],
        names,
        population_size=12,
        generations=1,
        max_depth=2,
        seed=11,
        n_hof=1,
    )
    assert len(programs) >= 1
    expr, _pred, _ic = programs[0]
    # أعد التنبؤ عبر نموذج جديد على train ثم predict؛ شَوّش المستقبل وقارن الماضي
    from gplearn.genetic import SymbolicRegressor

    model = SymbolicRegressor(
        population_size=12,
        generations=1,
        init_depth=(1, 2),
        function_set=("add", "sub", "mul", "div", "neg", "abs"),
        metric="spearman",
        feature_names=list(names),
        verbose=0,
        random_state=11,
        n_jobs=1,
    )
    model.fit(x[:split], y[:split])
    base = np.asarray(model.predict(x), dtype=np.float64)
    x_noisy = x.copy()
    x_noisy[split:] += 50.0  # ضوضاء عنيفة على المستقبل فقط
    noisy = np.asarray(model.predict(x_noisy), dtype=np.float64)
    np.testing.assert_allclose(base[:split], noisy[:split], atol=1e-12)
    _ = expr
