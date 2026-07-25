"""اختبارات البحث الرمزي (DEAP + gplearn) — تُتخطّى إن نقصت ``[gp]``."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("deap")
pytest.importorskip("gplearn")
pytest.importorskip("sklearn")

from nq.alpha.symbolic_gp import (
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
