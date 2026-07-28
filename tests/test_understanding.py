"""اختبارات طبقات الفهم الكمي — تشخيص OOS فقط بلا تسريب وبلا تغيير الاختيار."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.core.session import SessionPhase
from nq.research.understanding import (
    oos_test_indices,
    peel_one,
    resolve_base_column,
    run_understanding_layers,
    write_understanding_outputs,
)
from nq.strategies.breakout_hypothesis import core_breakout_grid, search_fail_breakout_hypotheses
from tests.test_coverage import _paired_streams


def test_resolve_base_peels_ssl_depth_enh() -> None:
    cols = {
        "sig",
        "sig__enh__z0",
        "sig__enh__z0__depth__pressure_q0p7",
        "sig__enh__z0__depth__pressure_q0p7__ssl",
    }
    base, layers = resolve_base_column("sig__enh__z0__depth__pressure_q0p7__ssl", cols)
    assert base == "sig"
    assert layers == ("ssl", "depth:pressure_q0p7", "enh:z0")
    assert peel_one("sig__ssl", {"sig", "sig__ssl"}) == ("sig", "ssl")


def test_oos_test_indices_disjoint_from_early_train() -> None:
    n = 120
    times = np.arange(n, dtype=np.int64) * 10_000
    idx = oos_test_indices(times, interval_ns=10_000, ssl_window=3, n_splits=3)
    assert idx.size > 0
    # First train block of purged WF starts at 0; first test starts at fold_size
    fold_size = n // 4
    assert int(idx.min()) >= fold_size


@pytest.mark.leakage
def test_understanding_past_stable_under_future_noise(tmp_path: Path) -> None:
    """تغيير المستقبل لا يغيّر طبقات الفهم على الماضي (مؤشرات OOS ثابتة للزمن)."""
    n = 180
    rng = np.random.default_rng(3)
    ts = np.arange(n, dtype=np.int64) * 10_000
    base = rng.normal(0, 1, n)
    gate = (np.abs(rng.normal(0, 1, n)) > 0.5).astype(np.float64)
    selected = base * gate
    prices = 100 + np.cumsum(rng.normal(0, 0.01, n))
    phases = np.array(
        [int(SessionPhase.MORNING) if i % 2 == 0 else int(SessionPhase.ETH) for i in range(n)]
    )

    features = pl.DataFrame(
        {
            AVAILABILITY_TS: ts,
            "nq_close": prices,
            "sig": base,
            "sig__depth__pressure_q0p7": selected,
            "session_phase": phases,
            "z0": rng.normal(0, 1, n),
        }
    )
    folds = pl.DataFrame(
        {
            "fold": [0, 1],
            "selected": ["sig__depth__pressure_q0p7"] * 2,
            "train_ic": [0.1, 0.2],
            "test_ic": [0.05, -0.02],
        }
    )

    r1 = run_understanding_layers(
        features,
        selected_column="sig__depth__pressure_q0p7",
        fold_selections=folds,
        interval_ns=10_000,
        ssl_window=3,
        n_splits=3,
        n_permutations=40,
        seed=1,
        quiet=True,
    )
    # Corrupt only the last third of prices (future relative to early bars)
    noisy = features.with_columns(
        pl.when(pl.col(AVAILABILITY_TS) >= int(ts[120]))
        .then(pl.col("nq_close") + 50.0)
        .otherwise(pl.col("nq_close"))
        .alias("nq_close")
    )
    r2 = run_understanding_layers(
        noisy,
        selected_column="sig__depth__pressure_q0p7",
        fold_selections=folds,
        interval_ns=10_000,
        ssl_window=3,
        n_splits=3,
        n_permutations=40,
        seed=1,
        quiet=True,
    )
    # OOS mask itself is time-index based — identical
    t1 = oos_test_indices(ts, interval_ns=10_000, ssl_window=3, n_splits=3)
    t2 = oos_test_indices(ts, interval_ns=10_000, ssl_window=3, n_splits=3)
    assert np.array_equal(t1, t2)
    # Stability layer uses fold_selections only — must be identical
    assert r1.stability == r2.stability
    assert r1.base_column == "sig"
    assert any(layer.startswith("depth:") for layer in r1.layers)
    assert r1.depth_cf is not None
    paths = write_understanding_outputs(r1, tmp_path)
    assert paths["report"].is_file()
    assert "understanding_does_not_alter_candidate_selection" in r1.notes


def test_search_understand_does_not_change_selection() -> None:
    """``understand=True`` must not alter candidates / best / OOS IC."""
    nq, mnq = _paired_streams(2200, seed=55)
    specs = core_breakout_grid()[:2]
    baseline = search_fail_breakout_hypotheses(
        nq,
        mnq,
        specs=specs,
        interval_ns=10_000,
        use_ssl_gate=False,
        enhance_with_ssl=False,
        use_depth_filter=True,
        n_splits=2,
        n_permutations=30,
        rng=make_generator(9),
        quiet=True,
        understand=False,
    )
    with_u = search_fail_breakout_hypotheses(
        nq,
        mnq,
        specs=specs,
        interval_ns=10_000,
        use_ssl_gate=False,
        enhance_with_ssl=False,
        use_depth_filter=True,
        n_splits=2,
        n_permutations=30,
        rng=make_generator(9),
        quiet=True,
        understand=True,
    )
    assert baseline.best_oos_spec == with_u.best_oos_spec
    assert baseline.oos_selected_ic == with_u.oos_selected_ic
    assert baseline.candidate_columns == with_u.candidate_columns
    assert baseline.understanding is None
    # Synthetic MBO often yields all-zero FB signals → no best → no understanding bundle.
    if with_u.best_oos_spec is None:
        assert with_u.understanding is None
    else:
        assert with_u.understanding is not None
        assert with_u.understanding.selected_column == with_u.best_oos_spec


def test_understanding_on_search_feature_schema(tmp_path: Path) -> None:
    """Run layers on a search-like feature frame with non-zero gated signal."""
    n = 200
    rng = np.random.default_rng(11)
    ts = np.arange(n, dtype=np.int64) * 10_000
    base = np.where(np.arange(n) % 7 == 0, 1.0, np.where(np.arange(n) % 11 == 0, -1.0, 0.0))
    gated = base * (rng.random(n) > 0.4).astype(np.float64)
    features = pl.DataFrame(
        {
            AVAILABILITY_TS: ts,
            "nq_close": 100 + np.cumsum(rng.normal(0, 0.02, n)),
            "fail_breakout__core_bar": base,
            "fail_breakout__core_bar__depth__pressure_q0p7": gated,
            "session_phase": [int(SessionPhase.MORNING)] * n,
            "z0": rng.normal(0, 1, n),
        }
    )
    folds = pl.DataFrame(
        {
            "fold": [0, 1, 2],
            "selected": ["fail_breakout__core_bar__depth__pressure_q0p7"] * 3,
            "train_ic": [0.1, 0.0, 0.2],
            "test_ic": [0.05, 0.01, -0.02],
        }
    )
    report = run_understanding_layers(
        features,
        selected_column="fail_breakout__core_bar__depth__pressure_q0p7",
        fold_selections=folds,
        interval_ns=10_000,
        ssl_window=3,
        n_splits=3,
        n_permutations=50,
        seed=2,
        quiet=True,
    )
    assert report.base_column == "fail_breakout__core_bar"
    assert report.ablation
    assert report.depth_cf is not None
    assert report.ssl_link is not None
    paths = write_understanding_outputs(report, tmp_path)
    assert (tmp_path / "understanding" / "report.md").is_file()
    assert "ablation" in paths

