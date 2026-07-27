"""اختبارات فلتر دخول مسار أحداث العمق (سببي، بلا إعادة كتابة القاعدة)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.simulation.depth_lifecycle import DEPTH_PATH_COLUMNS, depth_event_path_at_bar_close
from nq.strategies.breakout_hypothesis import core_breakout_grid, search_fail_breakout_hypotheses
from nq.strategies.depth_entry_filter import (
    attach_depth_path_to_features,
    generate_depth_entry_candidates,
)
from nq.strategies.fvg_hypothesis import default_fvg_grid, search_fail_fvg_hypotheses
from tests.test_coverage import _paired_streams


def test_depth_event_path_publishes_at_bucket_end() -> None:
    nq, _ = _paired_streams(800, seed=1)
    path = depth_event_path_at_bar_close(nq, interval_ns=10_000)
    assert path.height >= 1
    for c in DEPTH_PATH_COLUMNS:
        assert c in path.columns
    # availability = bucket_end
    assert (path[AVAILABILITY_TS] == path["bucket_end"]).all()


def test_depth_entry_candidates_do_not_invent_signal() -> None:
    n = 100
    features = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "sig": [1.0 if i % 8 == 0 else (-1.0 if i % 9 == 0 else 0.0) for i in range(n)],
            "depth_path_pressure": np.linspace(-1, 1, n),
            "depth_path_imbalance_delta": np.linspace(-0.5, 0.5, n),
        }
    )
    out, cols, specs = generate_depth_entry_candidates(features, ["sig"])
    assert len(cols) == len(specs) > 0
    assert all("__depth__" in c for c in cols)
    base = out["sig"].to_numpy()
    for c in cols:
        enh = out[c].to_numpy()
        assert np.all((enh == 0.0) | (np.sign(enh) == np.sign(base)) | (base == 0.0))


@pytest.mark.leakage
def test_depth_pressure_gate_past_stable_under_future_noise() -> None:
    n = 90
    features = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "sig": [1.0] * n,
            "depth_path_pressure": np.linspace(0.1, 1.5, n),
            "depth_path_imbalance_delta": np.linspace(0.0, 1.0, n),
        }
    )
    out1, cols, _ = generate_depth_entry_candidates(
        features, ["sig"], include_sign_agree=False, include_imbalance_delta=False, quantiles=(0.7,)
    )
    noisy = features.with_columns(
        pl.when(pl.col(AVAILABILITY_TS) >= 60)
        .then(pl.lit(100.0))
        .otherwise(pl.col("depth_path_pressure"))
        .alias("depth_path_pressure")
    )
    out2, _, _ = generate_depth_entry_candidates(
        noisy, ["sig"], include_sign_agree=False, include_imbalance_delta=False, quantiles=(0.7,)
    )
    col = cols[0]
    past1 = out1.filter(pl.col(AVAILABILITY_TS) < 60).select(col)
    past2 = out2.filter(pl.col(AVAILABILITY_TS) < 60).select(col)
    assert past1.equals(past2)


def test_attach_depth_path_asof_backward() -> None:
    nq, _ = _paired_streams(600, seed=2)
    clock = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(0, 50_000, 10_000)),
            "nq_close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sig": [1.0, 0.0, -1.0, 1.0, 0.0],
        }
    )
    joined = attach_depth_path_to_features(clock, nq, interval_ns=10_000)
    assert "depth_path_pressure" in joined.columns
    assert joined.height == clock.height


def test_breakout_search_with_depth_filter_smoke() -> None:
    nq, mnq = _paired_streams(2200, seed=33)
    result = search_fail_breakout_hypotheses(
        nq,
        mnq,
        specs=core_breakout_grid()[:2],
        interval_ns=10_000,
        use_ssl_gate=False,
        enhance_with_ssl=False,
        use_depth_filter=True,
        n_splits=2,
        n_permutations=30,
        rng=make_generator(4),
        quiet=True,
    )
    assert any("__depth__" in c for c in result.candidate_columns)
    assert "depth_path_pressure" in result.features.columns


def test_fvg_search_with_depth_filter_smoke() -> None:
    nq, mnq = _paired_streams(2200, seed=44)
    result = search_fail_fvg_hypotheses(
        nq,
        mnq,
        specs=default_fvg_grid()[:2],
        interval_ns=10_000,
        use_ssl_gate=False,
        use_depth_filter=True,
        n_splits=2,
        n_permutations=30,
        rng=make_generator(5),
        quiet=True,
    )
    assert any("__depth__" in c for c in result.candidate_columns)
