"""اختبارات طبقة العلم 1–9 (سببية + holdout مجمّد + معايرة)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior import BehaviorConfig, run_auction_behavior_analysis
from nq.auction_behavior.calibration import expected_calibration_error
from nq.auction_behavior.holdout import carve_frozen_holdout, evaluate_frozen_holdout_once
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    build_labeled_outcomes,
)
from nq.auction_behavior.science import ScienceConfig, run_behavior_science
from nq.contracts.temporal import AVAILABILITY_TS
from nq.validation.leakage import assert_availability_not_before_event
from tests.test_auction_behavior import _dense_trade_stream


def _long_stream(n_bars: int = 120) -> pl.DataFrame:
    return _dense_trade_stream(n_bars=n_bars, bar_ns=200)


@pytest.mark.leakage
def test_outcomes_respect_availability_order() -> None:
    frame = _long_stream(80)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            include_asia_london_projection=False,
            include_science=True,
            evaluate_holdout=False,
            n_splits=3,
            min_train_size=8,
            holdout_frac=0.2,
        ),
    )
    assert result.science is not None
    labeled = result.science.labeled
    if labeled.height == 0:
        pytest.skip("no labeled setups on synthetic stream")
    assert_availability_not_before_event(
        labeled[SETUP_AVAILABILITY_TS].to_numpy(),
        labeled[OUTCOME_AVAILABLE_TS].to_numpy(),
    )
    assert (labeled[OUTCOME_AVAILABLE_TS] >= labeled[SETUP_AVAILABILITY_TS]).all()


@pytest.mark.leakage
def test_frozen_holdout_refuses_second_touch() -> None:
    y = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: list(range(100)),
            OUTCOME_AVAILABLE_TS: list(range(1, 101)),
            "outcome_name": ["y_true_break"] * 100,
            "y": [0.0, 1.0] * 50,
            "p_hat": [0.4] * 100,
        }
    )
    pack = carve_frozen_holdout(y, holdout_frac=0.2)
    assert pack.holdout.height >= 1
    scored = pack.holdout.with_columns(pl.lit(0.4).alias("p_hat"))
    _eval1, touched = evaluate_frozen_holdout_once(pack, scored)
    assert touched.touched is True
    with pytest.raises(RuntimeError, match="already touched"):
        evaluate_frozen_holdout_once(touched, scored, allow_retouch=False)


def test_science_stack_runs_end_to_end() -> None:
    frame = _long_stream(160)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            include_asia_london_projection=False,
            include_science=True,
            evaluate_holdout=True,
            n_splits=4,
            min_train_size=10,
            holdout_frac=0.2,
            memory_lags=(1, 2),
        ),
    )
    assert result.validation.ok
    assert result.science is not None
    assert "struct_dist_vah_ticks" in result.blended.columns
    mem_cols = [c for c in result.blended.columns if "__rmean" in c or "__lag" in c]
    assert mem_cols
    diag = result.science.diagnostics
    assert "conditional_model" in diag["science_steps"]
    assert "frozen_final_holdout" in diag["science_steps"]
    assert diag["holdout_cut_ts"] != 0 or result.science.labeled.height == 0


def test_ece_bounds_and_perfect_calibration() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=4) == pytest.approx(0.0)


def test_build_labeled_outcomes_empty_safe() -> None:
    out = build_labeled_outcomes(pl.DataFrame({AVAILABILITY_TS: []}))
    assert out.height == 0
    assert SETUP_AVAILABILITY_TS in out.columns
    assert OUTCOME_AVAILABLE_TS in out.columns


def test_science_without_holdout_eval() -> None:
    frame = _long_stream(100)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=800,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            include_asia_london_projection=False,
            include_science=True,
            evaluate_holdout=False,
            holdout_frac=0.25,
            n_splits=3,
            min_train_size=8,
        ),
    )
    assert result.science is not None
    assert result.science.holdout_eval is None
    assert result.science.holdout.touched is False


def test_run_behavior_science_direct_on_blended() -> None:
    frame = _long_stream(90)
    base = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            include_asia_london_projection=False,
            include_science=False,
        ),
    )
    report = run_behavior_science(
        base.blended,
        config=ScienceConfig(
            outcome_window=6,
            n_splits=3,
            min_train_size=8,
            holdout_frac=0.2,
            evaluate_holdout=False,
            use_month_folds=False,
        ),
    )
    assert report.diagnostics["n_labeled"] >= 0
    if report.labeled.height:
        assert report.feature_names
