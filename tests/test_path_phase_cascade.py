"""طبقة التيك ثم الطور: فلتر AND قابل للخلع، بلا إعادة تدريب."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.auction_behavior.realized_path import Y_PHASE_EXTEND
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.path_phase_cascade import (
    MIN_P_PATH,
    MIN_P_PHASE,
    Y_PATH,
    assert_oof_fold_scores,
    build_cascade_fires,
    run_path_phase_cascade,
    write_cascade_report,
)

_ET = ZoneInfo("America/New_York")


def _ns(minute: int) -> int:
    stamp = dt.datetime(2025, 6, 3, 4, minute, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "run_path_phase_cascade" not in nq.research.__all__
    assert not hasattr(nq.research, "run_path_phase_cascade")


def test_rejects_wide_live_state_predictions() -> None:
    wide = pl.DataFrame({"availability_ts": [1], "p_y_path_further_beyond": [0.9]})
    with pytest.raises(ValueError, match="fold_scores"):
        assert_oof_fold_scores(wide)


def test_rejects_scores_without_phase_head() -> None:
    frame = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1],
            "outcome_name": [Y_PATH],
            "p_hat": [0.9],
            "prediction_is_oof": [True],
            "eligible_for_backtest": [True],
        }
    )
    with pytest.raises(ValueError, match="y_phase_extend"):
        assert_oof_fold_scores(frame)


def test_rejects_live_ineligible_rows() -> None:
    frame = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1, 1],
            "outcome_name": [Y_PATH, Y_PHASE_EXTEND],
            "p_hat": [0.9, 0.9],
            "eligible_for_backtest": [False, False],
        }
    )
    with pytest.raises(ValueError, match="ineligible"):
        assert_oof_fold_scores(frame)


def test_rejects_non_oof_flag() -> None:
    frame = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1, 1],
            "outcome_name": [Y_PATH, Y_PHASE_EXTEND],
            "p_hat": [0.9, 0.9],
            "prediction_is_oof": [False, False],
            "eligible_for_backtest": [True, True],
        }
    )
    with pytest.raises(ValueError, match="live predictions"):
        assert_oof_fold_scores(frame)


def test_and_gate_keeps_path_only_and_cascade() -> None:
    scores = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [10, 10, 20, 20, 30, 30],
            "outcome_name": [
                Y_PATH,
                Y_PHASE_EXTEND,
                Y_PATH,
                Y_PHASE_EXTEND,
                Y_PATH,
                Y_PHASE_EXTEND,
            ],
            "p_hat": [0.8, 0.7, 0.9, 0.4, 0.4, 0.9],
            "prediction_is_oof": [True] * 6,
            "eligible_for_backtest": [True] * 6,
        }
    )
    fires = build_cascade_fires(scores)
    path = fires.filter(pl.col("cohort") == "path_only")
    cas = fires.filter(pl.col("cohort") == "path_and_phase")
    assert sorted(path[SETUP_AVAILABILITY_TS].to_list()) == [10, 20]
    assert cas[SETUP_AVAILABILITY_TS].to_list() == [10]
    assert MIN_P_PATH == 0.5
    assert MIN_P_PHASE == 0.5


def test_skips_undirected_setups() -> None:
    ts = [_ns(i) for i in range(8)]
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: ts,
            "close": [200.0] * 8,
            "high": [210.0] * 8,
            "low": [199.0] * 8,
            "proj_break_direction": [0.0] * 8,
            "_behavior_story_run": [1] * 8,
        }
    )
    scores = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [ts[0], ts[0]],
            "outcome_name": [Y_PATH, Y_PHASE_EXTEND],
            "p_cal": [0.9, 0.9],
            "prediction_is_oof": [True, True],
            "eligible_for_backtest": [True, True],
        }
    )
    quality, diag = run_path_phase_cascade(blended=blended, fold_scores=scores)
    assert quality.height == 0
    assert diag["n_path_only_gate"] == 1
    assert diag["n_path_and_phase_gate"] == 1
    assert diag["n_path_only_unscored"] == 1


def test_cascade_improves_mae_on_synthetic_path() -> None:
    n = 12
    ts = [_ns(i) for i in range(n)]
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: ts + [_ns(20 + i) for i in range(n)],
            "close": [200.0] * n + [200.0] * n,
            "high": [200.0] + [222.0] * (n - 1) + [200.0] + [222.0] * (n - 1),
            "low": [200.0] + [198.0] * (n - 1) + [200.0] + [188.0] * (n - 1),
            "proj_break_direction": [1.0] * (2 * n),
            "_behavior_story_run": [1] * n + [2] * n,
        }
    )
    scores = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [ts[0], ts[0], _ns(20), _ns(20)],
            "outcome_name": [Y_PATH, Y_PHASE_EXTEND, Y_PATH, Y_PHASE_EXTEND],
            "p_cal": [0.8, 0.7, 0.8, 0.2],
            "prediction_is_oof": [True, True, True, True],
            "eligible_for_backtest": [True, True, True, True],
        }
    )
    quality, diag = run_path_phase_cascade(blended=blended, fold_scores=scores)
    path = quality.filter(pl.col("cohort") == "path_only")
    cas = quality.filter(pl.col("cohort") == "path_and_phase")
    assert path.height == 2
    assert cas.height == 1
    cas_mae = float(cas["mae_pts"].to_list()[0])
    path_mae = [float(x) for x in path["mae_pts"].to_list()]
    assert cas_mae < max(path_mae)
    assert bool(cas["hit_far"].to_list()[0]) is True
    assert diag["retrained"] is False
    assert diag["auto_exit"] is False
    assert diag["is_live_overlay"] is False
    assert diag["overlay_fire_set_used"] is False
    assert diag["thresholds_tuned_on_oof"] is False


def test_write_cascade_report(tmp_path: Path) -> None:
    n = 8
    ts = [_ns(i) for i in range(n)]
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: ts,
            "close": [200.0] * n,
            "high": [200.0] + [225.0] * (n - 1),
            "low": [200.0] + [199.0] * (n - 1),
            "proj_break_direction": [1.0] * n,
            "_behavior_story_run": [1] * n,
        }
    )
    scores = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [ts[0], ts[0]],
            "outcome_name": [Y_PATH, Y_PHASE_EXTEND],
            "p_cal": [0.9, 0.8],
            "prediction_is_oof": [True, True],
            "eligible_for_backtest": [True, True],
        }
    )
    quality, diag = run_path_phase_cascade(blended=blended, fold_scores=scores)
    out = write_cascade_report(quality, diag, tmp_path)
    text = (out / "CASCADE.md").read_text(encoding="utf-8")
    assert "path_and_phase" in text
    assert "Not the 5642 overlay fire set" in text
    assert (out / "summary.json").is_file()
