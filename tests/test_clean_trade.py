"""صفقة نظيفة: امتداد ATR مع MAE أصغر، بلا تسريب وبلا طبقة تنفيذ حيّة."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from nq.auction_behavior.clean_trade import (
    CLEAN_HORIZON_BARS,
    CLEAN_MAE_ATR_FRAC,
    CLEAN_OPERATING_P,
    CLEAN_ROUND_TRIP_COST_PTS,
    CLEAN_TARGET_ATR_FRAC,
    Y_CLEAN,
    build_clean_trade_outcomes,
    simulate_clean_exits,
    summarize_clean_oof,
)
from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.auction_behavior.realized_path import science_outcome_targets
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.validation.leakage import assert_availability_not_before_event

_ET = ZoneInfo("America/New_York")
_LONDON = int(VpLiquiditySession.LONDON)


def _ns(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    stamp = dt.datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _london_story(
    *,
    day: int,
    n: int,
    close: float,
    high: float,
    low: float,
    story: int,
    onset_at: int | None = None,
    dip_low: float | None = None,
    dip_bar: int | None = None,
    spike_high: float | None = None,
    spike_bar: int | None = None,
    direction: float = 1.0,
) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    for bar in range(n):
        is_onset = onset_at is not None and bar == onset_at
        bar_low = low if dip_bar is None or bar != dip_bar or dip_low is None else dip_low
        bar_high = (
            high if spike_bar is None or bar != spike_bar or spike_high is None else spike_high
        )
        if bar <= (onset_at if onset_at is not None else -1):
            bar_high = close
            bar_low = close
        rows.append(
            {
                AVAILABILITY_TS: _ns(2025, 6, day, 4, bar),
                VP_LIQUIDITY_SESSION: _LONDON,
                "close": close,
                "high": bar_high,
                "low": bar_low,
                "path_beyond_asia_ticks": 2.0
                if (onset_at is not None and bar >= onset_at)
                else 0.0,
                "vp_fsm_break": 1.0 if is_onset else 0.0,
                "vp_fsm_retest": 0.0,
                "proj_break_direction": direction,
                "_behavior_story_run": story,
            }
        )
    return pl.DataFrame(rows)


def _prior_range100() -> pl.DataFrame:
    prior = _london_story(day=2, n=3, close=100.0, high=150.0, low=50.0, story=0)
    return prior.with_columns(
        pl.lit(150.0).alias("high"),
        pl.lit(50.0).alias("low"),
        pl.lit(100.0).alias("close"),
    )


def test_science_keeps_prior_heads_and_adds_clean() -> None:
    names = science_outcome_targets(include_assumed_scripts=False)
    assert "y_path_further_beyond" in names
    assert "y_extend_5pts_25min" in names
    assert "y_phase_extend" in names
    assert Y_CLEAN in names
    assert names.count(Y_CLEAN) == 1
    assert CLEAN_HORIZON_BARS == 50
    assert CLEAN_TARGET_ATR_FRAC == 0.15
    assert CLEAN_MAE_ATR_FRAC == 0.08
    assert CLEAN_OPERATING_P == 0.5
    assert CLEAN_ROUND_TRIP_COST_PTS == 0.75


def test_clean_true_when_mfe_beats_mae() -> None:
    today = _london_story(day=3, n=9, close=200.0, high=216.0, low=197.0, story=1, onset_at=0)
    labels = build_clean_trade_outcomes(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    resolved = labels.filter(pl.col("label_status") == "resolved")
    assert resolved.height == 1
    assert float(resolved["y"][0]) == 1.0
    assert_availability_not_before_event(
        resolved[SETUP_AVAILABILITY_TS].to_numpy(),
        resolved[OUTCOME_AVAILABLE_TS].to_numpy(),
    )


def test_clean_false_when_mae_reaches_stop() -> None:
    today = _london_story(
        day=3,
        n=9,
        close=200.0,
        high=216.0,
        low=197.0,
        story=1,
        onset_at=0,
        dip_low=191.0,
        dip_bar=3,
    )
    labels = build_clean_trade_outcomes(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    assert float(labels.filter(pl.col("label_status") == "resolved")["y"][0]) == 0.0


def test_clean_false_when_mae_equals_stop_bound() -> None:
    today = _london_story(
        day=3,
        n=9,
        close=200.0,
        high=216.0,
        low=192.0,
        story=1,
        onset_at=0,
    )
    labels = build_clean_trade_outcomes(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    assert float(labels.filter(pl.col("label_status") == "resolved")["y"][0]) == 0.0


def test_clean_false_when_expansion_is_short() -> None:
    today = _london_story(day=3, n=9, close=200.0, high=210.0, low=197.0, story=1, onset_at=0)
    labels = build_clean_trade_outcomes(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    assert float(labels.filter(pl.col("label_status") == "resolved")["y"][0]) == 0.0


def test_clean_incomplete_window_is_censored() -> None:
    today = _london_story(day=3, n=4, close=200.0, high=216.0, low=197.0, story=1, onset_at=0)
    labels = build_clean_trade_outcomes(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    assert labels.filter(pl.col("label_status") == "censored").height == 1
    assert labels.filter(pl.col("label_status") == "resolved").height == 0


def test_clean_short_side_uses_downside_mfe() -> None:
    today = _london_story(
        day=3,
        n=9,
        close=200.0,
        high=203.0,
        low=184.0,
        story=1,
        onset_at=0,
        direction=-1.0,
    )
    labels = build_clean_trade_outcomes(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    assert float(labels.filter(pl.col("label_status") == "resolved")["y"][0]) == 1.0


def test_first_touch_takes_target_without_stop() -> None:
    today = _london_story(day=3, n=9, close=200.0, high=216.0, low=197.0, story=1, onset_at=0)
    exits = simulate_clean_exits(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    row = exits.row(0, named=True)
    assert row["exit_reason"] == "target"
    assert row["pnl_gross_pts"] == pytest.approx(15.0)
    assert row["pnl_net_pts"] == pytest.approx(15.0 - CLEAN_ROUND_TRIP_COST_PTS)
    assert float(row["y_clean"]) == 1.0


def test_first_touch_stop_beats_later_target() -> None:
    today = _london_story(
        day=3,
        n=9,
        close=200.0,
        high=197.0,
        low=197.0,
        story=1,
        onset_at=0,
        dip_low=191.0,
        dip_bar=2,
        spike_high=220.0,
        spike_bar=5,
    )
    exits = simulate_clean_exits(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    row = exits.row(0, named=True)
    assert row["exit_reason"] == "stop"
    assert row["pnl_gross_pts"] == pytest.approx(-8.0)
    assert float(row["y_clean"]) == 0.0


def test_same_bar_target_and_stop_counts_stop_first() -> None:
    today = _london_story(day=3, n=9, close=200.0, high=216.0, low=191.0, story=1, onset_at=0)
    exits = simulate_clean_exits(
        pl.concat([_prior_range100(), today]), window=8, group_col="_behavior_story_run"
    )
    row = exits.row(0, named=True)
    assert row["exit_reason"] == "stop"
    assert float(row["y_clean"]) == 0.0


def test_oof_summary_uses_declared_operating_point() -> None:
    today = _london_story(day=3, n=9, close=200.0, high=216.0, low=197.0, story=1, onset_at=0)
    frame = pl.concat([_prior_range100(), today])
    exits = simulate_clean_exits(frame, window=8, group_col="_behavior_story_run")
    setup = int(exits[SETUP_AVAILABILITY_TS][0])
    oof = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [setup, setup],
            "outcome_name": [Y_CLEAN, "y_path_further_beyond"],
            "p_cal": [0.61, 0.99],
        }
    )
    diag = summarize_clean_oof(exits, oof)
    assert diag["n_fires"] == 1
    assert diag["win_rate"] == pytest.approx(1.0)
    assert diag["mean_net_pts"] == pytest.approx(15.0 - CLEAN_ROUND_TRIP_COST_PTS)
    assert diag["is_live_overlay"] is False
    assert diag["thresholds_tuned_on_oof"] is False
    quiet = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [setup],
            "outcome_name": [Y_CLEAN],
            "p_cal": [0.49],
        }
    )
    assert summarize_clean_oof(exits, quiet)["n_fires"] == 0
