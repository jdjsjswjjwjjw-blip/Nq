"""طبقة ب: خروج فيتشز بعد إطلاق سببي — بلا وقف رقمي وبلا holdout."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.feature_exit import (
    FeatureExitConfig,
    render_feature_exit_markdown,
    run_feature_exit,
    run_feature_exit_from_period_dir,
    write_feature_exit_report,
)

_GROUP = "_behavior_story_run"


def _month_ns(year: int, month: int, day: int = 10) -> int:
    stamp = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def _bars(
    *,
    story: int,
    start: int,
    beyond: list[float],
    follow: list[float],
    balance: list[float] | None = None,
    fail: list[float] | None = None,
) -> pl.DataFrame:
    n = len(beyond)
    extreme: list[float] = []
    running = 0.0
    for value in beyond:
        running = max(running, value)
        extreme.append(running)
    return pl.DataFrame(
        {
            AVAILABILITY_TS: [start + i for i in range(n)],
            _GROUP: [story] * n,
            "path_beyond_asia_ticks": beyond,
            "path_extreme_ticks": extreme,
            "path_depth_follow": follow,
            "path_depth_confirm": follow,
            "path_change_fail": fail if fail is not None else [0.0] * n,
            "vp_balance": balance if balance is not None else [0.0] * n,
        }
    )


def _setup(
    *,
    story: int,
    ts: int,
    beyond: float,
    follow: float,
    p: float = 0.8,
    y: float = 1.0,
    outcome_ts: int | None = None,
    balance: float = 0.0,
) -> dict[str, float | int | str]:
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: ts + 1 if outcome_ts is None else outcome_ts,
        "outcome_name": "y_path_further_beyond",
        "y": y,
        "label_status": "resolved",
        _GROUP: story,
        "path_beyond_asia_ticks": beyond,
        "path_extreme_ticks": beyond,
        "p_y_path_further_beyond": p,
        "path_depth_follow": follow,
        "path_depth_confirm": follow,
        "path_change_fail": 0.0,
        "vp_balance": balance,
        "proj_outside_volume_share": 0.8,
    }


def _cfg() -> FeatureExitConfig:
    return FeatureExitConfig(holdout_months=None, round_trip_cost_pts=0.75, max_hold_bars=8)


def test_depth_follow_drop_exits_before_max_hold() -> None:
    blended = _bars(
        story=1,
        start=10,
        beyond=[80.0, 84.0, 88.0, 40.0],
        follow=[2.0, 2.0, 0.5, 0.4],
    )
    labeled = pl.DataFrame([_setup(story=1, ts=10, beyond=80.0, follow=2.0, outcome_ts=11)])
    report = run_feature_exit(labeled, blended, config=_cfg())
    assert report.diagnostics["removable_layer"] is True
    assert report.diagnostics["numeric_stop_take_not_used"] is True
    assert report.trades.height == 1
    row = report.trades.row(0, named=True)
    assert row["exit_reason"] == "depth_follow_drop"
    assert int(row["hold_bars"]) == 2
    assert float(row["realized_beyond_pts"]) == pytest.approx(2.0)
    assert float(row["net_pts"]) == pytest.approx(2.0 - 0.75)


def test_balance_return_exits() -> None:
    blended = _bars(
        story=1,
        start=10,
        beyond=[80.0, 82.0, 83.0],
        follow=[2.0, 2.0, 2.0],
        balance=[0.0, 0.0, 1.0],
    )
    labeled = pl.DataFrame([_setup(story=1, ts=10, beyond=80.0, follow=2.0, outcome_ts=11)])
    report = run_feature_exit(labeled, blended, config=_cfg())
    assert report.trades["exit_reason"][0] == "balance_return"
    assert int(report.trades["hold_bars"][0]) == 2


def test_max_hold_when_features_stay_alive() -> None:
    blended = _bars(
        story=1,
        start=10,
        beyond=[80.0, 81.0, 82.0, 83.0],
        follow=[2.0, 2.0, 2.0, 2.0],
    )
    labeled = pl.DataFrame([_setup(story=1, ts=10, beyond=80.0, follow=2.0, outcome_ts=11)])
    report = run_feature_exit(
        labeled, blended, config=FeatureExitConfig(holdout_months=None, max_hold_bars=2)
    )
    assert report.trades["exit_reason"][0] == "max_hold"
    assert int(report.trades["hold_bars"][0]) == 2


def test_label_window_does_not_clip_hold() -> None:
    blended = _bars(
        story=1,
        start=10,
        beyond=[80.0, 84.0, 88.0, 92.0],
        follow=[2.0, 2.0, 2.0, 2.0],
    )
    labeled = pl.DataFrame([_setup(story=1, ts=10, beyond=80.0, follow=2.0, outcome_ts=11)])
    report = run_feature_exit(
        labeled, blended, config=FeatureExitConfig(holdout_months=None, max_hold_bars=3)
    )
    assert int(report.trades["hold_bars"][0]) == 3
    assert float(report.trades["realized_beyond_pts"][0]) == pytest.approx(3.0)


def test_y_shuffle_does_not_change_exit() -> None:
    blended = _bars(
        story=1,
        start=10,
        beyond=[80.0, 84.0, 40.0],
        follow=[2.0, 0.4, 0.4],
    )
    labeled = pl.DataFrame([_setup(story=1, ts=10, beyond=80.0, follow=2.0, y=1.0, outcome_ts=12)])
    a = run_feature_exit(labeled, blended, config=_cfg())
    b = run_feature_exit(labeled.with_columns(pl.col("y") * 0.0), blended, config=_cfg())
    assert a.trades["exit_reason"].to_list() == b.trades["exit_reason"].to_list()
    assert a.trades["net_pts"].to_list() == b.trades["net_pts"].to_list()


def test_holdout_excluded() -> None:
    rows_b: list[pl.DataFrame] = []
    rows_l: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        holdout = month >= 9
        follow = [2.0, 0.2, 0.2] if holdout else [2.0, 2.0, 2.0]
        rows_b.append(_bars(story=month, start=ts, beyond=[80.0, 90.0, 100.0], follow=follow))
        rows_l.append(_setup(story=month, ts=ts, beyond=80.0, follow=2.0, outcome_ts=ts + 2))
    report = run_feature_exit(
        pl.DataFrame(rows_l),
        pl.concat(rows_b),
        config=FeatureExitConfig(holdout_months=4, max_hold_bars=3),
    )
    assert report.diagnostics["holdout_scored"] is False
    reasons = report.diagnostics["exit_reasons"]
    assert isinstance(reasons, dict)
    assert "depth_follow_drop" not in reasons


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame({AVAILABILITY_TS: [1], "order_id": [1], "action": ["A"], _GROUP: [1]})
    labeled = pl.DataFrame([_setup(story=1, ts=1, beyond=80.0, follow=2.0)])
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_feature_exit(labeled, raw, config=_cfg())


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    ts = _month_ns(2025, 5)
    blended = _bars(story=1, start=ts, beyond=[80.0, 84.0, 40.0], follow=[2.0, 0.4, 0.3])
    labeled = pl.DataFrame([_setup(story=1, ts=ts, beyond=80.0, follow=2.0, outcome_ts=ts + 1)])
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [ts], "p_y_path_further_beyond": [0.81]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    report = run_feature_exit_from_period_dir(period, config=_cfg())
    out = tmp_path / "out"
    write_feature_exit_report(report, out)
    text = (out / "FEATURE_EXIT.md").read_text(encoding="utf-8")
    assert "removable" in text.lower()
    assert "No numeric" in render_feature_exit_markdown(report)
