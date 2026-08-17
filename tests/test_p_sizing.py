"""طبقة ج: حجم من p وخروج انقلاب OOF — بلا تنبؤ حي وبلا holdout."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.p_sizing import (
    PSizingConfig,
    position_size,
    render_p_sizing_markdown,
    run_p_sizing,
    run_p_sizing_from_period_dir,
    write_p_sizing_report,
)

_GROUP = "_behavior_story_run"


def _month_ns(year: int, month: int, day: int = 10) -> int:
    stamp = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def _bars(*, story: int, start: int, beyond: list[float]) -> pl.DataFrame:
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
            "path_depth_follow": [2.0] * n,
            "path_depth_confirm": [2.0] * n,
            "vp_balance": [0.0] * n,
        }
    )


def _setup(
    *,
    story: int,
    ts: int,
    beyond: float,
    p: float,
    y: float = 1.0,
    outcome_ts: int | None = None,
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
        "path_depth_follow": 2.0,
        "path_depth_confirm": 2.0,
        "vp_balance": 0.0,
        "proj_outside_volume_share": 0.8,
    }


def _cfg() -> PSizingConfig:
    return PSizingConfig(holdout_months=None, round_trip_cost_pts=0.75, max_hold_bars=8)


def test_position_size_scales_with_p() -> None:
    assert position_size(0.49, enter_p=0.5, min_size=0.2) == 0.0
    assert position_size(0.5, enter_p=0.5, min_size=0.2) == pytest.approx(0.2)
    assert position_size(1.0, enter_p=0.5, min_size=0.2) == pytest.approx(1.0)


def test_p_flip_exits_on_fresh_oof_not_gap_bars() -> None:
    blended = _bars(story=1, start=10, beyond=[80.0, 84.0, 88.0, 40.0])
    labeled = pl.DataFrame(
        [
            _setup(story=1, ts=10, beyond=80.0, p=0.9, outcome_ts=11),
            _setup(story=1, ts=12, beyond=88.0, p=0.2, outcome_ts=13),
        ]
    )
    predictions = pl.DataFrame({AVAILABILITY_TS: [10, 12], "p_y_path_further_beyond": [0.9, 0.2]})
    report = run_p_sizing(labeled, blended, config=_cfg(), predictions=predictions)
    fires = report.trades.filter(pl.col("p_entry") >= 0.5)
    assert fires.height >= 1
    row = fires.row(0, named=True)
    assert row["exit_reason"] == "p_flip"
    assert int(row["hold_bars"]) == 2
    assert float(row["size"]) == pytest.approx(position_size(0.9, enter_p=0.5, min_size=0.2))
    assert float(row["realized_beyond_pts"]) == pytest.approx(2.0)


def test_no_fresh_oof_holds_to_cap() -> None:
    blended = _bars(story=1, start=10, beyond=[80.0, 84.0, 88.0, 92.0])
    labeled = pl.DataFrame([_setup(story=1, ts=10, beyond=80.0, p=0.8, outcome_ts=11)])
    predictions = pl.DataFrame({AVAILABILITY_TS: [10], "p_y_path_further_beyond": [0.8]})
    report = run_p_sizing(
        labeled,
        blended,
        config=PSizingConfig(holdout_months=None, max_hold_bars=3),
        predictions=predictions,
    )
    assert report.trades["exit_reason"][0] == "max_hold"
    assert int(report.trades["hold_bars"][0]) == 3
    assert report.diagnostics["p_flip_uses_fresh_oof_only"] is True
    assert report.diagnostics["live_predictions_not_used"] is True


def test_live_predictions_file_is_not_loaded(tmp_path: Path) -> None:
    ts = _month_ns(2025, 5)
    blended = _bars(story=1, start=ts, beyond=[80.0, 84.0, 88.0])
    labeled = pl.DataFrame([_setup(story=1, ts=ts, beyond=80.0, p=0.8, outcome_ts=ts + 1)])
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [ts], "p_y_path_further_beyond": [0.8]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    pl.DataFrame(
        {AVAILABILITY_TS: [ts + 1, ts + 2], "p_y_path_further_beyond": [0.05, 0.05]}
    ).write_parquet(period / "live_predictions.parquet")
    report = run_p_sizing_from_period_dir(period, config=_cfg())
    assert report.diagnostics["live_predictions_not_used"] is True
    assert report.trades["exit_reason"][0] == "max_hold"


def test_holdout_excluded() -> None:
    rows_b: list[pl.DataFrame] = []
    rows_l: list[dict[str, float | int | str]] = []
    oof_ts: list[int] = []
    oof_p: list[float] = []
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        rows_b.append(_bars(story=month, start=ts, beyond=[80.0, 90.0, 20.0]))
        rows_l.append(_setup(story=month, ts=ts, beyond=80.0, p=0.9, outcome_ts=ts + 2))
        oof_ts.append(ts)
        oof_p.append(0.9)
        oof_ts.append(ts + 2)
        oof_p.append(0.1)
        rows_l.append(_setup(story=month, ts=ts + 2, beyond=20.0, p=0.1, outcome_ts=ts + 3))
    predictions = pl.DataFrame({AVAILABILITY_TS: oof_ts, "p_y_path_further_beyond": oof_p})
    report = run_p_sizing(
        pl.DataFrame(rows_l),
        pl.concat(rows_b),
        config=PSizingConfig(holdout_months=4, max_hold_bars=3),
        predictions=predictions,
    )
    assert report.diagnostics["holdout_scored"] is False
    assert int(report.diagnostics["n_trades"]) == 8


def test_removable_markdown(tmp_path: Path) -> None:
    blended = _bars(story=1, start=10, beyond=[80.0, 81.0, 82.0])
    labeled = pl.DataFrame([_setup(story=1, ts=10, beyond=80.0, p=0.8, outcome_ts=11)])
    predictions = pl.DataFrame({AVAILABILITY_TS: [10], "p_y_path_further_beyond": [0.8]})
    report = run_p_sizing(labeled, blended, config=_cfg(), predictions=predictions)
    write_p_sizing_report(report, tmp_path)
    text = (tmp_path / "P_SIZING.md").read_text(encoding="utf-8")
    assert "removable" in text.lower()
    assert "Live predictions" in render_p_sizing_markdown(report)
