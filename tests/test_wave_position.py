"""موقع أول إشارة على الموجة المكتملة — ليس تسريبًا في الميزات."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.wave_position import (
    WAVE_BIN_COL,
    WavePositionConfig,
    assert_not_raw_mbo_stream,
    build_wave_geometry,
    classify_wave_bin,
    run_wave_position,
    run_wave_position_from_period_dir,
    write_wave_position_report,
)

_GROUP = "_behavior_story_run"


def _month_ns(year: int, month: int, day: int = 10) -> int:
    stamp = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def test_wave_bins_match_requested_cuts() -> None:
    frame = pl.DataFrame({"frac": [0.0, 0.199, 0.20, 0.399, 0.40, 0.599, 0.60, 1.0]})
    out = frame.with_columns(classify_wave_bin(pl.col("frac")).alias(WAVE_BIN_COL))
    assert out[WAVE_BIN_COL].to_list() == [
        "early_prediction",
        "early_prediction",
        "early_continuation",
        "early_continuation",
        "mid_wave",
        "mid_wave",
        "late_prediction",
        "late_prediction",
    ]


def _story_bars(*, story: int, start: int, beyond: list[float]) -> pl.DataFrame:
    extreme: list[float] = []
    running = 0.0
    for value in beyond:
        running = max(running, value)
        extreme.append(running)
    n = len(beyond)
    return pl.DataFrame(
        {
            AVAILABILITY_TS: [start + i for i in range(n)],
            _GROUP: [story] * n,
            "path_beyond_asia_ticks": beyond,
            "path_extreme_ticks": extreme,
            "vp_fsm_break": [1.0] + [0.0] * (n - 1),
        }
    )


def _setup(
    *,
    story: int,
    ts: int,
    y: float,
    beyond: float,
    extreme: float,
) -> dict[str, float | int | str]:
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: ts + 1,
        "outcome_name": "y_path_further_beyond",
        "y": y,
        "label_status": "resolved",
        _GROUP: story,
        "path_beyond_asia_ticks": beyond,
        "path_extreme_ticks": extreme,
        "vp_fsm_break": 1.0,
    }


def test_first_signal_at_onset_is_early_when_wave_continues() -> None:
    blended = _story_bars(story=1, start=10, beyond=[2.0, 10.0, 40.0, 50.0])
    labeled = pl.DataFrame([_setup(story=1, ts=10, y=1.0, beyond=2.0, extreme=2.0)])
    report = run_wave_position(
        labeled,
        blended,
        config=WavePositionConfig(holdout_months=None, min_peak_ticks=8.0),
    )
    row = report.first_signals.row(0, named=True)
    assert row[WAVE_BIN_COL] == "early_prediction"
    assert abs(float(row["wave_frac"]) - 2.0 / 50.0) < 1e-9


def test_signal_near_peak_is_late_prediction() -> None:
    blended = _story_bars(story=2, start=10, beyond=[2.0, 20.0, 40.0, 50.0])
    labeled = pl.DataFrame([_setup(story=2, ts=12, y=1.0, beyond=40.0, extreme=40.0)])
    report = run_wave_position(
        labeled,
        blended,
        config=WavePositionConfig(holdout_months=None, min_peak_ticks=8.0),
    )
    row = report.first_signals.row(0, named=True)
    assert row[WAVE_BIN_COL] == "late_prediction"
    assert float(row["wave_frac"]) == pytest.approx(0.8)


def test_holdout_waves_are_excluded() -> None:
    rows_b: list[pl.DataFrame] = []
    rows_l: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        holdout = month >= 9
        peak = [2.0, 10.0, 40.0, 50.0] if holdout else [2.0, 4.0, 6.0, 10.0]
        rows_b.append(_story_bars(story=month, start=ts, beyond=peak))
        rows_l.append(
            _setup(
                story=month,
                ts=ts,
                y=1.0,
                beyond=40.0 if holdout else 2.0,
                extreme=40.0 if holdout else 2.0,
            )
        )
    blended = pl.concat(rows_b)
    labeled = pl.DataFrame(rows_l)
    report = run_wave_position(
        labeled,
        blended,
        config=WavePositionConfig(holdout_months=4, min_peak_ticks=8.0),
    )
    assert report.diagnostics["holdout_scored"] is False
    assert report.diagnostics["holdout_excluded"] is True
    late = report.first_summary.filter(
        (pl.col("scope") == "develop")
        & (pl.col(WAVE_BIN_COL) == "late_prediction")
        & (pl.col("success_only"))
    )
    assert late.height == 1
    assert int(late["n"][0]) == 0


def test_y_shuffle_does_not_change_wave_frac() -> None:
    blended = _story_bars(story=1, start=1, beyond=[2.0, 25.0, 50.0])
    labeled = pl.DataFrame(
        [
            _setup(story=1, ts=1, y=1.0, beyond=2.0, extreme=2.0),
            _setup(story=1, ts=2, y=0.0, beyond=25.0, extreme=25.0),
        ]
    )
    cfg = WavePositionConfig(holdout_months=None)
    a = run_wave_position(labeled, blended, config=cfg)
    b = run_wave_position(labeled.with_columns(pl.col("y").reverse()), blended, config=cfg)
    left = a.all_setups.sort(SETUP_AVAILABILITY_TS)["wave_frac"].to_list()
    right = b.all_setups.sort(SETUP_AVAILABILITY_TS)["wave_frac"].to_list()
    assert left == right


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame(
        {
            AVAILABILITY_TS: [1],
            "order_id": [1],
            "action": ["A"],
            _GROUP: [1],
        }
    )
    with pytest.raises(ValueError, match="refuses raw MBO"):
        assert_not_raw_mbo_stream(raw)
    with pytest.raises(ValueError, match="refuses raw MBO"):
        build_wave_geometry(raw)


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    blended = pl.concat(
        [
            _story_bars(story=1, start=_month_ns(2025, 5), beyond=[2.0, 10.0, 50.0]),
            _story_bars(story=2, start=_month_ns(2025, 6), beyond=[2.0, 40.0, 50.0]),
        ]
    )
    labeled = pl.DataFrame(
        [
            _setup(story=1, ts=_month_ns(2025, 5), y=1.0, beyond=2.0, extreme=2.0),
            _setup(story=2, ts=_month_ns(2025, 6) + 1, y=1.0, beyond=40.0, extreme=40.0),
        ]
    )
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: labeled[SETUP_AVAILABILITY_TS].to_list()}).write_parquet(
        period / "oof_predictions.parquet"
    )
    (period / "summary.json").write_text(
        json.dumps({"diagnostics": {"science": {"holdout_cut_ts": _month_ns(2025, 8)}}}),
        encoding="utf-8",
    )
    report = run_wave_position_from_period_dir(
        period, config=WavePositionConfig(holdout_months=None)
    )
    out = tmp_path / "wave"
    write_wave_position_report(report, out)
    assert (out / "WAVE.md").is_file()
    text = (out / "WAVE.md").read_text(encoding="utf-8")
    assert "Holdout never scored" in text
    assert "early_prediction" in text
    assert report.diagnostics["holdout_scored"] is False
    assert report.diagnostics["wave_peak_is_diagnostic_lookahead_not_a_feature"] is True
