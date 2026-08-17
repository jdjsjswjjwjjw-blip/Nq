"""التقاط سببي بعد الإطلاق — بلا ذروة مكتملة وبلا holdout."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.causal_entry import (
    CausalEntryConfig,
    assert_not_raw_mbo_stream,
    compute_forward_capture,
    render_causal_entry_markdown,
    run_causal_entry,
    run_causal_entry_from_period_dir,
    ticks_to_nq_points,
    write_causal_entry_report,
)

_GROUP = "_behavior_story_run"


def _month_ns(year: int, month: int, day: int = 10) -> int:
    stamp = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


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
        }
    )


def _setup(
    *,
    story: int,
    ts: int,
    y: float,
    beyond: float,
    extreme: float,
    outcome_ts: int | None = None,
    p: float = 0.8,
) -> dict[str, float | int | str]:
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: ts + 4 if outcome_ts is None else outcome_ts,
        "outcome_name": "y_path_further_beyond",
        "y": y,
        "label_status": "resolved",
        _GROUP: story,
        "path_beyond_asia_ticks": beyond,
        "path_extreme_ticks": extreme,
        "p_y_path_further_beyond": p,
        "path_depth_confirm": 0.9,
        "proj_outside_volume_share": 0.8,
    }


def _cfg() -> CausalEntryConfig:
    return CausalEntryConfig(holdout_months=None)


def test_known_mfe_mae_inside_labeled_window() -> None:
    blended = _story_bars(story=1, start=10, beyond=[80.0, 90.0, 70.0, 100.0, 95.0])
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, y=1.0, beyond=80.0, extreme=80.0, outcome_ts=14)]
    )
    report = run_causal_entry(labeled, blended, config=_cfg())
    assert report.late_entries.height == 1
    row = report.late_entries.row(0, named=True)
    assert float(row["mfe_beyond_ticks"]) == pytest.approx(20.0)
    assert float(row["mae_beyond_ticks"]) == pytest.approx(10.0)
    assert float(row["realized_beyond_ticks"]) == pytest.approx(15.0)
    assert float(row["mfe_beyond_pts"]) == pytest.approx(5.0)
    assert float(row["mae_beyond_pts"]) == pytest.approx(2.5)
    assert row["window_empty"] is False


def test_late_confirmed_is_fixed_printed_threshold_not_peak_fraction() -> None:
    blended = _story_bars(story=1, start=10, beyond=[80.0, 90.0, 100.0])
    labeled = pl.DataFrame(
        [
            _setup(story=1, ts=10, y=1.0, beyond=80.0, extreme=80.0, outcome_ts=12),
            _setup(story=1, ts=10, y=1.0, beyond=20.0, extreme=20.0, outcome_ts=12),
        ]
    ).with_columns(
        pl.lit(10_000.0).alias("wave_peak_ticks"),
        pl.lit(0.05).alias("wave_frac"),
    )
    report = run_causal_entry(labeled, blended, config=_cfg())
    assert report.diagnostics["completed_wave_peak_not_used"] is True
    assert report.diagnostics["wave_frac_not_used_as_entry_filter"] is True
    assert int(report.diagnostics["n_late_confirmed"]) == 1
    assert int(report.diagnostics["n_all_fires"]) == 2
    assert float(report.late_entries["printed_at_entry_ticks"][0]) == pytest.approx(80.0)


def test_lookahead_peak_columns_do_not_change_capture() -> None:
    blended = _story_bars(story=1, start=10, beyond=[80.0, 90.0, 70.0, 100.0])
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, y=1.0, beyond=80.0, extreme=80.0, outcome_ts=13)]
    )
    clean = run_causal_entry(labeled, blended, config=_cfg())
    leaked = run_causal_entry(
        labeled.with_columns(
            pl.lit(8.0).alias("wave_peak_ticks"),
            pl.lit(0.99).alias("wave_frac"),
            pl.lit(1.0).alias("ticks_remaining_to_peak"),
        ),
        blended.with_columns(pl.lit(8.0).alias("wave_peak_ticks")),
        config=_cfg(),
    )
    assert float(clean.late_entries["mfe_beyond_ticks"][0]) == float(
        leaked.late_entries["mfe_beyond_ticks"][0]
    )
    assert float(clean.late_entries["mae_beyond_ticks"][0]) == float(
        leaked.late_entries["mae_beyond_ticks"][0]
    )


def test_holdout_entries_are_excluded() -> None:
    rows_b: list[pl.DataFrame] = []
    rows_l: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        holdout = month >= 9
        path = [80.0, 200.0, 220.0] if holdout else [80.0, 84.0, 86.0]
        rows_b.append(_story_bars(story=month, start=ts, beyond=path))
        rows_l.append(
            _setup(
                story=month,
                ts=ts,
                y=1.0,
                beyond=80.0,
                extreme=80.0,
                outcome_ts=ts + 2,
            )
        )
    report = run_causal_entry(
        pl.DataFrame(rows_l),
        pl.concat(rows_b),
        config=CausalEntryConfig(holdout_months=4),
    )
    assert report.diagnostics["holdout_scored"] is False
    assert report.diagnostics["holdout_excluded"] is True
    assert int(report.diagnostics["n_late_confirmed"]) == 8
    mfe = report.late_entries.get_column("mfe_beyond_ticks").max()
    assert isinstance(mfe, (int, float))
    assert float(mfe) < 50.0


def test_y_shuffle_does_not_change_path_geometry() -> None:
    blended = _story_bars(story=1, start=10, beyond=[80.0, 90.0, 70.0, 110.0])
    labeled = pl.DataFrame(
        [
            _setup(story=1, ts=10, y=1.0, beyond=80.0, extreme=80.0, outcome_ts=13),
            _setup(story=1, ts=11, y=0.0, beyond=90.0, extreme=90.0, outcome_ts=13),
        ]
    )
    cfg = _cfg()
    a = run_causal_entry(labeled, blended, config=cfg)
    b = run_causal_entry(labeled.with_columns(pl.col("y").reverse()), blended, config=cfg)
    left = a.all_entries.sort(SETUP_AVAILABILITY_TS)[
        ["mfe_beyond_ticks", "mae_beyond_ticks"]
    ].to_dicts()
    right = b.all_entries.sort(SETUP_AVAILABILITY_TS)[
        ["mfe_beyond_ticks", "mae_beyond_ticks"]
    ].to_dicts()
    assert left == right


def test_pre_expansion_fire_is_not_late_confirmed() -> None:
    blended = _story_bars(story=1, start=10, beyond=[8.0, 20.0, 40.0])
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, y=1.0, beyond=8.0, extreme=8.0, outcome_ts=12, p=0.9)]
    )
    report = run_causal_entry(labeled, blended, config=_cfg())
    assert int(report.diagnostics["n_all_fires"]) == 0
    assert int(report.diagnostics["n_late_confirmed"]) == 0


def test_window_after_entry_only() -> None:
    blended = _story_bars(story=1, start=10, beyond=[80.0, 200.0, 70.0, 90.0])
    labeled = pl.DataFrame(
        [_setup(story=1, ts=11, y=1.0, beyond=200.0, extreme=200.0, outcome_ts=13)]
    )
    captured = compute_forward_capture(labeled, blended)
    assert float(captured["mfe_beyond_ticks"][0]) == pytest.approx(0.0)
    assert float(captured["mae_beyond_ticks"][0]) == pytest.approx(130.0)
    assert float(captured["realized_beyond_ticks"][0]) == pytest.approx(-110.0)


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
    labeled = pl.DataFrame([_setup(story=1, ts=1, y=1.0, beyond=80.0, extreme=80.0)])
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_causal_entry(labeled, raw, config=_cfg())


def test_ticks_to_points() -> None:
    assert ticks_to_nq_points(80.0) == 20.0


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    blended = pl.concat(
        [
            _story_bars(story=1, start=_month_ns(2025, 5), beyond=[80.0, 90.0, 100.0]),
            _story_bars(story=2, start=_month_ns(2025, 9), beyond=[80.0, 400.0, 500.0]),
        ]
    )
    labeled = pl.DataFrame(
        [
            _setup(
                story=1,
                ts=_month_ns(2025, 5),
                y=1.0,
                beyond=80.0,
                extreme=80.0,
                outcome_ts=_month_ns(2025, 5) + 2,
            ),
            _setup(
                story=2,
                ts=_month_ns(2025, 9),
                y=1.0,
                beyond=80.0,
                extreme=80.0,
                outcome_ts=_month_ns(2025, 9) + 2,
            ),
        ]
    )
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame(
        {
            AVAILABILITY_TS: labeled[SETUP_AVAILABILITY_TS].to_list(),
            "p_y_path_further_beyond": [0.81, 0.91],
        }
    ).write_parquet(period / "oof_predictions.parquet")
    (period / "summary.json").write_text(
        json.dumps({"diagnostics": {"science": {"holdout_cut_ts": _month_ns(2025, 8)}}}),
        encoding="utf-8",
    )
    report = run_causal_entry_from_period_dir(period, config=_cfg())
    assert report.diagnostics["holdout_scored"] is False
    assert int(report.diagnostics["n_late_confirmed"]) == 1
    mfe = report.late_entries.get_column("mfe_beyond_ticks").max()
    assert isinstance(mfe, (int, float))
    assert float(mfe) < 50.0
    out = tmp_path / "causal"
    write_causal_entry_report(report, out)
    assert (out / "CAUSAL.md").is_file()
    text = (out / "CAUSAL.md").read_text(encoding="utf-8")
    assert "completed-wave peak" in text.lower() or "completed_wave_peak_not_used" in text
    assert "Holdout" in render_causal_entry_markdown(report) or "holdout" in text.lower()
