"""قمة مقابل نوافذ ضابطة: أرقام فقط، بلا نموذج."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import (
    HOUR_NS,
    compare_peak_controls,
    default_windows,
    first_high_ts,
    write_control_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_PX_ASK = 20_000_250_000
_HI = 20_001_000_000
_NS = SECOND_NS


def _t0() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "compare_peak_controls" not in nq.research.__all__
    assert not hasattr(nq.research, "compare_peak_controls")
    assert not hasattr(nq.research, "score_window")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mbo = make_stream(
        [("T", "B", _HI, 1, 9), ("T", "B", _HI, 1, 10)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    with pytest.raises(ValueError, match="multi-day"):
        compare_peak_controls(mbo, price_hi=_HI)


def test_default_windows_are_clock_aligned() -> None:
    high = _t0()
    peak, climb, drop = default_windows(high, window_s=30)
    assert peak.name == "peak"
    assert peak.end_ts - peak.start_ts == 30 * _NS
    assert peak.end_ts == high
    assert climb.end_ts == high - HOUR_NS
    assert climb.end_ts - climb.start_ts == 30 * _NS
    assert drop.start_ts == high + 30 * _NS
    assert drop.end_ts - drop.start_ts == 30 * _NS


def test_peak_counts_opposite_unfilled_not_fills_or_same_side() -> None:
    high = _t0()
    peak_t = high - _NS
    climb_t = high - HOUR_NS - _NS
    drop_t = high + 31 * _NS
    mbo = make_stream(
        [
            ("A", "A", _PX_ASK, 8, 1),
            ("C", "A", _PX_ASK, 8, 1),
            ("A", "B", _PX, 8, 2),
            ("C", "B", _PX, 8, 2),
            ("A", "A", _PX_ASK, 8, 3),
            ("F", "A", _PX_ASK, 8, 3),
            ("C", "A", _PX_ASK, 8, 3),
            ("T", "B", _PX_ASK, 2, 90),
            ("A", "A", _PX_ASK, 1, 4),
            ("C", "A", _PX_ASK, 1, 4),
            ("T", "B", _PX, 1, 91),
            ("A", "A", _PX_ASK, 4, 5),
            ("C", "A", _PX_ASK, 4, 5),
            ("T", "A", _PX, 3, 92),
            ("T", "B", _HI, 1, 99),
        ],
        event_ts=[
            peak_t - 5 * _NS,
            peak_t - 4 * _NS,
            peak_t - 5 * _NS,
            peak_t - 4 * _NS,
            peak_t - 3 * _NS,
            peak_t - 2 * _NS,
            peak_t - _NS,
            peak_t,
            climb_t - 2 * _NS,
            climb_t - _NS,
            climb_t,
            drop_t - 2 * _NS,
            drop_t - _NS,
            drop_t,
            high,
        ],
        sequence=list(range(1, 16)),
    )
    table, diag = compare_peak_controls(mbo, price_hi=_HI, large_min=3)
    assert first_high_ts(mbo, _HI) == high
    assert diag["not_backtest"] is True
    by = {row["name"]: row for row in table.iter_rows(named=True)}
    assert by["peak"]["opp_n"] == 1
    assert by["peak"]["opp_size"] == 8
    assert by["peak"]["large_n"] == 1
    assert by["peak"]["t_size"] == 2
    assert by["peak"]["opp_over_t"] == 4.0
    assert by["climb"]["opp_n"] == 1
    assert by["climb"]["large_n"] == 0
    assert by["drop"]["last_t_side"] == "A"
    assert by["drop"]["opp_n"] == 0


def test_events_outside_named_window_are_ignored() -> None:
    high = _t0()
    mbo = make_stream(
        [
            ("A", "A", _PX_ASK, 8, 1),
            ("C", "A", _PX_ASK, 8, 1),
            ("T", "B", _PX, 1, 90),
            ("T", "B", _HI, 1, 99),
        ],
        event_ts=[high - 40 * _NS, high - 39 * _NS, high - _NS, high],
        sequence=[1, 2, 3, 4],
    )
    table, _ = compare_peak_controls(mbo, price_hi=_HI)
    peak = table.filter(pl.col("name") == "peak")
    assert peak["opp_n"][0] == 0


def test_report_is_diagnostic_not_a_model(tmp_path: Path) -> None:
    high = _t0()
    mbo = make_stream(
        [
            ("T", "B", _PX, 1, 1),
            ("T", "B", _HI, 1, 2),
        ],
        event_ts=[high - _NS, high],
        sequence=[1, 2],
    )
    table, diag = compare_peak_controls(mbo, price_hi=_HI)
    written = write_control_report(table, diag, tmp_path)
    text = (written / "PEAK_CONTROL.md").read_text(encoding="utf-8")
    assert "Not spoofing" in text
    assert "not LSTM" in text
    assert (written / "summary.json").is_file()
