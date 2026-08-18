"""مقارنة يومين بنفس تعريفات CVD. ليست إشارة."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

import nq.research
from nq.research.cvd_day_compare import (
    compare_day_metrics,
    describe_tape_coverage,
    in_rth,
    scan_cvd_day,
    write_cvd_day_compare_report,
    write_day_scan_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_DAY = "2025-06-03"


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(2025, 6, 3, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _tape(mnq):
    return mnq.drop("order_id")


def test_compare_helpers_not_exported() -> None:
    assert "scan_cvd_day" not in nq.research.__all__
    assert "compare_day_metrics" not in nq.research.__all__
    assert not hasattr(nq.research, "scan_cvd_day")
    assert not hasattr(nq.research, "describe_tape_coverage")


def test_sunday_open_is_short_session_without_rth() -> None:
    t0 = _ns("18:01:00")
    t1 = _ns("18:20:00")
    mnq = make_stream(
        [("T", "B", _PX, 4, 1), ("T", "A", _PX, 2, 2)],
        event_ts=[t0, t1],
        sequence=[1, 2],
    )
    nq = make_stream(
        [("T", "A", _PX, 3, 0), ("T", "B", _PX, 1, 0)],
        event_ts=[t0, t1],
        sequence=[10, 11],
    ).drop("order_id")
    coverage = describe_tape_coverage(mnq, _tape(mnq), nq, label="open")
    assert coverage["has_rth"] is False
    assert coverage["coverage_class"] == "short_session"
    assert coverage["not_matched_rth"] is True
    assert coverage["t_hours"] < 1.0
    assert in_rth(str(coverage["mnq_tmin"])) is False


def test_coverage_window_ignores_rth_prints_outside() -> None:
    rth = _ns("11:01:00")
    eve = _ns("18:01:00")
    eve2 = _ns("18:20:00")
    mnq = make_stream(
        [("T", "B", _PX, 4, 1), ("T", "B", _PX, 2, 2), ("T", "B", _PX, 2, 3)],
        event_ts=[rth, eve, eve2],
        sequence=[1, 2, 3],
    )
    nq = make_stream(
        [("T", "B", _PX, 3, 0), ("T", "B", _PX, 1, 0), ("T", "B", _PX, 1, 0)],
        event_ts=[rth, eve, eve2],
        sequence=[10, 11, 12],
    ).drop("order_id")
    full = describe_tape_coverage(mnq, _tape(mnq), nq, label="full")
    night = describe_tape_coverage(
        mnq,
        _tape(mnq),
        nq,
        label="night",
        start_ts=_ns("18:00:00"),
        end_ts=_ns("19:00:00"),
    )
    assert full["has_rth"] is True
    assert night["has_rth"] is False
    assert night["coverage_class"] == "short_session"
    assert night["n_mnq_t"] == 2


def test_rth_clock_bounds() -> None:
    assert in_rth("2025-06-03T09:30:00-04:00") is True
    assert in_rth("2025-06-03T15:59:00-04:00") is True
    assert in_rth("2025-06-03T16:00:00-04:00") is False
    assert in_rth("2025-06-03T18:00:00-04:00") is False
    assert in_rth(None) is False


def test_compare_days_counts_and_report(tmp_path: Path) -> None:
    opp = _ns("11:01:00")
    align = _ns("11:06:00")
    a_mnq = make_stream(
        [
            ("T", "A", _PX, 500, 1),
            ("T", "B", _PX, 40, 2),
        ],
        event_ts=[opp, align],
        sequence=[1, 2],
    )
    a_nq = make_stream(
        [
            ("T", "B", _PX, 90, 0),
            ("T", "B", _PX, 20, 0),
        ],
        event_ts=[opp, align],
        sequence=[10, 11],
    ).drop("order_id")
    b_mnq = make_stream(
        [
            ("T", "B", _PX, 10, 1),
            ("T", "B", _PX, 10, 2),
        ],
        event_ts=[opp, align],
        sequence=[1, 2],
    )
    b_nq = make_stream(
        [
            ("T", "B", _PX, 8, 0),
            ("T", "B", _PX, 8, 0),
        ],
        event_ts=[opp, align],
        sequence=[10, 11],
    ).drop("order_id")
    a = scan_cvd_day(a_mnq, _tape(a_mnq), a_nq, label="day_a")
    b = scan_cvd_day(b_mnq, _tape(b_mnq), b_nq, label="day_b")
    assert a.summary["has_rth"] is True
    assert a.summary["not_pattern"] is True
    table = compare_day_metrics(a.summary, b.summary, a_label="day_a", b_label="day_b")
    opp_row = table.filter(pl.col("metric") == "n_delta_opposite").row(0, named=True)
    assert int(opp_row["day_a"]) >= 1
    assert int(opp_row["day_b"]) == 0
    written_a = write_day_scan_report(a, tmp_path / "a")
    assert (written_a / "CVD_DAY.md").exists()
    written = write_cvd_day_compare_report(
        table,
        a_summary=a.summary,
        b_summary=b.summary,
        output_dir=tmp_path / "cmp",
        a_label="day_a",
        b_label="day_b",
    )
    text = (written / "CVD_DAY_COMPARE.md").read_text(encoding="utf-8")
    assert "Not a lock" in text
    assert "n_delta_opposite" in text
    day_text = (written_a / "CVD_DAY.md").read_text(encoding="utf-8")
    assert "RTH MNQ→NQ" in day_text
    assert "11:00" in day_text
    assert "move15" in day_text
    assert "median_next15m_all_three" in a.summary
