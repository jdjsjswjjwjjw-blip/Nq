"""شريحة ساعة نيويورك: من 11:00 لا من بعد 30ث، وبعدها 5 دقائق."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import nq.research
from nq.research.clock_flow import (
    AFTER_S,
    clock_to_ns,
    clock_windows,
    compare_clock_range,
    write_clock_report,
)
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import SECOND_NS
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_PX_LOW = 19_800_000_000
_NS = SECOND_NS
_DAY = "2025-06-03"


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(2025, 6, 3, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "clock_to_ns" not in nq.research.__all__
    assert not hasattr(nq.research, "clock_to_ns")


def test_ny_1130_windows_start_at_eleven_and_cover_after() -> None:
    origin, wins = clock_windows(_DAY, "11:00:00", "11:30:00", bin_s=60)
    assert origin == _ns("11:00:00")
    assert AFTER_S == 300
    names = [w.name for w in wins]
    assert names[0] == "range"
    assert wins[0].start_ts == _ns("11:00:00")
    assert wins[0].end_ts == _ns("11:30:00")
    assert names[1].startswith("+0-")
    assert wins[1].start_ts == _ns("11:00:00")
    assert names[-1] == "after-0-300s"
    assert wins[-1].start_ts == _ns("11:30:00")
    assert wins[-1].end_ts == _ns("11:30:00") + 300 * _NS


def test_refuses_concatenated_multi_day() -> None:
    day_a = dt.datetime(2025, 6, 3, 11, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 11, 0, tzinfo=_ET)
    mnq = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _PX, 1, 2)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    tape = make_stream(
        [("T", "B", _PX, 1, 0)],
        event_ts=[int(day_a.timestamp() * 1_000_000_000)],
        sequence=[1],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mnq)
    with pytest.raises(ValueError, match="multi-day"):
        compare_clock_range(mnq, tape, tape, day=_DAY, start_clock="11:00:00", end_clock="11:30:00")


def test_range_vs_after_are_separate_and_nq_fill_is_nan() -> None:
    t_open = _ns("11:10:00")
    t_after = _ns("11:31:00")
    mnq = make_stream(
        [
            ("T", "B", _PX, 4, 1),
            ("C", "A", _PX, 8, 2),
            ("F", "A", _PX, 1, 2),
            ("T", "A", _PX_LOW, 2, 3),
        ],
        event_ts=[t_open, t_open + 1, t_open + 2, t_after],
        sequence=[1, 2, 3, 4],
    )
    tape = make_stream(
        [("T", "B", _PX, 4, 0), ("T", "A", _PX_LOW, 2, 0)],
        event_ts=[t_open, t_after],
        sequence=[1, 4],
    ).drop("order_id")
    nq = make_stream(
        [("T", "B", _PX, 5, 0), ("T", "A", _PX, 1, 0)],
        event_ts=[t_open, t_after],
        sequence=[10, 11],
    ).drop("order_id")
    table, diag = compare_clock_range(
        mnq, tape, nq, day=_DAY, start_clock="11:00:00", end_clock="11:30:00"
    )
    by = {r["name"]: r for r in table.iter_rows(named=True)}
    assert diag["tz"] == "America/New_York"
    assert by["range"]["nq_t_imbalance"] > 0
    assert by["range"]["nq_fill_ratio"] != by["range"]["nq_fill_ratio"]  # NaN
    assert by["after-0-300s"]["nq_t_imbalance"] < 0
    assert by["after-0-300s"]["min_px"] == _PX_LOW
    assert by["range"]["min_px"] == _PX
    sources = diag["sources"]
    assert diag["not_pattern"] is True
    assert len(sources) == 6
    nq_range = next(s for s in sources if s["name"] == "range" and s["source"] == "nq_trades")
    mnq_mbo = next(s for s in sources if s["name"] == "range" and s["source"] == "mnq_mbo")
    assert nq_range["ask_hit_share"] != nq_range["ask_hit_share"]
    assert mnq_mbo["f_ask_size"] == 1
    assert mnq_mbo["c_ask_size"] == 8


def test_clock_report(tmp_path: Path) -> None:
    t_open = _ns("11:05:00")
    mnq = make_stream(
        [("T", "B", _PX, 1, 1)],
        event_ts=[t_open],
        sequence=[1],
    )
    tape = make_stream(
        [("T", "B", _PX, 1, 0)],
        event_ts=[t_open],
        sequence=[1],
    ).drop("order_id")
    nq = make_stream(
        [("T", "B", _PX, 1, 0)],
        event_ts=[t_open],
        sequence=[9],
    ).drop("order_id")
    table, diag = compare_clock_range(
        mnq, tape, nq, day=_DAY, start_clock="11:00:00", end_clock="11:30:00"
    )
    written = write_clock_report(table, diag, tmp_path)
    text = (written / "CLOCK_FLOW.md").read_text(encoding="utf-8")
    assert "11:00:00–11:30:00" in text
    assert "America/New_York" in text
    assert "mnq_mbo" in text
    assert "nq_trades" in text
    assert "No pattern lock" in text
    assert (written / "clock_sources.parquet").exists()


def test_london_clock_is_twenty_nine_minutes() -> None:
    origin, wins = clock_windows(
        "2026-08-17",
        "03:12:00",
        "03:41:00",
        bin_s=60,
        tz_name="Europe/London",
    )
    start = clock_to_ns("2026-08-17", "03:12:00", "Europe/London")
    end = clock_to_ns("2026-08-17", "03:41:00", "Europe/London")
    assert origin == start
    assert wins[0].end_ts == end
    assert end - start == 29 * 60 * _NS


def test_price_lo_adds_low_and_level_windows() -> None:
    t_low = _ns("11:05:00")
    t_level = _ns("11:20:00")
    t_after = _ns("11:31:00")
    mnq = make_stream(
        [
            ("C", "A", _PX_LOW, 5, 2),
            ("F", "A", _PX_LOW, 1, 2),
            ("T", "A", _PX_LOW, 3, 1),
            ("T", "B", _PX, 4, 3),
            ("T", "A", _PX, 1, 4),
        ],
        event_ts=[t_low - 2, t_low - 1, t_low, t_level, t_after],
        sequence=[1, 2, 3, 4, 5],
    )
    tape = make_stream(
        [("T", "A", _PX_LOW, 3, 0), ("T", "B", _PX, 4, 0), ("T", "A", _PX, 1, 0)],
        event_ts=[t_low, t_level, t_after],
        sequence=[1, 4, 5],
    ).drop("order_id")
    nq = make_stream(
        [("T", "A", _PX_LOW, 2, 0), ("T", "B", _PX, 6, 0), ("T", "A", _PX, 1, 0)],
        event_ts=[t_low, t_level, t_after],
        sequence=[10, 11, 12],
    ).drop("order_id")
    table, diag = compare_clock_range(
        mnq,
        tape,
        nq,
        day=_DAY,
        start_clock="11:00:00",
        end_clock="11:30:00",
        price_lo=_PX,
    )
    names = {r["name"] for r in table.iter_rows(named=True)}
    assert diag["tz"] == "America/New_York"
    assert diag["low_ts"] == t_low
    assert diag["low_px"] == _PX_LOW
    assert diag["level_ts"] == t_level
    assert "low" in names
    assert "level" in names
    assert "low+0-300s" in names
    assert "level+0-30s" in names
    sources = diag["sources"]
    assert len(sources) == 36
    level_nq = next(s for s in sources if s["name"] == "level" and s["source"] == "nq_trades")
    assert level_nq["ask_hit_share"] != level_nq["ask_hit_share"]
    low_mbo = next(s for s in sources if s["name"] == "low" and s["source"] == "mnq_mbo")
    assert low_mbo["c_ask_size"] == 5
