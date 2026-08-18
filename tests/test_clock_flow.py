"""شريحة ساعة نيويورك: من 11:00 لا من بعد 30ث، وبعدها 5 دقائق."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.research.clock_flow import (
    AFTER_S,
    clock_to_ns,
    clock_windows,
    compare_clock_range,
    scan_cvd_align_expansion,
    scan_cvd_opposite,
    scan_cvd_prealign,
    write_clock_report,
    write_cvd_align_expansion_report,
    write_cvd_opposite_report,
    write_cvd_prealign_report,
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
    assert "scan_cvd_opposite" not in nq.research.__all__
    assert "scan_cvd_align_expansion" not in nq.research.__all__
    assert "scan_cvd_prealign" not in nq.research.__all__
    assert not hasattr(nq.research, "scan_cvd_opposite")
    assert not hasattr(nq.research, "scan_cvd_align_expansion")
    assert not hasattr(nq.research, "scan_cvd_prealign")


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
    assert "CVD" in text
    assert (written / "clock_sources.parquet").exists()


def test_ny_open_five_minute_bins() -> None:
    origin, wins = clock_windows(_DAY, "09:25:00", "09:40:00", bin_s=300)
    assert origin == _ns("09:25:00")
    names = [w.name for w in wins]
    assert names[0] == "range"
    assert names[1] == "+0-300s"
    assert names[2] == "+300-600s"
    assert names[3] == "+600-900s"
    assert wins[1].start_ts == _ns("09:25:00")
    assert wins[2].start_ts == _ns("09:30:00")
    assert wins[3].start_ts == _ns("09:35:00")
    assert wins[3].end_ts == _ns("09:40:00")
    assert names[-1] == "after-0-300s"


def test_cvd_is_cumulative_signed_size_from_session_start() -> None:
    t0 = _ns("10:00:00")
    t_open = _ns("11:10:00")
    t_after = _ns("11:31:00")
    mnq = make_stream(
        [
            ("T", "B", _PX, 10, 1),
            ("T", "A", _PX, 3, 2),
            ("T", "B", _PX, 4, 3),
            ("T", "A", _PX_LOW, 2, 4),
        ],
        event_ts=[t0, t0 + 1, t_open, t_after],
        sequence=[1, 2, 3, 4],
    )
    tape = make_stream(
        [
            ("T", "B", _PX, 10, 0),
            ("T", "A", _PX, 3, 0),
            ("T", "B", _PX, 4, 0),
            ("T", "A", _PX_LOW, 2, 0),
        ],
        event_ts=[t0, t0 + 1, t_open, t_after],
        sequence=[1, 2, 3, 4],
    ).drop("order_id")
    nq = make_stream(
        [
            ("T", "B", _PX, 8, 0),
            ("T", "A", _PX, 1, 0),
            ("T", "B", _PX, 5, 0),
            ("T", "A", _PX, 2, 0),
        ],
        event_ts=[t0, t0 + 1, t_open, t_after],
        sequence=[10, 11, 12, 13],
    ).drop("order_id")
    table, diag = compare_clock_range(
        mnq, tape, nq, day=_DAY, start_clock="11:00:00", end_clock="11:30:00"
    )
    nq_range = next(
        s for s in diag["sources"] if s["name"] == "range" and s["source"] == "nq_trades"
    )
    nq_after = next(
        s for s in diag["sources"] if s["name"] == "after-0-300s" and s["source"] == "nq_trades"
    )
    mnq_range = next(
        s for s in diag["sources"] if s["name"] == "range" and s["source"] == "mnq_mbo"
    )
    assert nq_range["cvd_before"] == 7
    assert nq_range["cvd_end"] == 12
    assert nq_range["cvd_delta"] == 5
    assert nq_range["cvd_notional_end"] == 12 * 20.0
    assert nq_after["cvd_before"] == 12
    assert nq_after["cvd_end"] == 10
    assert nq_after["cvd_delta"] == -2
    assert mnq_range["cvd_before"] == 7
    assert mnq_range["cvd_delta"] == 4
    by = {r["name"]: r for r in table.iter_rows(named=True)}
    assert by["range"]["nq_cvd_end"] == 12
    assert diag["not_cvd_threshold"] is True
    assert diag["cvd_origin"] == "first_T_in_day_file"


def test_stack_bins_includes_five_minute_open_windows() -> None:
    t_pre = _ns("09:27:00")
    t_open = _ns("09:32:00")
    t_late = _ns("09:37:00")
    mnq = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "A", _PX, 1, 2), ("T", "B", _PX, 1, 3)],
        event_ts=[t_pre, t_open, t_late],
        sequence=[1, 2, 3],
    )
    tape = make_stream(
        [("T", "B", _PX, 1, 0), ("T", "A", _PX, 1, 0), ("T", "B", _PX, 1, 0)],
        event_ts=[t_pre, t_open, t_late],
        sequence=[1, 2, 3],
    ).drop("order_id")
    nq = make_stream(
        [("T", "B", _PX, 2, 0), ("T", "A", _PX, 2, 0), ("T", "B", _PX, 3, 0)],
        event_ts=[t_pre, t_open, t_late],
        sequence=[10, 11, 12],
    ).drop("order_id")
    _table, diag = compare_clock_range(
        mnq,
        tape,
        nq,
        day=_DAY,
        start_clock="09:25:00",
        end_clock="09:40:00",
        bin_s=300,
        stack_bins=True,
    )
    names = {s["name"] for s in diag["sources"]}
    assert "+0-300s" in names
    assert "+300-600s" in names
    assert "+600-900s" in names
    assert len(diag["sources"]) == 15
    pre = next(s for s in diag["sources"] if s["name"] == "+0-300s" and s["source"] == "nq_trades")
    opn = next(
        s for s in diag["sources"] if s["name"] == "+300-600s" and s["source"] == "nq_trades"
    )
    assert pre["cvd_end"] == 2
    assert opn["cvd_end"] == 0
    assert opn["cvd_delta"] == -2


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


def test_cvd_opposite_flags_delta_not_zero() -> None:
    t_buy = _ns("11:01:00")
    t_both = _ns("11:06:00")
    mnq = make_stream(
        [("T", "B", _PX, 5, 1), ("T", "B", _PX, 2, 2)],
        event_ts=[t_buy, t_both],
        sequence=[1, 2],
    )
    tape = make_stream(
        [("T", "B", _PX, 5, 0), ("T", "B", _PX, 2, 0)],
        event_ts=[t_buy, t_both],
        sequence=[1, 2],
    ).drop("order_id")
    nq = make_stream(
        [("T", "A", _PX, 3, 0), ("T", "B", _PX, 4, 0)],
        event_ts=[t_buy, t_both],
        sequence=[10, 11],
    ).drop("order_id")
    table, diag = scan_cvd_opposite(mnq, tape, nq, bin_s=300)
    assert diag["not_pattern"] is True
    opp = table.filter(pl.col("delta_opposite"))
    same = table.filter(~pl.col("delta_opposite"))
    assert opp.height == 1
    assert same.height >= 1
    row = opp.row(0, named=True)
    assert row["mnq_cvd_delta"] > 0
    assert row["nq_cvd_delta"] < 0
    aligned = same.filter(pl.col("mnq_cvd_delta") > 0)
    assert aligned.height == 1
    assert aligned["nq_cvd_delta"][0] > 0


def test_cvd_opposite_report(tmp_path: Path) -> None:
    t_buy = _ns("11:01:00")
    mnq = make_stream(
        [("T", "B", _PX, 4, 1)],
        event_ts=[t_buy],
        sequence=[1],
    )
    tape = make_stream(
        [("T", "B", _PX, 4, 0)],
        event_ts=[t_buy],
        sequence=[1],
    ).drop("order_id")
    nq = make_stream(
        [("T", "A", _PX, 1, 0)],
        event_ts=[t_buy],
        sequence=[9],
    ).drop("order_id")
    table, diag = scan_cvd_opposite(mnq, tape, nq, bin_s=300)
    written = write_cvd_opposite_report(table, diag, tmp_path)
    text = (written / "CVD_OPPOSITE.md").read_text(encoding="utf-8")
    assert "ΔCVD opposite" in text
    assert (written / "cvd_delta_opposite.parquet").exists()


def _tape_like(mnq: pl.DataFrame) -> pl.DataFrame:
    return mnq.drop("order_id")


def test_strong_opposite_then_align_flags_wide_range() -> None:
    quiet = _ns("10:51:00")
    opp = _ns("11:01:00")
    align = _ns("11:06:00")
    px = _PX
    px_r = _PX + 1_000_000_000
    px_w = _PX + 4_000_000_000
    mnq = make_stream(
        [
            ("T", "B", px, 2, 1),
            ("T", "B", px_r, 2, 2),
            ("T", "A", px, 500, 3),
            ("T", "A", px_r, 20, 4),
            ("T", "B", px, 30, 5),
            ("T", "B", px_w, 30, 6),
        ],
        event_ts=[quiet, quiet + _NS, opp, opp + _NS, align, align + _NS],
        sequence=[1, 2, 3, 4, 5, 6],
    )
    nq = make_stream(
        [
            ("T", "B", px, 2, 0),
            ("T", "B", px, 2, 0),
            ("T", "B", px, 10, 0),
            ("T", "B", px, 10, 0),
            ("T", "B", px, 8, 0),
            ("T", "B", px, 8, 0),
        ],
        event_ts=[quiet, quiet + _NS, opp, opp + _NS, align, align + _NS],
        sequence=[10, 11, 12, 13, 14, 15],
    ).drop("order_id")
    table, diag = scan_cvd_align_expansion(
        mnq,
        _tape_like(mnq),
        nq,
        bin_s=300,
        strong_mnq=500,
        strong_nq=80,
    )
    assert diag["not_pattern"] is True
    assert table.height == 1
    row = table.row(0, named=True)
    assert row["aligned"] is True
    assert row["mnq_cvd_delta"] < 0
    assert row["nq_cvd_delta"] > 0
    assert row["align_mnq_delta"] > 0
    assert row["align_nq_delta"] > 0
    assert row["wide_vs_median"] is True
    assert row["moved_with_align"] is True
    assert row["moved_with_nq_opp"] is True


def test_strong_opposite_align_without_wide_range() -> None:
    quiet = _ns("10:51:00")
    opp = _ns("11:01:00")
    align = _ns("11:06:00")
    px = _PX
    px_r = _PX + 1_000_000_000
    mnq = make_stream(
        [
            ("T", "B", px, 2, 1),
            ("T", "B", px_r, 2, 2),
            ("T", "A", px, 500, 3),
            ("T", "A", px_r, 20, 4),
            ("T", "B", px, 30, 5),
            ("T", "B", px_r, 30, 6),
        ],
        event_ts=[quiet, quiet + _NS, opp, opp + _NS, align, align + _NS],
        sequence=[1, 2, 3, 4, 5, 6],
    )
    nq = make_stream(
        [
            ("T", "B", px, 2, 0),
            ("T", "B", px, 2, 0),
            ("T", "B", px, 10, 0),
            ("T", "B", px, 10, 0),
            ("T", "B", px, 8, 0),
            ("T", "B", px, 8, 0),
        ],
        event_ts=[quiet, quiet + _NS, opp, opp + _NS, align, align + _NS],
        sequence=[10, 11, 12, 13, 14, 15],
    ).drop("order_id")
    table, diag = scan_cvd_align_expansion(
        mnq,
        _tape_like(mnq),
        nq,
        bin_s=300,
        strong_mnq=500,
        strong_nq=80,
    )
    assert diag["n_aligned"] == 1
    assert table.row(0, named=True)["wide_vs_median"] is False


def test_zero_delta_is_not_alignment(tmp_path: Path) -> None:
    opp = _ns("11:01:00")
    zero = _ns("11:06:00")
    align = _ns("11:11:00")
    mnq = make_stream(
        [
            ("T", "A", _PX, 500, 1),
            ("T", "B", _PX, 1, 2),
            ("T", "A", _PX, 1, 3),
            ("T", "B", _PX, 20, 4),
        ],
        event_ts=[opp, zero, zero + _NS, align],
        sequence=[1, 2, 3, 4],
    )
    nq = make_stream(
        [
            ("T", "B", _PX, 9, 0),
            ("T", "B", _PX, 4, 0),
        ],
        event_ts=[opp, align],
        sequence=[10, 11],
    ).drop("order_id")
    table, diag = scan_cvd_align_expansion(
        mnq,
        _tape_like(mnq),
        nq,
        bin_s=300,
        strong_mnq=500,
        strong_nq=80,
    )
    assert table.height == 1
    row = table.row(0, named=True)
    assert row["aligned"] is True
    assert int(row["time_to_align_s"]) == 300
    written = write_cvd_align_expansion_report(table, diag, tmp_path)
    text = (written / "CVD_ALIGN_EXPANSION.md").read_text(encoding="utf-8")
    assert "Not a pattern lock" in text
    assert (written / "cvd_align_expansion.parquet").exists()


def test_prealign_minutes_cover_flip_and_are_not_a_pattern(tmp_path: Path) -> None:
    opp = _ns("11:01:00")
    confirm = _ns("11:05:10")
    mnq = make_stream(
        [
            ("A", "B", _PX, 9, 99),
            ("T", "A", _PX, 500, 1),
            ("A", "B", _PX, 3, 100),
            ("T", "B", _PX, 40, 2),
        ],
        event_ts=[opp - _NS, opp, confirm - _NS, confirm],
        sequence=[1, 2, 3, 4],
    )
    nq = make_stream(
        [
            ("T", "B", _PX, 10, 0),
            ("T", "B", _PX, 8, 0),
        ],
        event_ts=[opp, confirm],
        sequence=[10, 11],
    ).drop("order_id")
    minutes, summaries, diag = scan_cvd_prealign(
        mnq,
        _tape_like(mnq),
        nq,
        strong_mnq=500,
        strong_nq=80,
    )
    assert diag["not_pattern"] is True
    assert diag["not_book_hidden"] is True
    assert summaries.height == 1
    rel = set(minutes["rel_s"].to_list())
    assert -300 in rel
    assert 0 in rel
    confirm_row = minutes.filter(pl.col("rel_s") == 0).row(0, named=True)
    assert confirm_row["mnq_cvd_delta"] > 0
    pre_sell = minutes.filter(pl.col("rel_s") == -240).row(0, named=True)
    assert pre_sell["mnq_cvd_delta"] < 0
    written = write_cvd_prealign_report(minutes, summaries, diag, tmp_path)
    text = (written / "CVD_PREALIGN.md").read_text(encoding="utf-8")
    assert "not a lock" in text
    assert (written / "cvd_prealign_minutes.parquet").exists()
