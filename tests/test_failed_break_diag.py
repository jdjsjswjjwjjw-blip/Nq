"""تشخيص كسر فاشل: Fill_Ratio عند t، ووقف بديل، بلا holdout."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.research.failed_break_diag import (
    scan_tick_diagnostics,
    scan_year_idrive_diag,
    write_failed_break_diag_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_NS = 1_000_000_000
_DAY = dt.date(2025, 6, 3)
_BASE = 100 * 1_000_000_000


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(_DAY.year, _DAY.month, _DAY.day, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _range_tape(*, after: list[tuple[str, int]]) -> tuple[list[str], list[int], list[str]]:
    origin = dt.datetime(_DAY.year, _DAY.month, _DAY.day, 4, 0, tzinfo=_ET)
    clocks: list[str] = []
    prices: list[int] = []
    sides: list[str] = []
    for i in range(6):
        stamp = origin + dt.timedelta(minutes=30 * i)
        clocks.append(stamp.strftime("%H:%M:%S"))
        prices.append(_BASE)
        sides.append("B")
        clocks.append((stamp + dt.timedelta(minutes=1)).strftime("%H:%M:%S"))
        prices.append(_BASE + 1_000_000_000)
        sides.append("B")
    clocks.extend(["07:04:56", "07:04:58", "07:05:00"])
    prices.extend([_BASE, _BASE + 1_000_000_000, int(101.5 * 1_000_000_000)])
    sides.extend(["A", "A", "A"])
    for clock, px in after:
        clocks.append(clock)
        prices.append(px)
        sides.append("A")
    return clocks, prices, sides


def _t_stream(clocks: list[str], prices: list[int], sides: list[str]):
    events = [("T", sides[i], prices[i], 2, i + 1) for i in range(len(clocks))]
    return make_stream(
        events,
        event_ts=[_ns(c) for c in clocks],
        sequence=list(range(1, len(clocks) + 1)),
    )


def test_not_exported() -> None:
    assert not hasattr(nq.research, "scan_tick_diagnostics")
    assert not hasattr(nq.research, "scan_year_idrive_diag")
    assert "scan_tick_diagnostics" not in nq.research.__all__


def test_low_ask_fill_on_failed_upside_break() -> None:
    clocks, prices, sides = _range_tape(after=[("07:05:02", _BASE)])
    trades = _t_stream(clocks, prices, sides)
    mbo = make_stream(
        [("C", "A", _BASE + 1_000_000_000, 40, 900), ("F", "A", _BASE + 1_000_000_000, 4, 901)],
        event_ts=[_ns("07:04:56"), _ns("07:04:57")],
        sequence=[900, 901],
    )
    breaks, _, diag = scan_tick_diagnostics(
        trades, mbo, lookback=3, atr_window=3, path_ticks=3, min_votes=9, skip_open=False
    )
    assert breaks.height >= 1
    row = breaks.row(0, named=True)
    assert row["side"] == "short"
    assert row["fill_ratio_5"] == pytest.approx(4.0 / 44.0)
    assert row["hyp_lt_020_5"] is True
    assert row["outcome"] == "failed"
    assert diag["not_fb_lock"] is True


def test_high_ask_fill_on_held_upside_break() -> None:
    clocks, prices, sides = _range_tape(after=[("07:05:02", int(110.0 * 1_000_000_000))])
    trades = _t_stream(clocks, prices, sides)
    mbo = make_stream(
        [("C", "A", _BASE + 1_000_000_000, 4, 900), ("F", "A", _BASE + 1_000_000_000, 40, 901)],
        event_ts=[_ns("07:04:56"), _ns("07:04:57")],
        sequence=[900, 901],
    )
    breaks, _, _ = scan_tick_diagnostics(
        trades, mbo, lookback=3, atr_window=3, path_ticks=3, min_votes=9, skip_open=False
    )
    row = breaks.row(0, named=True)
    assert row["fill_ratio_5"] == pytest.approx(40.0 / 44.0)
    assert row["hyp_lt_020_5"] is False
    assert row["outcome"] == "held"


def test_stop_rules_are_predeclared(tmp_path: Path) -> None:
    clocks, prices, sides = _range_tape(after=[("07:05:02", int(101.25 * 1_000_000_000))])
    trades = _t_stream(clocks, prices, sides)
    breaks, stops, diag = scan_tick_diagnostics(
        trades, lookback=3, atr_window=3, path_ticks=3, min_votes=1, skip_open=False
    )
    rules = set(stops["rule"].to_list())
    assert "range_plus_atr" in rules
    assert "print_plus_2" in rules
    assert "time_only" in rules
    plus2 = stops.filter(pl.col("rule") == "print_plus_2")
    row = plus2.row(0, named=True)
    assert row["sl"] == pytest.approx(row["entry"] + 2.0)
    written = write_failed_break_diag_report(breaks, stops, diag, tmp_path)
    text = (written / "FAILED_BREAK_DIAG.md").read_text(encoding="utf-8")
    assert "not a lock" in text
    assert "NQ tape is not in IDrive" in text


def test_stop_behind_strongest_ask_wall() -> None:
    clocks, prices, sides = _range_tape(after=[("07:05:02", int(101.25 * 1_000_000_000))])
    trades = _t_stream(clocks, prices, sides)
    mbo = make_stream(
        [
            ("A", "A", int(102.0 * 1_000_000_000), 20, 10),
            ("A", "A", int(103.0 * 1_000_000_000), 200, 11),
        ],
        event_ts=[_ns("04:00:00"), _ns("04:00:01")],
        sequence=[10, 11],
    )
    _, stops, _ = scan_tick_diagnostics(
        trades, mbo, lookback=3, atr_window=3, path_ticks=3, min_votes=1, skip_open=False
    )
    wall = stops.filter(pl.col("rule") == "behind_strongest_wall")
    assert wall.height == 1
    row = wall.row(0, named=True)
    assert row["wall_px"] == pytest.approx(103.0)
    assert row["wall_sz"] == 200
    assert row["sl"] == pytest.approx(103.25)
    assert row["sl"] > row["entry"]


def test_idrive_diag_skips_holdout(tmp_path: Path) -> None:
    clocks, prices, sides = _range_tape(after=[("07:05:02", _BASE)])
    tape = _t_stream(clocks, prices, sides)
    june = tmp_path / "MES_MBO_2025_06"
    sept = tmp_path / "MES_MBO_2025_09"
    june.mkdir()
    sept.mkdir()
    tape.write_parquet(june / "glbx-mdp3-20250603.continuous.clean.parquet")
    tape.write_parquet(sept / "glbx-mdp3-20250902.continuous.clean.parquet")
    breaks, _, diag = scan_year_idrive_diag(
        tmp_path, lookback=3, atr_window=3, path_ticks=3, min_votes=1, skip_open=False
    )
    assert diag["n_skipped_holdout"] == 1
    assert diag["n_days"] == 1
    assert diag["nq_tape"] == "unavailable_idrive_mnq_only"
    assert breaks["day_id"].to_list() == ["2025-06-03"] * breaks.height
