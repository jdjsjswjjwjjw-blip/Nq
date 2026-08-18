"""فشل كسر مبكر: الدخول عند الطبعة، لا عند range_high، ولا holdout."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.failed_break_flow import (
    HOLDOUT_START_DATE,
    iter_idrive_session_files,
    scan_blended_early_fail,
    scan_tick_early_fail,
    scan_year_blended,
    scan_year_idrive_tick,
    write_failed_break_flow_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_NS = 1_000_000_000
_DAY = dt.date(2025, 6, 3)


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(_DAY.year, _DAY.month, _DAY.day, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _prints(clocks: list[str], prices: list[int], sides: list[str] | None = None) -> pl.DataFrame:
    n = len(clocks)
    if sides is not None:
        side = sides
    else:
        side = ["B" if prices[i] > prices[max(0, i - 1)] else "A" for i in range(n)]
    events = [("T", side[i], prices[i], 2, i + 1) for i in range(n)]
    return make_stream(
        events,
        event_ts=[_ns(c) for c in clocks],
        sequence=list(range(1, n + 1)),
    ).drop("order_id")


def test_not_exported() -> None:
    assert "scan_tick_early_fail" not in nq.research.__all__
    assert not hasattr(nq.research, "scan_tick_early_fail")
    assert not hasattr(nq.research, "scan_year_blended")
    assert not hasattr(nq.research, "scan_year_idrive_tick")


def test_tick_entry_is_print_not_range_high() -> None:
    origin = dt.datetime(_DAY.year, _DAY.month, _DAY.day, 4, 0, tzinfo=_ET)
    clocks: list[str] = []
    prices: list[int] = []
    sides: list[str] = []
    base = 100 * 1_000_000_000
    for i in range(6):
        stamp = origin + dt.timedelta(minutes=30 * i)
        clocks.append(stamp.strftime("%H:%M:%S"))
        prices.append(base)
        sides.append("B")
        clocks.append((stamp + dt.timedelta(minutes=1)).strftime("%H:%M:%S"))
        prices.append(base + 1_000_000_000)
        sides.append("B")
    clocks.extend(["07:04:56", "07:04:58", "07:05:00", "07:05:01"])
    prices.extend(
        [
            base,
            base + 1_000_000_000,
            int(101.5 * 1_000_000_000),
            int(101.25 * 1_000_000_000),
        ]
    )
    sides.extend(["A", "A", "A", "A"])
    tape = _prints(clocks, prices, sides=sides)
    table, diag = scan_tick_early_fail(
        tape, lookback=3, atr_window=3, path_ticks=3, min_votes=1, skip_open=False
    )
    assert diag["not_fb_lock"] is True
    assert diag["not_path_head_on_ticks"] is True
    assert table.height >= 1
    row = table.row(0, named=True)
    assert row["entry"] == pytest.approx(101.5)
    assert row["entry"] != pytest.approx(row["range_high"])
    assert row["range_high"] == pytest.approx(101.0)
    assert row["leak_pts"] == pytest.approx(0.5)


def test_blended_entry_is_close_not_high() -> None:
    start = _ns("04:00:00")
    rows: list[dict[str, float | int]] = []
    px = 100.0
    for i in range(250):
        ts = start + i * 30 * _NS
        h = px + 1.0
        low = px - 1.0
        close = px
        if i == 200:
            h = 120.0
            close = 99.5
        rows.append(
            {
                AVAILABILITY_TS: ts,
                "high": h,
                "low": low,
                "close": close,
                "vp_of_delta": -5.0,
                "lf_liquidity_withdrawal": 1.0,
                "path_change_fail": 1.0,
                "p_y_path_further_beyond": 0.2,
            }
        )
        px = close
    blended = pl.DataFrame(rows)
    table, diag = scan_blended_early_fail(blended, lookback=3, atr_window=3, min_votes=2)
    assert diag["fill_rule"] == "bar_close_at_signal"
    assert table.height >= 1
    row = table.row(0, named=True)
    assert row["entry"] == pytest.approx(99.5)
    assert row["entry"] != pytest.approx(row["range_high"])
    assert row["range_high"] == pytest.approx(101.0)
    assert row["p_path"] == pytest.approx(0.2)


def test_blended_nano_prices_scale_to_points() -> None:
    start = _ns("04:00:00")
    rows: list[dict[str, float | int]] = []
    px = 21_000_000_000_000.0
    for i in range(250):
        ts = start + i * 30 * _NS
        h = px + 1_000_000_000.0
        close = px
        if i == 200:
            h = px + 20_000_000_000.0
            close = px - 500_000_000.0
        rows.append(
            {
                AVAILABILITY_TS: ts,
                "high": h,
                "low": px - 1_000_000_000.0,
                "close": close,
                "vp_of_delta": -5.0,
                "lf_liquidity_withdrawal": 1.0,
                "path_change_fail": 1.0,
            }
        )
    blended = pl.DataFrame(rows)
    table, _ = scan_blended_early_fail(blended, lookback=3, atr_window=3, min_votes=2)
    assert table.height >= 1
    row = table.row(0, named=True)
    assert row["entry"] == pytest.approx(20999.5)
    assert row["entry"] < 1_000_000


def test_year_skips_holdout(tmp_path: Path) -> None:
    assert HOLDOUT_START_DATE == "2025-09-01"
    for name, close in (("2025-08-29", 100.0), ("2025-09-02", 100.0)):
        day = tmp_path / name
        day.mkdir()
        start = _ns("04:00:00")
        rows = []
        px = close
        for i in range(250):
            ts = start + i * 30 * _NS
            rows.append(
                {
                    AVAILABILITY_TS: ts,
                    "high": px + (20.0 if i == 200 else 1.0),
                    "low": px - 1.0,
                    "close": px,
                    "vp_of_delta": -2.0,
                    "lf_liquidity_withdrawal": 1.0,
                    "path_change_fail": 1.0,
                }
            )
        pl.DataFrame(rows).write_parquet(day / "blended.parquet")
    table, diag = scan_year_blended(tmp_path, min_votes=2)
    assert diag["n_skipped_holdout"] == 1
    assert diag["n_days"] == 1
    written = write_failed_break_flow_report(table, diag, tmp_path / "out")
    text = (written / "FAILED_BREAK_FLOW.md").read_text(encoding="utf-8")
    assert "Not a lock" in text
    assert "not a live tick overlay" in text
    assert "2025-09-01" in text


def test_tick_refuses_concat_days() -> None:
    day_a = dt.datetime(2025, 6, 3, 11, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 11, 0, tzinfo=_ET)
    trades = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _PX, 1, 2)],
        event_ts=[int(day_a.timestamp() * _NS), int(day_b.timestamp() * _NS)],
        sequence=[1, 2],
    ).drop("order_id")
    with pytest.raises(ValueError, match="concatenated"):
        scan_tick_early_fail(trades)


def _range_break_tape() -> pl.DataFrame:
    origin = dt.datetime(_DAY.year, _DAY.month, _DAY.day, 4, 0, tzinfo=_ET)
    clocks: list[str] = []
    prices: list[int] = []
    sides: list[str] = []
    base = 100 * 1_000_000_000
    for i in range(6):
        stamp = origin + dt.timedelta(minutes=30 * i)
        clocks.append(stamp.strftime("%H:%M:%S"))
        prices.append(base)
        sides.append("B")
        clocks.append((stamp + dt.timedelta(minutes=1)).strftime("%H:%M:%S"))
        prices.append(base + 1_000_000_000)
        sides.append("B")
    clocks.extend(["07:04:56", "07:04:58", "07:05:00", "07:05:01"])
    prices.extend(
        [
            base,
            base + 1_000_000_000,
            int(101.5 * 1_000_000_000),
            int(101.25 * 1_000_000_000),
        ]
    )
    sides.extend(["A", "A", "A", "A"])
    return _prints(clocks, prices, sides=sides)


def test_idrive_year_skips_holdout_and_scans_one_file_per_day(tmp_path: Path) -> None:
    tape = _range_break_tape()
    june = tmp_path / "MES_MBO_2025_06"
    sept = tmp_path / "MES_MBO_2025_09"
    june.mkdir()
    sept.mkdir()
    tape.write_parquet(june / "glbx-mdp3-20250603.continuous.clean.parquet")
    tape.write_parquet(sept / "glbx-mdp3-20250902.continuous.clean.parquet")
    found = iter_idrive_session_files(tmp_path)
    assert [day for day, _ in found] == ["2025-06-03", "2025-09-02"]
    table, diag = scan_year_idrive_tick(
        tmp_path,
        lookback=3,
        atr_window=3,
        path_ticks=3,
        min_votes=1,
        skip_open=False,
    )
    assert diag["source"] == "idrive_tick"
    assert diag["n_skipped_holdout"] == 1
    assert diag["n_days"] == 1
    assert diag["n_skipped_error"] == 0
    assert table.height >= 1
    assert table["day_id"].to_list() == ["2025-06-03"] * table.height
    row = table.row(0, named=True)
    assert row["entry"] == pytest.approx(101.5)
    assert row["entry"] != pytest.approx(row["range_high"])


def test_idrive_year_counts_concat_error_without_abort(tmp_path: Path) -> None:
    day_a = dt.datetime(2025, 6, 3, 11, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 11, 0, tzinfo=_ET)
    bad = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _PX, 1, 2)],
        event_ts=[int(day_a.timestamp() * _NS), int(day_b.timestamp() * _NS)],
        sequence=[1, 2],
    )
    month = tmp_path / "MES_MBO_2025_06"
    month.mkdir()
    bad.write_parquet(month / "glbx-mdp3-20250603.continuous.clean.parquet")
    _range_break_tape().write_parquet(month / "glbx-mdp3-20250604.continuous.clean.parquet")
    _, diag = scan_year_idrive_tick(
        tmp_path,
        lookback=3,
        atr_window=3,
        path_ticks=3,
        min_votes=1,
        skip_open=False,
    )
    assert diag["n_skipped_error"] == 1
    assert diag["n_days"] == 1
    assert any("concatenated" in err for err in diag["errors"])
