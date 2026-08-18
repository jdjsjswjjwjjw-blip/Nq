"""كسر فاشل سببي: الملء عند افتتاح التالية، لا عند range_high."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.failed_breakout import (
    _tape_prices_to_fixed,
    scan_failed_breakout,
    simulate_from_bars,
    write_failed_breakout_report,
)
from nq.research.mbo_trade_overlap import prepare_trades_tape
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.fvg import NS_30M, build_ohlcv_bars
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_NS = 1_000_000_000
_DAY = dt.date(2025, 6, 3)


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(_DAY.year, _DAY.month, _DAY.day, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _bar(
    clock: str,
    *,
    o: float,
    h: float,
    low: float,
    c: float,
    volume: float = 1_000.0,
) -> dict[str, float | int]:
    start = _ns(clock)
    end = start + NS_30M
    return {
        BUCKET_START: start,
        BUCKET_END: end,
        AVAILABILITY_TS: end,
        "o": o,
        "h": h,
        "l": low,
        "c": c,
        "volume": volume,
        "range": h - low,
    }


def _warmup(n: int, start_clock: str = "04:00:00") -> list[dict[str, float | int]]:
    hour, minute, _ = (int(p) for p in start_clock.split(":"))
    origin = dt.datetime(_DAY.year, _DAY.month, _DAY.day, hour, minute, tzinfo=_ET)
    rows: list[dict[str, float | int]] = []
    for i in range(n):
        stamp = origin + dt.timedelta(minutes=30 * i)
        clock = stamp.strftime("%H:%M:%S")
        rows.append(_bar(clock, o=100.0, h=101.0, low=99.0, c=100.0, volume=1_000.0))
    return rows


def _m30(rows: list[dict[str, float | int]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _run(
    rows: list[dict[str, float | int]],
    *,
    lookback: int = 3,
    atr_window: int = 3,
    vol_window: int = 3,
    sma_period: int = 0,
    hold_bars: int = 3,
    reward_ratio: float = 2.0,
    range_mult: float = 0.0,
    vol_mult: float = 0.0,
    skip_open: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    table, diag = simulate_from_bars(
        _m30(rows),
        pl.DataFrame(),
        lookback=lookback,
        atr_window=atr_window,
        vol_window=vol_window,
        sma_period=sma_period,
        hold_bars=hold_bars,
        reward_ratio=reward_ratio,
        range_mult=range_mult,
        vol_mult=vol_mult,
        skip_open=skip_open,
    )
    return table, diag


def test_not_exported() -> None:
    assert "scan_failed_breakout" not in nq.research.__all__
    assert "simulate_from_bars" not in nq.research.__all__
    assert not hasattr(nq.research, "scan_failed_breakout")
    assert not hasattr(nq.research, "simulate_from_bars")


def test_entry_is_next_open_not_range_high() -> None:
    rows = _warmup(7)
    # 07:00: lookback highs are 101. Failed break: high 110, close 100.
    rows[6] = _bar("07:00:00", o=100.0, h=110.0, low=99.0, c=100.0, volume=2_000.0)
    rows.append(_bar("07:30:00", o=100.5, h=101.0, low=99.5, c=100.2, volume=1_000.0))
    rows.append(_bar("08:00:00", o=100.2, h=100.8, low=99.8, c=100.1, volume=1_000.0))
    rows.append(_bar("08:30:00", o=100.1, h=100.4, low=99.9, c=100.0, volume=1_000.0))
    table, diag = _run(rows)
    assert diag["n_fb_pattern"] >= 1
    assert table.height == 1
    row = table.row(0, named=True)
    assert row["side"] == "short"
    assert row["entry"] == pytest.approx(100.5)
    assert row["entry"] != pytest.approx(row["range_high"])
    assert row["range_high"] == pytest.approx(101.0)
    assert row["signal_h"] == pytest.approx(110.0)
    assert row["sl"] > row["signal_h"]
    assert row["sl"] != pytest.approx(row["range_high"])
    assert row["leak_pts"] == pytest.approx(abs(101.0 - 100.5))


def test_stop_beyond_signal_wick_not_range_high() -> None:
    rows = _warmup(7)
    rows[6] = _bar("07:00:00", o=100.0, h=110.0, low=99.0, c=100.0, volume=2_000.0)
    rows.append(_bar("07:30:00", o=100.5, h=101.0, low=99.5, c=100.2, volume=1_000.0))
    rows.append(_bar("08:00:00", o=100.2, h=100.8, low=99.8, c=100.1, volume=1_000.0))
    rows.append(_bar("08:30:00", o=100.1, h=100.4, low=99.9, c=100.0, volume=1_000.0))
    table, _ = _run(rows)
    row = table.row(0, named=True)
    assert row["sl"] == pytest.approx(row["signal_h"] + 2.0)
    assert row["prior_swing"] == pytest.approx(99.0)


def test_same_bar_hits_stop_before_target() -> None:
    rows = _warmup(7)
    rows[6] = _bar("07:00:00", o=100.0, h=110.0, low=99.0, c=100.0, volume=2_000.0)
    # Fill bar trades through both SL and TP. SL-first must win.
    rows.append(_bar("07:30:00", o=100.5, h=200.0, low=0.0, c=100.0, volume=1_000.0))
    table, _ = _run(rows, hold_bars=1)
    row = table.row(0, named=True)
    assert row["exit_reason"] == "sl"
    assert row["exit"] == pytest.approx(row["sl"])
    assert row["exit"] != pytest.approx(row["tp"])
    assert row["pnl_pts"] < 0


def test_gap_through_stop_is_skipped() -> None:
    rows = _warmup(7)
    rows[6] = _bar("07:00:00", o=100.0, h=110.0, low=99.0, c=100.0, volume=2_000.0)
    rows.append(_bar("07:30:00", o=200.0, h=201.0, low=199.0, c=200.0, volume=1_000.0))
    table, diag = _run(rows)
    assert table.height == 0
    assert diag["n_fb_pattern"] >= 1
    assert diag["n_skipped_gap_sl"] >= 1


def test_time_exit_uses_last_close_when_neither_side_hit() -> None:
    rows = _warmup(7)
    rows[6] = _bar("07:00:00", o=100.0, h=110.0, low=99.0, c=100.0, volume=2_000.0)
    rows.append(_bar("07:30:00", o=100.5, h=101.0, low=99.5, c=100.4, volume=1_000.0))
    rows.append(_bar("08:00:00", o=100.4, h=100.8, low=100.0, c=100.2, volume=1_000.0))
    table, _ = _run(rows, hold_bars=2)
    row = table.row(0, named=True)
    assert row["exit_reason"] == "time"
    assert row["exit"] == pytest.approx(100.2)


def test_sma_min_periods_skips_when_not_ready() -> None:
    rows = _warmup(7)
    rows[6] = _bar("07:00:00", o=100.0, h=110.0, low=99.0, c=100.0, volume=2_000.0)
    rows.append(_bar("07:30:00", o=100.5, h=101.0, low=99.5, c=100.2, volume=1_000.0))
    table, diag = _run(rows, sma_period=50)
    assert diag["n_fb_pattern"] >= 1
    assert diag["n_skipped_sma"] >= 1
    assert table.height == 0
    assert diag["sma_filter"] == "sma50_hourly_min_periods"


def test_long_failed_low_fill_is_next_open() -> None:
    rows = _warmup(7)
    rows[6] = _bar("07:00:00", o=100.0, h=101.0, low=90.0, c=100.0, volume=2_000.0)
    rows.append(_bar("07:30:00", o=99.5, h=100.5, low=99.0, c=100.0, volume=1_000.0))
    rows.append(_bar("08:00:00", o=100.0, h=100.4, low=99.6, c=100.1, volume=1_000.0))
    table, _ = _run(rows)
    row = table.row(0, named=True)
    assert row["side"] == "long"
    assert row["entry"] == pytest.approx(99.5)
    assert row["entry"] != pytest.approx(row["range_low"])
    assert row["sl"] < row["signal_l"]


def test_dollar_tape_stays_in_index_points() -> None:
    stamps = [_ns("10:00:01"), _ns("10:00:02"), _ns("10:00:03")]
    raw = pl.DataFrame(
        {
            "event_ts": stamps,
            "ingest_ts": stamps,
            "sequence": [1, 2, 3],
            "action": ["T", "T", "T"],
            "side": ["B", "A", "B"],
            "price": [30215.5, 30216.0, 30215.75],
            "size": [1, 1, 1],
        }
    )
    tape, unit = _tape_prices_to_fixed(prepare_trades_tape(raw))
    assert unit == "dollar"
    bars = build_ohlcv_bars(tape, interval_ns=NS_30M)
    assert bars.height == 1
    assert bars["o"][0] == pytest.approx(30215.5)
    assert bars["h"][0] == pytest.approx(30216.0)


def test_nano_tape_still_scales_like_mbo() -> None:
    stamps = [_ns("10:00:01"), _ns("10:00:02")]
    raw = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "A", _PX + 1_000_000_000, 1, 2)],
        event_ts=stamps,
        sequence=[1, 2],
    ).drop("order_id")
    tape, unit = _tape_prices_to_fixed(prepare_trades_tape(raw))
    assert unit == "fixed_point"
    bars = build_ohlcv_bars(tape, interval_ns=NS_30M)
    assert bars["o"][0] == pytest.approx(20.0)


def test_scan_refuses_concatenated_days() -> None:
    day_a = dt.datetime(2025, 6, 3, 11, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 11, 0, tzinfo=_ET)
    trades = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _PX, 1, 2)],
        event_ts=[
            int(day_a.timestamp() * _NS),
            int(day_b.timestamp() * _NS),
        ],
        sequence=[1, 2],
    ).drop("order_id")
    with pytest.raises(ValueError, match="concatenated"):
        scan_failed_breakout(trades, sma_period=0)


def test_report_states_next_open_not_lock(tmp_path: Path) -> None:
    rows = _warmup(7)
    rows[6] = _bar("07:00:00", o=100.0, h=110.0, low=99.0, c=100.0, volume=2_000.0)
    rows.append(_bar("07:30:00", o=100.5, h=101.0, low=99.5, c=100.2, volume=1_000.0))
    table, diag = _run(rows)
    written = write_failed_breakout_report(table, diag, tmp_path)
    text = (written / "FAILED_BREAKOUT.md").read_text(encoding="utf-8")
    assert "next bar open" in text
    assert "Not a lock" in text
    assert "hanging limit" in text
    assert diag["not_hanging_limit"] is True
    assert diag["not_fb_lock"] is True
    assert diag["not_mbo_book"] is True
