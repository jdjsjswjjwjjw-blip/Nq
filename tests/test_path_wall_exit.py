"""وقف مسار خلف جدار الإبطال: وصف، بلا holdout، بلا تصدير."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.path_wall_exit import (
    scan_path_wall_day,
    scan_year_path_wall,
    write_path_wall_exit_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_NS = 1_000_000_000
_DAY = dt.date(2025, 6, 3)
_BAR = 30 * _NS


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(_DAY.year, _DAY.month, _DAY.day, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _blended(
    *,
    n: int = 8,
    onset_i: int = 2,
    close: float = 101.25,
    vah: float = 100.0,
    val: float = 90.0,
    beyond: float = 4.0,
    direction: float = 1.0,
    after_low: float = 100.5,
    after_high: float = 102.0,
    nano: bool = False,
) -> pl.DataFrame:
    start = _ns("04:00:00")
    scale = (1.0 / PRICE_SCALE) if nano else 1.0
    rows: list[dict[str, object]] = []
    for i in range(n):
        px = close if i >= onset_i else (vah - 1.0 if direction > 0 else val + 1.0)
        lo = after_low if i > onset_i else px - 0.25
        hi = after_high if i > onset_i else px + 0.25
        rows.append(
            {
                AVAILABILITY_TS: start + i * _BAR,
                "open": px * scale,
                "high": hi * scale,
                "low": lo * scale,
                "close": px * scale,
                "asia_vah": vah * scale,
                "asia_val": val * scale,
                "path_beyond_asia_ticks": beyond if i >= onset_i else 0.0,
                "vp_fsm_break": 0.0,
                "vp_fsm_retest": 0.0,
                "proj_break_direction": direction if i >= onset_i else 0.0,
            }
        )
    return pl.DataFrame(rows)


def test_not_exported() -> None:
    assert not hasattr(nq.research, "scan_path_wall_day")
    assert not hasattr(nq.research, "scan_year_path_wall")
    assert "scan_path_wall_day" not in nq.research.__all__


def test_long_stop_behind_strongest_bid_wall() -> None:
    blended = _blended(direction=1.0, after_low=100.5, after_high=102.0)
    mbo = make_stream(
        [
            ("A", "B", int(100.0 / PRICE_SCALE), 200, 10),
            ("A", "B", int(99.0 / PRICE_SCALE), 20, 11),
        ],
        event_ts=[_ns("03:59:00"), _ns("03:59:01")],
        sequence=[10, 11],
    )
    table, diag = scan_path_wall_day(blended, mbo, horizon_bars=5, day_id="2025-06-03")
    assert table.height >= 1
    row = table.row(0, named=True)
    assert row["side"] == "long"
    assert row["entry"] == pytest.approx(101.25)
    assert row["wall_px"] == pytest.approx(100.0)
    assert row["wall_sz"] == 200
    assert row["sl"] == pytest.approx(99.75)
    assert row["sl"] < row["entry"]
    assert row["hit_sl"] is False
    assert row["mae_pts"] == pytest.approx(0.75)
    assert diag["exits_remain_manual"] is True
    assert diag["not_overlay"] is True
    assert diag["not_fb_lock"] is True


def test_short_stop_behind_strongest_ask_wall() -> None:
    blended = _blended(
        close=88.75,
        vah=100.0,
        val=90.0,
        direction=-1.0,
        after_low=87.0,
        after_high=89.5,
    )
    mbo = make_stream(
        [
            ("A", "A", int(90.0 / PRICE_SCALE), 20, 10),
            ("A", "A", int(91.0 / PRICE_SCALE), 180, 11),
        ],
        event_ts=[_ns("03:59:00"), _ns("03:59:01")],
        sequence=[10, 11],
    )
    table, _ = scan_path_wall_day(blended, mbo, horizon_bars=5)
    row = table.row(0, named=True)
    assert row["side"] == "short"
    assert row["wall_px"] == pytest.approx(91.0)
    assert row["wall_sz"] == 180
    assert row["sl"] == pytest.approx(91.25)
    assert row["sl"] > row["entry"]


def test_hit_sl_when_adverse_reaches_wall() -> None:
    blended = _blended(direction=1.0, after_low=99.5, after_high=101.5)
    mbo = make_stream(
        [("A", "B", int(100.0 / PRICE_SCALE), 50, 10)],
        event_ts=[_ns("03:59:00")],
        sequence=[10],
    )
    table, _ = scan_path_wall_day(blended, mbo, horizon_bars=5)
    row = table.row(0, named=True)
    assert row["hit_sl"] is True
    assert row["exit_reason"] == "sl"
    assert row["mae_pts"] >= row["risk_pts"] - 1e-9


def test_nano_close_scaled_to_points() -> None:
    blended = _blended(nano=True, direction=1.0, after_low=100.5, after_high=102.0)
    mbo = make_stream(
        [("A", "B", int(100.0 / PRICE_SCALE), 80, 10)],
        event_ts=[_ns("03:59:00")],
        sequence=[10],
    )
    table, _ = scan_path_wall_day(blended, mbo, horizon_bars=5)
    row = table.row(0, named=True)
    assert row["entry"] == pytest.approx(101.25)
    assert row["entry"] < 1_000_000
    assert row["wall_px"] == pytest.approx(100.0)


def test_year_skips_holdout(tmp_path: Path) -> None:
    mbo = make_stream(
        [("A", "B", int(100.0 / PRICE_SCALE), 80, 10)],
        event_ts=[_ns("03:59:00")],
        sequence=[10],
    )
    for name in ("2025-08-29", "2025-09-02"):
        day = tmp_path / "year" / name
        day.mkdir(parents=True)
        _blended().write_parquet(day / "blended.parquet")
        month = tmp_path / "idrive" / f"MES_MBO_{name[:4]}_{name[5:7]}"
        month.mkdir(parents=True, exist_ok=True)
        stamp = name.replace("-", "")
        mbo.write_parquet(month / f"glbx-mdp3-{stamp}.continuous.clean.parquet")
    table, diag = scan_year_path_wall(tmp_path / "year", tmp_path / "idrive")
    assert diag["n_skipped_holdout"] == 1
    assert diag["n_days"] == 1
    assert diag["nq_tape"] == "unavailable_idrive_mnq_only"
    assert table["day_id"].to_list() == ["2025-08-29"] * table.height
    written = write_path_wall_exit_report(table, diag, tmp_path / "out")
    text = (written / "PATH_WALL_EXIT.md").read_text(encoding="utf-8")
    assert "not a lock" in text
    assert "Exits stay manual" in text
    assert "2025-09-01" in text
    assert "NQ tape is not in IDrive" in text
