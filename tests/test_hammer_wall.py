"""مطرقة حجم الصفقة × جدار MBP-10. ليست إشارة."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

import nq.research
from nq.research.hammer_wall import (
    HYP_THICK_L1,
    HYP_THIN_L1,
    prepare_mbp10,
    scan_hammer_wall,
    write_hammer_wall_report,
)
from nq.research.opposite_phantom import SECOND_NS
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_NS = SECOND_NS


def _ns(clock: str) -> int:
    hour, minute, second = (int(p) for p in clock.split(":"))
    stamp = dt.datetime(2025, 6, 3, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _tape(mnq: pl.DataFrame) -> pl.DataFrame:
    return mnq.drop("order_id")


def _mbp(*, clock: str, ask: int, bid: int, seq: int) -> pl.DataFrame:
    ts = _ns(clock)
    return pl.DataFrame(
        {
            "event_ts": [ts],
            "ingest_ts": [ts],
            "sequence": [seq],
            "ask_sz_00": [ask],
            "bid_sz_00": [bid],
            "ask_sz_01": [1],
            "ask_sz_02": [1],
            "bid_sz_01": [1],
            "bid_sz_02": [1],
        }
    )


def test_not_exported() -> None:
    assert "scan_hammer_wall" not in nq.research.__all__
    assert not hasattr(nq.research, "scan_hammer_wall")
    assert not hasattr(nq.research, "prepare_mbp10")


def test_asof_uses_snapshot_before_bin_not_during() -> None:
    before = _mbp(clock="10:59:50", ask=40, bid=8, seq=1)
    during = _mbp(clock="11:00:20", ask=900, bid=8, seq=2)
    mbp = pl.concat([before, during])
    trades = [("T", "B", _PX, 5, i + 1) for i in range(4)]
    stamps = [_ns("11:00:10") + i * 50_000_000 for i in range(4)]
    later = [_ns("11:06:00") + i * _NS for i in range(2)]
    mnq = make_stream(
        [*trades, ("T", "B", _PX + 1_000_000_000, 1, 9), ("T", "B", _PX + 2_000_000_000, 1, 10)],
        event_ts=[*stamps, *later],
        sequence=list(range(1, 7)),
    )
    nq = make_stream(
        [("T", "B", _PX, 1, 0), ("T", "B", _PX, 1, 0)],
        event_ts=[stamps[0], later[0]],
        sequence=[100, 101],
    ).drop("order_id")
    table, diag = scan_hammer_wall(
        mnq,
        _tape(mnq),
        nq,
        mbp,
        bin_s=60,
    )
    assert diag["not_mbo_book"] is True
    assert diag["not_hammer_wall_lock"] is True
    row = table.filter(pl.col("clock").str.contains("11:00")).row(0, named=True)
    assert row["ask_l1"] == 40
    assert row["ask_l1"] != 900
    assert row["avg_trade_size"] > 3
    assert row["hyp_pass"] is True


def test_thick_ask_does_not_pass_and_fail_recipe_needs_both(tmp_path: Path) -> None:
    mbp = _mbp(clock="10:59:50", ask=HYP_THICK_L1 + 20, bid=8, seq=1)
    mnq = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _PX, 1, 2)],
        event_ts=[_ns("11:00:10"), _ns("11:00:20")],
        sequence=[1, 2],
    )
    nq = _tape(mnq)
    table, diag = scan_hammer_wall(mnq, _tape(mnq), nq, mbp, bin_s=60)
    row = table.row(0, named=True)
    assert row["hyp_thin_ask"] is False
    assert row["hyp_thick_ask"] is True
    assert row["hyp_pass"] is False
    assert row["hyp_fail_recipe"] is True
    assert row["ask_l1"] > HYP_THIN_L1
    written = write_hammer_wall_report(table, diag, tmp_path)
    text = (written / "HAMMER_WALL.md").read_text(encoding="utf-8")
    assert "No MBO book rebuild" in text
    assert "not a lock" in text


def test_prepare_mbp10_sums_l3() -> None:
    frame = _mbp(clock="11:00:00", ask=5, bid=6, seq=1)
    out = prepare_mbp10(frame)
    row = out.row(0, named=True)
    assert row["ask_l1"] == 5
    assert row["ask_l3"] == 7
    assert row["bid_l3"] == 8
