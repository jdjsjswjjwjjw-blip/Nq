"""اختبارات مُحاكي البصمة السعرية."""

from __future__ import annotations

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.simulation.common import extract_trades
from nq.simulation.footprint import footprint_cells, footprint_summary
from nq.validation import assert_availability_not_before_event
from tests.mbo_factory import make_stream

_TRADES = make_stream(
    [
        ("T", "B", 100, 5, 0),
        ("T", "A", 100, 2, 0),
        ("T", "B", 101, 3, 0),
        ("T", "B", 100, 1, 0),
    ],
    event_ts=[0, 1, 2, 11],
    sequence=[1, 2, 3, 4],
)


def test_extract_trades_signed_volume() -> None:
    trades = extract_trades(_TRADES)
    assert trades["buy_volume"].to_list() == [5, 0, 3, 1]
    assert trades["sell_volume"].to_list() == [0, 2, 0, 0]
    assert trades["signed_volume"].to_list() == [5, -2, 3, 1]


def test_footprint_cells() -> None:
    cells = footprint_cells(_TRADES, interval_ns=10)
    b0 = cells.filter(pl.col("bucket_start") == 0).sort("price")
    assert b0["price"].to_list() == [100, 101]
    assert b0["delta"].to_list() == [3, 3]  # 100: 5-2, 101: 3-0
    assert b0["total_volume"].to_list() == [7, 3]
    assert abs(b0["imbalance"].to_list()[0] - 3 / 7) < 1e-12


def test_footprint_summary_cumulative_delta_and_absorption() -> None:
    summary = footprint_summary(_TRADES, interval_ns=10).sort("bucket_start")
    assert summary["delta"].to_list() == [6, 1]  # bucket0: 8-2, bucket10: 1-0
    assert summary["cumulative_delta"].to_list() == [6, 7]
    assert summary["price_range"].to_list() == [1, 0]
    # bucket0: total 10 over range 1 -> 10/(1+1)=5 ; bucket10: 1/(0+1)=1
    assert summary["absorption_ratio"].to_list() == [5.0, 1.0]


def test_footprint_is_point_in_time() -> None:
    cells = footprint_cells(_TRADES, interval_ns=10)
    # كل خلية متاحة فقط عند إغلاق نافذتها: availability_ts > كل event_ts داخلها.
    trades = extract_trades(_TRADES)
    joined = trades.join(
        cells.select("bucket_start", AVAILABILITY_TS),
        left_on=(pl.col(EVENT_TS) // 10 * 10),
        right_on="bucket_start",
        how="left",
    )
    assert_availability_not_before_event(
        joined[EVENT_TS].to_list(), joined[AVAILABILITY_TS].to_list()
    )
