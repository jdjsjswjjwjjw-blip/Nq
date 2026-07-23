"""اختبارات مُحاكي تدفّق الأوامر."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from nq.orderbook import reconstruct
from nq.simulation.order_flow import ofi_by_bucket, order_flow_imbalance, order_flow_summary
from tests.mbo_factory import make_stream

_TRADES = make_stream(
    [
        ("T", "B", 100, 5, 0),
        ("T", "A", 100, 2, 0),
        ("T", "B", 101, 3, 0),
        ("T", "A", 100, 4, 0),
    ],
    event_ts=[0, 1, 2, 11],
    sequence=[1, 2, 3, 4],
)


def test_order_flow_summary() -> None:
    summary = order_flow_summary(_TRADES, interval_ns=10).sort("bucket_start")
    assert summary["buy_volume"].to_list() == [8, 0]
    assert summary["sell_volume"].to_list() == [2, 4]
    assert summary["delta"].to_list() == [6, -4]
    assert summary["cumulative_delta"].to_list() == [6, 2]
    assert summary["buy_trades"].to_list() == [2, 0]
    assert summary["sell_trades"].to_list() == [1, 1]
    assert summary["consumption"].to_list() == [10, 4]


def test_order_flow_cumulative_delta_resets_at_cme_session_boundary() -> None:
    et = ZoneInfo("America/New_York")
    first = int(dt.datetime(2024, 7, 15, 16, 59, 30, tzinfo=et).timestamp() * 1e9)
    second = int(dt.datetime(2024, 7, 15, 18, 1, 30, tzinfo=et).timestamp() * 1e9)
    trades = make_stream(
        [("T", "B", 100, 5, 0), ("T", "B", 100, 7, 0)],
        event_ts=[first, second],
        sequence=[1, 2],
    )
    summary = order_flow_summary(trades, interval_ns=60_000_000_000).sort("bucket_start")
    assert summary["cumulative_delta"].to_list() == [5, 7]


def test_order_flow_rejects_contract_roll_without_explicit_lifecycle_config() -> None:
    trades = make_stream(
        [("T", "B", 100, 5, 0), ("T", "B", 101, 7, 0)],
        event_ts=[0, 1],
        sequence=[1, 2],
        symbol="NQU4",
    ).with_columns(pl.Series("symbol", ["NQU4", "NQZ4"], dtype=pl.Utf8()))

    with pytest.raises(ValueError, match="contract roll"):
        order_flow_summary(trades, interval_ns=10)


def _tob() -> pl.DataFrame:
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("A", "A", 102, 4, 2),
            ("A", "B", 101, 3, 3),  # bid improves 100->101, adds size
        ]
    )
    return reconstruct(frame).top_of_book


def test_ofi_first_event_zero_and_increases_on_bid_improvement() -> None:
    ofi = order_flow_imbalance(_tob())
    values = ofi["ofi"].to_list()
    assert values[0] == 0  # no previous event
    # bid price improved (100->101) on last event -> positive bid contribution
    assert values[-1] > 0
    assert ofi["ofi_cumulative"].to_list()[-1] == sum(values)


def test_ofi_by_bucket_causal_availability() -> None:
    bucketed = ofi_by_bucket(_tob(), interval_ns=10)
    assert bucketed["availability_ts"].to_list() == bucketed["bucket_end"].to_list()
