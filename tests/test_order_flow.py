"""اختبارات مُحاكي تدفّق الأوامر."""

from __future__ import annotations

import polars as pl

from nq.orderbook import reconstruct
from nq.simulation.order_flow import (
    ofi_by_bucket,
    order_acceleration_columns,
    order_flow_imbalance,
    order_flow_summary,
)
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


def test_order_acceleration_rate_and_early_imbalance() -> None:
    """تسارع استهلاك عدواني داخل التوازن = اختلال مبكر موقّع بالدلتا."""
    frame = pl.DataFrame(
        {
            "consumption": [10.0, 10.0, 10.0, 40.0, 10.0],
            "delta": [1.0, -1.0, 1.0, 8.0, -2.0],
            "is_balanced": [True, True, True, True, False],
        }
    )
    out = order_acceleration_columns(frame, lookback=3, accel_mult=1.5)
    # البرميل 3: استهلاك 40 / متوسط(10,10,10)=10 → rate=4 ≥ 1.5، متوازن → early=+1
    assert out["order_accel_rate"].to_list()[3] == 4.0
    assert out["early_imbalance"].to_list()[3] == 1.0
    # بلا تسارع لا اختلال مبكر
    assert out["early_imbalance"].to_list()[0] == 0.0
    assert out["early_imbalance"].to_list()[1] == 0.0


def test_order_acceleration_onset_when_flip_to_imbalance() -> None:
    """أول برميل قلب للتوازن→اختلال مع تسارع يُشعل early_imbalance."""
    frame = pl.DataFrame(
        {
            "consumption": [5.0, 5.0, 5.0, 20.0],
            "delta": [0.0, 1.0, -1.0, -9.0],
            "is_balanced": [True, True, True, False],
        }
    )
    out = order_acceleration_columns(frame, lookback=3, accel_mult=1.5)
    assert out["early_imbalance"].to_list()[3] == -1.0


def test_order_acceleration_no_lookahead_past_only() -> None:
    """الأساس يستخدم نوافذ سابقة فقط — تسارع لاحق لا يلوّث معدّل سابق."""
    frame = pl.DataFrame(
        {
            "buy_volume": [2.0, 2.0, 2.0, 30.0],
            "sell_volume": [0.0, 0.0, 0.0, 0.0],
            "delta": [2.0, 2.0, 2.0, 30.0],
            "is_balanced": [True, True, True, True],
        }
    )
    out = order_acceleration_columns(frame, lookback=2, accel_mult=1.5)
    # البرميل 2: استهلاك 2 / متوسط(2,2)=2 → rate=1 < 1.5
    assert abs(out["order_accel_rate"].to_list()[2]) <= 1.0 + 1e-9
    assert out["early_imbalance"].to_list()[2] == 0.0
    assert out["early_imbalance"].to_list()[3] == 1.0
