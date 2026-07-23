"""اختبارات مُحاكي السيولة."""

from __future__ import annotations

from nq.simulation.liquidity import detect_icebergs, liquidity_summary
from tests.mbo_factory import make_stream


def test_liquidity_summary_add_pull_net() -> None:
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("A", "B", 100, 3, 2),
            ("C", "B", 100, 4, 1),  # cancel removes 4
            ("A", "B", 100, 2, 3),  # bucket 10
        ],
        event_ts=[0, 1, 2, 11],
        sequence=[1, 2, 3, 4],
    )
    summary = liquidity_summary(frame, interval_ns=10).sort("bucket_start")
    assert summary["added_volume"].to_list() == [8, 2]
    assert summary["pulled_volume"].to_list() == [4, 0]
    assert summary["net_liquidity"].to_list() == [4, 2]
    assert summary["availability_ts"].to_list() == [10, 20]


def test_detect_iceberg_flagged() -> None:
    # يُظهر 2 فقط لكن يُنفّذ 6 مع إعادة تعبئة مرتين -> آيسبرغ.
    frame = make_stream(
        [
            ("A", "B", 100, 2, 1),
            ("T", "A", 100, 2, 0),
            ("A", "B", 100, 2, 2),
            ("T", "A", 100, 2, 0),
            ("A", "B", 100, 2, 3),
            ("T", "A", 100, 2, 0),
        ],
        event_ts=[0, 1, 2, 3, 4, 5],
        sequence=[1, 2, 3, 4, 5, 6],
    )
    icebergs = detect_icebergs(frame, min_refills=2, min_hidden_ratio=2.0)
    row = icebergs.filter(icebergs["price"] == 100)
    assert row["peak_display"].to_list() == [2]
    assert row["executed"].to_list() == [6]
    assert row["replenish_count"].to_list() == [2]
    assert row["is_iceberg"].to_list() == [True]


def test_no_iceberg_for_normal_level() -> None:
    frame = make_stream(
        [
            ("A", "B", 100, 10, 1),
            ("T", "A", 100, 2, 0),
        ],
        event_ts=[0, 1],
        sequence=[1, 2],
    )
    icebergs = detect_icebergs(frame)
    assert icebergs.filter(icebergs["price"] == 100)["is_iceberg"].to_list() == [False]


def test_detect_icebergs_empty() -> None:
    icebergs = detect_icebergs(make_stream([]))
    assert icebergs.height == 0
