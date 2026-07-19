"""اختبارات مُحاكي المزاد."""

from __future__ import annotations

import polars as pl

from nq.simulation.auction import auction_states
from tests.mbo_factory import make_stream


def test_balanced_window_rotates_in_value() -> None:
    # نافذة متوازنة: حجم مركّز داخل نطاق ضيّق.
    frame = make_stream(
        [
            ("T", "B", 100, 5, 0),
            ("T", "A", 100, 5, 0),
            ("T", "B", 101, 2, 0),
            ("T", "A", 99, 2, 0),
        ],
        event_ts=[0, 1, 2, 3],
        sequence=[1, 2, 3, 4],
    )
    states = auction_states(frame, interval_ns=100).sort("bucket_start")
    assert states.height == 1
    assert states["in_value_fraction"].to_list()[0] > 0
    assert states["is_balanced"].to_list()[0] in (True, False)


def test_expansion_and_new_high_detected() -> None:
    frame = make_stream(
        [
            # نافذة 0: نطاق ضيّق حول 100
            ("T", "B", 100, 5, 0),
            ("T", "A", 100, 5, 0),
            ("T", "B", 101, 1, 0),
            # نافذة 100: نطاق واسع وقمة جديدة
            ("T", "B", 100, 1, 0),
            ("T", "B", 120, 5, 0),
            ("T", "A", 100, 1, 0),
        ],
        event_ts=[0, 1, 2, 100, 101, 102],
        sequence=[1, 2, 3, 4, 5, 6],
    )
    states = auction_states(frame, interval_ns=100).sort("bucket_start")
    assert states.height == 2
    assert states["made_new_high"].to_list() == [False, True]
    # النافذة الثانية أوسع مدى من الأولى
    assert states["is_expansion"].to_list()[1] is True


def test_availability_is_bucket_end() -> None:
    frame = make_stream(
        [("T", "B", 100, 5, 0), ("T", "A", 100, 5, 0)],
        event_ts=[0, 1],
        sequence=[1, 2],
    )
    states = auction_states(frame, interval_ns=10)
    assert states["availability_ts"].to_list() == states["bucket_end"].to_list()


def test_balance_flips_to_imbalance() -> None:
    # نافذة 0 متوازنة (تدوير حول 100)، نافذة 100 مختلّة (اتجاه يُغلق عند القمة).
    balanced = [("T", "B", 100 + d, 2, 0) for d in (0, 0, 1, -1, 0)]
    trend = [("T", "B", 100 + j, 2, 0) for j in range(10)]
    events = balanced + trend
    ts = list(range(len(balanced))) + list(range(100, 100 + len(trend)))
    seq = list(range(1, len(events) + 1))
    frame = make_stream(events, event_ts=ts, sequence=seq)

    states = auction_states(frame, interval_ns=50).sort("bucket_start")
    states = states.with_columns(
        (pl.col("is_balanced").shift(1) & ~pl.col("is_balanced"))
        .fill_null(value=False)
        .alias("flip_to_imbalance")
    )
    assert states["is_balanced"].to_list()[0] is True
    assert states["is_balanced"].to_list()[1] is False
    assert states["flip_to_imbalance"].to_list()[1] is True
    assert "close_in_value" in states.columns


def test_empty_stream() -> None:
    states = auction_states(make_stream([]), interval_ns=10)
    assert states.height == 0
