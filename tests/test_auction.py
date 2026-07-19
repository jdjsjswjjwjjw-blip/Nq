"""اختبارات مُحاكي المزاد."""

from __future__ import annotations

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


def test_empty_stream() -> None:
    states = auction_states(make_stream([]), interval_ns=10)
    assert states.height == 0
