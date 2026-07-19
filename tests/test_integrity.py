"""اختبارات فحوص سلامة تدفّق MBO."""

from __future__ import annotations

from nq.orderbook import check_integrity
from tests.mbo_factory import make_stream


def test_clean_stream_is_ok() -> None:
    frame = make_stream([("A", "B", 100, 1, 1), ("A", "A", 101, 1, 2)])
    report = check_integrity(frame)
    assert report.ok
    assert report.out_of_order == 0
    assert report.sequence_non_monotonic == 0
    assert report.sequence_skips == 0


def test_out_of_order_detected() -> None:
    frame = make_stream(
        [("A", "B", 100, 1, 1), ("A", "A", 101, 1, 2), ("A", "B", 99, 1, 3)],
        event_ts=[0, 2, 1],
        sequence=[1, 2, 3],
    )
    report = check_integrity(frame)
    assert report.out_of_order == 1
    assert not report.ok


def test_sequence_skip_detected() -> None:
    frame = make_stream(
        [("A", "B", 100, 1, 1), ("A", "A", 101, 1, 2)],
        event_ts=[0, 1],
        sequence=[1, 5],
    )
    report = check_integrity(frame)
    assert report.sequence_skips == 1
    assert report.ok  # skips alone do not invalidate the stream


def test_non_monotonic_sequence_detected() -> None:
    frame = make_stream(
        [("A", "B", 100, 1, 1), ("A", "A", 101, 1, 2)],
        event_ts=[0, 1],
        sequence=[5, 5],
    )
    report = check_integrity(frame)
    assert report.sequence_non_monotonic == 1
    assert not report.ok


def test_empty_frame() -> None:
    empty = make_stream([])
    report = check_integrity(empty)
    assert report.n_events == 0
    assert report.ok
