"""اختبارات إعادة بناء دفتر الأوامر."""

from __future__ import annotations

import polars as pl
import pytest

from nq.orderbook import reconstruct, reconstruct_by_instrument, scan_book_tob_and_depth
from nq.simulation.depth_lifecycle import depth_at_bar_close_multi
from tests.mbo_factory import make_stream


def test_top_of_book_series() -> None:
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("A", "A", 102, 4, 2),
            ("C", "N", 0, 0, 1),
        ]
    )
    result = reconstruct(frame)
    tob = result.top_of_book
    assert tob.height == 3
    assert tob["best_bid"].to_list() == [100, 100, None]
    assert tob["best_ask"].to_list() == [None, 102, 102]
    assert result.integrity.ok


def test_final_book_state() -> None:
    frame = make_stream([("A", "B", 100, 5, 1), ("A", "A", 102, 4, 2)])
    result = reconstruct(frame)
    assert result.book.best_bid() == (100, 5)
    assert result.book.best_ask() == (102, 4)


def test_crossed_book_detected() -> None:
    frame = make_stream(
        [
            ("A", "A", 100, 2, 1),  # ask at 100
            ("A", "B", 101, 2, 2),  # bid at 101 -> crossed
        ]
    )
    result = reconstruct(frame)
    assert result.integrity.crossed_book_events >= 1


def test_unknown_refs_flow_into_integrity() -> None:
    frame = make_stream([("A", "B", 100, 5, 1), ("C", "N", 0, 0, 42)])
    result = reconstruct(frame)
    assert result.integrity.unknown_order_refs == 1
    assert not result.integrity.ok


def test_reconstruct_without_recording() -> None:
    frame = make_stream([("A", "B", 100, 5, 1)])
    result = reconstruct(frame, record_top_of_book=False)
    assert result.top_of_book.height == 0
    assert result.book.best_bid() == (100, 5)


def test_multi_instrument_rejected() -> None:
    a = make_stream([("A", "B", 100, 1, 1)], instrument_id=1)
    b = make_stream([("A", "B", 100, 1, 1)], instrument_id=2)
    with pytest.raises(ValueError, match="single instrument"):
        reconstruct(pl.concat([a, b]))


def test_reconstruct_by_instrument_splits() -> None:
    nq = make_stream([("A", "B", 100, 1, 1)], instrument_id=1, symbol="NQ")
    mnq = make_stream([("A", "A", 200, 2, 1)], instrument_id=2, symbol="MNQ")
    results = reconstruct_by_instrument(pl.concat([nq, mnq]))
    assert set(results) == {1, 2}
    assert results[1].book.best_bid() == (100, 1)
    assert results[2].book.best_ask() == (200, 2)


def test_empty_frame_reconstruction() -> None:
    result = reconstruct(make_stream([]))
    assert result.top_of_book.height == 0
    assert result.integrity.n_events == 0


def test_unified_scan_matches_separate_tob_and_depth() -> None:
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("A", "A", 103, 4, 2),
            ("A", "B", 99, 3, 3),
            ("M", "A", 102, 6, 2),
            ("F", "B", 100, 2, 1),  # Fill: book unchanged (Databento)
            ("C", "N", 0, 3, 3),  # Cancel full size of order 3
        ],
        event_ts=[0, 4, 11, 14, 25, 31],
    )
    intervals = (10, 20)

    unified, unified_depth = scan_book_tob_and_depth(
        frame,
        interval_ns_list=intervals,
        n_levels=3,
    )
    separate = reconstruct(frame)
    separate_depth = depth_at_bar_close_multi(
        frame,
        interval_ns_list=intervals,
        n_levels=3,
    )

    assert unified.top_of_book.equals(separate.top_of_book)
    assert unified.integrity == separate.integrity
    assert unified.book.orders == separate.book.orders
    for interval in intervals:
        assert unified_depth[interval].equals(separate_depth[interval])
