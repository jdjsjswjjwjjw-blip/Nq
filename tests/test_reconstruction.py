"""اختبارات إعادة بناء دفتر الأوامر."""

from __future__ import annotations

import polars as pl
import pytest

from nq.orderbook import reconstruct, reconstruct_by_instrument
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


def test_strict_reconstruction_rejects_crossed_book() -> None:
    frame = make_stream(
        [
            ("A", "A", 100, 2, 1),
            ("A", "B", 101, 2, 2),
        ]
    )
    with pytest.raises(ValueError, match="crossed_book_events"):
        reconstruct(frame, strict=True)


def test_strict_reconstruction_rejects_sequence_skips() -> None:
    frame = make_stream(
        [("A", "B", 100, 1, 1), ("A", "A", 101, 1, 2)],
        event_ts=[0, 1],
        sequence=[1, 5],
    )
    with pytest.raises(ValueError, match="sequence_skips"):
        reconstruct(frame, strict=True)


def test_unknown_modify_does_not_create_resting_liquidity() -> None:
    frame = make_stream([("M", "B", 100, 5, 42)])
    result = reconstruct(frame)
    assert result.integrity.unknown_order_refs == 1
    assert result.book.best_bid() is None
    with pytest.raises(ValueError, match="unknown_order_refs"):
        reconstruct(frame, strict=True)


def test_duplicate_add_does_not_duplicate_resting_liquidity() -> None:
    frame = make_stream(
        [
            ("A", "B", 100, 5, 1),
            ("A", "B", 101, 7, 1),
        ]
    )
    result = reconstruct(frame)
    assert result.integrity.unknown_order_refs == 1
    assert result.book.best_bid() == (100, 5)
    with pytest.raises(ValueError, match="unknown_order_refs"):
        reconstruct(frame, strict=True)


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
