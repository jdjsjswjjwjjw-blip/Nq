"""اختبارات أدوات الحتمية والترتيب الزمني السببي."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.contracts import MBO_SCHEMA
from nq.core import (
    assert_sorted_causal,
    is_sorted_causal,
    make_generator,
    seed_everything,
    sort_causal,
)


def test_seed_everything_is_reproducible() -> None:
    g1 = seed_everything(123)
    a = g1.standard_normal(5)
    g2 = seed_everything(123)
    b = g2.standard_normal(5)
    np.testing.assert_array_equal(a, b)


def test_make_generator_reproducible_and_isolated() -> None:
    a = make_generator(7).integers(0, 1000, size=10)
    b = make_generator(7).integers(0, 1000, size=10)
    np.testing.assert_array_equal(a, b)


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
    with pytest.raises(ValueError, match="non-negative"):
        make_generator(-1)


def _frame(event_ts: list[int], sequence: list[int]) -> pl.DataFrame:
    rows = len(event_ts)
    return pl.DataFrame(
        {
            "event_ts": event_ts,
            "ingest_ts": [t + 1 for t in event_ts],
            "sequence": sequence,
            "instrument_id": [1] * rows,
            "symbol": ["NQ"] * rows,
            "action": ["A"] * rows,
            "side": ["B"] * rows,
            "price": [1] * rows,
            "size": [1] * rows,
            "order_id": list(range(rows)),
            "flags": [0] * rows,
        },
        schema=MBO_SCHEMA,
    )


def test_causal_sort_orders_by_event_then_sequence() -> None:
    frame = _frame([2, 1, 1], [1, 2, 1])
    ordered = sort_causal(frame)
    assert ordered["event_ts"].to_list() == [1, 1, 2]
    assert ordered["sequence"].to_list() == [1, 2, 1]
    assert is_sorted_causal(ordered)


def test_is_sorted_causal_detects_disorder() -> None:
    assert not is_sorted_causal(_frame([2, 1], [1, 1]))
    assert is_sorted_causal(_frame([1, 2], [1, 1]))


def test_assert_sorted_causal_raises() -> None:
    with pytest.raises(ValueError, match="causal-order violation"):
        assert_sorted_causal(_frame([3, 1, 2], [1, 1, 1]))


def test_assert_sorted_causal_returns_frame_when_ok() -> None:
    frame = _frame([1, 2, 3], [1, 1, 1])
    assert assert_sorted_causal(frame) is frame


def test_single_row_frame_is_causally_sorted() -> None:
    assert is_sorted_causal(_frame([5], [1]))
