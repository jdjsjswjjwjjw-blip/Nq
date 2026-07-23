"""اختبارات عقد بيانات MBO."""

from __future__ import annotations

import polars as pl
import pytest

from nq.contracts import MBO_SCHEMA, MboAction, MboEvent, MboSide, validate_mbo_frame


def _valid_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_ts": [1_000, 1_001, 1_002],
            "ingest_ts": [1_005, 1_006, 1_007],
            "sequence": [1, 2, 3],
            "instrument_id": [42, 42, 42],
            "symbol": ["NQ", "NQ", "MNQ"],
            "action": ["A", "T", "C"],
            "side": ["B", "N", "A"],
            "price": [20_000_000_000, 20_000_250_000, 20_000_500_000],
            "size": [3, 1, 2],
            "order_id": [10, 0, 11],
            "flags": [0, 0, 0],
        },
        schema=MBO_SCHEMA,
    )


def test_valid_frame_passes() -> None:
    frame = _valid_frame()
    assert validate_mbo_frame(frame) is frame


def test_missing_column_rejected() -> None:
    frame = _valid_frame().drop("price")
    with pytest.raises(ValueError, match="missing columns"):
        validate_mbo_frame(frame)


def test_unexpected_column_rejected() -> None:
    frame = _valid_frame().with_columns(pl.lit(1).alias("rogue"))
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_mbo_frame(frame)


def test_dtype_mismatch_rejected() -> None:
    frame = _valid_frame().with_columns(pl.col("price").cast(pl.Float64))
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_mbo_frame(frame)


def test_point_in_time_violation_rejected() -> None:
    frame = _valid_frame().with_columns(
        pl.Series("ingest_ts", [1_005, 999, 1_007])  # 999 < event_ts 1_001
    )
    with pytest.raises(ValueError, match="ingest_ts < event_ts"):
        validate_mbo_frame(frame)


def test_empty_frame_with_schema_passes() -> None:
    empty = pl.DataFrame(schema=MBO_SCHEMA)
    assert validate_mbo_frame(empty).height == 0


def test_mbo_event_point_in_time_guard() -> None:
    with pytest.raises(ValueError, match="point-in-time"):
        MboEvent(
            event_ts=100,
            ingest_ts=50,
            sequence=1,
            instrument_id=1,
            symbol="NQ",
            action=MboAction.ADD,
            side=MboSide.BID,
            price=1,
            size=1,
            order_id=1,
        )


def test_mbo_event_negative_size_rejected() -> None:
    with pytest.raises(ValueError, match="size must be non-negative"):
        MboEvent(
            event_ts=1,
            ingest_ts=2,
            sequence=1,
            instrument_id=1,
            symbol="NQ",
            action=MboAction.ADD,
            side=MboSide.BID,
            price=1,
            size=-1,
            order_id=1,
        )


def test_enum_values() -> None:
    assert MboAction.TRADE.value == "T"
    assert MboSide.BID.value == "B"
