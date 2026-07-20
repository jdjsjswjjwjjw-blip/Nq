"""اختبارات محول Databento داخل ingestion."""

from __future__ import annotations

import polars as pl

from nq.contracts.mbo import MBO_SCHEMA, PRICE_SCALE
from nq.ingestion.databento import is_databento_frame, normalize_databento_frame
from nq.ingestion.reader import load_mbo_frame, sanitize_mbo_frame


def test_is_databento_frame_detects_vendor_columns() -> None:
    frame = pl.DataFrame({"ts_event": [1], "action": ["A"], "side": ["B"]})
    assert is_databento_frame(frame) is True


def test_normalize_databento_to_mbo_schema() -> None:
    frame = pl.DataFrame(
        {
            "ts_event": [100, 200],
            "ts_recv": [101, 201],
            "sequence": [1, 2],
            "instrument_id": [1, 1],
            "symbol": ["NQ", "NQ"],
            "action": ["A", "T"],
            "side": ["B", "A"],
            "price": [20_000_000_000, 20_000_000_000],
            "size": [5, 3],
            "order_id": [10, 0],
        }
    )
    normalized = normalize_databento_frame(frame)
    assert set(normalized.columns) == set(MBO_SCHEMA.keys())
    assert normalized["action"].to_list() == ["A", "T"]


def test_normalize_databento_rtype_and_flags_no_duplicate() -> None:
    frame = pl.DataFrame(
        {
            "ts_event": [100],
            "ts_recv": [101],
            "sequence": [1],
            "instrument_id": [1],
            "symbol": ["NQ"],
            "action": ["A"],
            "side": ["B"],
            "price": [20_000_000_000],
            "size": [5],
            "order_id": [10],
            "rtype": [9],
            "flags": [3],
        }
    )
    normalized = normalize_databento_frame(frame)
    assert normalized["flags"].to_list() == [3]


def test_normalize_databento_float_price_to_fixed_point() -> None:
    frame = pl.DataFrame(
        {
            "ts_event": [100],
            "ts_recv": [101],
            "sequence": [1],
            "instrument_id": [1],
            "symbol": ["NQ"],
            "action": ["T"],
            "side": ["B"],
            "price": [19100.25],
            "size": [1],
            "order_id": [0],
        }
    )
    normalized = normalize_databento_frame(frame)
    expected = round(19100.25 / PRICE_SCALE)
    assert normalized["price"].to_list() == [expected]


def test_load_mbo_frame_auto_normalizes_databento() -> None:
    frame = pl.DataFrame(
        {
            "ts_event": [100, 200],
            "ts_recv": [100, 200],
            "sequence": [1, 2],
            "instrument_id": [1, 1],
            "symbol": ["NQ", "NQ"],
            "action": ["A", "A"],
            "side": ["B", "A"],
            "price": [20_000_000_000, 20_001_000_000],
            "size": [5, 3],
            "order_id": [1, 2],
        }
    )
    loaded = load_mbo_frame(frame)
    assert "event_ts" in loaded.columns


def test_sanitize_mbo_frame_drops_null_price_trades() -> None:
    frame = pl.DataFrame(
        {
            "event_ts": [1, 2],
            "ingest_ts": [1, 2],
            "sequence": [1, 2],
            "instrument_id": [1, 1],
            "symbol": ["NQ", "NQ"],
            "action": ["T", "R"],
            "side": ["B", "N"],
            "price": [100, None],
            "size": [1, 0],
            "order_id": [1, 0],
            "flags": [0, 0],
        }
    ).cast({"action": pl.Utf8, "side": pl.Utf8})
    cleaned = sanitize_mbo_frame(frame)
    assert cleaned.height == 2
