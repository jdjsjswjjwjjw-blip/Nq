"""اختبارات محول Databento داخل ingestion."""

from __future__ import annotations

import polars as pl

from nq.contracts.mbo import MBO_SCHEMA
from nq.ingestion.databento import is_databento_frame, normalize_databento_frame
from nq.ingestion.reader import load_mbo_frame


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
