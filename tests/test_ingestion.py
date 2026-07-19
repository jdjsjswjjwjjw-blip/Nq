"""اختبارات قارئ MBO التدفّقي."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nq.ingestion import iter_mbo_batches, load_mbo_frame
from tests.mbo_factory import make_stream


def _stream() -> pl.DataFrame:
    return make_stream(
        [
            ("A", "B", 20_000_000_000, 3, 1),
            ("A", "A", 20_001_000_000, 2, 2),
            ("C", "N", 0, 0, 1),
        ]
    )


def test_load_from_dataframe_sorts_causal() -> None:
    unsorted = _stream().sort("event_ts", descending=True)
    loaded = load_mbo_frame(unsorted)
    assert loaded["event_ts"].to_list() == [0, 1, 2]


def test_load_validates_contract() -> None:
    bad = _stream().drop("price")
    with pytest.raises(ValueError, match="missing columns"):
        load_mbo_frame(bad)


def test_roundtrip_parquet(tmp_path: Path) -> None:
    path = tmp_path / "mbo.parquet"
    _stream().write_parquet(path)
    loaded = load_mbo_frame(path)
    assert loaded.height == 3


def test_roundtrip_arrow(tmp_path: Path) -> None:
    path = tmp_path / "mbo.arrow"
    _stream().write_ipc(path)
    loaded = load_mbo_frame(path)
    assert loaded.height == 3


def test_unsupported_format_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mbo.txt"
    path.write_text("nope")
    with pytest.raises(ValueError, match="unsupported MBO file format"):
        load_mbo_frame(path)


def test_iter_batches_preserves_global_order() -> None:
    frame = make_stream([("A", "B", 100, 1, i) for i in range(1, 11)])
    batches = list(iter_mbo_batches(frame, batch_size=4))
    assert [b.height for b in batches] == [4, 4, 2]
    stitched = pl.concat(batches)
    assert stitched["event_ts"].to_list() == list(range(10))


def test_invalid_batch_size_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be"):
        list(iter_mbo_batches(_stream(), batch_size=0))
