"""اختبارات قارئ MBO التدفّقي."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nq.contracts.mbo import MBO_SCHEMA
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


@pytest.mark.parametrize("suffix", [".parquet", ".arrow"])
def test_iter_file_batches_are_bounded_and_honor_max_rows(tmp_path: Path, suffix: str) -> None:
    frame = make_stream([("A", "B", 100, 1, i) for i in range(1, 12)])
    path = tmp_path / f"mbo{suffix}"
    if suffix == ".parquet":
        frame.write_parquet(path, row_group_size=3)
    else:
        frame.write_ipc(path)

    batches = list(iter_mbo_batches(path, batch_size=4, max_rows=9))

    assert sum(batch.height for batch in batches) == 9
    assert all(0 < batch.height <= 4 for batch in batches)
    assert pl.concat(batches)["event_ts"].to_list() == list(range(9))


def test_iter_file_batches_reject_cross_batch_time_reversal(tmp_path: Path) -> None:
    first = make_stream([("A", "B", 100, 1, i) for i in range(5, 9)], event_ts=[4, 5, 6, 7])
    second = make_stream([("A", "B", 100, 1, i) for i in range(1, 5)], event_ts=[0, 1, 2, 3])
    path = tmp_path / "reversed.parquet"
    pl.concat([first, second]).write_parquet(path, row_group_size=4)

    stream = iter_mbo_batches(path, batch_size=4)
    assert next(stream)["event_ts"].to_list() == [4, 5, 6, 7]
    with pytest.raises(ValueError, match="causal-order violation across streaming batches"):
        next(stream)


def test_iter_file_batches_does_not_materialize_with_full_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mbo.parquet"
    make_stream([("A", "B", 100, 1, i) for i in range(1, 7)]).write_parquet(path)

    def fail_full_load(*args: object, **kwargs: object) -> pl.DataFrame:
        del args, kwargs
        raise AssertionError("full loader must not be used")

    monkeypatch.setattr("nq.ingestion.reader.load_mbo_frame", fail_full_load)
    assert [batch.height for batch in iter_mbo_batches(path, batch_size=2)] == [2, 2, 2]


def test_bounded_loader_keeps_global_earliest_without_full_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "unordered.parquet"
    frame = make_stream(
        [("A", "B", 100, 1, i) for i in range(1, 13)],
        event_ts=[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3],
    )
    frame.write_parquet(path, row_group_size=4)

    def fail_full_load(*args: object, **kwargs: object) -> pl.DataFrame:
        del args, kwargs
        raise AssertionError("bounded loader must not call the full reader")

    monkeypatch.setattr("nq.ingestion.reader._read_columnar", fail_full_load)
    loaded = load_mbo_frame(path, max_rows=5)
    assert loaded["event_ts"].to_list() == [0, 1, 2, 3, 4]


def test_iter_csv_normalizes_inferred_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "mbo.csv"
    make_stream([("A", "B", 100, 1, i) for i in range(1, 6)]).write_csv(path)
    batches = list(iter_mbo_batches(path, batch_size=2))
    stitched = pl.concat(batches)
    assert stitched.schema == MBO_SCHEMA
    assert stitched.height == 5


def test_streaming_databento_generated_sequence_is_global(tmp_path: Path) -> None:
    path = tmp_path / "databento.parquet"
    frame = pl.DataFrame(
        {
            "ts_event": [100] * 4,
            "ts_recv": [100] * 4,
            "instrument_id": [1] * 4,
            "symbol": ["NQ"] * 4,
            "action": ["A"] * 4,
            "side": ["B"] * 4,
            "price": [20_000_000_000] * 4,
            "size": [1] * 4,
            "order_id": [1, 2, 3, 4],
        }
    )
    frame.write_parquet(path, row_group_size=2)
    stitched = pl.concat(iter_mbo_batches(path, batch_size=2))
    assert stitched["sequence"].to_list() == [0, 1, 2, 3]


def test_invalid_batch_size_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be"):
        list(iter_mbo_batches(_stream(), batch_size=0))
