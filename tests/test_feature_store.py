"""اختبارات مخزن الميزات point-in-time."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nq.features import FEATURE_STORE_SCHEMA, FeatureStore, wide_to_features
from nq.simulation.footprint import footprint_summary
from tests.mbo_factory import make_stream


def _wide() -> pl.DataFrame:
    # مخرَج مُحاكٍ مُبسّط: نافذتان مع دلتا وحجم، متاح عند bucket_end.
    return pl.DataFrame(
        {
            "bucket_start": [0, 10],
            "availability_ts": [10, 20],
            "delta": [5, -3],
            "total_volume": [10, 8],
        }
    )


def test_wide_to_features_long_schema() -> None:
    feats = wide_to_features(
        _wide(), value_columns=["delta", "total_volume"], version="v1", instrument_id=1
    )
    assert set(feats.columns) == set(FEATURE_STORE_SCHEMA)
    assert feats.height == 4  # 2 windows * 2 features
    assert set(feats["feature"].unique().to_list()) == {"delta", "total_volume"}


def test_ingest_and_len_and_versions() -> None:
    store = FeatureStore()
    store.ingest_wide(_wide(), value_columns=["delta"], version="v1", instrument_id=1)
    store.ingest_wide(_wide(), value_columns=["delta"], version="v2", instrument_id=1)
    assert len(store) == 4
    assert store.versions() == ["v1", "v2"]


def test_point_in_time_violation_rejected() -> None:
    bad = pl.DataFrame(
        {
            "feature": ["x"],
            "instrument_id": [1],
            "value": [1.0],
            "event_ts": [100],
            "availability_ts": [50],  # < event_ts
            "version": ["v1"],
        }
    )
    with pytest.raises(ValueError, match="point-in-time violation"):
        FeatureStore(bad)


def test_as_of_returns_only_available_and_latest() -> None:
    store = FeatureStore().ingest_wide(
        _wide(), value_columns=["delta"], version="v1", instrument_id=1
    )
    # عند t=15 لا تتوفّر إلا نافذة bucket_end=10 (القيمة 5)
    at15 = store.as_of(15)
    assert at15["value"].to_list() == [5.0]
    # عند t=25 تتوفّر النافذتان -> الأحدث (bucket_end=20, القيمة -3)
    at25 = store.as_of(25)
    assert at25["value"].to_list() == [-3.0]
    # عند t=5 لا شيء متاح بعد
    assert store.as_of(5).height == 0


def test_point_in_time_join_no_future_leak() -> None:
    store = FeatureStore().ingest_wide(
        _wide(), value_columns=["delta", "total_volume"], version="v1", instrument_id=1
    )
    query = pl.DataFrame({"t": [5, 12, 25]})
    joined = store.point_in_time_join(query, ts_col="t")
    # t=5: لا ميزات بعد (null)؛ t=12: قيمة النافذة الأولى (5)؛ t=25: الأحدث (-3)
    assert joined["delta"].to_list() == [None, 5.0, -3.0]
    assert joined["total_volume"].to_list() == [None, 10.0, 8.0]


def test_snapshot_series_forward_fill() -> None:
    store = FeatureStore().ingest_wide(
        _wide(), value_columns=["delta"], version="v1", instrument_id=1
    )
    snap = store.snapshot_series()
    assert snap["availability_ts"].to_list() == [10, 20]
    assert snap["delta"].to_list() == [5.0, -3.0]


def test_integration_with_real_simulator_output() -> None:
    stream = make_stream(
        [
            ("T", "B", 100, 5, 0),
            ("T", "A", 100, 2, 0),
            ("T", "B", 101, 3, 0),
        ],
        event_ts=[0, 1, 2],
        sequence=[1, 2, 3],
    )
    summary = footprint_summary(stream, interval_ns=10)
    store = FeatureStore().ingest_wide(
        summary,
        value_columns=["delta", "cumulative_delta", "absorption_ratio"],
        version="footprint_v1",
        instrument_id=1,
    )
    assert len(store) == 3
    snapshot = store.as_of(10)
    assert "delta" in snapshot["feature"].to_list()


def test_parquet_roundtrip(tmp_path: Path) -> None:
    store = FeatureStore().ingest_wide(
        _wide(), value_columns=["delta"], version="v1", instrument_id=1
    )
    path = tmp_path / "store.parquet"
    store.to_parquet(path)
    reloaded = FeatureStore.read_parquet(path)
    assert len(reloaded) == len(store)
    assert reloaded.as_of(25)["value"].to_list() == [-3.0]
