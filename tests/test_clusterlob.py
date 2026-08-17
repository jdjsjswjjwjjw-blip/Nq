"""ClusterLOB: سمات الأمر، k-means على القطار فقط، OFI عنقودي مقابل مُجمَّع."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

import nq.research
from nq.core.determinism import make_generator
from nq.research.clusterlob import (
    HOLDOUT_START_DATE,
    N_CLUSTERS,
    WINDOW_NS,
    Y_TARGET,
    assign_clusters,
    causal_zscore,
    extract_add_features,
    extract_day_stream,
    fit_kmeans_train_only,
    ofi_feature_matrix,
    run_clusterlob,
    write_clusterlob_report,
)
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_PX_ASK = 20_000_250_000


def _setup_ts() -> int:
    stamp = dt.datetime(2025, 6, 3, 4, 0, 30, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "run_clusterlob" not in nq.research.__all__
    assert not hasattr(nq.research, "run_clusterlob")
    assert not hasattr(nq.research, "extract_day_stream")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    ts = [
        int(day_a.timestamp() * 1_000_000_000),
        int(day_b.timestamp() * 1_000_000_000),
    ]
    mbo = make_stream(
        [("A", "B", _PX, 1, 1), ("A", "A", _PX_ASK, 1, 2)],
        event_ts=ts,
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    with pytest.raises(ValueError, match="multi-day"):
        extract_day_stream(mbo)


def test_t1_and_t_prev_on_two_adds_at_same_price() -> None:
    t0 = _setup_ts()
    mbo = make_stream(
        [
            ("A", "B", _PX, 2, 1),
            ("A", "A", _PX_ASK, 2, 2),
            ("A", "B", _PX, 3, 3),
            ("A", "B", _PX, 1, 4),
        ],
        event_ts=[t0, t0 + 1, t0 + 10, t0 + 15],
    )
    feats = extract_add_features(mbo)
    assert feats.height == 4
    first_bid = feats.filter(pl.col("order_id") == 1)
    second_bid = feats.filter(pl.col("order_id") == 3)
    third_bid = feats.filter(pl.col("order_id") == 4)
    assert float(first_bid["t_1"][0]) == 0.0
    assert float(first_bid["t_prev"][0]) == 0.0
    assert float(second_bid["t_1"][0]) == 10.0
    assert float(second_bid["t_prev"][0]) == 10.0
    assert float(third_bid["t_1"][0]) == 15.0
    assert float(third_bid["t_prev"][0]) == 5.0
    assert float(second_bid["v"][0]) == 2.0


def test_ignores_events_after_setup() -> None:
    t = _setup_ts()
    mbo = make_stream(
        [
            ("A", "B", _PX, 4, 1),
            ("A", "B", _PX, 8, 3),
        ],
        event_ts=[t - 1_000, t + 1_000],
    )
    stream = extract_day_stream(mbo)
    oid_cluster = {1: 0, 3: 0}
    pooled, clustered = ofi_feature_matrix(
        stream.event_ts,
        stream.ingest_ts,
        stream.contrib_s,
        stream.contrib_c,
        stream.order_id,
        oid_cluster,
        np.asarray([t], dtype=np.int64),
        window_ns=WINDOW_NS,
    )
    assert float(pooled[0, 1]) != 0.0
    late_only, _ = ofi_feature_matrix(
        stream.event_ts,
        stream.ingest_ts,
        stream.contrib_s,
        stream.contrib_c,
        stream.order_id,
        oid_cluster,
        np.asarray([t - 2_000], dtype=np.int64),
        window_ns=WINDOW_NS,
    )
    assert float(late_only[0, 1]) == 0.0
    assert float(late_only[0, 2]) == 0.0
    assert clustered.shape == (1, 1 + 2 * N_CLUSTERS)


def test_kmeans_fit_on_train_only() -> None:
    rng = make_generator(7)
    train = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.1, size=(40, 6)),
            rng.normal(loc=5.0, scale=0.1, size=(40, 6)),
            rng.normal(loc=-5.0, scale=0.1, size=(40, 6)),
        ]
    )
    test = rng.normal(loc=50.0, scale=0.1, size=(30, 6))
    km = fit_kmeans_train_only(train, seed=7, subsample=120)
    assert km.centroids_ is not None
    frozen = km.centroids_.copy()
    assign_clusters(test, km)
    assert np.allclose(km.centroids_, frozen)
    leaked = fit_kmeans_train_only(np.vstack([train, test]), seed=7, subsample=150)
    assert leaked.centroids_ is not None
    assert float(np.max(np.abs(leaked.centroids_ - frozen))) > 0.5


def test_opposed_cluster_ofi_not_equal_pooled() -> None:
    ts = np.asarray([10, 20, 30, 40], dtype=np.int64)
    ingest = ts.copy()
    contrib_s = np.asarray([5, -5, 5, -5], dtype=np.int64)
    contrib_c = np.asarray([1, -1, 1, -1], dtype=np.int64)
    oids = np.asarray([1, 2, 3, 4], dtype=np.int64)
    oid_cluster = {1: 0, 3: 0, 2: 1, 4: 1}
    pooled, clustered = ofi_feature_matrix(
        ts,
        ingest,
        contrib_s,
        contrib_c,
        oids,
        oid_cluster,
        np.asarray([40], dtype=np.int64),
        window_ns=100,
    )
    assert float(pooled[0, 1]) == 0.0
    assert float(clustered[0, 1]) == 10.0
    assert float(clustered[0, 3]) == -10.0
    assert float(clustered[0, 5]) == 0.0


def test_causal_zscore_ignores_future_rows() -> None:
    raw = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    z = causal_zscore(raw, window=3)
    assert float(z[0, 0]) == 0.0
    z_without_last = causal_zscore(raw[:3], window=3)
    assert float(z[2, 0]) == float(z_without_last[2, 0])


def test_report_writes(tmp_path: Path) -> None:
    scored = pl.DataFrame(
        {
            "setup_availability_ts": [1, 2],
            "y": [1.0, 0.0],
            "p_pooled": [0.6, 0.4],
            "p_clustered": [0.7, 0.3],
            "fold": [0, 0],
        }
    )
    diagnostics = {
        "pooled": {"n": 2.0, "auc": 0.5, "brier_skill": 0.0},
        "clustered": {"n": 2.0, "auc": 0.5, "brier_skill": 0.0},
        "holdout_touched": False,
        "concatenated_raw_mbo": False,
        "reconstructed_order_book": True,
    }
    written = write_clusterlob_report(scored, diagnostics, tmp_path)
    text = (written / "CLUSTERLOB.md").read_text(encoding="utf-8")
    assert "Do not claim" in text
    assert "pooled OFI" in text
    assert Y_TARGET == "y_phase_extend"
    assert HOLDOUT_START_DATE == "2025-09-01"


def test_run_clusterlob_two_days_holdout_untouched() -> None:
    t = _setup_ts()
    day_ns = 86_400_000_000_000
    mbo_a = make_stream(
        [
            ("A", "B", _PX, 2, 1),
            ("A", "A", _PX_ASK, 2, 2),
            ("A", "B", _PX, 1, 3),
            ("A", "A", _PX_ASK, 1, 4),
        ],
        event_ts=[t - 4_000, t - 3_000, t - 2_000, t - 1_000],
    )
    mbo_b = make_stream(
        [
            ("A", "B", _PX, 3, 1),
            ("A", "A", _PX_ASK, 1, 2),
            ("A", "B", _PX, 2, 3),
            ("A", "A", _PX_ASK, 2, 4),
        ],
        event_ts=[t + day_ns - 4_000, t + day_ns - 3_000, t + day_ns - 2_000, t + day_ns - 1_000],
    )
    n = 12
    gap = 1_000_000_000
    labels_a = pl.DataFrame(
        {
            "setup_availability_ts": [t + i * gap for i in range(n)],
            "y": [float(i % 2) for i in range(n)],
            "outcome_name": ["y_phase_extend"] * n,
            "label_status": ["resolved"] * n,
        }
    )
    t_b = t + day_ns
    labels_b = pl.DataFrame(
        {
            "setup_availability_ts": [t_b + i * gap for i in range(n)],
            "y": [float((i + 1) % 2) for i in range(n)],
            "outcome_name": ["y_phase_extend"] * n,
            "label_status": ["resolved"] * n,
        }
    )
    scored, diag = run_clusterlob(
        [(mbo_a, labels_a), (mbo_b, labels_b)],
        seed=0,
        holdout_start="2025-09-01",
    )
    assert diag["holdout_touched"] is False
    assert diag["concatenated_raw_mbo"] is False
    assert diag["reconstructed_order_book"] is True
    assert diag["kmeans_train_only"] is True
    assert scored.height >= 0
