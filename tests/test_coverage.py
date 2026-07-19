"""اختبارات المحطة 9: مراقب التغطية البنيوية (Structural Coverage Monitor)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.alpha import run_full_research_pipeline
from nq.core.determinism import make_generator
from nq.coverage import (
    distance_correlation,
    mbo_window_descriptors,
    run_all_metrics,
    run_coverage_pipeline,
)
from nq.coverage.blocks import resolve_block_columns
from nq.simulation.cross_market import cross_market_features
from tests.mbo_factory import Event, make_stream, random_add_cancel_stream


def _paired_streams(n_events: int, *, seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    nq = random_add_cancel_stream(n_events, seed=seed)
    mnq = random_add_cancel_stream(n_events, seed=seed + 1)
    return nq, mnq


def test_distance_correlation_independent_near_zero() -> None:
    rng = make_generator(0)
    x = rng.normal(0, 1, 200)
    y = rng.normal(0, 1, 200)
    dcor = distance_correlation(x, y)
    assert dcor < 0.3


def test_distance_correlation_monotone_high() -> None:
    x = np.linspace(0, 1, 100)
    y = x**2
    dcor = distance_correlation(x, y)
    assert dcor > 0.8


def test_mbo_window_descriptors_nonempty() -> None:
    events: list[Event] = [
        ("A", "B", 20_000_000_000, 5, 1),
        ("A", "A", 20_001_000_000, 3, 2),
        ("C", "N", 0, 0, 1),
    ]
    frame = make_stream(events, event_ts=[0, 500, 1000])
    desc = mbo_window_descriptors(frame, interval_ns=1_000)
    assert desc.height >= 1
    assert "add_count" in desc.columns
    assert "cancel_ratio" in desc.columns


def test_resolve_block_columns() -> None:
    cols = resolve_block_columns(["nq_delta", "lead_lag", "nq_close"])
    assert "order_flow" in cols
    assert "nq_delta" in cols["order_flow"]


def test_run_coverage_pipeline_smoke() -> None:
    nq, mnq = _paired_streams(2000, seed=10)
    report = run_coverage_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        n_splits=2,
        rng=make_generator(0),
    )
    assert report.metrics.height > 0
    assert "metric" in report.metrics.columns
    md = report.report.to_markdown()
    assert "Structural Coverage" in md


def test_run_all_metrics_on_cross_market_features() -> None:
    nq, mnq = _paired_streams(2500, seed=20)
    features = cross_market_features(nq, mnq, interval_ns=10_000, lead_lag_window=2)
    nq_desc = mbo_window_descriptors(nq, interval_ns=10_000)
    results = run_all_metrics(
        features,
        nq_desc,
        n_permutations=200,
        n_splits=2,
        rng=make_generator(1),
    )
    names = {r.name for r in results}
    assert "mfig" in names
    assert "qduf" in names
    assert "psg" in names
    assert len(results) >= 3


def test_run_full_research_pipeline_integration() -> None:
    nq, mnq = _paired_streams(2000, seed=30)
    result = run_full_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        coverage_splits=2,
        rng=make_generator(2),
    )
    assert result.coverage.metrics.height > 0
    assert result.alpha.evaluations.height >= 0
    assert "Structural Coverage" in result.coverage.report.to_markdown()


def test_coverage_empty_frame() -> None:
    empty = pl.DataFrame(
        schema={
            "event_ts": pl.Int64(),
            "ingest_ts": pl.Int64(),
            "sequence": pl.UInt64(),
            "instrument_id": pl.Int32(),
            "symbol": pl.Utf8(),
            "action": pl.Utf8(),
            "side": pl.Utf8(),
            "price": pl.Int64(),
            "size": pl.Int64(),
            "order_id": pl.Int64(),
            "flags": pl.Int32(),
        }
    )
    report = run_coverage_pipeline(empty, empty, interval_ns=1_000)
    assert report.alerts == []
    assert report.metrics.height == 0
