"""اختبارات محرّك الميزات اللحظية (Streaming State Machine)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.determinism import make_generator
from nq.features.streaming import (
    sample_streaming_to_interval,
    streaming_event_features,
)
from nq.research.orchestrator import PipelineConfig, run_research_pipeline
from tests.test_coverage import _paired_streams


def test_streaming_event_features_availability_equals_event_ts() -> None:
    nq, mnq = _paired_streams(800, seed=1)
    events = streaming_event_features(nq, mnq)
    assert events.height > 0
    assert (events[AVAILABILITY_TS] == events[EVENT_TS]).all()
    assert "trap_setup" in events.columns
    assert "nq_close" in events.columns
    assert "nq_bid" in events.columns


def test_sample_streaming_keeps_last_state_at_bucket_end() -> None:
    nq, mnq = _paired_streams(800, seed=2)
    events = streaming_event_features(nq, mnq)
    interval = 10_000
    sampled = sample_streaming_to_interval(events, interval_ns=interval)
    assert sampled.height > 0
    assert sampled.height <= events.height
    # عيّنة البحث: availability = نهاية الفاصل؛ المحتوى = آخر حالة داخله
    for row in sampled.iter_rows(named=True):
        assert int(row[AVAILABILITY_TS]) % interval == 0


def test_streaming_past_stable_under_future_perturbation() -> None:
    nq, mnq = _paired_streams(1200, seed=3)
    baseline = streaming_event_features(nq, mnq)
    cut = int(baseline[AVAILABILITY_TS][baseline.height // 2])
    past = baseline.filter(pl.col(AVAILABILITY_TS) <= cut)["trap_setup"].to_list()

    rng = np.random.default_rng(0)
    prices = nq["price"].to_list()
    ts = nq["event_ts"].to_list()
    for i, t in enumerate(ts):
        if t > cut:
            prices[i] = int(prices[i] + rng.integers(50, 200) * 1_000_000)
    perturbed = nq.with_columns(pl.Series("price", prices))
    after = streaming_event_features(perturbed, mnq)
    after_past = after.filter(pl.col(AVAILABILITY_TS) <= cut)["trap_setup"].to_list()
    assert after_past == past


def test_orchestrator_uses_streaming_features_by_default() -> None:
    nq, mnq = _paired_streams(2000, seed=4)
    cfg = PipelineConfig(
        feature_mode="streaming",
        n_permutations=100,
        parallel_coverage=False,
    )
    result = run_research_pipeline(
        nq,
        mnq,
        config=cfg,
        interval_ns=10_000,
        rng=make_generator(0),
    )
    assert "phase_balance" in result.features.columns or "trap_setup" in result.features.columns
    assert "nq_close" in result.features.columns
    assert "fail_fvg" in result.features.columns


def test_orchestrator_batch_mode_still_works() -> None:
    nq, mnq = _paired_streams(2000, seed=5)
    cfg = PipelineConfig(
        feature_mode="batch",
        n_permutations=100,
        parallel_coverage=False,
    )
    result = run_research_pipeline(
        nq,
        mnq,
        config=cfg,
        interval_ns=10_000,
        rng=make_generator(1),
    )
    assert "lead_lag" in result.features.columns
    assert "nq_close" in result.features.columns
