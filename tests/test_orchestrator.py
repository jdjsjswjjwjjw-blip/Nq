"""اختبارات المنسّق الموحّد: خط واحد من MBO إلى التقرير."""

from __future__ import annotations

import polars as pl
import pytest

from nq.core.determinism import make_generator
from nq.models.ssl_pipeline import run_ssl_pipeline
from nq.research.assistant import ResearchAssistant
from nq.research.orchestrator import (
    PipelineConfig,
    _load_pipeline_frames,
    _validate_contract_input,
    run_research_pipeline,
)
from nq.research.unified import build_unified_report
from nq.simulation.cross_market import cross_market_features
from tests.test_coverage import _paired_streams


def test_contract_guard_rejects_rollover_stitching() -> None:
    nq, _ = _paired_streams(20, seed=900)
    mixed = pl.concat(
        [
            nq.head(10),
            nq.tail(10).with_columns(pl.lit(99, dtype=pl.UInt32).alias("instrument_id")),
        ]
    )
    with pytest.raises(ValueError, match="run each futures contract independently"):
        _validate_contract_input(mixed, expected_family="NQ", role="NQ")


def test_contract_guard_rejects_nq_as_mnq() -> None:
    nq, _ = _paired_streams(20, seed=901)
    with pytest.raises(ValueError, match="expected MNQ"):
        _validate_contract_input(nq, expected_family="MNQ", role="MNQ")


def test_dual_mode_rejects_reusing_same_source() -> None:
    nq, _ = _paired_streams(20, seed=902)
    with pytest.raises(ValueError, match="requires separate NQ and MNQ"):
        _load_pipeline_frames(nq, nq, PipelineConfig(cross_market_mode="dual"))


def test_run_ssl_pipeline_produces_report() -> None:
    nq, mnq = _paired_streams(2500, seed=40)
    features = cross_market_features(nq, mnq, interval_ns=10_000, lead_lag_window=2)
    result = run_ssl_pipeline(
        features,
        window=3,
        n_components=3,
        n_splits=2,
        rng=make_generator(0),
    )
    assert result.metrics.height >= 0
    md = result.report.to_markdown()
    assert "SSL Foundation Model" in md


def test_unified_report_has_three_channels() -> None:
    assistant = ResearchAssistant()
    empty = assistant.write_report([], title="empty")
    unified = build_unified_report(
        ssl_report=empty,
        coverage_report=empty,
        alpha_report=empty,
    )
    md = unified.to_markdown()
    assert "قناة 1 — SSL" in md
    assert "قناة 2 — المراقب M9" in md
    assert "قناة 3 — LLM" in md


def test_run_research_pipeline_unified_report() -> None:
    nq, mnq = _paired_streams(2500, seed=50)
    result = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        parallel_coverage=True,
        rng=make_generator(1),
        quiet=True,
    )
    md = result.report.to_markdown()
    assert "قناة 1 — SSL" in md
    assert "قناة 2 — المراقب M9" in md
    assert "قناة 3 — LLM" in md
    assert "session_phase" in result.features.columns
    assert result.ssl.metrics is not None
    assert result.coverage.metrics.height >= 0
    assert result.alpha.evaluations.height >= 0


def test_run_research_pipeline_includes_failed_fvg_signal() -> None:
    nq, mnq = _paired_streams(2500, seed=70)
    result = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        parallel_coverage=False,
        rng=make_generator(3),
        quiet=True,
    )
    assert "fail_fvg" in result.features.columns
    assert "effort_range_ratio" in result.features.columns
    names = result.alpha.evaluations["name"].to_list() if result.alpha.evaluations.height else []
    assert "fail_fvg" in names or result.alpha.evaluations.height == 0


def test_run_research_pipeline_includes_auction_vp_signals() -> None:
    nq, mnq = _paired_streams(2500, seed=71)
    result = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        parallel_coverage=False,
        rng=make_generator(4),
        quiet=True,
    )
    for col in (
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_close_in_value",
        "vp_flip_to_imbalance",
    ):
        assert col in result.features.columns
    names = result.alpha.evaluations["name"].to_list() if result.alpha.evaluations.height else []
    assert "vp_balance" in names or result.alpha.evaluations.height == 0


def test_run_research_pipeline_sequential_coverage() -> None:
    nq, mnq = _paired_streams(2000, seed=60)
    parallel = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        parallel_coverage=True,
        rng=make_generator(2),
        quiet=True,
    )
    sequential = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        parallel_coverage=False,
        rng=make_generator(2),
        quiet=True,
    )
    assert parallel.ssl.metrics.equals(sequential.ssl.metrics)
    assert parallel.alpha.selected == sequential.alpha.selected
