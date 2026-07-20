"""اختبارات المنسّق الموحّد: خط واحد من MBO إلى التقرير."""

from __future__ import annotations

from nq.core.determinism import make_generator
from nq.models.ssl_pipeline import run_ssl_pipeline
from nq.research.assistant import ResearchAssistant
from nq.research.orchestrator import run_research_pipeline
from nq.research.unified import build_unified_report
from nq.simulation.cross_market import cross_market_features
from tests.test_coverage import _paired_streams


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
    )
    md = result.report.to_markdown()
    assert "قناة 1 — SSL" in md
    assert "قناة 2 — المراقب M9" in md
    assert "قناة 3 — LLM" in md
    assert "session_phase" in result.features.columns
    assert result.ssl.metrics is not None
    assert result.coverage.metrics.height >= 0
    assert result.alpha.evaluations.height >= 0


def test_run_research_pipeline_sequential_coverage() -> None:
    nq, mnq = _paired_streams(2000, seed=60)
    parallel = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        parallel_coverage=True,
        rng=make_generator(2),
    )
    sequential = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=200,
        parallel_coverage=False,
        rng=make_generator(2),
    )
    assert parallel.ssl.metrics.equals(sequential.ssl.metrics)
    assert parallel.alpha.selected == sequential.alpha.selected
