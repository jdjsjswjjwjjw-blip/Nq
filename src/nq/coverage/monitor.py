"""مراقب التغطية البنيوية (Structural Coverage Monitor).

يجمع مقاييس العمى البنيوي الستة، يُنتج ``Evidence`` وتنبيهات، ويكتب تقريرًا
موثّقًا عبر ``ResearchAssistant``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.coverage.mbo_descriptors import mbo_window_descriptors
from nq.coverage.metrics import MetricResult, metric_to_evidence, run_all_metrics
from nq.coverage.types import CoverageAlert, CoverageReport
from nq.research.assistant import ResearchAssistant
from nq.research.evidence import Evidence
from nq.research.findings import Finding
from nq.simulation.cross_market import cross_market_features
from nq.statistics.hypothesis import verify_hypotheses

_SEVERITY_HIGH = "high"
_SEVERITY_MEDIUM = "medium"


def _severity_for(metric_name: str) -> str:
    if metric_name.startswith(("mfig", "qduf", "cer:")):
        return _SEVERITY_HIGH
    return _SEVERITY_MEDIUM


def _metrics_frame(results: list[MetricResult]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "metric": [r.name for r in results],
            "value": [r.value for r in results],
            "pvalue": [r.pvalue for r in results],
            "sample_size": [r.sample_size for r in results],
            "triggered": [r.triggered for r in results],
            "detail": [r.detail for r in results],
        }
    )


def build_coverage_report(
    results: list[MetricResult],
    *,
    assistant: ResearchAssistant | None = None,
    alpha: float = 0.05,
    title: str = "Structural Coverage — Blind-Spot Report",
) -> CoverageReport:
    """يبني تقرير تغطية من نتائج المقاييس."""
    research = assistant if assistant is not None else ResearchAssistant(alpha=alpha)
    evidence_list: list[Evidence] = []
    alerts: list[CoverageAlert] = []
    findings: list[Finding] = []

    triggered = [r for r in results if r.triggered]
    pvalues = {r.name: r.pvalue for r in triggered}
    verified = verify_hypotheses(pvalues, alpha=alpha) if pvalues else None
    reject_set: set[str] = set()
    if verified is not None:
        reject_set = set(verified.filter(~pl.col("reject"))["hypothesis"].to_list())

    for result in results:
        ev = metric_to_evidence(result)
        if result.triggered and result.name not in reject_set:
            research.store.add(ev)
            evidence_list.append(ev)
            alerts.append(
                CoverageAlert(
                    metric=result.name,
                    evidence_id=ev.id,
                    severity=_severity_for(result.name),
                    detail=result.detail,
                )
            )
            claim = f"عمى بنيوي مكتشف: {result.detail}"
            findings.append(research.generate_hypothesis(claim, ev, category="coverage"))

    report = research.write_report(findings, title=title)
    return CoverageReport(
        alerts=alerts,
        evidence=evidence_list,
        metrics=_metrics_frame(results),
        report=report,
    )


def run_coverage_on_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    features: pl.DataFrame,
    *,
    interval_ns: int,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_splits: int = 3,
    embargo: int = 0,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
    assistant: ResearchAssistant | None = None,
) -> CoverageReport:
    """يشغّل مراقب التغطية على ميزات مُسبَقة البناء (بدون إعادة حساب المحاكيات)."""
    if features.height == 0:
        research = assistant if assistant is not None else ResearchAssistant(alpha=alpha)
        empty_report = research.write_report([], title="Structural Coverage — Blind-Spot Report")
        return CoverageReport(
            alerts=[],
            evidence=[],
            metrics=pl.DataFrame(
                schema={
                    "metric": pl.Utf8(),
                    "value": pl.Float64(),
                    "pvalue": pl.Float64(),
                    "sample_size": pl.Int64(),
                    "triggered": pl.Boolean(),
                    "detail": pl.Utf8(),
                }
            ),
            report=empty_report,
        )

    nq_desc = mbo_window_descriptors(nq, interval_ns=interval_ns)
    mnq_desc = mbo_window_descriptors(mnq, interval_ns=interval_ns)
    _time_cols = {AVAILABILITY_TS, "bucket_start", "bucket_end"}
    mnq_renamed = mnq_desc.rename({c: f"mnq_{c}" for c in mnq_desc.columns if c not in _time_cols})
    combined_desc = nq_desc.join(mnq_renamed, on=AVAILABILITY_TS, how="left")

    results = run_all_metrics(
        features,
        combined_desc,
        price_col=price_col,
        n_splits=n_splits,
        embargo=embargo,
        alpha=alpha,
        n_permutations=n_permutations,
        rng=rng,
    )
    return build_coverage_report(results, assistant=assistant, alpha=alpha)


def run_coverage_pipeline(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_splits: int = 3,
    embargo: int = 0,
    n_permutations: int = 2000,
    lead_lag_window: int = 2,
    rng: np.random.Generator | None = None,
) -> CoverageReport:
    """خط تغطية كامل: من MBO الخام إلى تقرير عمى بنيوي موثّق."""
    features = cross_market_features(
        nq, mnq, interval_ns=interval_ns, lead_lag_window=lead_lag_window
    )
    return run_coverage_on_features(
        nq,
        mnq,
        features,
        interval_ns=interval_ns,
        price_col=price_col,
        alpha=alpha,
        n_splits=n_splits,
        embargo=embargo,
        n_permutations=n_permutations,
        rng=rng,
    )
