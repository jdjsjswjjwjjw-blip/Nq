"""منسّق البحث الموحّد (Unified Research Orchestrator).

عند تشغيل SSL:
1. يُبنى إطار الميزات مرة واحدة من MBO.
2. تُشغَّل المحطة 9 (المراقب) **بالتوازي في الخلفية** مع SSL.
3. تُشغَّل قناة الألفا/LLM بعد اكتمال SSL.
4. يُدمَج كل شيء في ``UnifiedResearchReport`` شامل.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.alpha.discovery import AlphaDiscovery, discover_alpha_from_features
from nq.alpha.execution import ExecutionMode
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.temporal_policy import TemporalPolicy
from nq.coverage.monitor import run_coverage_on_features
from nq.coverage.types import CoverageReport
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_pipeline
from nq.research.assistant import LanguageModel, ResearchAssistant
from nq.research.unified import UnifiedResearchReport, build_unified_report
from nq.simulation.cross_market import cross_market_features

_DEFAULT_SIGNAL_COLUMNS = (
    "nq_delta",
    "mnq_delta",
    "lead_lag",
    "trap_setup",
    "divergence",
)


@dataclass(frozen=True, slots=True)
class UnifiedResearchResult:
    """مخرجات المنسّق الكامل: SSL + M9 + ألفا + تقرير موحّد."""

    ssl: SSLPipelineResult
    coverage: CoverageReport
    alpha: AlphaDiscovery
    report: UnifiedResearchReport


def _run_coverage_task(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    features: pl.DataFrame,
    *,
    interval_ns: int,
    price_col: str,
    alpha: float,
    n_splits: int,
    embargo: int,
    n_permutations: int,
    seed: int,
) -> CoverageReport:
    """مهمة الخلفية: مراقب التغطية M9 (بذرة مستقلة لضمان الحتمية)."""
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
        rng=np.random.default_rng(seed),
    )


def run_ssl_research_pipeline(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    horizon: int = 1,
    signal_columns: Sequence[str] | None = None,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_permutations: int = 2000,
    lead_lag_window: int = 2,
    ssl_window: int = 5,
    ssl_components: int = 4,
    coverage_splits: int = 3,
    coverage_embargo: int | None = None,
    execution_mode: ExecutionMode = "mid",
    slippage_ticks: float = 0.5,
    tick_size: float = 0.25,
    commission_bps: float = 0.0,
    parallel_coverage: bool = True,
    language_model: LanguageModel | None = None,
    rng: np.random.Generator | None = None,
) -> UnifiedResearchResult:
    """خط البحث الرئيسي: SSL + M9 (خلفية) + ألفا/LLM → تقرير شامل.

    * ``parallel_coverage=True`` (افتراضي): المحطة 9 تشتغل في thread منفصل
      أثناء تشغيل SSL على المسار الرئيسي.
    * عند اكتمال القنوات الثلاث يُعاد ``UnifiedResearchReport`` جاهزًا
      للعرض عبر ``.to_markdown()``.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    seed = int(generator.integers(0, 2**31))

    policy = TemporalPolicy.for_run(interval_ns=interval_ns, window=ssl_window)
    embargo_val = (
        coverage_embargo
        if coverage_embargo is not None
        else policy.embargo_time_units(interval_ns=interval_ns)
    )
    purge_val = policy.purge_samples()

    features = cross_market_features(
        nq, mnq, interval_ns=interval_ns, lead_lag_window=lead_lag_window
    )

    ssl_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)
    alpha_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)

    if parallel_coverage and features.height > 0:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="coverage-m9") as executor:
            coverage_future = executor.submit(
                _run_coverage_task,
                nq,
                mnq,
                features,
                interval_ns=interval_ns,
                price_col=price_col,
                alpha=alpha,
                n_splits=coverage_splits,
                embargo=embargo_val,
                n_permutations=n_permutations,
                seed=seed,
            )
            ssl_result = run_ssl_pipeline(
                features,
                window=ssl_window,
                n_components=ssl_components,
                n_splits=coverage_splits,
                embargo=embargo_val,
                purge_samples=purge_val,
                interval_ns=interval_ns,
                alpha=alpha,
                rng=generator,
                assistant=ssl_assistant,
            )
            columns = (
                list(signal_columns)
                if signal_columns is not None
                else [c for c in _DEFAULT_SIGNAL_COLUMNS if c in features.columns]
            )
            alpha_result = discover_alpha_from_features(
                features,
                signal_columns=columns,
                price_col=price_col,
                time_col=AVAILABILITY_TS,
                horizon=horizon,
                execution_mode=execution_mode,
                slippage_ticks=slippage_ticks,
                tick_size=tick_size,
                commission_bps=commission_bps,
                alpha=alpha,
                n_permutations=n_permutations,
                rng=generator,
                assistant=alpha_assistant,
            )
            coverage_result = coverage_future.result()
    else:
        ssl_result = run_ssl_pipeline(
            features,
            window=ssl_window,
            n_components=ssl_components,
            n_splits=coverage_splits,
            embargo=embargo_val,
            purge_samples=purge_val,
            interval_ns=interval_ns,
            alpha=alpha,
            rng=generator,
            assistant=ssl_assistant,
        )
        columns = (
            list(signal_columns)
            if signal_columns is not None
            else [c for c in _DEFAULT_SIGNAL_COLUMNS if c in features.columns]
        )
        alpha_result = discover_alpha_from_features(
            features,
            signal_columns=columns,
            price_col=price_col,
            time_col=AVAILABILITY_TS,
            horizon=horizon,
            alpha=alpha,
            n_permutations=n_permutations,
            rng=generator,
            assistant=alpha_assistant,
        )
        coverage_result = run_coverage_on_features(
            nq,
            mnq,
            features,
            interval_ns=interval_ns,
            price_col=price_col,
            alpha=alpha,
            n_splits=coverage_splits,
            embargo=embargo_val,
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed),
        )

    narrative = ""
    if language_model is not None:
        all_claims = " ".join(
            o.finding.claim
            for report in (
                ssl_result.report,
                coverage_result.report,
                alpha_result.report,
            )
            for o in report.verified
        )
        if all_claims:
            narrative = language_model.complete(
                "لخّص الاستنتاجات الموثّقة التالية من قنوات SSL والمراقب M9 والألفا "
                "دون إضافة أي ادعاء جديد:\n" + all_claims
            )

    unified = build_unified_report(
        ssl_report=ssl_result.report,
        coverage_report=coverage_result.report,
        alpha_report=alpha_result.report,
        narrative=narrative,
    )

    return UnifiedResearchResult(
        ssl=ssl_result,
        coverage=coverage_result,
        alpha=alpha_result,
        report=unified,
    )
