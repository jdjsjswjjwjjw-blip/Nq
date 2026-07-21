"""Failed FVG كطبقة بحث داخل الخط الموحّد.

المسار الأساسي هو ``run_research_pipeline`` (إشارة ``fail_fvg`` بجانب
``trap_setup`` / ``lead_lag``). هذا الملف واجهة مريحة تُركّز الفرز على
إشارات Failed FVG دون fork معماري.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nq.alpha.discovery import AlphaDiscovery
from nq.models.ssl_pipeline import SSLPipelineResult
from nq.research.assistant import ResearchReport
from nq.research.orchestrator import (
    PipelineConfig,
    UnifiedResearchResult,
    run_research_pipeline,
)
from nq.research.unified import UnifiedResearchReport

_FAIL_FVG_FOCUS = (
    "fail_fvg",
    "lead_lag",
    "trap_setup",
    "divergence",
    "nq_delta",
    "mnq_delta",
)


@dataclass(frozen=True, slots=True)
class FailFvgResearchResult:
    """غلاف مريح فوق ``UnifiedResearchResult`` لتركيز Failed FVG."""

    features: pl.DataFrame
    alpha: AlphaDiscovery
    ssl: SSLPipelineResult | None
    report: ResearchReport
    unified: UnifiedResearchReport
    signal_columns: tuple[str, ...]

    @classmethod
    def from_unified(
        cls,
        result: UnifiedResearchResult,
        *,
        signal_columns: tuple[str, ...],
    ) -> FailFvgResearchResult:
        return cls(
            features=result.features,
            alpha=result.alpha,
            ssl=result.ssl,
            report=result.alpha.report,
            unified=result.report,
            signal_columns=signal_columns,
        )


def run_fail_fvg_research(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    use_ssl_gate: bool = True,
    ssl_window: int = 5,
    ssl_components: int = 4,
    horizon: int = 1,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    max_rows: int | None = None,
    rng: np.random.Generator | None = None,
    output_dir: Path | str | None = None,
) -> FailFvgResearchResult:
    """يشغّل Failed FVG عبر الخط الموحّد (نفس طبقات trap_setup / lead_lag).

    ``use_ssl_gate`` يُبقي اسم التوافق؛ الفرز يبقى على ``fail_fvg`` داخل
    ``run_research_pipeline`` مع SSL tick/bucket حسب الإعداد.
    """
    _ = use_ssl_gate  # التوافق مع الواجهة السابقة؛ البوابة عبر ssl_mode في الخط الموحّد
    cfg = PipelineConfig(
        include_failed_fvg=True,
        cross_market_mode="nq_only" if mnq is None else "dual",
        max_rows=max_rows,
        horizon=horizon,
        alpha=alpha,
        n_permutations=n_permutations,
        ssl_window=ssl_window,
        ssl_components=ssl_components,
        signal_columns=_FAIL_FVG_FOCUS,
    )
    partner = mnq if mnq is not None else nq
    result = run_research_pipeline(
        nq,
        partner,
        config=cfg,
        signal_columns=_FAIL_FVG_FOCUS,
        output_dir=output_dir,
        rng=rng,
    )
    return FailFvgResearchResult.from_unified(result, signal_columns=_FAIL_FVG_FOCUS)


__all__ = [
    "FailFvgResearchResult",
    "run_fail_fvg_research",
]
