"""Failed Breakout كطبقة بحث داخل الخط الموحّد.

أمر تشغيل منفصل يمرّ بنفس المحرك: ميزات + SSL ‖ M9 ‖ ألفا + مخرجات.
يضيّق الفرز على ``fail_breakout`` — مع دخول سببي (إغلاق الشمعة) بلا ملء
وهمي عند مستوى الكسر.
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

_FAIL_BREAKOUT_FOCUS = (
    "fail_breakout",
    "fb_effort_range_ratio",
    "fb_effort_volume_ratio",
    "trap_setup",
    "nq_delta",
    "mnq_delta",
    "lead_lag",
)


@dataclass(frozen=True, slots=True)
class FailBreakoutResearchResult:
    """غلاف مريح فوق ``UnifiedResearchResult`` لتركيز Failed Breakout."""

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
    ) -> FailBreakoutResearchResult:
        return cls(
            features=result.features,
            alpha=result.alpha,
            ssl=result.ssl,
            report=result.alpha.report,
            unified=result.report,
            signal_columns=signal_columns,
        )


def run_fail_breakout_research(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    ssl_window: int = 5,
    ssl_components: int = 4,
    horizon: int = 1,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    max_rows: int | None = None,
    rng: np.random.Generator | None = None,
    output_dir: Path | str | None = None,
    quiet: bool = False,
) -> FailBreakoutResearchResult:
    """يشغّل Failed Breakout عبر الخط الموحّد (أمر تشغيل منفصل)."""
    cfg = PipelineConfig(
        include_failed_breakout=True,
        include_failed_fvg=False,
        include_auction_vp=False,
        cross_market_mode="nq_only" if mnq is None else "dual",
        max_rows=max_rows,
        horizon=horizon,
        alpha=alpha,
        n_permutations=n_permutations,
        ssl_window=ssl_window,
        ssl_components=ssl_components,
        signal_columns=_FAIL_BREAKOUT_FOCUS,
        quiet=quiet,
    )
    partner = mnq if mnq is not None else nq
    result = run_research_pipeline(
        nq,
        partner,
        config=cfg,
        signal_columns=_FAIL_BREAKOUT_FOCUS,
        output_dir=output_dir,
        rng=rng,
    )
    return FailBreakoutResearchResult.from_unified(
        result, signal_columns=_FAIL_BREAKOUT_FOCUS
    )


__all__ = [
    "FailBreakoutResearchResult",
    "run_fail_breakout_research",
]
