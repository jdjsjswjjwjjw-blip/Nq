"""Volume Profile / Auction كطبقة بحث داخل الخط الموحّد.

المسار الأساسي هو ``run_research_pipeline`` (إشارات ``vp_balance`` /
``vp_imbalance`` / ``vp_expansion`` بجانب باقي القناة). هذا الملف واجهة
مريحة تُركّز الفرز على فرضيات الملف الحجمي والتوازن/الاختلال دون fork.
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

_VP_AUCTION_FOCUS = (
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_flip_to_imbalance",
    "vp_pullback_defense",
    "nq_delta",
)


@dataclass(frozen=True, slots=True)
class VpAuctionResearchResult:
    """غلاف مريح فوق ``UnifiedResearchResult`` لتركيز Volume Profile / Auction."""

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
    ) -> VpAuctionResearchResult:
        return cls(
            features=result.features,
            alpha=result.alpha,
            ssl=result.ssl,
            report=result.alpha.report,
            unified=result.report,
            signal_columns=signal_columns,
        )


def run_vp_auction_research(
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
) -> VpAuctionResearchResult:
    """يشغّل فرضيات VP + التوازن/الاختلال عبر الخط الموحّد (NQ-only افتراضيًا)."""
    cfg = PipelineConfig(
        include_auction_vp=True,
        include_failed_fvg=False,
        include_failed_breakout=False,
        cross_market_mode="nq_only" if mnq is None else "dual",
        max_rows=max_rows,
        horizon=horizon,
        alpha=alpha,
        n_permutations=n_permutations,
        ssl_window=ssl_window,
        ssl_components=ssl_components,
        signal_columns=_VP_AUCTION_FOCUS,
        quiet=quiet,
    )
    partner = mnq if mnq is not None else nq
    result = run_research_pipeline(
        nq,
        partner,
        config=cfg,
        signal_columns=_VP_AUCTION_FOCUS,
        output_dir=output_dir,
        rng=rng,
    )
    return VpAuctionResearchResult.from_unified(result, signal_columns=_VP_AUCTION_FOCUS)


__all__ = [
    "VpAuctionResearchResult",
    "run_vp_auction_research",
]
