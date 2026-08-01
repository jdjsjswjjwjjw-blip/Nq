"""Volume Profile / Auction كطبقة بحث داخل الخط الموحّد.

المسار الأساسي يبني إشارات ``vp_*`` عبر ``run_research_pipeline``. هذا الملف
يركّز الفرز على فرضيات الملف الحجمي **باختيار walk-forward purged** (وليس
شاشة عيّنة كاملة وحدها) — نفس صرامة FB/FVG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nq.alpha.discovery import AlphaDiscovery
from nq.core.temporal_policy import TemporalPolicy
from nq.models.ssl_pipeline import SSLPipelineResult
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.orchestrator import (
    PipelineConfig,
    UnifiedResearchResult,
    run_research_pipeline,
)
from nq.research.progress import PipelineProgress, resolve_progress
from nq.research.unified import UnifiedResearchReport
from nq.strategies.fvg_hypothesis import walk_forward_select_hypotheses

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
    fold_df: pl.DataFrame
    oos_ic: float
    oos_pvalue: float
    oos_n: int
    best_signal: str | None
    exploratory_only: bool

    @classmethod
    def from_unified(
        cls,
        result: UnifiedResearchResult,
        *,
        signal_columns: tuple[str, ...],
        fold_df: pl.DataFrame,
        oos_ic: float,
        oos_pvalue: float,
        oos_n: int,
        best_signal: str | None,
        exploratory_only: bool,
        report: ResearchReport,
    ) -> VpAuctionResearchResult:
        return cls(
            features=result.features,
            alpha=result.alpha,
            ssl=result.ssl,
            report=report,
            unified=result.report,
            signal_columns=signal_columns,
            fold_df=fold_df,
            oos_ic=oos_ic,
            oos_pvalue=oos_pvalue,
            oos_n=oos_n,
            best_signal=best_signal,
            exploratory_only=exploratory_only,
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
    n_splits: int = 3,
    max_rows: int | None = None,
    rng: np.random.Generator | None = None,
    output_dir: Path | str | None = None,
    quiet: bool = False,
    exploratory_full_sample: bool = False,
) -> VpAuctionResearchResult:
    """يشغّل فرضيات VP عبر الخط الموحّد ثم يختار بـ walk-forward OOS.

    ``exploratory_full_sample=True`` يبقي شاشة العيّنة الكاملة للتقرير فقط؛
    الحكم الإحصائي يبقى من مسار WF (ما لم تُفرَّغ المرشّحات).
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    log = resolve_progress(None, quiet=quiet)

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
        rng=generator,
    )

    interval_ns = int(cfg.interval_ns)
    policy = TemporalPolicy.for_run(
        interval_ns=interval_ns, window=ssl_window, horizon=horizon
    )
    embargo = policy.embargo_time_units(interval_ns=interval_ns)
    candidates = tuple(c for c in _VP_AUCTION_FOCUS if c in result.features.columns)
    log.step("VP walk-forward selection", f"candidates={len(candidates)}")
    fold_df, oos_ic, oos_p, oos_n, best = walk_forward_select_hypotheses(
        result.features,
        candidates,
        price_col="nq_close",
        horizon=horizon,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=policy.purge_samples(),
        n_permutations=n_permutations,
        selection_aware_null=True,
        rng=generator,
        progress=log,
    )

    assistant = ResearchAssistant(alpha=alpha)
    detail = (
        f"best_oos={best!r}; oos_ic={oos_ic:.4g}; oos_p={oos_p:.4g}; "
        f"n={oos_n}; exploratory_full_sample={exploratory_full_sample}; "
        f"horizon={horizon}; selection_aware_null=True"
    )
    evidence = Evidence(
        id="vp_search:oos_ic",
        source="vp_auction_walk_forward",
        metric="IC",
        value=oos_ic,
        pvalue=oos_p,
        sample_size=oos_n,
        detail=detail,
    )
    claim = (
        f"فرضية Volume Profile المختارة بـ walk-forward (best={best!r}) "
        f"تحقق IC خارج العينة = {oos_ic:.4g} (p={oos_p:.4g})."
    )
    findings = [
        assistant.generate_hypothesis(
            claim,
            evidence,
            requires_significance=True,
            category="vp_auction_search",
        )
    ]
    if exploratory_full_sample:
        findings.append(
            assistant.generate_hypothesis(
                "شاشة العيّنة الكاملة استكشافية فقط — ليست أساس الاختيار.",
                Evidence(
                    id="vp_search:exploratory_note",
                    source="vp_auction_walk_forward",
                    metric="note",
                    value=0.0,
                    detail="full-sample alpha screen is exploratory",
                ),
                requires_significance=False,
                category="vp_auction_search",
            )
        )
    report = assistant.write_report(
        findings,
        title="Volume Profile / Auction — Walk-Forward Selection",
    )

    return VpAuctionResearchResult.from_unified(
        result,
        signal_columns=_VP_AUCTION_FOCUS,
        fold_df=fold_df,
        oos_ic=oos_ic,
        oos_pvalue=oos_p,
        oos_n=oos_n,
        best_signal=best,
        exploratory_only=exploratory_full_sample,
        report=report,
    )


__all__ = [
    "VpAuctionResearchResult",
    "run_vp_auction_research",
]
