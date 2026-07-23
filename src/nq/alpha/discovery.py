"""اكتشاف الألفا من الميزات وخط البحث الكامل (Alpha Discovery & Pipeline).

يجمع كامل المسار: من إطار الميزات (المُشتق سببيًا من MBO) إلى إشارات مرشّحة،
تقييمها وفرزها إحصائيًا مع تصحيح التعدّد، ثم تقرير بحثي موثّق. كل شيء حتمي
وقابل لإعادة الإنتاج من البيانات الخام.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nq.alpha.signals import (
    ExecutionMode,
    align_forward_returns,
    evaluate_signal,
    evaluate_signal_intraday,
    screen_signals,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.findings import Finding
from nq.simulation.execution import (
    depth_matrices_from_frame,
    directional_execution_returns,
    execution_forward_returns_depth,
)

if TYPE_CHECKING:
    from nq.coverage.types import CoverageReport

_DEFAULT_SIGNAL_COLUMNS = ("nq_delta", "mnq_delta", "lead_lag", "trap_setup", "divergence")


@dataclass(frozen=True, slots=True)
class AlphaDiscovery:
    """مخرجات اكتشاف الألفا: تقييمات مفرزة، إشارات مختارة، وتقرير موثّق."""

    evaluations: pl.DataFrame
    selected: list[str]
    report: ResearchReport


def discover_alpha_from_features(
    frame: pl.DataFrame,
    *,
    signal_columns: Sequence[str],
    price_col: str,
    time_col: str = AVAILABILITY_TS,
    horizon: int = 1,
    execution_mode: ExecutionMode = "mid",
    bid_col: str = "nq_bid",
    ask_col: str = "nq_ask",
    slippage_ticks: float = 0.5,
    tick_size: float = 0.25,
    commission_bps: float = 0.0,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
    assistant: ResearchAssistant | None = None,
    progress: object | None = None,
) -> AlphaDiscovery:
    """يقيّم ويفرز إشارات مرشّحة من إطار ميزات، ويكتب تقريرًا موثّقًا."""
    generator = rng if rng is not None else np.random.default_rng(0)
    research = assistant if assistant is not None else ResearchAssistant(alpha=alpha)
    log = progress

    if frame.height == 0:
        if log is not None:
            log.op("ألفا: إطار فارغ — لا إشارات للتقييم")  # type: ignore[union-attr]
        empty = screen_signals([], alpha=alpha)
        return AlphaDiscovery(empty, [], research.write_report([], title="Alpha Discovery"))

    cols = list(signal_columns)
    if log is not None:
        log.op(  # type: ignore[union-attr]
            f"ألفا: تقييم {len(cols)} إشارة · mode={execution_mode} · "
            f"n_perm={n_permutations} · rows={frame.height:,}"
        )

    evaluations = []
    if execution_mode == "intraday":
        if bid_col not in frame.columns or ask_col not in frame.columns:
            raise ValueError(
                f"intraday execution requires {bid_col!r} and {ask_col!r} in feature frame"
            )
        bid = frame[bid_col].to_numpy().astype(np.float64)
        ask = frame[ask_col].to_numpy().astype(np.float64)
        use_depth = "depth_bid_sz_1" in frame.columns and "depth_ask_sz_1" in frame.columns
        depth_long = depth_short = None
        if use_depth:
            if log is not None:
                log.op("ألفا: عوائد أمامية بمسح عمق ظاهر (دخول+خروج)")  # type: ignore[union-attr]
            bid_px, bid_sz, ask_px, ask_sz = depth_matrices_from_frame(frame, n_levels=5)
            depth_long, depth_short = execution_forward_returns_depth(
                bid_px,
                bid_sz,
                ask_px,
                ask_sz,
                horizon=horizon,
                order_qty=1,
                n_levels=5,
                commission_bps=commission_bps,
                fallback_bid=bid,
                fallback_ask=ask,
                slippage_ticks=slippage_ticks,
                tick_size=tick_size,
            )
        for i, col in enumerate(cols, start=1):
            mode_tag = "depth-walk" if use_depth else "intraday"
            if log is not None:
                log.op(f"ألفا [{i}/{len(cols)}]: تقييم {col!r} ({mode_tag})")  # type: ignore[union-attr]
            if use_depth and depth_long is not None and depth_short is not None:
                directional = directional_execution_returns(
                    frame[col].to_numpy().astype(np.float64),
                    depth_long,
                    depth_short,
                )
                evaluations.append(
                    evaluate_signal(
                        col,
                        frame[col].to_numpy().astype(np.float64),
                        directional,
                        n_permutations=n_permutations,
                        rng=generator,
                    )
                )
            else:
                evaluations.append(
                    evaluate_signal_intraday(
                        col,
                        frame[col].to_numpy().astype(np.float64),
                        bid,
                        ask,
                        horizon=horizon,
                        slippage_ticks=slippage_ticks,
                        tick_size=tick_size,
                        commission_bps=commission_bps,
                        n_permutations=n_permutations,
                        rng=generator,
                    )
                )
    else:
        prices = frame[price_col].to_numpy().astype(np.float64)
        forward = align_forward_returns(prices, horizon=horizon)
        evaluations = []
        for i, col in enumerate(cols, start=1):
            if log is not None:
                log.op(f"ألفا [{i}/{len(cols)}]: تقييم {col!r} (mid)")  # type: ignore[union-attr]
            evaluations.append(
                evaluate_signal(
                    col,
                    frame[col].to_numpy().astype(np.float64),
                    forward,
                    n_permutations=n_permutations,
                    rng=generator,
                )
            )
    if log is not None:
        log.op("ألفا: فرز/تصحيح تعدّد (screen_signals)")  # type: ignore[union-attr]
    screened = screen_signals(evaluations, alpha=alpha)

    findings: list[Finding] = []
    selected: list[str] = []
    for row in screened.filter(pl.col("selected")).iter_rows(named=True):
        selected.append(row["name"])
        evidence = Evidence(
            id=f"alpha:{row['name']}",
            source="alpha_screen",
            metric="IC",
            value=float(row["ic"]),
            pvalue=float(row["adjusted_pvalue"]),
            sample_size=int(row["n"]),
            detail=f"predictive alpha of signal '{row['name']}' (horizon-forward IC)",
        )
        claim = (
            f"إشارة '{row['name']}' تحمل ألفا تنبّئيًا دالًّا "
            f"(IC={row['ic']:.3f}, adj_p={row['adjusted_pvalue']:.4g}, Sharpe={row['sharpe']:.3f})."
        )
        findings.append(research.generate_hypothesis(claim, evidence, category="alpha"))

    if log is not None:
        log.op(  # type: ignore[union-attr]
            f"ألفا: selected={selected!r} · evals={screened.height}"
        )
    report = research.write_report(findings, title="Novel Alpha Signals — Research Report")
    return AlphaDiscovery(evaluations=screened, selected=selected, report=report)


@dataclass(frozen=True, slots=True)
class FullResearchResult:
    """مخرجات الخط البحثي الكامل: تغطية + ألفا."""

    coverage: CoverageReport
    alpha: AlphaDiscovery


def run_full_research_pipeline(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path,
    *,
    interval_ns: int = 1_000_000_000,
    horizon: int = 1,
    signal_columns: Sequence[str] | None = None,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_permutations: int = 2000,
    latency_ns: int = 0,
    lead_lag_window: int = 2,
    coverage_splits: int = 3,
    execution_mode: ExecutionMode = "intraday",
    rng: np.random.Generator | None = None,
) -> FullResearchResult:
    """يُفوِّض إلى الخط الموحّد ويُعيد تغطية + ألفا فقط."""
    from nq.research.orchestrator import PipelineConfig, run_research_pipeline  # noqa: PLC0415

    cfg = PipelineConfig(
        interval_ns=interval_ns,
        horizon=horizon,
        latency_ns=latency_ns,
        lead_lag_window=lead_lag_window,
        coverage_splits=coverage_splits,
        execution_mode=execution_mode,
        alpha=alpha,
        n_permutations=n_permutations,
    )
    result = run_research_pipeline(nq, mnq, config=cfg, rng=rng)
    return FullResearchResult(coverage=result.coverage, alpha=result.alpha)


def run_research_pipeline(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path,
    *,
    interval_ns: int = 1_000_000_000,
    horizon: int = 1,
    signal_columns: Sequence[str] | None = None,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_permutations: int = 2000,
    latency_ns: int = 0,
    lead_lag_window: int = 2,
    execution_mode: ExecutionMode = "intraday",
    slippage_ticks: float = 0.5,
    tick_size: float = 0.25,
    commission_bps: float = 0.0,
    rng: np.random.Generator | None = None,
    quiet: bool = True,
) -> AlphaDiscovery:
    """اختصار للخط الموحّد — يُعيد قناة الألفا فقط (للتوافق مع الاختبارات)."""
    from nq.research.orchestrator import PipelineConfig, run_research_pipeline  # noqa: PLC0415

    cfg = PipelineConfig(
        interval_ns=interval_ns,
        horizon=horizon,
        latency_ns=latency_ns,
        lead_lag_window=lead_lag_window,
        execution_mode=execution_mode,
        alpha=alpha,
        n_permutations=n_permutations,
        slippage_ticks=slippage_ticks,
        tick_size=tick_size,
        commission_bps=commission_bps,
        quiet=quiet,
    )
    return run_research_pipeline(
        nq,
        mnq,
        config=cfg,
        signal_columns=signal_columns,
        price_col=price_col,
        rng=rng,
    ).alpha
