"""اكتشاف الألفا من الميزات وخط البحث الكامل (Alpha Discovery & Pipeline).

يجمع كامل المسار: من إطار الميزات (المُشتق سببيًا من MBO) إلى إشارات مرشّحة،
تقييمها وفرزها إحصائيًا مع تصحيح التعدّد، ثم تقرير بحثي موثّق. كل شيء حتمي
وقابل لإعادة الإنتاج من البيانات الخام.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import polars as pl

from nq.alpha.signals import (
    ExecutionMode,
    SignalEvaluation,
    align_forward_returns,
    evaluate_signal,
    evaluate_signal_intraday,
    screen_signals,
)
from nq.contracts.instruments import NQ_METADATA
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
_ALPHA_ROW_INDEX = "_alpha_row"
_MIN_HOLDOUT_LABELS = 3


class _ProgressLike(Protocol):
    def op(self, message: str) -> None: ...


@dataclass(frozen=True, slots=True)
class AlphaDiscovery:
    """مخرجات اكتشاف الألفا: تقييمات مفرزة، إشارات مختارة، وتقرير موثّق."""

    evaluations: pl.DataFrame
    selected: list[str]
    report: ResearchReport


def _take_rows(frame: pl.DataFrame, idx: np.ndarray) -> pl.DataFrame:
    if idx.size == 0:
        return frame.head(0)
    return (
        frame.with_row_index(_ALPHA_ROW_INDEX)
        .filter(pl.col(_ALPHA_ROW_INDEX).is_in(idx.astype(int).tolist()))
        .drop(_ALPHA_ROW_INDEX)
    )


def _temporal_holdout_indices(
    frame: pl.DataFrame,
    *,
    time_col: str,
    horizon: int,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_stop = frame.height - horizon
    if label_stop < _MIN_HOLDOUT_LABELS:
        empty = np.asarray([], dtype=np.intp)
        return empty, empty, empty

    train_end = label_stop // 3
    val_end = (2 * label_stop) // 3
    if train_end < 1 or val_end <= train_end or val_end >= label_stop:
        empty = np.asarray([], dtype=np.intp)
        return empty, empty, empty

    all_idx = np.arange(label_stop, dtype=np.intp)
    train_idx = all_idx[:train_end]
    validation_idx = all_idx[train_end:val_end]
    final_idx = all_idx[val_end:label_stop]

    # Remove any row whose forward label would cross into the next selection stage.
    train_idx = train_idx[train_idx + horizon < train_end]
    validation_idx = validation_idx[validation_idx + horizon < val_end]

    if embargo > 0 and time_col in frame.columns:
        times = frame[time_col].to_numpy().astype(np.int64)
        train_cutoff = int(times[train_end]) - embargo
        validation_cutoff = int(times[val_end]) - embargo
        train_idx = train_idx[times[train_idx] <= train_cutoff]
        validation_idx = validation_idx[times[validation_idx] <= validation_cutoff]

    return train_idx, validation_idx, final_idx


def _evaluate_alpha_candidates(
    frame: pl.DataFrame,
    *,
    cols: Sequence[str],
    price_col: str,
    horizon: int,
    execution_mode: ExecutionMode,
    bid_col: str,
    ask_col: str,
    slippage_ticks: float,
    tick_size: float,
    commission_bps: float,
    n_permutations: int,
    generator: np.random.Generator,
    log: _ProgressLike | None = None,
    stage: str = "",
) -> list[SignalEvaluation]:
    evaluations: list[SignalEvaluation] = []
    if frame.height == 0 or not cols:
        return evaluations
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
                log.op(f"ألفا {stage}: عوائد أمامية بمسح عمق ظاهر")
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
            if log is not None:
                log.op(f"ألفا [{i}/{len(cols)}] {stage}: {col!r}")
            values = frame[col].to_numpy().astype(np.float64)
            if use_depth and depth_long is not None and depth_short is not None:
                directional = directional_execution_returns(values, depth_long, depth_short)
                evaluations.append(
                    evaluate_signal(
                        col,
                        values,
                        directional,
                        n_permutations=n_permutations,
                        rng=generator,
                    )
                )
            else:
                evaluations.append(
                    evaluate_signal_intraday(
                        col,
                        values,
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
        return evaluations

    prices = frame[price_col].to_numpy().astype(np.float64)
    forward = align_forward_returns(prices, horizon=horizon)
    for i, col in enumerate(cols, start=1):
        if log is not None:
            log.op(f"ألفا [{i}/{len(cols)}] {stage}: {col!r} (mid)")
        evaluations.append(
            evaluate_signal(
                col,
                frame[col].to_numpy().astype(np.float64),
                forward,
                n_permutations=n_permutations,
                rng=generator,
            )
        )
    return evaluations


def _prefix_screen(screened: pl.DataFrame, prefix: str) -> pl.DataFrame:
    if screened.height == 0:
        return pl.DataFrame(schema={"name": pl.Utf8()})
    return screened.select(
        "name",
        pl.col("n").alias(f"{prefix}_n"),
        pl.col("ic").alias(f"{prefix}_ic"),
        pl.col("ic_pvalue").alias(f"{prefix}_ic_pvalue"),
        pl.col("adjusted_pvalue").alias(f"{prefix}_adjusted_pvalue"),
        pl.col("sharpe").alias(f"{prefix}_sharpe"),
        pl.col("selected").alias(f"{prefix}_selected"),
    )


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
    tick_size: float = NQ_METADATA.tick_size,
    commission_bps: float = 0.0,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
    assistant: ResearchAssistant | None = None,
    progress: _ProgressLike | None = None,
) -> AlphaDiscovery:
    """يكتشف الإشارات عبر train/validation/final-OOS دون اختيار على العينة النهائية."""
    generator = rng if rng is not None else np.random.default_rng(0)
    research = assistant if assistant is not None else ResearchAssistant(alpha=alpha)
    log = progress

    if frame.height == 0:
        if log is not None:
            log.op("ألفا: إطار فارغ — لا إشارات للتقييم")
        empty = screen_signals([], alpha=alpha)
        return AlphaDiscovery(empty, [], research.write_report([], title="Alpha Discovery"))

    if time_col not in frame.columns:
        raise ValueError(f"time column {time_col!r} not found")
    frame = frame.sort(time_col)
    cols = [c for c in signal_columns if c in frame.columns]
    if log is not None:
        log.op(
            f"ألفا: train/validation/final-OOS · signals={len(cols)} · mode={execution_mode} · "
            f"n_perm={n_permutations} · rows={frame.height:,}"
        )

    train_idx, validation_idx, final_idx = _temporal_holdout_indices(
        frame,
        time_col=time_col,
        horizon=horizon,
    )
    train_frame = _take_rows(frame, train_idx)
    validation_frame = _take_rows(frame, validation_idx)
    final_frame = _take_rows(frame, final_idx)
    if log is not None and (
        train_frame.height == 0 or validation_frame.height == 0 or final_frame.height == 0
    ):
        for i, col in enumerate(cols, start=1):
            log.op(f"ألفا [{i}/{len(cols)}] skipped: insufficient temporal holdout for {col!r}")

    train_screened = screen_signals(
        _evaluate_alpha_candidates(
            train_frame,
            cols=cols,
            price_col=price_col,
            horizon=horizon,
            execution_mode=execution_mode,
            bid_col=bid_col,
            ask_col=ask_col,
            slippage_ticks=slippage_ticks,
            tick_size=tick_size,
            commission_bps=commission_bps,
            n_permutations=n_permutations,
            generator=generator,
            log=log,
            stage="train",
        ),
        alpha=alpha,
    )
    train_candidates = train_screened.filter(pl.col("selected"))["name"].to_list()
    validation_screened = screen_signals(
        _evaluate_alpha_candidates(
            validation_frame,
            cols=train_candidates,
            price_col=price_col,
            horizon=horizon,
            execution_mode=execution_mode,
            bid_col=bid_col,
            ask_col=ask_col,
            slippage_ticks=slippage_ticks,
            tick_size=tick_size,
            commission_bps=commission_bps,
            n_permutations=n_permutations,
            generator=generator,
            log=log,
            stage="validation",
        ),
        alpha=alpha,
    )
    selected = validation_screened.filter(pl.col("selected"))["name"].to_list()
    final_screened = screen_signals(
        _evaluate_alpha_candidates(
            final_frame,
            cols=selected,
            price_col=price_col,
            horizon=horizon,
            execution_mode=execution_mode,
            bid_col=bid_col,
            ask_col=ask_col,
            slippage_ticks=slippage_ticks,
            tick_size=tick_size,
            commission_bps=commission_bps,
            n_permutations=n_permutations,
            generator=generator,
            log=log,
            stage="final_oos",
        ),
        alpha=alpha,
    )
    if log is not None:
        log.op("ألفا: final-OOS evaluation only; selection frozen from validation")

    if final_screened.height > 0:
        screened = final_screened.with_columns(pl.col("name").is_in(selected).alias("selected"))
        screened = screened.join(_prefix_screen(train_screened, "train"), on="name", how="left")
        screened = screened.join(
            _prefix_screen(validation_screened, "validation"),
            on="name",
            how="left",
        )
    else:
        screened = final_screened

    findings: list[Finding] = []
    for row in screened.filter(pl.col("selected")).iter_rows(named=True):
        evidence = Evidence(
            id=f"alpha:{row['name']}",
            source="alpha_final_oos",
            metric="IC",
            value=float(row["ic"]),
            pvalue=float(row["adjusted_pvalue"]),
            sample_size=int(row["n"]),
            detail=(
                f"signal '{row['name']}' selected on train/validation only; "
                "IC reported on final OOS"
            ),
        )
        claim = (
            f"إشارة '{row['name']}' مختارة زمنيًا ثم مقيمة على final-OOS "
            f"(IC={row['ic']:.3f}, adj_p={row['adjusted_pvalue']:.4g}, Sharpe={row['sharpe']:.3f})."
        )
        findings.append(research.generate_hypothesis(claim, evidence, category="alpha"))

    if log is not None:
        log.op(f"ألفا: selected={selected!r} · evals={screened.height}")
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
    tick_size: float = NQ_METADATA.tick_size,
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
