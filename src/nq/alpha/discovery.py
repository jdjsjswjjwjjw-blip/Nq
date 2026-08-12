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
    screen_signals,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.models.splitting import WalkForwardFold, purged_walk_forward_split
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.findings import Finding
from nq.research.progress import ProgressLike
from nq.simulation.execution import (
    depth_matrices_from_frame,
    directional_execution_returns,
    execution_forward_returns,
    execution_forward_returns_depth,
)
from nq.statistics.metrics import information_coefficient

if TYPE_CHECKING:
    from nq.coverage.types import CoverageReport

_DEFAULT_SIGNAL_COLUMNS = ("nq_delta", "mnq_delta", "lead_lag", "trap_setup", "divergence")
_MIN_EVAL_SAMPLES = 8
_MIN_STABILITY_FOLDS = 2
_MIN_SIGN_CONSISTENCY = 2.0 / 3.0
_SPARSE_ACTIVE_RATE = 0.5


@dataclass(frozen=True, slots=True)
class AlphaDiscovery:
    """مخرجات اكتشاف الألفا: تقييمات مفرزة، إشارات مختارة، وتقرير موثّق."""

    evaluations: pl.DataFrame
    selected: list[str]
    report: ResearchReport


def _is_sparse_event_signal(values: np.ndarray) -> bool:
    """هل العمود نبضة اتجاهية نادرة (−1/0/+1) حيث الصفر = لا حدث؟"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    unique = np.unique(finite)
    return bool(
        np.all(np.isin(unique, np.asarray([-1.0, 0.0, 1.0])))
        and float(np.mean(finite != 0.0)) < _SPARSE_ACTIVE_RATE
    )


def _fold_sign_stability(
    values: np.ndarray,
    forward: np.ndarray,
    folds: Sequence[WalkForwardFold],
    *,
    active_only: bool,
) -> tuple[int, float, bool]:
    """ثبات اتجاه IC على طيات OOS بدل الاكتفاء بتجميع قد يخفي الانقلاب."""
    fold_ics: list[float] = []
    for fold in folds:
        v = values[fold.test_idx]
        f = forward[fold.test_idx]
        mask = np.isfinite(v) & np.isfinite(f)
        if active_only:
            mask &= v != 0.0
        v, f = v[mask], f[mask]
        if v.size < _MIN_EVAL_SAMPLES or float(np.std(v)) == 0.0 or float(np.std(f)) == 0.0:
            continue
        fold_ics.append(float(information_coefficient(v, f, method="spearman")))
    if len(fold_ics) < _MIN_STABILITY_FOLDS:
        return len(fold_ics), 0.0, False
    arr = np.asarray(fold_ics, dtype=np.float64)
    aggregate_sign = 1.0 if float(np.mean(arr)) >= 0.0 else -1.0
    consistency = float(np.mean(np.sign(arr) == aggregate_sign))
    return len(fold_ics), consistency, consistency >= _MIN_SIGN_CONSISTENCY


def discover_alpha_from_features(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    signal_columns: Sequence[str],
    price_col: str,
    time_col: str = AVAILABILITY_TS,
    horizon: int = 1,
    n_splits: int = 3,
    embargo: int = 0,
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
    progress: ProgressLike | None = None,
) -> AlphaDiscovery:
    """يقيّم ويفرز إشارات مرشّحة من إطار ميزات، ويكتب تقريرًا موثّقًا."""
    generator = rng if rng is not None else np.random.default_rng(0)
    research = assistant if assistant is not None else ResearchAssistant(alpha=alpha)
    log = progress

    if frame.height == 0:
        if log is not None:
            log.op("ألفا: إطار فارغ — لا إشارات للتقييم")
        empty = screen_signals([], alpha=alpha)
        return AlphaDiscovery(empty, [], research.write_report([], title="Alpha Discovery"))

    cols = list(signal_columns)
    if log is not None:
        log.op(
            f"ألفا: تقييم {len(cols)} إشارة · mode={execution_mode} · "
            f"n_perm={n_permutations} · rows={frame.height:,}"
        )

    evaluations = []
    stability_rows: list[dict[str, float | int | str | bool]] = []
    # تقييم خارج العيّنة فقط (purged walk-forward) — لا IC داخل العيّنة أبدًا.
    times = frame[time_col].to_numpy().astype(np.int64)
    folds: list[WalkForwardFold] = []
    n = int(times.shape[0])
    for splits in range(min(max(int(n_splits), 1), max(n - 1, 1)), 0, -1):
        try:
            candidate = purged_walk_forward_split(
                times,
                n_splits=splits,
                embargo=max(int(embargo), 0),
                purge_samples=max(int(horizon), 0),
                min_train_size=8,
            )
        except ValueError:
            continue
        if candidate:
            folds = candidate
            break
    if not folds:
        if log is not None:
            log.op("ألفا: طيّات غير كافية — رفض التقييم داخل العيّنة")
        empty = screen_signals([], alpha=alpha)
        return AlphaDiscovery(empty, [], research.write_report([], title="Alpha Discovery"))
    oos_idx = np.unique(np.concatenate([f.test_idx for f in folds]))
    if log is not None:
        log.op(f"ألفا: OOS purged · folds={len(folds)} · oos_rows={oos_idx.size:,}")

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
                log.op("ألفا: عوائد أمامية بمسح عمق ظاهر (دخول+خروج)")
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
                progress=log,
            )
        # عوائد كاملة السلسلة ثم قصّ OOS — لا إعادة حساب على شريحة مقطوعة.
        mid = (bid + ask) * 0.5
        mid_fwd = align_forward_returns(mid, horizon=horizon)
        for i, col in enumerate(cols, start=1):
            mode_tag = "depth-walk" if use_depth else "intraday"
            if log is not None:
                log.op(f"ألفا [{i}/{len(cols)}]: تقييم {col!r} ({mode_tag}/OOS)")
            vals = frame[col].to_numpy().astype(np.float64)
            active_only = _is_sparse_event_signal(vals)
            if use_depth and depth_long is not None and depth_short is not None:
                directional = directional_execution_returns(vals, depth_long, depth_short)
            else:
                long_fwd, short_fwd = execution_forward_returns(
                    bid,
                    ask,
                    horizon=horizon,
                    slippage_ticks=slippage_ticks,
                    tick_size=tick_size,
                    commission_bps=commission_bps,
                )
                directional = directional_execution_returns(vals, long_fwd, short_fwd)
            evaluations.append(
                evaluate_signal(
                    col,
                    vals[oos_idx],
                    mid_fwd[oos_idx],
                    strategy_returns=directional[oos_idx],
                    n_permutations=n_permutations,
                    rng=generator,
                    active_only=active_only,
                    progress=log,
                    progress_label=f"ألفا-perm:{col}",
                )
            )
            stable_n, consistency, stable = _fold_sign_stability(
                vals, mid_fwd, folds, active_only=active_only
            )
            stability_rows.append(
                {
                    "name": col,
                    "evaluation_scope": "active_events" if active_only else "all_rows",
                    "active_rate": float(np.mean(np.isfinite(vals) & (vals != 0.0))),
                    "valid_oos_folds": stable_n,
                    "fold_sign_consistency": consistency,
                    "sign_stable": stable,
                }
            )
    else:
        prices = frame[price_col].to_numpy().astype(np.float64)
        forward = align_forward_returns(prices, horizon=horizon)
        for i, col in enumerate(cols, start=1):
            if log is not None:
                log.op(f"ألفا [{i}/{len(cols)}]: تقييم {col!r} (mid/OOS)")
            vals = frame[col].to_numpy().astype(np.float64)
            active_only = _is_sparse_event_signal(vals)
            evaluations.append(
                evaluate_signal(
                    col,
                    vals[oos_idx],
                    forward[oos_idx],
                    n_permutations=n_permutations,
                    rng=generator,
                    active_only=active_only,
                    progress=log,
                    progress_label=f"ألفا-perm:{col}",
                )
            )
            stable_n, consistency, stable = _fold_sign_stability(
                vals, forward, folds, active_only=active_only
            )
            stability_rows.append(
                {
                    "name": col,
                    "evaluation_scope": "active_events" if active_only else "all_rows",
                    "active_rate": float(np.mean(np.isfinite(vals) & (vals != 0.0))),
                    "valid_oos_folds": stable_n,
                    "fold_sign_consistency": consistency,
                    "sign_stable": stable,
                }
            )
    if log is not None:
        log.op("ألفا: فرز/تصحيح تعدّد (screen_signals)")
    screened = screen_signals(evaluations, alpha=alpha)
    if screened.height and stability_rows:
        screened = (
            screened.join(pl.DataFrame(stability_rows), on="name", how="left")
            .rename({"selected": "significance_selected"})
            .with_columns(
                (pl.col("significance_selected") & pl.col("sign_stable")).alias("selected")
            )
            .sort("adjusted_pvalue")
        )

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
