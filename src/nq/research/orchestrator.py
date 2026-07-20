"""منسّق البحث الموحّد — خط واحد من MBO إلى التقرير.

``run_research_pipeline`` نقطة الدخول الوحيدة:

1. تحميل NQ/MNQ (Parquet/Arrow/Databento أو إطار جاهز).
2. بناء الميزات (cross-market + session + latency).
3. تشغيل SSL + M9 (بالتوازي) + ألفا intraday.
4. دمج التقرير الموحّد وحفظه اختياريًا.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from nq.alpha.signals import ExecutionMode
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import seed_everything
from nq.core.temporal_policy import TemporalPolicy
from nq.coverage.monitor import run_coverage_on_features
from nq.coverage.types import CoverageReport
from nq.ingestion.reader import load_mbo_frame
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_pipeline, run_ssl_tick_pipeline
from nq.research.assistant import LanguageModel, ResearchAssistant
from nq.research.unified import UnifiedResearchReport, build_unified_report
from nq.simulation.cross_market import cross_market_features

if TYPE_CHECKING:
    from nq.alpha.discovery import AlphaDiscovery

SslMode = Literal["bucket", "tick"]
CrossMarketMode = Literal["dual", "nq_only"]

_DEFAULT_SIGNAL_COLUMNS = (
    "nq_delta",
    "mnq_delta",
    "lead_lag",
    "trap_setup",
    "divergence",
    "session_phase",
)


@dataclass(frozen=True, slots=True)
class UnifiedResearchResult:
    """مخرجات الخط الكامل: SSL + M9 + ألفا + تقرير موحّد."""

    features: pl.DataFrame
    ssl: SSLPipelineResult
    coverage: CoverageReport
    alpha: AlphaDiscovery
    report: UnifiedResearchReport


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """إعدادات الخط الموحّد — تُقرأ من TOML أو تُمرَّر يدويًا."""

    interval_ns: int = 1_000_000_000
    horizon: int = 1
    latency_ns: int = 0
    lead_lag_window: int = 2
    ssl_window: int = 5
    ssl_components: int = 4
    coverage_splits: int = 3
    coverage_embargo: int | None = None
    execution_mode: ExecutionMode = "intraday"
    slippage_ticks: float = 0.5
    tick_size: float = 0.25
    commission_bps: float = 0.0
    alpha: float = 0.05
    n_permutations: int = 2000
    global_seed: int = 0
    parallel_coverage: bool = True
    ssl_mode: SslMode = "tick"
    cross_market_mode: CrossMarketMode = "dual"
    max_rows: int | None = None

    @classmethod
    def from_toml(cls, path: Path | str) -> PipelineConfig:
        config_path = Path(path)
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        temporal = raw.get("temporal", {})
        cross = raw.get("cross_market", {})
        exec_cfg = raw.get("execution", {})
        ssl = raw.get("ssl", {})
        data = raw.get("data", {})
        det = raw.get("determinism", {})
        max_rows_raw = data.get("max_rows")
        max_rows = None if max_rows_raw in (None, 0) else int(max_rows_raw)
        return cls(
            interval_ns=int(temporal.get("interval_ns", 1_000_000_000)),
            horizon=int(temporal.get("horizon", 1)),
            latency_ns=int(cross.get("latency_ns", 0)),
            lead_lag_window=int(cross.get("lead_lag_window", 2)),
            ssl_window=int(ssl.get("window", 5)),
            ssl_components=int(ssl.get("n_components", 4)),
            coverage_splits=int(ssl.get("n_splits", 3)),
            execution_mode=str(exec_cfg.get("mode", "intraday")),  # type: ignore[arg-type]
            slippage_ticks=float(exec_cfg.get("slippage_ticks", 0.5)),
            tick_size=float(exec_cfg.get("tick_size", 0.25)),
            commission_bps=float(exec_cfg.get("commission_bps", 0.0)),
            alpha=float(raw.get("statistics", {}).get("alpha", 0.05)),
            n_permutations=int(raw.get("statistics", {}).get("n_permutations", 2000)),
            global_seed=int(det.get("global_seed", 0)),
            ssl_mode=str(ssl.get("mode", "tick")),  # type: ignore[arg-type]
            cross_market_mode=str(data.get("cross_market_mode", "dual")),  # type: ignore[arg-type]
            max_rows=max_rows,
        )


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


def _resolve_signal_columns(
    features: pl.DataFrame,
    signal_columns: Sequence[str] | None,
) -> list[str]:
    if signal_columns is not None:
        return list(signal_columns)
    return [c for c in _DEFAULT_SIGNAL_COLUMNS if c in features.columns]


def run_ssl_research_pipeline(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    features: pl.DataFrame,
    *,
    interval_ns: int,
    horizon: int = 1,
    signal_columns: Sequence[str] | None = None,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_permutations: int = 2000,
    ssl_window: int = 5,
    ssl_components: int = 4,
    coverage_splits: int = 3,
    coverage_embargo: int | None = None,
    execution_mode: ExecutionMode = "intraday",
    slippage_ticks: float = 0.5,
    tick_size: float = 0.25,
    commission_bps: float = 0.0,
    parallel_coverage: bool = True,
    ssl_mode: SslMode = "tick",
    language_model: LanguageModel | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SSLPipelineResult, CoverageReport, AlphaDiscovery, UnifiedResearchReport]:
    """يشغّل SSL + M9 (خلفية) + ألفا → تقرير شامل (الميزات مُبنية مسبقًا)."""
    from nq.alpha.discovery import discover_alpha_from_features  # noqa: PLC0415

    generator = rng if rng is not None else np.random.default_rng(0)
    seed = int(generator.integers(0, 2**31))

    policy = TemporalPolicy.for_run(interval_ns=interval_ns, window=ssl_window)
    embargo_val = (
        coverage_embargo
        if coverage_embargo is not None
        else policy.embargo_time_units(interval_ns=interval_ns)
    )
    purge_val = policy.purge_samples()
    columns = _resolve_signal_columns(features, signal_columns)

    ssl_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)
    alpha_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)

    def _run_ssl() -> SSLPipelineResult:
        if ssl_mode == "tick":
            return run_ssl_tick_pipeline(
                nq,
                mnq,
                window=ssl_window,
                n_components=ssl_components,
                n_splits=coverage_splits,
                embargo=embargo_val,
                purge_samples=purge_val,
                alpha=alpha,
                rng=generator,
                assistant=ssl_assistant,
            )
        return run_ssl_pipeline(
            features,
            feature_columns=columns or None,
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

    if parallel_coverage and (features.height > 0 or ssl_mode == "tick"):
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
            ssl_result = _run_ssl()
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
        ssl_result = _run_ssl()
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
    return ssl_result, coverage_result, alpha_result, unified


def _resolve_pipeline_config(
    config: PipelineConfig | None,
    config_path: Path | str | None,
    *,
    interval_ns: int | None,
    latency_ns: int | None,
    horizon: int | None,
    execution_mode: ExecutionMode | None,
    parallel_coverage: bool | None,
    n_permutations: int | None,
) -> PipelineConfig:
    cfg = config
    if cfg is None:
        path = config_path if config_path is not None else Path("configs/research.toml")
        cfg = PipelineConfig.from_toml(path) if Path(path).is_file() else PipelineConfig()
    if interval_ns is not None:
        cfg = replace(cfg, interval_ns=interval_ns)
    if latency_ns is not None:
        cfg = replace(cfg, latency_ns=latency_ns)
    if horizon is not None:
        cfg = replace(cfg, horizon=horizon)
    if execution_mode is not None:
        cfg = replace(cfg, execution_mode=execution_mode)
    if parallel_coverage is not None:
        cfg = replace(cfg, parallel_coverage=parallel_coverage)
    if n_permutations is not None:
        cfg = replace(cfg, n_permutations=n_permutations)
    return cfg


def _load_pipeline_frames(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path,
    cfg: PipelineConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """يُحمّل NQ/MNQ مع دعم nq_only و max_rows."""
    nq_frame = nq if isinstance(nq, pl.DataFrame) else load_mbo_frame(nq, max_rows=cfg.max_rows)
    if cfg.cross_market_mode == "nq_only":
        return nq_frame, nq_frame
    mnq_frame = (
        mnq if isinstance(mnq, pl.DataFrame) else load_mbo_frame(mnq, max_rows=cfg.max_rows)
    )
    return nq_frame, mnq_frame


def run_research_pipeline(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path,
    *,
    config: PipelineConfig | None = None,
    config_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    interval_ns: int | None = None,
    latency_ns: int | None = None,
    horizon: int | None = None,
    signal_columns: Sequence[str] | None = None,
    price_col: str = "nq_close",
    execution_mode: ExecutionMode | None = None,
    parallel_coverage: bool | None = None,
    n_permutations: int | None = None,
    language_model: LanguageModel | None = None,
    rng: np.random.Generator | None = None,
) -> UnifiedResearchResult:
    """الخط الموحّد: تحميل MBO → ميزات → SSL‖M9 → ألفا → تقرير.

    Parameters
    ----------
    nq, mnq:
        إطار Polars جاهز أو مسار ملف (Parquet/Arrow/Databento).
    config / config_path:
        إعدادات من ``configs/research.toml`` أو كائن ``PipelineConfig``.
    output_dir:
        عند التحديد يُحفظ ``report.md`` والمقاييس في هذا المجلد.
    """
    cfg = _resolve_pipeline_config(
        config,
        config_path,
        interval_ns=interval_ns,
        latency_ns=latency_ns,
        horizon=horizon,
        execution_mode=execution_mode,
        parallel_coverage=parallel_coverage,
        n_permutations=n_permutations,
    )

    seed_everything(cfg.global_seed)
    generator = rng if rng is not None else np.random.default_rng(cfg.global_seed)

    nq_frame, mnq_frame = _load_pipeline_frames(nq, mnq, cfg)

    features = cross_market_features(
        nq_frame,
        mnq_frame,
        interval_ns=cfg.interval_ns,
        lead_lag_window=cfg.lead_lag_window,
        latency_ns=cfg.latency_ns,
    )

    ssl_result, coverage_result, alpha_result, unified = run_ssl_research_pipeline(
        nq_frame,
        mnq_frame,
        features,
        interval_ns=cfg.interval_ns,
        horizon=cfg.horizon,
        signal_columns=signal_columns,
        price_col=price_col,
        alpha=cfg.alpha,
        n_permutations=cfg.n_permutations,
        ssl_window=cfg.ssl_window,
        ssl_components=cfg.ssl_components,
        coverage_splits=cfg.coverage_splits,
        coverage_embargo=cfg.coverage_embargo,
        execution_mode=cfg.execution_mode,
        ssl_mode=cfg.ssl_mode,
        slippage_ticks=cfg.slippage_ticks,
        tick_size=cfg.tick_size,
        commission_bps=cfg.commission_bps,
        parallel_coverage=cfg.parallel_coverage,
        language_model=language_model,
        rng=generator,
    )

    result = UnifiedResearchResult(
        features=features,
        ssl=ssl_result,
        coverage=coverage_result,
        alpha=alpha_result,
        report=unified,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(unified.to_markdown(), encoding="utf-8")
        if ssl_result.metrics.height > 0:
            ssl_result.metrics.write_parquet(out / "ssl_metrics.parquet")
        if coverage_result.metrics.height > 0:
            coverage_result.metrics.write_parquet(out / "coverage_metrics.parquet")
        if alpha_result.evaluations.height > 0:
            alpha_result.evaluations.write_parquet(out / "alpha_evaluations.parquet")
        features.write_parquet(out / "features.parquet")

    return result


__all__ = [
    "PipelineConfig",
    "UnifiedResearchResult",
    "run_research_pipeline",
    "run_ssl_research_pipeline",
]
