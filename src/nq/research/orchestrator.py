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
from nq.contracts.instruments import NQ_METADATA
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import seed_everything
from nq.core.temporal_policy import TemporalPolicy
from nq.coverage.monitor import run_coverage_on_features
from nq.coverage.types import CoverageReport
from nq.features.streaming import (
    STREAMING_SIGNAL_COLUMNS,
    build_streaming_research_features,
)
from nq.ingestion.reader import load_mbo_frame
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_pipeline, run_ssl_tick_pipeline
from nq.research.assistant import LanguageModel, ResearchAssistant
from nq.research.progress import PipelineProgress, resolve_progress
from nq.research.unified import UnifiedResearchReport, build_unified_report
from nq.simulation.auction import auction_signal_frame
from nq.simulation.breakout import failed_breakout_features
from nq.simulation.cross_market import cross_market_features, single_market_features
from nq.simulation.depth_lifecycle import attach_depth_asof, depth_at_bar_close
from nq.simulation.fvg import failed_fvg_features

if TYPE_CHECKING:
    from nq.alpha.discovery import AlphaDiscovery

SslMode = Literal["bucket", "tick"]
CrossMarketMode = Literal["dual", "nq_only"]
FeatureMode = Literal["streaming", "batch"]

_DEFAULT_SIGNAL_COLUMNS = (
    "nq_delta",
    "mnq_delta",
    "trap_setup",
    "phase_balance",
    "phase_expansion",
    "in_value_area",
    "near_vah",
    "poc_dist_norm",
    "session_phase",
    "fail_fvg",
    "fail_breakout",
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_flip_to_imbalance",
)

_BATCH_SIGNAL_COLUMNS = (
    "nq_delta",
    "mnq_delta",
    "lead_lag",
    "trap_setup",
    "divergence",
    "session_phase",
    "fail_fvg",
    "fail_breakout",
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_flip_to_imbalance",
)

_VP_AUCTION_SIGNAL_COLUMNS = (
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_in_value_frac",
    "vp_pullback_defense",
    "vp_poc_migration",
    "vp_flip_to_imbalance",
)

_FB_SIGNAL_COLUMNS = (
    "fail_breakout",
    "fb_break_level",
    "fb_entry_ref",
    "fb_effort_range_ratio",
    "fb_effort_volume_ratio",
    "fb_effort_result_ratio",
    "fb_bar_volume",
    "fb_cum_volume",
    "fb_delta",
    "fb_cum_delta",
    "fb_vol_imbalance",
    "fb_absorption",
    "fb_risk_pts",
    "fb_depth_at_break",
    "fb_depth_imbalance",
    "fb_depth_cum_bid",
    "fb_depth_cum_ask",
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
    tick_size: float = NQ_METADATA.tick_size
    commission_bps: float = 0.0
    alpha: float = 0.05
    n_permutations: int = 2000
    global_seed: int = 0
    parallel_coverage: bool = True
    ssl_mode: SslMode = "tick"
    cross_market_mode: CrossMarketMode = "dual"
    max_rows: int | None = None
    include_failed_fvg: bool = True
    include_auction_vp: bool = True
    include_failed_breakout: bool = True
    feature_mode: FeatureMode = "streaming"
    signal_columns: tuple[str, ...] | None = None
    quiet: bool = False

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
        signals = raw.get("signals", {})
        features_cfg = raw.get("features", {})
        det = raw.get("determinism", {})
        run_cfg = raw.get("run", {})
        max_rows_raw = data.get("max_rows")
        max_rows = None if max_rows_raw in (None, 0) else int(max_rows_raw)
        signal_cols = signals.get("columns")
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
            tick_size=float(exec_cfg.get("tick_size", NQ_METADATA.tick_size)),
            commission_bps=float(exec_cfg.get("commission_bps", 0.0)),
            alpha=float(raw.get("statistics", {}).get("alpha", 0.05)),
            n_permutations=int(raw.get("statistics", {}).get("n_permutations", 2000)),
            global_seed=int(det.get("global_seed", 0)),
            ssl_mode=str(ssl.get("mode", "tick")),  # type: ignore[arg-type]
            cross_market_mode=str(data.get("cross_market_mode", "dual")),  # type: ignore[arg-type]
            max_rows=max_rows,
            include_failed_fvg=bool(signals.get("include_failed_fvg", True)),
            include_auction_vp=bool(signals.get("include_auction_vp", True)),
            include_failed_breakout=bool(signals.get("include_failed_breakout", True)),
            feature_mode=str(features_cfg.get("mode", signals.get("feature_mode", "streaming"))),  # type: ignore[arg-type]
            signal_columns=tuple(signal_cols) if signal_cols else None,
            quiet=bool(run_cfg.get("quiet", False)),
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
    progress: PipelineProgress | None = None,
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
        progress=progress,
    )


def _resolve_signal_columns(
    features: pl.DataFrame,
    signal_columns: Sequence[str] | None,
    *,
    config_columns: Sequence[str] | None = None,
) -> list[str]:
    if signal_columns is not None:
        return [c for c in signal_columns if c in features.columns]
    if config_columns is not None:
        return [c for c in config_columns if c in features.columns]
    ordered = list(
        dict.fromkeys([*_DEFAULT_SIGNAL_COLUMNS, *_BATCH_SIGNAL_COLUMNS, *STREAMING_SIGNAL_COLUMNS])
    )
    return [c for c in ordered if c in features.columns]


def _attach_failed_fvg(features: pl.DataFrame, nq: pl.DataFrame) -> pl.DataFrame:
    """يلحق إشارة Failed FVG بإطار البحث الموحّد (asof خلفي — بلا تسريب)."""
    fvg = failed_fvg_features(nq)
    if fvg.height == 0 or features.height == 0:
        return features.with_columns(
            pl.lit(0.0).alias("fail_fvg"),
            pl.lit(0.0).alias("effort_range_ratio"),
            pl.lit(0.0).alias("effort_volume_ratio"),
        )
    keep = [
        c
        for c in (
            AVAILABILITY_TS,
            "fail_fvg",
            "effort_range_ratio",
            "effort_volume_ratio",
        )
        if c in fvg.columns
    ]
    right = fvg.select(keep).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    # تجنّب تضارب أعمدة إن وُجدت سابقًا
    drop_existing = [c for c in keep if c != AVAILABILITY_TS and c in left.columns]
    if drop_existing:
        left = left.drop(drop_existing)
    joined = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
    return joined.with_columns(
        pl.col("fail_fvg").fill_null(0.0),
        pl.col("effort_range_ratio").fill_null(0.0),
        pl.col("effort_volume_ratio").fill_null(0.0),
    )


def _attach_auction_vp(
    features: pl.DataFrame,
    nq: pl.DataFrame,
    *,
    interval_ns: int,
) -> pl.DataFrame:
    """يلحق إشارات Volume Profile / المزاد (توازن·اختلال·تمدّد) asof خلفي."""
    signals = auction_signal_frame(nq, interval_ns=interval_ns)
    zero_exprs = [pl.lit(0.0).alias(c) for c in _VP_AUCTION_SIGNAL_COLUMNS]
    if signals.height == 0 or features.height == 0:
        return features.with_columns(zero_exprs)

    keep = [c for c in (AVAILABILITY_TS, *_VP_AUCTION_SIGNAL_COLUMNS) if c in signals.columns]
    right = signals.select(keep).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop_existing = [c for c in keep if c != AVAILABILITY_TS and c in left.columns]
    if drop_existing:
        left = left.drop(drop_existing)
    joined = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
    fills = [pl.col(c).fill_null(0.0) for c in _VP_AUCTION_SIGNAL_COLUMNS if c in joined.columns]
    return joined.with_columns(fills) if fills else joined


def _attach_failed_breakout(features: pl.DataFrame, nq: pl.DataFrame) -> pl.DataFrame:
    """يلحق Failed Breakout asof خلفي — إشارة + عمق عند مستوى الكسر (سببي)."""
    from nq.contracts.mbo import PRICE_SCALE  # noqa: PLC0415

    fb = failed_breakout_features(nq)
    zero_exprs = [pl.lit(0.0).alias(c) for c in _FB_SIGNAL_COLUMNS]
    if fb.height == 0 or features.height == 0:
        return features.with_columns(zero_exprs)

    # عمق عند إغلاق شمعة الإشارة (30m) — لا طمس السلم
    depth = depth_at_bar_close(nq, interval_ns=30 * 60 * 1_000_000_000, n_levels=5)
    if depth.height > 0:
        fb = attach_depth_asof(
            fb,
            depth,
            columns=[
                "depth_cum_bid",
                "depth_cum_ask",
                "depth_imbalance",
                "depth_bid_px_1",
                "depth_bid_sz_1",
                "depth_ask_px_1",
                "depth_ask_sz_1",
                "depth_bid_px_2",
                "depth_bid_sz_2",
                "depth_ask_px_2",
                "depth_ask_sz_2",
                "depth_bid_px_3",
                "depth_bid_sz_3",
                "depth_ask_px_3",
                "depth_ask_sz_3",
                "depth_bid_px_4",
                "depth_bid_sz_4",
                "depth_ask_px_4",
                "depth_ask_sz_4",
                "depth_bid_px_5",
                "depth_bid_sz_5",
                "depth_ask_px_5",
                "depth_ask_sz_5",
            ],
        )
        # سيولة ظاهرة عند مستوى الكسر: نبحث أقرب مستوى ضمن السلم
        levels_bid_px = [f"depth_bid_px_{k}" for k in range(1, 6)]
        levels_bid_sz = [f"depth_bid_sz_{k}" for k in range(1, 6)]
        levels_ask_px = [f"depth_ask_px_{k}" for k in range(1, 6)]
        levels_ask_sz = [f"depth_ask_sz_{k}" for k in range(1, 6)]

        def _depth_at_break(row: dict) -> float:
            level = float(row.get("fb_break_level") or 0.0)
            signal = float(row.get("fail_breakout") or 0.0)
            if level <= 0 or signal == 0.0:
                return 0.0
            # SHORT بعد فشل كسر أعلى → سيولة عروض عند المستوى؛ LONG → طلبات
            px_cols = levels_ask_px if signal < 0 else levels_bid_px
            sz_cols = levels_ask_sz if signal < 0 else levels_bid_sz
            best = 0.0
            best_dist = float("inf")
            for px_c, sz_c in zip(px_cols, sz_cols, strict=True):
                px = row.get(px_c)
                sz = row.get(sz_c)
                if px is None or sz is None:
                    continue
                dist = abs(float(px) - level)
                if dist < best_dist:
                    best_dist = dist
                    best = float(sz)
            # تطابق ضمن تيك تقريبًا
            if best_dist <= max(PRICE_SCALE * 4, 1e-6):
                return best
            return 0.0

        at_break = [_depth_at_break(r) for r in fb.iter_rows(named=True)]
        fb = fb.with_columns(
            pl.Series("fb_depth_at_break", at_break),
            pl.col("depth_imbalance").fill_null(0.0).alias("fb_depth_imbalance"),
            pl.col("depth_cum_bid").fill_null(0.0).alias("fb_depth_cum_bid"),
            pl.col("depth_cum_ask").fill_null(0.0).alias("fb_depth_cum_ask"),
        )
    else:
        fb = fb.with_columns(
            pl.lit(0.0).alias("fb_depth_at_break"),
            pl.lit(0.0).alias("fb_depth_imbalance"),
            pl.lit(0.0).alias("fb_depth_cum_bid"),
            pl.lit(0.0).alias("fb_depth_cum_ask"),
        )

    keep = [c for c in (AVAILABILITY_TS, *_FB_SIGNAL_COLUMNS) if c in fb.columns]
    right = fb.select(keep).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop_existing = [c for c in keep if c != AVAILABILITY_TS and c in left.columns]
    if drop_existing:
        left = left.drop(drop_existing)
    joined = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
    fills = [pl.col(c).fill_null(0.0) for c in _FB_SIGNAL_COLUMNS if c in joined.columns]
    return joined.with_columns(fills) if fills else joined


def _attach_causal_depth(
    features: pl.DataFrame,
    nq: pl.DataFrame,
    *,
    interval_ns: int,
    progress: PipelineProgress | None = None,
) -> pl.DataFrame:
    """يلحق سلم عمق NQ عند إغلاق كل فاصل بحثي (مراقبة + تنفيذ/خروج)."""
    log = progress if progress is not None else PipelineProgress(enabled=False)
    log.op(f"depth_at_bar_close levels=5 · interval_ns={interval_ns}")
    depth = depth_at_bar_close(nq, interval_ns=interval_ns, n_levels=5)
    if depth.height == 0:
        log.op("عمق: لا لقطات — تخطّي")
        return features
    # لا تستبدل nq_bid/nq_ask إن وُجدت من streaming؛ أبقِ أعمدة depth_*
    cols = [c for c in depth.columns if c.startswith("depth_")]
    # إن لم توجد عروض L1 في الإطار، ألحقها من لقطة العمق
    if "nq_bid" not in features.columns and "nq_bid" in depth.columns:
        cols = [*cols, "nq_bid", "nq_ask"]
    out = attach_depth_asof(features, depth, columns=cols)
    log.op(f"عمق مُلحق: {len(cols)} عمود · rows={out.height:,}")
    return out


def _build_research_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    cfg: PipelineConfig,
    *,
    progress: PipelineProgress | None = None,
) -> pl.DataFrame:
    """يبني إطار البحث: streaming (افتراضي) أو batch، ثم FVG/Auction asof."""
    log = progress if progress is not None else PipelineProgress(enabled=False)
    if cfg.feature_mode == "streaming":
        if cfg.cross_market_mode == "nq_only":
            log.step(
                "بناء الميزات (NQ-only)",
                f"NQ={nq.height:,} · interval_ns={cfg.interval_ns}",
            )
            features = single_market_features(nq, interval_ns=cfg.interval_ns)
        else:
            log.step(
                "بناء الميزات (streaming state-machine)",
                f"NQ={nq.height:,} · MNQ={mnq.height:,} · interval_ns={cfg.interval_ns}",
            )
            features = build_streaming_research_features(
                nq,
                mnq,
                interval_ns=cfg.interval_ns,
                progress=log,
            )
    elif cfg.cross_market_mode == "nq_only":
        log.step(
            "بناء الميزات (batch NQ-only)",
            f"NQ={nq.height:,} · interval_ns={cfg.interval_ns}",
        )
        features = single_market_features(nq, interval_ns=cfg.interval_ns)
    else:
        log.step(
            "بناء الميزات (batch cross-market)",
            f"NQ={nq.height:,} · MNQ={mnq.height:,} · interval_ns={cfg.interval_ns}",
        )
        log.op("حساب cross_market_features (batch)")
        features = cross_market_features(
            nq,
            mnq,
            interval_ns=cfg.interval_ns,
            lead_lag_window=cfg.lead_lag_window,
            latency_ns=cfg.latency_ns,
        )
    log.note(f"إطار الميزات الأساسي: {features.height:,} صف × {features.width} عمود")
    log.step("إلحاق عمق الدفتر السببي (دخول/مراقبة/تنفيذ/خروج)")
    features = _attach_causal_depth(features, nq, interval_ns=cfg.interval_ns, progress=log)
    if cfg.include_failed_fvg:
        log.step("إلحاق Failed FVG (asof خلفي)")
        log.op("failed_fvg_features + join_asof backward")
        features = _attach_failed_fvg(features, nq)
        log.op(f"بعد FVG: {features.height:,} صف")
    if cfg.include_auction_vp:
        log.step("إلحاق Volume Profile / Auction (asof خلفي)")
        log.op("auction_signal_frame + join_asof backward")
        features = _attach_auction_vp(features, nq, interval_ns=cfg.interval_ns)
        log.op(f"بعد Auction/VP: {features.height:,} صف")
    if cfg.include_failed_breakout:
        log.step("إلحاق Failed Breakout + عمق عند مستوى الكسر")
        log.op("failed_breakout_features + depth_at_break + join_asof backward")
        features = _attach_failed_breakout(features, nq)
        log.op(f"بعد Failed Breakout: {features.height:,} صف")
    return features


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
    tick_size: float = NQ_METADATA.tick_size,
    commission_bps: float = 0.0,
    parallel_coverage: bool = True,
    ssl_mode: SslMode = "tick",
    language_model: LanguageModel | None = None,
    rng: np.random.Generator | None = None,
    progress: PipelineProgress | None = None,
) -> tuple[SSLPipelineResult, CoverageReport, AlphaDiscovery, UnifiedResearchReport]:
    """يشغّل SSL + M9 (خلفية) + ألفا → تقرير شامل (الميزات مُبنية مسبقًا)."""
    from nq.alpha.discovery import discover_alpha_from_features  # noqa: PLC0415

    log = progress if progress is not None else PipelineProgress(enabled=False)
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
    log.note(
        f"إشارات الفرز: {len(columns)} · ssl_mode={ssl_mode} · parallel_m9={parallel_coverage}"
    )

    ssl_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)
    alpha_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)

    def _run_ssl() -> SSLPipelineResult:
        if ssl_mode == "tick":
            log.op("استدعاء run_ssl_tick_pipeline")
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
                progress=log,
            )
        log.op("استدعاء run_ssl_pipeline (bucket)")
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
            progress=log,
        )

    if parallel_coverage and (features.height > 0 or ssl_mode == "tick"):
        log.step("تشغيل SSL ‖ M9 بالتوازي", f"mode={ssl_mode}")
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
                progress=log,
            )
            log.note("M9 يعمل في الخلفية")
            log.note(f"SSL يبدأ الآن (mode={ssl_mode})")
            ssl_result = _run_ssl()
            log.note(f"SSL انتهى — metrics={ssl_result.metrics.height}")
            log.step("اكتشاف الألفا (intraday)", f"signals={len(columns)}")
            log.op(f"تقييم إشارات: {columns}")
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
                progress=log,
            )
            log.op(
                f"ألفا انتهى — evals={alpha_result.evaluations.height} · "
                f"selected={alpha_result.selected!r}"
            )
            log.step("انتظار نتيجة M9")
            coverage_result = coverage_future.result()
            log.note(f"M9 انتهى — metrics={coverage_result.metrics.height}")
    else:
        log.step("تشغيل SSL (تسلسلي)", f"mode={ssl_mode}")
        ssl_result = _run_ssl()
        log.note(f"SSL انتهى — metrics={ssl_result.metrics.height}")
        log.step("اكتشاف الألفا (intraday)", f"signals={len(columns)}")
        log.op(f"تقييم إشارات: {columns}")
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
            progress=log,
        )
        log.op(
            f"ألفا انتهى — evals={alpha_result.evaluations.height} · "
            f"selected={alpha_result.selected!r}"
        )
        log.step("تشغيل المراقب M9 (تسلسلي)")
        log.op("run_coverage_on_features")
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
            progress=log,
        )
        log.note(f"M9 انتهى — metrics={coverage_result.metrics.height}")

    narrative = ""
    if language_model is not None:
        log.step("تلخيص الأدلة عبر LanguageModel")
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
        else:
            log.note("لا توجد ادعاءات موثّقة للتلخيص")

    log.step("دمج التقرير الموحّد (SSL ‖ M9 ‖ ألفا)")
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
    *,
    progress: PipelineProgress | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """يُحمّل NQ/MNQ مع دعم nq_only و max_rows."""
    log = progress if progress is not None else PipelineProgress(enabled=False)
    log.op("تحميل NQ")
    nq_frame = (
        nq
        if isinstance(nq, pl.DataFrame)
        else load_mbo_frame(nq, max_rows=cfg.max_rows, progress=log)
    )
    if isinstance(nq, pl.DataFrame) and cfg.max_rows is not None:
        log.op(f"قص NQ DataFrame إلى max_rows={cfg.max_rows:,}")
        nq_frame = load_mbo_frame(nq_frame, max_rows=cfg.max_rows, progress=log)
    if cfg.cross_market_mode == "nq_only":
        log.op("وضع nq_only — لا يتم إنشاء MNQ اصطناعي")
        return nq_frame, nq_frame.head(0)
    log.op("تحميل MNQ")
    mnq_frame = (
        mnq
        if isinstance(mnq, pl.DataFrame)
        else load_mbo_frame(mnq, max_rows=cfg.max_rows, progress=log)
    )
    if isinstance(mnq, pl.DataFrame) and cfg.max_rows is not None:
        mnq_frame = load_mbo_frame(mnq_frame, max_rows=cfg.max_rows, progress=log)
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
    progress: PipelineProgress | bool | None = None,
    quiet: bool | None = None,
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
    progress / quiet:
        طباعة تقدّم الخطوات على stderr. الافتراضي: مفعّل.
        ``quiet=True`` أو ``progress=False`` يعطّل الطباعة.
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
    if quiet is not None:
        cfg = replace(cfg, quiet=quiet)

    log = resolve_progress(progress, quiet=cfg.quiet)
    feature_extra = (
        int(cfg.include_failed_fvg) + int(cfg.include_auction_vp) + int(cfg.include_failed_breakout)
    )
    save_step = 1 if output_dir is not None else 0
    llm_step = 1 if language_model is not None else 0
    # load + feature_base + depth + extras + ssl/m9/alpha path (~4) + unify + optional save/llm
    total_steps = 3 + feature_extra + 4 + llm_step + save_step
    if output_dir is not None:
        out_early = Path(output_dir)
        out_early.mkdir(parents=True, exist_ok=True)
        log.attach_log(out_early / "progress.log")
    log.begin("الخط الموحّد MBO → تقرير", total_steps=total_steps)
    log.line("كل عملية تُطبع سطرًا بسطر — راقب progress.log أو stderr")

    try:
        log.step(
            "تهيئة الحتمية + تحميل MBO",
            (
                f"mode={cfg.cross_market_mode} · features={cfg.feature_mode} · "
                f"ssl={cfg.ssl_mode} · max_rows={cfg.max_rows}"
            ),
        )
        log.op(f"seed_everything({cfg.global_seed})")
        seed_everything(cfg.global_seed)
        generator = rng if rng is not None else np.random.default_rng(cfg.global_seed)
        nq_frame, mnq_frame = _load_pipeline_frames(nq, mnq, cfg, progress=log)
        log.note(
            f"NQ={nq_frame.height:,} صف · MNQ={mnq_frame.height:,} صف"
            + (" (nq_only)" if cfg.cross_market_mode == "nq_only" else "")
        )

        features = _build_research_features(nq_frame, mnq_frame, cfg, progress=log)
        resolved_signals = signal_columns if signal_columns is not None else cfg.signal_columns

        ssl_result, coverage_result, alpha_result, unified = run_ssl_research_pipeline(
            nq_frame,
            mnq_frame,
            features,
            interval_ns=cfg.interval_ns,
            horizon=cfg.horizon,
            signal_columns=resolved_signals,
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
            progress=log,
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
            log.step("حفظ المخرجات", str(out.resolve()))
            out.mkdir(parents=True, exist_ok=True)
            log.op("كتابة report.md")
            (out / "report.md").write_text(unified.to_markdown(), encoding="utf-8")
            if ssl_result.metrics.height > 0:
                log.op("كتابة ssl_metrics.parquet")
                ssl_result.metrics.write_parquet(out / "ssl_metrics.parquet")
            if coverage_result.metrics.height > 0:
                log.op("كتابة coverage_metrics.parquet")
                coverage_result.metrics.write_parquet(out / "coverage_metrics.parquet")
            if alpha_result.evaluations.height > 0:
                log.op("كتابة alpha_evaluations.parquet")
                alpha_result.evaluations.write_parquet(out / "alpha_evaluations.parquet")
            log.op("كتابة features.parquet")
            features.write_parquet(out / "features.parquet")
            log.note(f"كُتبت الملفات في {out.resolve()}")

        log.done(
            f"features={features.height:,} · "
            f"ssl_metrics={ssl_result.metrics.height} · "
            f"m9_metrics={coverage_result.metrics.height} · "
            f"alpha_evals={alpha_result.evaluations.height}"
        )
        return result
    except Exception as exc:
        log.fail(exc)
        raise


__all__ = [
    "PipelineConfig",
    "PipelineProgress",
    "UnifiedResearchResult",
    "run_research_pipeline",
    "run_ssl_research_pipeline",
]
