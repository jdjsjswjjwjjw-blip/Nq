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
from nq.features.streaming import (
    STREAMING_SIGNAL_COLUMNS,
    build_streaming_research_features,
)
from nq.ingestion.reader import load_mbo_frame
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_pipeline, run_ssl_tick_pipeline
from nq.models.tick_stream import TickStream
from nq.orderbook import reconstruct, scan_book_tob_and_depth
from nq.research.assistant import LanguageModel, ResearchAssistant
from nq.research.progress import PipelineProgress, resolve_progress
from nq.research.unified import UnifiedResearchReport, build_unified_report
from nq.simulation.auction import VP_PROFILE_INTERVAL_NS, auction_signal_frame
from nq.simulation.bottom_book import (
    BOTTOM_BOOK_COLUMNS,
    attach_bottom_book_asof,
    bottom_book_features_at_bar_close,
)
from nq.simulation.breakout import FB_PULSE_ZERO_FILL, failed_breakout_features
from nq.simulation.cross_market import cross_market_features
from nq.simulation.depth_lifecycle import (
    attach_depth_asof,
    depth_at_bar_close,
    depth_at_bar_close_multi,
)
from nq.simulation.depth_noise import DepthNoiseConfig, filter_depth_noise
from nq.simulation.fvg import failed_fvg_features

if TYPE_CHECKING:
    from nq.alpha.discovery import AlphaDiscovery

SslMode = Literal["bucket", "tick"]
CrossMarketMode = Literal["dual", "nq_only"]
FeatureMode = Literal["streaming", "batch"]

# أنطولوجيا ألفا افتراضية موحّدة مع symbolic: deltas/trap/fail + vp_* فقط.
# أعمدة streaming VA (in_value_area/near_vah/poc_dist_norm) تبقى في الإطار
# للمراقبة/M9 لكنها ليست مرشّحات ألفا افتراضية (تجنّب خلط الأنطولوجيتين).
_DEFAULT_SIGNAL_COLUMNS = (
    "nq_delta",
    "mnq_delta",
    "trap_setup",
    "phase_balance",
    "phase_expansion",
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
    "vp_upper",
    "vp_mid",
    "vp_lower",
    "vp_rel_upper",
    "vp_rel_mid",
    "vp_rel_lower",
    "vp_excess_upper",
    "vp_excess_lower",
    "vp_of_delta",
    "vp_absorb",
    "vp_look_fail",
    "vp_order_accel",
    "vp_early_imbalance",
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_in_value_frac",
    "vp_pullback_defense",
    "vp_poc_migration",
    "vp_flip_to_imbalance",
    "vp_liquidity_session",
    "vp_fr_active",
    "vp_fr_accepted_expansion",
    "vp_fr_in_balance",
    "vp_fr_exit",
    "vp_fr_upper",
    "vp_fr_mid",
    "vp_fr_lower",
    "vp_fr_start_ts",
    "vp_fr_end_ts",
    "vp_fsm_break",
    "vp_fsm_build",
    "vp_fsm_accel",
    "vp_fsm_retest",
    "vp_fsm_expand",
    "vp_auction_setup",
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
    tick_size: float = 0.25
    commission_bps: float = 0.0
    alpha: float = 0.05
    n_permutations: int = 2000
    global_seed: int = 0
    parallel_coverage: bool = False
    ssl_mode: SslMode = "tick"
    cross_market_mode: CrossMarketMode = "dual"
    max_rows: int | None = None
    include_failed_fvg: bool = True
    include_auction_vp: bool = True
    include_failed_breakout: bool = True
    feature_mode: FeatureMode = "streaming"
    signal_columns: tuple[str, ...] | None = None
    quiet: bool = False
    # [streaming] — ساعة البحث vs دقة ميكرو
    research_interval_ns: int | None = None
    micro_interval_ns: int = 100_000_000
    use_micro_interval: bool = False
    # عمق موحّد: ضوضاء + أسفل الدفتر
    filter_depth_noise: bool = True
    include_bottom_book: bool = True
    #: رينج Volume Profile (افتراضي 5 دقائق) — مستقل عن ساعة البحث/الفعل.
    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS

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
        streaming = raw.get("streaming", {})
        det = raw.get("determinism", {})
        run_cfg = raw.get("run", {})
        max_rows_raw = data.get("max_rows")
        max_rows = None if max_rows_raw in (None, 0) else int(max_rows_raw)
        signal_cols = signals.get("columns")

        temporal_iv = int(temporal.get("interval_ns", 1_000_000_000))
        research_iv_raw = streaming.get("research_interval_ns")
        research_iv = int(research_iv_raw) if research_iv_raw is not None else None
        micro_iv = int(streaming.get("micro_interval_ns", 100_000_000))
        use_micro = bool(streaming.get("use_micro_interval", False))
        # ترتيب الأولوية: micro (إن طُلب) → research_interval من [streaming] → temporal
        if use_micro:
            interval_ns = micro_iv
        elif research_iv is not None:
            interval_ns = research_iv
        else:
            interval_ns = temporal_iv

        return cls(
            interval_ns=interval_ns,
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
            include_failed_fvg=bool(signals.get("include_failed_fvg", True)),
            include_auction_vp=bool(signals.get("include_auction_vp", True)),
            include_failed_breakout=bool(signals.get("include_failed_breakout", True)),
            feature_mode=str(features_cfg.get("mode", signals.get("feature_mode", "streaming"))),  # type: ignore[arg-type]
            signal_columns=tuple(signal_cols) if signal_cols else None,
            quiet=bool(run_cfg.get("quiet", False)),
            parallel_coverage=bool(run_cfg.get("parallel_coverage", False)),
            research_interval_ns=research_iv,
            micro_interval_ns=micro_iv,
            use_micro_interval=use_micro,
            filter_depth_noise=bool(raw.get("depth", {}).get("filter_noise", True)),
            include_bottom_book=bool(raw.get("depth", {}).get("include_bottom_book", True)),
            profile_interval_ns=int(
                temporal.get(
                    "profile_interval_ns",
                    signals.get("profile_interval_ns", VP_PROFILE_INTERVAL_NS),
                )
            ),
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
    nq_top_of_book: pl.DataFrame | None = None,
    mnq_top_of_book: pl.DataFrame | None = None,
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
        nq_top_of_book=nq_top_of_book,
        mnq_top_of_book=mnq_top_of_book,
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
    # افتراضي: لا تخلط streaming VA مع vp_* في شاشة الألفا
    streaming_va = frozenset({"in_value_area", "near_vah", "near_val", "poc_dist_norm"})
    ordered = list(
        dict.fromkeys([*_DEFAULT_SIGNAL_COLUMNS, *_BATCH_SIGNAL_COLUMNS, *STREAMING_SIGNAL_COLUMNS])
    )
    return [c for c in ordered if c in features.columns and c not in streaming_va]


def _attach_failed_fvg(
    features: pl.DataFrame,
    nq: pl.DataFrame,
    *,
    progress: PipelineProgress | None = None,
) -> pl.DataFrame:
    """يلحق إشارة Failed FVG بإطار البحث الموحّد (نبضة تطابقية — بلا sticky).

    ``fail_fvg`` يُصفَّر عند غياب التطابق؛ نسب الجهد تبقى null (لا تطابق ≠ صفر)
    بنفس سياسة FB للجهد/العمق.
    """
    log = progress if progress is not None else PipelineProgress(enabled=False)
    fvg = failed_fvg_features(nq, progress=log)
    if fvg.height == 0 or features.height == 0:
        return features.with_columns(
            pl.lit(0.0).alias("fail_fvg"),
            pl.lit(None).cast(pl.Float64).alias("effort_range_ratio"),
            pl.lit(None).cast(pl.Float64).alias("effort_volume_ratio"),
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
    drop_existing = [c for c in keep if c != AVAILABILITY_TS and c in left.columns]
    if drop_existing:
        left = left.drop(drop_existing)
    joined = left.join(right, on=AVAILABILITY_TS, how="left")
    # نبضة فقط؛ الجهد يبقى null خارج أوقات الإشارة
    return joined.with_columns(pl.col("fail_fvg").fill_null(0.0))


def _attach_auction_vp(
    features: pl.DataFrame,
    nq: pl.DataFrame,
    *,
    interval_ns: int,
    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS,
    progress: PipelineProgress | None = None,
) -> pl.DataFrame:
    """يلحق إشارات Volume Profile / المزاد (رينج 5د · فعل 30ث) asof خلفي."""
    log = progress if progress is not None else PipelineProgress(enabled=False)
    signals = auction_signal_frame(
        nq,
        signal_interval_ns=interval_ns,
        profile_interval_ns=profile_interval_ns,
        progress=log,
    )
    return _attach_auction_vp_signals(features, signals)


def _attach_auction_vp_signals(
    features: pl.DataFrame,
    signals: pl.DataFrame,
) -> pl.DataFrame:
    """يلحق إشارات مزاد محسوبة مسبقًا بـ asof خلفي."""
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


def _fb_empty_column_exprs() -> list[pl.Expr]:
    """أعمدة FB عند غياب الصفوف: نبضة=0 · جهد/عمق=null (لا تطابق ≠ صفر)."""
    exprs: list[pl.Expr] = []
    for col in _FB_SIGNAL_COLUMNS:
        if col in FB_PULSE_ZERO_FILL:
            exprs.append(pl.lit(0.0).alias(col))
        else:
            exprs.append(pl.lit(None).cast(pl.Float64).alias(col))
    return exprs


def _attach_failed_breakout(  # noqa: PLR0915
    features: pl.DataFrame,
    nq: pl.DataFrame,
    *,
    depth_30m: pl.DataFrame | None = None,
    progress: PipelineProgress | None = None,
) -> pl.DataFrame:
    """يلحق Failed Breakout بنبضة تطابقية + عمق عند مستوى الكسر (سببي).

    ``fail_breakout`` نبضة عند ``availability_ts`` فقط (ليست sticky asof).
    ``fb_depth_at_break``: NaN = لا تطابق مستوى / لا دفتر؛ 0.0 = تطابق بحجم صفر.
    """
    import numpy as np  # noqa: PLC0415

    log = progress if progress is not None else PipelineProgress(enabled=False)
    log.op("failed_breakout_features (إشارة فوليوم)")
    fb = failed_breakout_features(nq, progress=log)
    if fb.height == 0 or features.height == 0:
        log.op("Failed Breakout: لا صفوف — نبضة=0 · عمق/جهد=null")
        return features.with_columns(_fb_empty_column_exprs())

    # عمق عند إغلاق شمعة الإشارة (30m) — يُعاد استخدام المسح الموحّد إن وُجد
    interval_30m = 30 * 60 * 1_000_000_000
    if depth_30m is None:
        log.op("مسح عمق FB عند إغلاق 30m · levels=5")
        depth = depth_at_bar_close(nq, interval_ns=interval_30m, n_levels=5, progress=log)
    else:
        log.op("إعادة استخدام عمق 30m من المسح الموحّد (بدون مرور ثانٍ)")
        depth = depth_30m
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
        # سيولة ظاهرة عند مستوى الكسر: أقرب مستوى ضمن ≤ 4 تيكات (1.0$)
        # الأسعار بالدولار (بعد PRICE_SCALE في snapshot_to_row) — لا تستخدم PRICE_SCALE*4
        match_tol = 0.25 * 4
        levels_bid_px = [f"depth_bid_px_{k}" for k in range(1, 6)]
        levels_bid_sz = [f"depth_bid_sz_{k}" for k in range(1, 6)]
        levels_ask_px = [f"depth_ask_px_{k}" for k in range(1, 6)]
        levels_ask_sz = [f"depth_ask_sz_{k}" for k in range(1, 6)]

        level_arr = fb["fb_break_level"].to_numpy().astype(np.float64)
        signal_arr = fb["fail_breakout"].to_numpy().astype(np.float64)
        n_fb = int(fb.height)
        at_break = np.full(n_fb, np.nan, dtype=np.float64)

        def _stack(cols: list[str], *, fill: float) -> np.ndarray:
            cols_present = [c for c in cols if c in fb.columns]
            if not cols_present:
                return np.full((n_fb, len(cols)), fill, dtype=np.float64)
            return np.column_stack([fb[c].to_numpy().astype(np.float64) for c in cols])

        bid_px = _stack(levels_bid_px, fill=np.nan)
        bid_sz = _stack(levels_bid_sz, fill=0.0)
        ask_px = _stack(levels_ask_px, fill=np.nan)
        ask_sz = _stack(levels_ask_sz, fill=0.0)

        def _assign(mask: np.ndarray, px: np.ndarray, sz: np.ndarray) -> None:
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            dist = np.abs(px[idx] - level_arr[idx][:, None])
            dist = np.where(np.isfinite(px[idx]), dist, np.inf)
            best_j = np.argmin(dist, axis=1)
            best_d = dist[np.arange(idx.size), best_j]
            ok = best_d <= match_tol
            if not np.any(ok):
                return
            chosen = idx[ok]
            at_break[chosen] = sz[chosen, best_j[ok]]

        log.op(f"حساب fb_depth_at_break على {n_fb:,} صف (متّجهي · match≤{match_tol})")
        active = (level_arr > 0) & (signal_arr != 0.0)
        _assign(active & (signal_arr < 0), ask_px, ask_sz)
        _assign(active & (signal_arr > 0), bid_px, bid_sz)
        fb = fb.with_columns(
            pl.Series("fb_depth_at_break", at_break),
            pl.col("depth_imbalance").alias("fb_depth_imbalance"),
            pl.col("depth_cum_bid").alias("fb_depth_cum_bid"),
            pl.col("depth_cum_ask").alias("fb_depth_cum_ask"),
        )
        log.op("أعمدة عمق FB جاهزة (at_break / imbalance / cum)")
    else:
        log.op("عمق 30m فارغ — fb_depth_* = null (لا تطابق دفتر)")
        fb = fb.with_columns(
            pl.lit(None).cast(pl.Float64).alias("fb_depth_at_break"),
            pl.lit(None).cast(pl.Float64).alias("fb_depth_imbalance"),
            pl.lit(None).cast(pl.Float64).alias("fb_depth_cum_bid"),
            pl.lit(None).cast(pl.Float64).alias("fb_depth_cum_ask"),
        )

    keep = [c for c in (AVAILABILITY_TS, *_FB_SIGNAL_COLUMNS) if c in fb.columns]
    right = fb.select(keep).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop_existing = [c for c in keep if c != AVAILABILITY_TS and c in left.columns]
    if drop_existing:
        left = left.drop(drop_existing)
    joined = left.join(right, on=AVAILABILITY_TS, how="left")
    fills = [
        pl.col(c).fill_null(0.0)
        for c in _FB_SIGNAL_COLUMNS
        if c in joined.columns and c in FB_PULSE_ZERO_FILL
    ]
    return joined.with_columns(fills) if fills else joined


def _attach_causal_depth(
    features: pl.DataFrame,
    nq: pl.DataFrame,
    *,
    interval_ns: int,
    depth: pl.DataFrame | None = None,
    cleaned_mbo: pl.DataFrame | None = None,
    filter_noise: bool = True,
    include_bottom_book: bool = True,
    progress: PipelineProgress | None = None,
) -> pl.DataFrame:
    """يلحق سلم عمق NQ (+ أسفل الدفتر) عند إغلاق كل فاصل بحثي.

    يصفّي ضوضاء العمق سببيًا قبل اللقطة عندما لا تُمرَّر ``cleaned_mbo``.
    """
    log = progress if progress is not None else PipelineProgress(enabled=False)
    mbo = cleaned_mbo
    if mbo is None:
        if filter_noise:
            log.op(f"depth_noise_filter قبل لقطة العمق · events={nq.height:,}")
            mbo = filter_depth_noise(nq, config=DepthNoiseConfig())
            log.op(f"بعد الفلتر: {mbo.height:,} حدث (أُسقط {nq.height - mbo.height:,})")
        else:
            mbo = nq
    if depth is None:
        log.op(f"depth_at_bar_close ساعة البحث · levels=5 · interval_ns={interval_ns}")
        depth = depth_at_bar_close(mbo, interval_ns=interval_ns, n_levels=5, progress=log)
    else:
        log.op("إعادة استخدام عمق ساعة البحث من المسح الموحّد")
    out = features
    if depth.height == 0:
        log.op("عمق: لا لقطات — تخطّي L1–L5")
    else:
        cols = [c for c in depth.columns if c.startswith("depth_")]
        if "nq_bid" not in out.columns and "nq_bid" in depth.columns:
            cols = [*cols, "nq_bid", "nq_ask"]
        out = attach_depth_asof(out, depth, columns=cols, fill_missing=False)
        log.op(f"عمق مُلحق: {len(cols)} عمود · rows={out.height:,}")

    if include_bottom_book:
        log.op(f"bottom_book L2–L5 · interval_ns={interval_ns}")
        bottom = bottom_book_features_at_bar_close(
            mbo,
            interval_ns=interval_ns,
            filter_noise=False,
            progress=log,
        )
        out = attach_bottom_book_asof(out, bottom)
        log.op(f"bottom_book مُلحق: {len(BOTTOM_BOOK_COLUMNS)} عمود")
    return out


def _build_research_features(  # noqa: PLR0915
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    cfg: PipelineConfig,
    *,
    progress: PipelineProgress | None = None,
) -> tuple[
    pl.DataFrame,
    TickStream | None,
    pl.DataFrame | None,
    pl.DataFrame | None,
]:
    """يبني إطار البحث: streaming/batch ثم عمق (+noise/bottom) ثم FVG نبضة / Auction asof / FB نبضة.

    يُعيد أيضًا ``TickStream`` عند الوضع streaming، وTOB المحسوب مسبقًا في
    الوضع batch لإعادة استخدامه في cross-market وM9.
    """
    log = progress if progress is not None else PipelineProgress(enabled=False)
    tick_stream: TickStream | None = None
    nq_top_of_book: pl.DataFrame | None = None
    mnq_top_of_book: pl.DataFrame | None = None
    if cfg.use_micro_interval:
        log.note(
            f"ساعة البحث = micro_interval_ns={cfg.micro_interval_ns} (use_micro_interval=true)"
        )
    elif cfg.research_interval_ns is not None:
        log.note(f"ساعة البحث من [streaming].research_interval_ns={cfg.research_interval_ns}")
    market_pair = (
        f"NQ={nq.height:,} (nq_only)" if nq is mnq else f"NQ={nq.height:,} · MNQ={mnq.height:,}"
    )
    interval_30m = 30 * 60 * 1_000_000_000
    depth_intervals: list[int] = [cfg.interval_ns]
    if cfg.include_failed_breakout:
        depth_intervals.append(interval_30m)
    depth_intervals = list(dict.fromkeys(int(value) for value in depth_intervals))

    cleaned_nq = nq
    if cfg.filter_depth_noise:
        log.step("فلتر ضوضاء العمق (سببي) قبل لقطات الدفتر")
        cleaned_nq = filter_depth_noise(nq, config=DepthNoiseConfig())
        log.op(f"depth_noise: {nq.height:,} → {cleaned_nq.height:,}")

    depth_by_iv: dict[int, pl.DataFrame]
    if cfg.feature_mode == "streaming":
        log.step(
            "بناء الميزات (streaming state-machine)",
            f"{market_pair} · interval_ns={cfg.interval_ns}",
        )
        features, tick_stream = build_streaming_research_features(
            nq,
            mnq,
            interval_ns=cfg.interval_ns,
            progress=log,
            return_tick=True,
        )
        log.step(
            "إلحاق عمق الدفتر السببي (مسح موحّد)",
            f"intervals={depth_intervals} · bottom_book={cfg.include_bottom_book}",
        )
        depth_by_iv = depth_at_bar_close_multi(
            cleaned_nq,
            interval_ns_list=tuple(depth_intervals),
            n_levels=5,
            progress=log,
        )
    else:
        log.step(
            "مسح دفتر موحّد للعمق وTOB",
            f"intervals={depth_intervals} · bottom_book={cfg.include_bottom_book}",
        )
        nq_scan, depth_by_iv = scan_book_tob_and_depth(
            cleaned_nq,
            interval_ns_list=tuple(depth_intervals),
            n_levels=5,
            progress=log,
            progress_label="book_scan:NQ",
        )
        nq_top_of_book = nq_scan.top_of_book
        if nq is mnq:
            mnq_top_of_book = nq_top_of_book
        else:
            mnq_top_of_book = reconstruct(
                mnq,
                progress=log,
                progress_label="book_scan:MNQ",
            ).top_of_book

        log.step(
            "بناء الميزات (batch cross-market)",
            f"{market_pair} · interval_ns={cfg.interval_ns}",
        )
        log.op("حساب cross_market_features (batch)")
        features = cross_market_features(
            nq,
            mnq,
            interval_ns=cfg.interval_ns,
            lead_lag_window=cfg.lead_lag_window,
            latency_ns=cfg.latency_ns,
            progress=log,
            nq_top_of_book=nq_top_of_book,
            mnq_top_of_book=mnq_top_of_book,
        )
    log.note(f"إطار الميزات الأساسي: {features.height:,} صف × {features.width} عمود")
    features = _attach_causal_depth(
        features,
        nq,
        interval_ns=cfg.interval_ns,
        depth=depth_by_iv[cfg.interval_ns],
        cleaned_mbo=cleaned_nq,
        filter_noise=False,
        include_bottom_book=cfg.include_bottom_book,
        progress=log,
    )

    if cfg.include_failed_fvg:
        log.step("إلحاق Failed FVG (نبضة تطابقية)")
        log.op("failed_fvg_features + pulse join (exact availability_ts)")
        features = _attach_failed_fvg(features, nq, progress=log)
        log.op(f"بعد FVG: {features.height:,} صف")
    if cfg.include_auction_vp:
        log.step("إلحاق Volume Profile / Auction (asof خلفي)")
        log.op("auction_signal_frame + join_asof backward")
        features = _attach_auction_vp(
            features,
            nq,
            interval_ns=cfg.interval_ns,
            profile_interval_ns=cfg.profile_interval_ns,
            progress=log,
        )
        log.op(f"بعد Auction/VP: {features.height:,} صف")
    if cfg.include_failed_breakout:
        log.step("إلحاق Failed Breakout + عمق عند مستوى الكسر")
        log.op("failed_breakout_features + depth_at_break(30m) + pulse join")
        features = _attach_failed_breakout(
            features,
            nq,
            depth_30m=depth_by_iv.get(interval_30m),
            progress=log,
        )
        log.op(f"بعد Failed Breakout: {features.height:,} صف")
    return features, tick_stream, nq_top_of_book, mnq_top_of_book


def run_ssl_research_pipeline(  # noqa: PLR0915
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
    parallel_coverage: bool = False,
    ssl_mode: SslMode = "tick",
    language_model: LanguageModel | None = None,
    rng: np.random.Generator | None = None,
    progress: PipelineProgress | None = None,
    tick_stream: TickStream | None = None,
    nq_top_of_book: pl.DataFrame | None = None,
    mnq_top_of_book: pl.DataFrame | None = None,
) -> tuple[SSLPipelineResult, CoverageReport, AlphaDiscovery, UnifiedResearchReport]:
    """يشغّل SSL + M9 + ألفا → تقرير شامل (الميزات مُبنية مسبقًا).

    افتراضيًا تسلسلي (``parallel_coverage=False``) حتى يبقى اللوج مسارًا خطيًا
    مقروءًا. عند التوازي تُستخدم بادئات قنوات ``[SSL]`` / ``[M9]``.
    ``tick_stream`` يُمرَّر لـ SSL-tick عند توفره من مرحلة الميزات streaming.
    """
    from nq.alpha.discovery import discover_alpha_from_features  # noqa: PLC0415

    log = progress if progress is not None else PipelineProgress(enabled=False)
    generator = rng if rng is not None else np.random.default_rng(0)
    seed = int(generator.integers(0, 2**31))

    policy = TemporalPolicy.for_run(interval_ns=interval_ns, window=ssl_window, horizon=horizon)
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
    if parallel_coverage:
        log.note(
            "توازي SSL‖M9 مفعّل — راقب بادئة [SSL]/[M9] (للمسار الخطي: parallel_coverage=false)"
        )

    ssl_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)
    alpha_assistant = ResearchAssistant(alpha=alpha, language_model=language_model)

    def _run_ssl() -> SSLPipelineResult:
        with log.channel("SSL"):
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
                    tick_stream=tick_stream,
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

    def _run_m9() -> CoverageReport:
        with log.channel("M9"):
            return _run_coverage_task(
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
                nq_top_of_book=nq_top_of_book,
                mnq_top_of_book=mnq_top_of_book,
            )

    if parallel_coverage and (features.height > 0 or ssl_mode == "tick"):
        log.step("تشغيل SSL ‖ M9 بالتوازي", f"mode={ssl_mode}")
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="coverage-m9") as executor:
            coverage_future = executor.submit(_run_m9)
            log.note("M9 يعمل في الخلفية (قناة [M9])")
            log.note(f"SSL يبدأ الآن (قناة [SSL] · mode={ssl_mode})")
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
        coverage_result = _run_m9()
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
        nq_frame = nq_frame.head(cfg.max_rows)
    if cfg.cross_market_mode == "nq_only":
        log.op("وضع nq_only — سوق NQ فقط (بدون تحميل/إعادة بناء MNQ)")
        return nq_frame, nq_frame
    log.op("تحميل MNQ")
    mnq_frame = (
        mnq
        if isinstance(mnq, pl.DataFrame)
        else load_mbo_frame(mnq, max_rows=cfg.max_rows, progress=log)
    )
    if isinstance(mnq, pl.DataFrame) and cfg.max_rows is not None:
        mnq_frame = mnq_frame.head(cfg.max_rows)
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
            f"NQ={nq_frame.height:,} صف"
            + (
                " (nq_only — بدون MNQ)"
                if cfg.cross_market_mode == "nq_only"
                else f" · MNQ={mnq_frame.height:,} صف"
            )
        )

        features, tick_stream, nq_top_of_book, mnq_top_of_book = _build_research_features(
            nq_frame,
            mnq_frame,
            cfg,
            progress=log,
        )
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
            tick_stream=tick_stream,
            nq_top_of_book=nq_top_of_book,
            mnq_top_of_book=mnq_top_of_book,
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
