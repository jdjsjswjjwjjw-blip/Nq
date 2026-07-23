"""بحث فرضيات Failed Breakout — تركيز فوليوم بلا تسريب.

* شبكة واسعة من أوضاع الفوليوم: فردي (bar) / تراكمي (cum) /
  دلتا (delta) / جهد مقابل نتيجة (effort_result).
* اختيار walk-forward purged (train فقط → OOS).
* بوابة SSL سببية + تعزيزات سياق/فوليوم — لا إعادة كتابة القاعدة أثناء التدريب.

الدخول التقييمي عبر مسار الألفا (close/bid-ask) — ليس ملء عند ``fb_break_level``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import seed_everything
from nq.core.temporal_policy import TemporalPolicy
from nq.ingestion.reader import load_mbo_frame
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_tick_pipeline
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.progress import PipelineProgress, resolve_progress
from nq.simulation.breakout import VolMode, failed_breakout_features, failed_breakout_from_bars
from nq.simulation.cross_market import cross_market_features, single_market_features
from nq.simulation.fvg import NS_PER_MIN, build_ohlcv_bars
from nq.strategies.fvg_hypothesis import (
    apply_causal_ssl_gate,
    exploratory_screen_candidates,
    walk_forward_select_hypotheses,
)
from nq.strategies.ssl_enhancements import (
    EnhancementSpec,
    generate_ssl_enhancement_candidates,
)

_SSL_GATE_QUANTILE = 0.7

_VOLUME_FEATURE_COLUMNS = (
    "fb_effort_range_ratio",
    "fb_effort_volume_ratio",
    "fb_effort_result_ratio",
    "fb_bar_volume",
    "fb_cum_volume",
    "fb_delta",
    "fb_cum_delta",
    "fb_vol_imbalance",
    "fb_absorption",
)


@dataclass(frozen=True, slots=True)
class BreakoutHypothesisSpec:
    """فرضية Failed Breakout بإطار + إعدادات فوليوم/جهد ثابتة."""

    name: str
    signal_interval_ns: int
    trend_interval_ns: int
    lookback: int = 5
    range_mult: float = 1.1
    vol_mult: float = 1.2
    result_mult: float = 1.2
    vol_window: int = 20
    cum_window: int = 5
    vol_mode: VolMode = "bar"
    sma_period: int = 50
    require_sma_filter: bool = True

    def column(self) -> str:
        return f"fail_breakout__{self.name}"


@dataclass(frozen=True, slots=True)
class BreakoutHypothesisSearchResult:
    features: pl.DataFrame
    specs: tuple[BreakoutHypothesisSpec, ...]
    candidate_columns: tuple[str, ...]
    enhancement_columns: tuple[str, ...]
    enhancement_specs: tuple[EnhancementSpec, ...]
    fold_selections: pl.DataFrame
    exploratory_screen: pl.DataFrame
    oos_selected_ic: float
    best_oos_spec: str | None
    ssl: SSLPipelineResult | None
    report: ResearchReport


def _tag_float(value: float) -> str:
    return str(value).replace(".", "p")


def volume_breakout_grid() -> tuple[BreakoutHypothesisSpec, ...]:
    """شبكة فرضيات فوليوم واسعة (حتمية) — فردي/تراكمي/دلتا/جهد×نتيجة.

    الأبعاد:
    * إطار إشارة 15m/30m
    * lookback ∈ {3,5,8}
    * vol_mode ∈ {bar, cum, delta, effort_result}
    * ملفات عتبة فوليوم (window × mult × result)
    * SMA on/off
    """
    signal_mins = (15, 30)
    lookbacks = (3, 5, 8)
    vol_modes: tuple[VolMode, ...] = ("bar", "cum", "delta", "effort_result")
    # (vol_window, vol_mult, result_mult, cum_window)
    vol_profiles = (
        (10, 1.2, 1.2, 3),
        (20, 1.5, 1.5, 5),
        (40, 2.0, 1.8, 8),
    )
    sma_modes = (True, False)
    specs: list[BreakoutHypothesisSpec] = []
    for sig_m in signal_mins:
        for lb in lookbacks:
            for mode in vol_modes:
                for vw, vm, rm, cw in vol_profiles:
                    for use_sma in sma_modes:
                        tag = "sma" if use_sma else "nosma"
                        name = (
                            f"s{sig_m}_lb{lb}_{mode}_"
                            f"vw{vw}_v{_tag_float(vm)}_"
                            f"er{_tag_float(rm)}_{tag}"
                        )
                        specs.append(
                            BreakoutHypothesisSpec(
                                name=name,
                                signal_interval_ns=sig_m * NS_PER_MIN,
                                trend_interval_ns=60 * NS_PER_MIN,
                                lookback=lb,
                                range_mult=1.1,
                                vol_mult=vm,
                                result_mult=rm,
                                vol_window=vw,
                                cum_window=cw,
                                vol_mode=mode,
                                require_sma_filter=use_sma,
                            )
                        )
    return tuple(specs)


def classic_breakout_grid() -> tuple[BreakoutHypothesisSpec, ...]:
    """الشبكة القديمة (جهد حجم فردي فقط) — للتوافق/المقارنة."""
    signal_mins = (15, 30)
    lookbacks = (3, 5, 8)
    thresholds = ((1.1, 1.2), (1.2, 1.3), (1.3, 1.5))
    sma_modes = (True, False)
    specs: list[BreakoutHypothesisSpec] = []
    for sig_m in signal_mins:
        for lb in lookbacks:
            for rm, vm in thresholds:
                for use_sma in sma_modes:
                    tag = "sma" if use_sma else "nosma"
                    name = f"s{sig_m}_lb{lb}_r{_tag_float(rm)}_v{_tag_float(vm)}_{tag}"
                    specs.append(
                        BreakoutHypothesisSpec(
                            name=name,
                            signal_interval_ns=sig_m * NS_PER_MIN,
                            trend_interval_ns=60 * NS_PER_MIN,
                            lookback=lb,
                            range_mult=rm,
                            vol_mult=vm,
                            vol_mode="bar",
                            require_sma_filter=use_sma,
                        )
                    )
    return tuple(specs)


def default_breakout_grid() -> tuple[BreakoutHypothesisSpec, ...]:
    """الافتراضي = شبكة الفوليوم الواسعة."""
    return volume_breakout_grid()


def core_breakout_grid() -> tuple[BreakoutHypothesisSpec, ...]:
    """نواة مضغوطة: وضع فوليوم واحد لكل عائلة + تعزيزات SSL لاحقًا."""
    return (
        BreakoutHypothesisSpec(
            name="core_bar_vw20_v1p5_sma",
            signal_interval_ns=30 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=5,
            range_mult=1.1,
            vol_mode="bar",
            vol_window=20,
            vol_mult=1.5,
            require_sma_filter=True,
        ),
        BreakoutHypothesisSpec(
            name="core_cum_vw20_v1p5_sma",
            signal_interval_ns=30 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=5,
            range_mult=1.1,
            vol_mode="cum",
            vol_window=20,
            vol_mult=1.5,
            cum_window=5,
            require_sma_filter=True,
        ),
        BreakoutHypothesisSpec(
            name="core_delta_vw20_v1p5_nosma",
            signal_interval_ns=30 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=5,
            range_mult=1.1,
            vol_mode="delta",
            vol_window=20,
            vol_mult=1.5,
            require_sma_filter=False,
        ),
        BreakoutHypothesisSpec(
            name="core_effort_result_vw20_er1p5_sma",
            signal_interval_ns=30 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=5,
            range_mult=1.1,
            vol_mode="effort_result",
            vol_window=20,
            vol_mult=1.5,
            result_mult=1.5,
            require_sma_filter=True,
        ),
        BreakoutHypothesisSpec(
            name="core15_bar_vw10_v1p2_sma",
            signal_interval_ns=15 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=3,
            range_mult=1.1,
            vol_mode="bar",
            vol_window=10,
            vol_mult=1.2,
            require_sma_filter=True,
        ),
        BreakoutHypothesisSpec(
            name="core15_effort_result_vw40_er1p8_nosma",
            signal_interval_ns=15 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=8,
            range_mult=1.1,
            vol_mode="effort_result",
            vol_window=40,
            vol_mult=2.0,
            result_mult=1.8,
            require_sma_filter=False,
        ),
        BreakoutHypothesisSpec(
            name="core_cum_vw40_v2p0_nosma",
            signal_interval_ns=30 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=8,
            range_mult=1.1,
            vol_mode="cum",
            vol_window=40,
            vol_mult=2.0,
            cum_window=8,
            require_sma_filter=False,
        ),
        BreakoutHypothesisSpec(
            name="core_delta_vw10_v1p2_sma",
            signal_interval_ns=15 * NS_PER_MIN,
            trend_interval_ns=60 * NS_PER_MIN,
            lookback=5,
            range_mult=1.1,
            vol_mode="delta",
            vol_window=10,
            vol_mult=1.2,
            require_sma_filter=True,
        ),
    )


def materialize_breakout_hypotheses(
    nq: pl.DataFrame,
    specs: Sequence[BreakoutHypothesisSpec],
    *,
    clock: pl.DataFrame,
) -> pl.DataFrame:
    """يبني أعمدة فرضيات على ساعة مشتركة (asof خلفي فقط)."""
    if AVAILABILITY_TS not in clock.columns:
        raise ValueError(f"clock requires {AVAILABILITY_TS}")
    left = clock.select(AVAILABILITY_TS).unique().sort(AVAILABILITY_TS)
    if left.height == 0 or not specs:
        return left

    bars_cache: dict[int, pl.DataFrame] = {}

    def _bars(interval_ns: int) -> pl.DataFrame:
        cached = bars_cache.get(interval_ns)
        if cached is None:
            cached = build_ohlcv_bars(nq, interval_ns=interval_ns)
            bars_cache[interval_ns] = cached
        return cached

    out = left
    for spec in specs:
        raw = failed_breakout_from_bars(
            _bars(spec.signal_interval_ns),
            trend_bars=_bars(spec.trend_interval_ns) if spec.require_sma_filter else None,
            lookback=spec.lookback,
            range_mult=spec.range_mult,
            vol_mult=spec.vol_mult,
            result_mult=spec.result_mult,
            vol_window=spec.vol_window,
            cum_window=spec.cum_window,
            vol_mode=spec.vol_mode,
            sma_period=spec.sma_period,
            require_sma_filter=spec.require_sma_filter,
        )
        col = spec.column()
        if raw.height == 0:
            out = out.with_columns(pl.lit(0.0).alias(col))
            continue
        right = (
            raw.select(AVAILABILITY_TS, pl.col("fail_breakout").alias(col))
            .sort(AVAILABILITY_TS)
            .unique(subset=[AVAILABILITY_TS], keep="last")
        )
        if col in out.columns:
            out = out.drop(col)
        out = out.join_asof(right, on=AVAILABILITY_TS, strategy="backward").with_columns(
            pl.col(col).fill_null(0.0)
        )
    return out


def _attach_volume_context(features: pl.DataFrame, nq: pl.DataFrame) -> pl.DataFrame:
    """يلحق أعمدة فوليوم سببية افتراضية للتعزيز/السياق (asof خلفي)."""
    fb = failed_breakout_features(nq, require_sma_filter=False, rth_only=False)
    keep = [c for c in (AVAILABILITY_TS, *_VOLUME_FEATURE_COLUMNS) if c in fb.columns]
    if len(keep) < 2 or features.height == 0:
        zeros = [pl.lit(0.0).alias(c) for c in _VOLUME_FEATURE_COLUMNS if c not in features.columns]
        return features.with_columns(zeros) if zeros else features
    right = fb.select(keep).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop = [c for c in keep if c != AVAILABILITY_TS and c in left.columns]
    if drop:
        left = left.drop(drop)
    joined = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
    fills = [pl.col(c).fill_null(0.0) for c in _VOLUME_FEATURE_COLUMNS if c in joined.columns]
    return joined.with_columns(fills) if fills else joined


def search_fail_breakout_hypotheses(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    specs: Sequence[BreakoutHypothesisSpec] | None = None,
    interval_ns: int = 1_000_000_000,
    horizon: int = 1,
    use_ssl_gate: bool = True,
    enhance_with_ssl: bool = True,
    ssl_window: int = 5,
    ssl_components: int = 4,
    n_splits: int = 3,
    alpha: float = 0.05,
    n_permutations: int = 200,
    max_rows: int | None = None,
    latency_ns: int = 0,
    global_seed: int = 0,
    output_dir: Path | str | None = None,
    rng: np.random.Generator | None = None,
    progress: PipelineProgress | bool | None = None,
    quiet: bool = False,
) -> BreakoutHypothesisSearchResult:
    """يبحث أفضل إعداد فوليوم/تعزيز Failed Breakout بـ walk-forward + SSL.

    عند ``enhance_with_ssl=True`` (افتراضي مع البحث):
    يستخدم نواة فوليوم مضغوطة + يولّد مرشّحي تعزيز من ``z*`` والسياق/الفوليوم،
    ثم يختار بـ walk-forward (OOS هو الحكم).
    بدون تعزيز: الشبكة الكاملة ``volume_breakout_grid`` (~144 فرضية).
    """
    log = resolve_progress(progress, quiet=quiet)
    need_ssl = use_ssl_gate or enhance_with_ssl
    save_step = 1 if output_dir is not None else 0
    ssl_steps = (1 if need_ssl else 0) + (1 if enhance_with_ssl else 0) + (1 if use_ssl_gate else 0)
    if output_dir is not None:
        out_early = Path(output_dir)
        out_early.mkdir(parents=True, exist_ok=True)
        log.attach_log(out_early / "progress.log")
    log.begin(
        "بحث فرضيات Failed Breakout (فوليوم) + تعزيزات SSL",
        total_steps=7 + ssl_steps + save_step,
    )
    log.line("كل عملية تُطبع سطرًا بسطر — راقب progress.log أو stderr")
    try:
        log.step("تهيئة + تحميل MBO", f"max_rows={max_rows}")
        seed_everything(global_seed)
        generator = rng if rng is not None else np.random.default_rng(global_seed)
        nq_frame = (
            nq
            if isinstance(nq, pl.DataFrame)
            else load_mbo_frame(nq, max_rows=max_rows, progress=log)
        )
        if mnq is None:
            mnq_frame = nq_frame.head(0)
            log.note(f"NQ={nq_frame.height:,} (nq_only)")
        else:
            mnq_frame = (
                mnq
                if isinstance(mnq, pl.DataFrame)
                else load_mbo_frame(mnq, max_rows=max_rows, progress=log)
            )
            log.note(f"NQ={nq_frame.height:,} · MNQ={mnq_frame.height:,}")

        if specs is not None:
            grid = tuple(specs)
        elif enhance_with_ssl:
            grid = core_breakout_grid()
            log.note(f"نواة فوليوم مضغوطة ({len(grid)}) لتوليد تعزيزات SSL")
        else:
            grid = volume_breakout_grid()
            log.note(f"شبكة فوليوم كاملة: {len(grid)} فرضية")

        log.step("بناء ساعة البحث", f"interval_ns={interval_ns}")
        if mnq is None:
            clock = single_market_features(nq_frame, interval_ns=interval_ns)
        else:
            clock = cross_market_features(
                nq_frame,
                mnq_frame,
                interval_ns=interval_ns,
                lead_lag_window=2,
                latency_ns=latency_ns,
            )
        log.step("تجسيد فرضيات FB الفوليوم", f"specs={len(grid)}")
        hyp = materialize_breakout_hypotheses(nq_frame, grid, clock=clock)
        base = clock.sort(AVAILABILITY_TS)
        hyp_cols = [s.column() for s in grid]
        drop = [c for c in hyp_cols if c in base.columns]
        if drop:
            base = base.drop(drop)
        features = base.join_asof(
            hyp.sort(AVAILABILITY_TS),
            on=AVAILABILITY_TS,
            strategy="backward",
        )
        for col in hyp_cols:
            if col in features.columns:
                features = features.with_columns(pl.col(col).fill_null(0.0))

        log.step("إلحاق سياق فوليوم سببي (asof خلفي)")
        features = _attach_volume_context(features, nq_frame)
        log.note(f"أعمدة فوليوم: {[c for c in _VOLUME_FEATURE_COLUMNS if c in features.columns]}")

        ssl_result: SSLPipelineResult | None = None
        enhancement_columns: tuple[str, ...] = ()
        enhancement_specs: tuple[EnhancementSpec, ...] = ()
        candidates: list[str] = list(hyp_cols)

        if need_ssl:
            log.step("تشغيل SSL tick (تمثيلات للتعزيز/البوابة)", f"window={ssl_window}")
            ssl_result = run_ssl_tick_pipeline(
                nq_frame,
                mnq_frame,
                window=ssl_window,
                n_components=ssl_components,
                n_splits=max(2, n_splits),
                alpha=alpha,
                rng=generator,
                progress=log,
            )

        if enhance_with_ssl and ssl_result is not None:
            log.step("توليد مرشّحي تعزيز SSL/سياق/فوليوم", f"bases={len(hyp_cols)}")
            features, enh_cols, enh_specs = generate_ssl_enhancement_candidates(
                features,
                ssl_result.embeddings,
                hyp_cols,
            )
            enhancement_columns = enh_cols
            enhancement_specs = enh_specs
            candidates.extend(list(enh_cols))
            log.note(f"تعزيزات مولَّدة: {len(enh_cols)}")

        if use_ssl_gate and ssl_result is not None:
            log.step("بوابة SSL كلاسيكية على الأساس", f"q={_SSL_GATE_QUANTILE}")
            features, gated = apply_causal_ssl_gate(
                features,
                ssl_result.embeddings,
                hyp_cols,
                z_col="z0",
                quantile=_SSL_GATE_QUANTILE,
            )
            candidates.extend(list(gated))
            log.note(f"أعمدة بوابة: {len(gated)}")

        seen: set[str] = set()
        uniq: list[str] = []
        for c in candidates:
            if c in features.columns and c not in seen:
                seen.add(c)
                uniq.append(c)
        candidates_t = tuple(uniq)
        log.note(f"إجمالي المرشّحين للاختيار: {len(candidates_t)}")

        policy = TemporalPolicy.for_run(interval_ns=interval_ns, window=ssl_window)
        embargo = policy.embargo_time_units(interval_ns=interval_ns)
        log.step("اختيار walk-forward (OOS)", f"n_splits={n_splits}")
        fold_df, oos_ic, oos_p, oos_n, best = walk_forward_select_hypotheses(
            features,
            candidates_t,
            price_col="nq_close",
            horizon=horizon,
            n_splits=n_splits,
            embargo=embargo,
            purge_samples=policy.purge_samples(),
            n_permutations=n_permutations,
            rng=generator,
            progress=log,
        )
        enh_won = best is not None and "__enh__" in str(best)
        log.note(
            f"best_oos={best!r} · oos_ic={oos_ic:.4g} · "
            f"{'تعزيز SSL فاز' if enh_won else 'أساس فوليوم/بوابة'}"
        )

        log.step("شاشة استكشافية")
        explor = exploratory_screen_candidates(
            features,
            candidates_t,
            price_col="nq_close",
            horizon=horizon,
            alpha=alpha,
            n_permutations=n_permutations,
            rng=generator,
        )

        log.step("تقرير البحث")
        assistant = ResearchAssistant(alpha=alpha)
        detail = (
            f"best_oos_spec={best!r}; volume_specs={len(grid)}; "
            f"enhancements={len(enhancement_columns)}; enhance_won={enh_won}; "
            f"walk-forward nested selection"
        )
        evidence = Evidence(
            id="fb_search:oos_ic",
            source="breakout_hypothesis_search",
            metric="IC",
            value=oos_ic,
            pvalue=oos_p,
            sample_size=oos_n,
            detail=detail,
        )
        claim = (
            f"فرضية فوليوم/تعزيز Failed Breakout المختارة بـ walk-forward "
            f"(best={best!r}) تحقق IC خارج العينة = {oos_ic:.4g} (p={oos_p:.4g})."
        )
        findings = [
            assistant.generate_hypothesis(
                claim,
                evidence,
                requires_significance=True,
                category="breakout_search",
            )
        ]
        report = assistant.write_report(
            findings,
            title=(
                "Failed Breakout Volume Search — Walk-Forward + SSL Enhancements"
                if enhance_with_ssl
                else "Failed Breakout Volume Hypothesis Search — Walk-Forward"
            ),
        )
        result = BreakoutHypothesisSearchResult(
            features=features,
            specs=grid,
            candidate_columns=candidates_t,
            enhancement_columns=enhancement_columns,
            enhancement_specs=enhancement_specs,
            fold_selections=fold_df,
            exploratory_screen=explor,
            oos_selected_ic=oos_ic,
            best_oos_spec=best,
            ssl=ssl_result,
            report=report,
        )
        if output_dir is not None:
            out = Path(output_dir)
            log.step("حفظ المخرجات", str(out.resolve()))
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.md").write_text(report.to_markdown(), encoding="utf-8")
            features.write_parquet(out / "features.parquet")
            fold_df.write_parquet(out / "fold_selections.parquet")
            explor.write_parquet(out / "exploratory_screen.parquet")
            if enhancement_specs:
                pl.DataFrame(
                    {
                        "column": [s.column() for s in enhancement_specs],
                        "base": [s.base_column for s in enhancement_specs],
                        "name": [s.name for s in enhancement_specs],
                        "kind": [s.kind for s in enhancement_specs],
                    }
                ).write_parquet(out / "enhancement_specs.parquet")
            if ssl_result is not None and ssl_result.metrics.height > 0:
                ssl_result.metrics.write_parquet(out / "ssl_metrics.parquet")
        log.done(f"best={best!r} · oos_ic={oos_ic:.4g}")
        return result
    except Exception as exc:
        log.fail(exc)
        raise


__all__ = [
    "BreakoutHypothesisSearchResult",
    "BreakoutHypothesisSpec",
    "classic_breakout_grid",
    "core_breakout_grid",
    "default_breakout_grid",
    "materialize_breakout_hypotheses",
    "search_fail_breakout_hypotheses",
    "volume_breakout_grid",
]
