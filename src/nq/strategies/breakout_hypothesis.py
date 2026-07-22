"""بحث فرضيات Failed Breakout (تايم فريم + إعدادات) بلا تسريب.

* شبكة محدودة من lookback / عتبات جهد / فلتر SMA.
* اختيار walk-forward purged (train فقط → OOS).
* بوابة SSL سببية كفلتر تأكيد (لا اختراع إشارة من المستقبل).

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
from nq.simulation.breakout import failed_breakout_from_bars
from nq.simulation.cross_market import cross_market_features
from nq.simulation.fvg import NS_PER_MIN, build_ohlcv_bars
from nq.strategies.fvg_hypothesis import (
    apply_causal_ssl_gate,
    exploratory_screen_candidates,
    walk_forward_select_hypotheses,
)

_SSL_GATE_QUANTILE = 0.7


@dataclass(frozen=True, slots=True)
class BreakoutHypothesisSpec:
    """فرضية Failed Breakout بإطار وإعدادات جهد ثابتة."""

    name: str
    signal_interval_ns: int
    trend_interval_ns: int
    lookback: int = 5
    range_mult: float = 1.1
    vol_mult: float = 1.2
    sma_period: int = 50
    require_sma_filter: bool = True

    def column(self) -> str:
        return f"fail_breakout__{self.name}"


@dataclass(frozen=True, slots=True)
class BreakoutHypothesisSearchResult:
    features: pl.DataFrame
    specs: tuple[BreakoutHypothesisSpec, ...]
    candidate_columns: tuple[str, ...]
    fold_selections: pl.DataFrame
    exploratory_screen: pl.DataFrame
    oos_selected_ic: float
    best_oos_spec: str | None
    ssl: SSLPipelineResult | None
    report: ResearchReport


def default_breakout_grid() -> tuple[BreakoutHypothesisSpec, ...]:
    """شبكة حتمية صغيرة: أطر إشارة + lookback + عتبات + SMA on/off."""
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
                    name = (
                        f"s{sig_m}_lb{lb}_"
                        f"r{str(rm).replace('.', 'p')}_"
                        f"v{str(vm).replace('.', 'p')}_{tag}"
                    )
                    specs.append(
                        BreakoutHypothesisSpec(
                            name=name,
                            signal_interval_ns=sig_m * NS_PER_MIN,
                            trend_interval_ns=60 * NS_PER_MIN,
                            lookback=lb,
                            range_mult=rm,
                            vol_mult=vm,
                            require_sma_filter=use_sma,
                        )
                    )
    return tuple(specs)


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


def search_fail_breakout_hypotheses(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    specs: Sequence[BreakoutHypothesisSpec] | None = None,
    interval_ns: int = 1_000_000_000,
    horizon: int = 1,
    use_ssl_gate: bool = True,
    ssl_window: int = 5,
    ssl_components: int = 4,
    n_splits: int = 3,
    alpha: float = 0.05,
    n_permutations: int = 200,
    max_rows: int | None = None,
    global_seed: int = 0,
    output_dir: Path | str | None = None,
    rng: np.random.Generator | None = None,
    progress: PipelineProgress | bool | None = None,
    quiet: bool = False,
) -> BreakoutHypothesisSearchResult:
    """يبحث أفضل إعداد Failed Breakout بـ walk-forward + بوابة SSL اختيارية."""
    log = resolve_progress(progress, quiet=quiet)
    save_step = 1 if output_dir is not None else 0
    gate_step = 1 if use_ssl_gate else 0
    log.begin(
        "بحث فرضيات Failed Breakout (walk-forward)",
        total_steps=6 + gate_step + save_step,
    )
    try:
        log.step("تهيئة + تحميل MBO", f"max_rows={max_rows}")
        seed_everything(global_seed)
        generator = rng if rng is not None else np.random.default_rng(global_seed)
        nq_frame = nq if isinstance(nq, pl.DataFrame) else load_mbo_frame(nq, max_rows=max_rows)
        if mnq is None:
            mnq_frame = nq_frame
            log.note(f"NQ={nq_frame.height:,} (nq_only)")
        else:
            mnq_frame = (
                mnq if isinstance(mnq, pl.DataFrame) else load_mbo_frame(mnq, max_rows=max_rows)
            )
            log.note(f"NQ={nq_frame.height:,} · MNQ={mnq_frame.height:,}")

        grid = tuple(specs) if specs is not None else default_breakout_grid()
        log.step("بناء ساعة البحث", f"interval_ns={interval_ns}")
        clock = cross_market_features(
            nq_frame,
            mnq_frame,
            interval_ns=interval_ns,
            lead_lag_window=2,
            latency_ns=0,
        )
        log.step("تجسيد شبكة فرضيات FB", f"candidates={len(grid)}")
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

        ssl_result: SSLPipelineResult | None = None
        candidates: tuple[str, ...] = tuple(hyp_cols)
        if use_ssl_gate:
            log.step("SSL tick + بوابة تأكيد سببية", f"window={ssl_window}")
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
            features, gated = apply_causal_ssl_gate(
                features,
                ssl_result.embeddings,
                hyp_cols,
                z_col="z0",
                quantile=_SSL_GATE_QUANTILE,
            )
            candidates = gated
            log.note(f"مرشّحون بعد البوابة: {len(candidates)} / {len(hyp_cols)}")

        policy = TemporalPolicy.for_run(interval_ns=interval_ns, window=ssl_window)
        embargo = policy.embargo_time_units(interval_ns=interval_ns)
        log.step("اختيار walk-forward", f"n_splits={n_splits}")
        fold_df, oos_ic, oos_p, oos_n, best = walk_forward_select_hypotheses(
            features,
            candidates,
            price_col="nq_close",
            horizon=horizon,
            n_splits=n_splits,
            embargo=embargo,
            purge_samples=policy.purge_samples(),
            n_permutations=n_permutations,
            rng=generator,
        )
        log.note(f"best_oos={best!r} · oos_ic={oos_ic:.4g}")

        log.step("شاشة استكشافية")
        explor = exploratory_screen_candidates(
            features,
            candidates,
            price_col="nq_close",
            horizon=horizon,
            alpha=alpha,
            n_permutations=n_permutations,
            rng=generator,
        )

        log.step("تقرير البحث")
        assistant = ResearchAssistant(alpha=alpha)
        evidence = Evidence(
            id="fb_search:oos_ic",
            source="breakout_hypothesis_search",
            metric="IC",
            value=oos_ic,
            pvalue=oos_p,
            sample_size=oos_n,
            detail=f"best_oos_spec={best!r}; walk-forward nested selection",
        )
        claim = (
            f"فرضية Failed Breakout المختارة بـ walk-forward "
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
            title="Failed Breakout Hypothesis Search — Walk-Forward + SSL Gate",
        )
        result = BreakoutHypothesisSearchResult(
            features=features,
            specs=grid,
            candidate_columns=candidates,
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
    "default_breakout_grid",
    "materialize_breakout_hypotheses",
    "search_fail_breakout_hypotheses",
]
