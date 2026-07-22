"""بحث فرضيات Failed FVG (تايم فريم + إعدادات) بلا تسريب زمني.

المبادئ الملزِمة هنا:

1. **التسريب:** كل إشارة سببية (``availability_ts``)؛ اختيار الإعدادات بـ
   walk-forward purged (تدريب → اختيار → اختبار خارج العينة فقط).
2. **الصرامة:** IC + permutation + BH على المرشّحين؛ مسار OOS هو الحكم.
3. **الأداء:** كاش شموع OHLCV حسب ``interval_ns``.
4. **MBO فقط:** الفرضيات من ``failed_fvg_from_bars`` على شريط الصفقات.

بوابة SSL: asof خلفي للتمثيلات ``z*`` + عتبة سببية (كمّية ماضية فقط).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nq.alpha.signals import align_forward_returns, evaluate_signal, screen_signals
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import seed_everything
from nq.core.temporal_policy import TemporalPolicy
from nq.ingestion.reader import load_mbo_frame
from nq.models.splitting import purged_walk_forward_split
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_tick_pipeline
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.progress import PipelineProgress, resolve_progress
from nq.simulation.cross_market import cross_market_features
from nq.simulation.fvg import (
    NS_PER_MIN,
    build_ohlcv_bars,
    failed_fvg_from_bars,
)

_MIN_ROWS_FOR_SEARCH = 20
_MIN_OOS_SAMPLES = 8
_SSL_GATE_WINDOW = 50
_SSL_GATE_MIN_SAMPLES = 10
_SSL_GATE_QUANTILE = 0.7


@dataclass(frozen=True, slots=True)
class FvgHypothesisSpec:
    """فرضية Failed FVG بإطار زمني وعتبات جهد ثابتة (قاعدة سببية)."""

    name: str
    h1_interval_ns: int
    signal_interval_ns: int
    fvg_window_ns: int
    vol_price_mult: float = 1.2
    vol_volume_mult: float = 1.3

    def column(self) -> str:
        return f"fail_fvg__{self.name}"


@dataclass(frozen=True, slots=True)
class FvgHypothesisSearchResult:
    """مخرجات بحث الفرضيات: شبكة مرشّحين + اختيار walk-forward + تقرير."""

    features: pl.DataFrame
    specs: tuple[FvgHypothesisSpec, ...]
    candidate_columns: tuple[str, ...]
    fold_selections: pl.DataFrame
    exploratory_screen: pl.DataFrame
    oos_selected_ic: float
    best_oos_spec: str | None
    ssl: SSLPipelineResult | None
    report: ResearchReport


def default_fvg_grid() -> tuple[FvgHypothesisSpec, ...]:
    """شبكة صغيرة حتمية: أطر إشارة/FVG + نوافذ + عتبات جهد."""
    pairs = (
        (15, 30),
        (15, 60),
        (30, 60),
        (30, 120),
        (5, 15),
        (10, 30),
        (45, 90),
    )
    windows = (60, 90, 120)
    thresholds = ((1.1, 1.2), (1.2, 1.3), (1.3, 1.5), (1.5, 1.8))
    specs: list[FvgHypothesisSpec] = []
    for sig_m, fvg_m in pairs:
        if fvg_m < sig_m:
            continue
        for win_m in windows:
            for vp, vv in thresholds:
                name = (
                    f"s{sig_m}_f{fvg_m}_w{win_m}_"
                    f"p{str(vp).replace('.', 'p')}_v{str(vv).replace('.', 'p')}"
                )
                specs.append(
                    FvgHypothesisSpec(
                        name=name,
                        h1_interval_ns=fvg_m * NS_PER_MIN,
                        signal_interval_ns=sig_m * NS_PER_MIN,
                        fvg_window_ns=win_m * NS_PER_MIN,
                        vol_price_mult=vp,
                        vol_volume_mult=vv,
                    )
                )
    return tuple(specs)


def materialize_fvg_hypotheses(
    nq: pl.DataFrame,
    specs: Sequence[FvgHypothesisSpec],
    *,
    clock: pl.DataFrame,
) -> pl.DataFrame:
    """يبني أعمدة فرضيات على ساعة تقييم مشتركة (asof خلفي فقط)."""
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
        raw = failed_fvg_from_bars(
            _bars(spec.h1_interval_ns),
            _bars(spec.signal_interval_ns),
            fvg_window_ns=spec.fvg_window_ns,
            vol_price_mult=spec.vol_price_mult,
            vol_volume_mult=spec.vol_volume_mult,
        )
        col = spec.column()
        if raw.height == 0:
            out = out.with_columns(pl.lit(0.0).alias(col))
            continue
        right = (
            raw.select(AVAILABILITY_TS, pl.col("fail_fvg").alias(col))
            .sort(AVAILABILITY_TS)
            .unique(subset=[AVAILABILITY_TS], keep="last")
        )
        if col in out.columns:
            out = out.drop(col)
        out = out.join_asof(right, on=AVAILABILITY_TS, strategy="backward").with_columns(
            pl.col(col).fill_null(0.0)
        )
    return out


def apply_causal_ssl_gate(
    features: pl.DataFrame,
    embeddings: pl.DataFrame,
    signal_columns: Sequence[str],
    *,
    z_col: str = "z0",
    quantile: float = 0.7,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """بوابة SSL سببية: asof خلفي + كمّية ماضية لـ ``|z|`` (بدون مستقبل).

    يُنتج أعمدة ``{signal}__ssl`` = الإشارة × بوابة (0/1).
    """
    gated = tuple(f"{c}__ssl" for c in signal_columns)
    if embeddings.height == 0 or z_col not in embeddings.columns:
        zeros = features.with_columns([pl.lit(0.0).alias(c) for c in gated])
        return zeros, gated

    right = embeddings.select(AVAILABILITY_TS, z_col).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop = [c for c in (z_col,) if c in left.columns]
    if drop:
        left = left.drop(drop)
    joined = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
    abs_z = pl.col(z_col).abs().fill_null(0.0)
    # كمّية ماضية فقط: shift(1) يمنع استخدام قيمة اللحظة الحالية في العتبة
    past_q = abs_z.shift(1).rolling_quantile(
        quantile, window_size=_SSL_GATE_WINDOW, min_samples=_SSL_GATE_MIN_SAMPLES
    )
    gate = (abs_z >= past_q.fill_null(float("inf"))).cast(pl.Float64)
    gated_exprs = [
        (pl.col(c).fill_null(0.0) * pl.col("_ssl_gate")).alias(f"{c}__ssl") for c in signal_columns
    ]
    with_gate = joined.with_columns(gate.alias("_ssl_gate")).with_columns(gated_exprs)
    return with_gate.drop("_ssl_gate"), gated


def _ic_on_slice(
    values: np.ndarray,
    forward: np.ndarray,
    idx: np.ndarray,
    *,
    name: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    if idx.size == 0:
        return 0.0
    ev = evaluate_signal(
        name,
        values[idx],
        forward[idx],
        n_permutations=n_permutations,
        rng=rng,
    )
    return float(ev.ic)


def walk_forward_select_hypotheses(
    features: pl.DataFrame,
    candidate_columns: Sequence[str],
    *,
    price_col: str = "nq_close",
    horizon: int = 1,
    n_splits: int = 3,
    embargo: int = 0,
    purge_samples: int = 0,
    n_permutations: int = 200,
    rng: np.random.Generator | None = None,
    progress: object | None = None,
) -> tuple[pl.DataFrame, float, float, int, str | None]:
    """اختيار فرضية على التدريب فقط؛ قياس IC خارج العينة على الاختبار.

    يُعيد: (fold_df, oos_ic, oos_pvalue, oos_n, best_name)
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    log = progress
    work = features.sort(AVAILABILITY_TS)
    times = work[AVAILABILITY_TS].to_numpy()
    prices = work[price_col].to_numpy().astype(np.float64)
    forward = align_forward_returns(prices, horizon=horizon)
    cols = [c for c in candidate_columns if c in work.columns]
    empty = pl.DataFrame(
        schema={
            "fold": pl.Int64(),
            "selected": pl.Utf8(),
            "train_ic": pl.Float64(),
            "test_ic": pl.Float64(),
        }
    )
    if not cols or work.height < _MIN_ROWS_FOR_SEARCH:
        if log is not None:
            log.op(  # type: ignore[union-attr]
                f"walk-forward: تخطّي (cols={len(cols)} · rows={work.height})"
            )
        return empty, 0.0, 1.0, 0, None

    folds = purged_walk_forward_split(
        times,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=purge_samples,
        min_train_size=max(10, work.height // (n_splits + 2)),
    )
    if log is not None:
        log.op(  # type: ignore[union-attr]
            f"walk-forward: {len(folds)} طيّات · candidates={len(cols)} · "
            f"n_perm={n_permutations}"
        )
    rows: list[dict[str, float | int | str]] = []
    oos_values = np.full(work.height, np.nan, dtype=np.float64)
    oos_fwd = np.full(work.height, np.nan, dtype=np.float64)
    for fold_i, fold in enumerate(folds):
        if log is not None:
            log.op(  # type: ignore[union-attr]
                f"WF fold {fold_i + 1}/{len(folds)} "
                f"(train={len(fold.train_idx):,} · test={len(fold.test_idx):,})"
            )
        best_name = cols[0]
        best_ic = -1e18
        for col_i, col in enumerate(cols, start=1):
            vals = work[col].to_numpy().astype(np.float64)
            ic = _ic_on_slice(
                vals,
                forward,
                fold.train_idx,
                name=col,
                n_permutations=n_permutations,
                rng=generator,
            )
            if abs(ic) > abs(best_ic) or (abs(ic) == abs(best_ic) and ic > best_ic):
                best_ic = ic
                best_name = col
            if log is not None:
                log.heartbeat(  # type: ignore[union-attr]
                    col_i,
                    len(cols),
                    label=f"WF fold {fold_i + 1} candidates",
                )
        test_vals = work[best_name].to_numpy().astype(np.float64)
        test_ic = _ic_on_slice(
            test_vals,
            forward,
            fold.test_idx,
            name=best_name,
            n_permutations=n_permutations,
            rng=generator,
        )
        oos_values[fold.test_idx] = test_vals[fold.test_idx]
        oos_fwd[fold.test_idx] = forward[fold.test_idx]
        rows.append(
            {
                "fold": fold_i,
                "selected": best_name,
                "train_ic": float(best_ic),
                "test_ic": float(test_ic),
            }
        )
        if log is not None:
            log.op(  # type: ignore[union-attr]
                f"WF fold {fold_i + 1}: selected={best_name!r} · "
                f"train_ic={best_ic:.4g} · test_ic={test_ic:.4g}"
            )

    fold_df = pl.DataFrame(rows) if rows else empty
    mask = np.isfinite(oos_values) & np.isfinite(oos_fwd)
    oos_n = int(mask.sum())
    if oos_n >= _MIN_OOS_SAMPLES and float(np.std(oos_values[mask])) > 0:
        oos_ev = evaluate_signal(
            "wf_selected",
            oos_values[mask],
            oos_fwd[mask],
            n_permutations=n_permutations,
            rng=generator,
        )
        oos_ic = float(oos_ev.ic)
        oos_p = float(oos_ev.ic_pvalue)
    else:
        oos_ic = 0.0
        oos_p = 1.0
    selected_name: str | None = None
    if fold_df.height > 0:
        counts = (
            fold_df.group_by("selected").len().sort(["len", "selected"], descending=[True, False])
        )
        selected_name = str(counts["selected"][0])
    return fold_df, oos_ic, oos_p, oos_n, selected_name


def exploratory_screen_candidates(
    features: pl.DataFrame,
    candidate_columns: Sequence[str],
    *,
    price_col: str = "nq_close",
    horizon: int = 1,
    alpha: float = 0.05,
    n_permutations: int = 200,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """فرز BH استكشافي على المرشّحين (ليس أساس اختيار الإعداد على نفس العيّنة)."""
    generator = rng if rng is not None else np.random.default_rng(0)
    work = features.sort(AVAILABILITY_TS)
    forward = align_forward_returns(work[price_col].to_numpy().astype(np.float64), horizon=horizon)
    evaluations = []
    for col in candidate_columns:
        if col not in work.columns:
            continue
        evaluations.append(
            evaluate_signal(
                col,
                work[col].to_numpy().astype(np.float64),
                forward,
                n_permutations=n_permutations,
                rng=generator,
            )
        )
    return screen_signals(evaluations, alpha=alpha)


def search_fail_fvg_hypotheses(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    specs: Sequence[FvgHypothesisSpec] | None = None,
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
) -> FvgHypothesisSearchResult:
    """يبحث أفضل إعداد/تايم فريم Failed FVG بـ walk-forward + بوابة SSL اختيارية."""
    log = resolve_progress(progress, quiet=quiet)
    save_step = 1 if output_dir is not None else 0
    gate_step = 1 if use_ssl_gate else 0
    if output_dir is not None:
        out_early = Path(output_dir)
        out_early.mkdir(parents=True, exist_ok=True)
        log.attach_log(out_early / "progress.log")
    log.begin(
        "بحث فرضيات Failed FVG (walk-forward)",
        total_steps=6 + gate_step + save_step,
    )
    log.line("كل عملية تُطبع سطرًا بسطر — راقب progress.log أو stderr")
    try:
        log.step("تهيئة الحتمية + تحميل MBO", f"max_rows={max_rows}")
        seed_everything(global_seed)
        generator = rng if rng is not None else np.random.default_rng(global_seed)

        nq_frame = (
            nq
            if isinstance(nq, pl.DataFrame)
            else load_mbo_frame(nq, max_rows=max_rows, progress=log)
        )
        if mnq is None:
            mnq_frame = nq_frame
            log.note(f"NQ={nq_frame.height:,} صف (nq_only)")
        else:
            mnq_frame = (
                mnq
                if isinstance(mnq, pl.DataFrame)
                else load_mbo_frame(mnq, max_rows=max_rows, progress=log)
            )
            log.note(f"NQ={nq_frame.height:,} · MNQ={mnq_frame.height:,}")

        grid = tuple(specs) if specs is not None else default_fvg_grid()
        log.step("بناء ساعة البحث cross-market", f"interval_ns={interval_ns}")
        clock = cross_market_features(
            nq_frame,
            mnq_frame,
            interval_ns=interval_ns,
            lead_lag_window=2,
            latency_ns=0,
        )
        log.step("تجسيد شبكة فرضيات FVG", f"candidates={len(grid)}")
        hyp = materialize_fvg_hypotheses(nq_frame, grid, clock=clock)
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
        log.note(f"features={features.height:,} صف × {features.width} عمود")

        ssl_result: SSLPipelineResult | None = None
        candidates: tuple[str, ...] = tuple(hyp_cols)
        if use_ssl_gate:
            log.step("تشغيل SSL tick + بوابة سببية", f"window={ssl_window}")
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
        log.step(
            "اختيار walk-forward (purged)",
            f"n_splits={n_splits} · candidates={len(candidates)}",
        )
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
            progress=log,
        )
        log.note(f"best_oos={best!r} · oos_ic={oos_ic:.4g} · p={oos_p:.4g} · n={oos_n}")

        log.step("شاشة استكشافية للمرشّحين")
        explor = exploratory_screen_candidates(
            features,
            candidates,
            price_col="nq_close",
            horizon=horizon,
            alpha=alpha,
            n_permutations=n_permutations,
            rng=generator,
        )

        log.step("كتابة تقرير البحث الموثّق")
        assistant = ResearchAssistant(alpha=alpha)
        evidence = Evidence(
            id="fvg_search:oos_ic",
            source="fvg_hypothesis_search",
            metric="IC",
            value=oos_ic,
            pvalue=oos_p,
            sample_size=oos_n,
            detail=f"best_oos_spec={best!r}; walk-forward nested selection",
        )
        claim = (
            f"فرضية Failed FVG المختارة بـ walk-forward "
            f"(best={best!r}) تحقق IC خارج العينة = {oos_ic:.4g} (p={oos_p:.4g})."
        )
        findings = [
            assistant.generate_hypothesis(
                claim,
                evidence,
                requires_significance=True,
                category="fvg_search",
            )
        ]
        report = assistant.write_report(
            findings,
            title="Failed FVG Hypothesis Search — Walk-Forward + SSL Gate",
        )

        result = FvgHypothesisSearchResult(
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
            log.note(f"كُتبت الملفات في {out.resolve()}")

        log.done(f"best={best!r} · oos_ic={oos_ic:.4g}")
        return result
    except Exception as exc:
        log.fail(exc)
        raise


__all__ = [
    "FvgHypothesisSearchResult",
    "FvgHypothesisSpec",
    "apply_causal_ssl_gate",
    "default_fvg_grid",
    "materialize_fvg_hypotheses",
    "search_fail_fvg_hypotheses",
    "walk_forward_select_hypotheses",
]
