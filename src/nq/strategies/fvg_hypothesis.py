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
from nq.core.temporal_policy import (
    TemporalPolicy,
    align_horizon_to_context,
    resolve_grid_context_interval,
)
from nq.ingestion.reader import load_mbo_frame
from nq.models.splitting import purged_walk_forward_split
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_tick_pipeline
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.capacity import (
    LEAN_GATE_QUANTILES,
    SEARCH_N_PERMUTATIONS,
    UNDERSTAND_N_PERMUTATIONS,
)
from nq.research.evidence import Evidence
from nq.research.progress import PipelineProgress, ProgressLike, resolve_progress
from nq.research.understanding import (
    UnderstandingReport,
    run_understanding_layers,
    write_understanding_outputs,
)
from nq.simulation.cross_market import cross_market_features
from nq.simulation.fvg import (
    NS_PER_MIN,
    build_ohlcv_bars,
    failed_fvg_from_bars,
)
from nq.statistics.metrics import information_coefficient
from nq.strategies.depth_entry_filter import (
    DepthEntrySpec,
    attach_depth_path_to_features,
    count_signal_hits,
    generate_depth_entry_candidates,
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
    understanding: UnderstandingReport | None = None


def default_fvg_grid() -> tuple[FvgHypothesisSpec, ...]:
    """شبكة كاملة: أطر إشارة/FVG + نوافذ + عتبات جهد-بلا-نتيجة.

    ``vol_price_mult`` = سقف مدى/ATR؛ ``vol_volume_mult`` = أرضية حجم.
    تُستبعد النوافذ أقصر من إطار تشكيل FVG.
    """
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
            if win_m < fvg_m:
                continue
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


def core_fvg_grid() -> tuple[FvgHypothesisSpec, ...]:
    """نواة مضغوطة للبحث ضمن القدرة — نفس القواعد السببية."""
    pairs = ((15, 30), (15, 60), (30, 60), (30, 120))
    windows = (60, 120)
    thresholds = ((1.2, 1.3), (1.5, 1.8))
    specs: list[FvgHypothesisSpec] = []
    for sig_m, fvg_m in pairs:
        for win_m in windows:
            if win_m < fvg_m:
                continue
            for vp, vv in thresholds:
                name = (
                    f"core_s{sig_m}_f{fvg_m}_w{win_m}_"
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
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يبني أعمدة فرضيات على ساعة تقييم مشتركة (نبضة تطابقية عند bucket_end)."""
    if AVAILABILITY_TS not in clock.columns:
        raise ValueError(f"clock requires {AVAILABILITY_TS}")
    if clock.height == 0 or not specs:
        return clock

    log = progress
    n_specs = len(specs)
    if log is not None:
        log.op(f"تجسيد {n_specs} فرضية FVG (كل فرضية تُطبع)")

    bars_cache: dict[int, pl.DataFrame] = {}

    def _bars(interval_ns: int) -> pl.DataFrame:
        cached = bars_cache.get(interval_ns)
        if cached is None:
            if log is not None:
                log.op(f"بناء OHLCV interval_ns={interval_ns}")
            cached = build_ohlcv_bars(nq, interval_ns=interval_ns)
            bars_cache[interval_ns] = cached
            if log is not None:
                log.op(f"OHLCV جاهز: {cached.height:,} شمعة")
        return cached

    out = clock.sort(AVAILABILITY_TS)
    for i, spec in enumerate(specs, start=1):
        if log is not None:
            log.op(
                f"فرضية [{i}/{n_specs}] {spec.name} · h1={spec.h1_interval_ns} · "
                f"sig={spec.signal_interval_ns}"
            )
        raw = failed_fvg_from_bars(
            _bars(spec.h1_interval_ns),
            _bars(spec.signal_interval_ns),
            fvg_window_ns=spec.fvg_window_ns,
            vol_price_mult=spec.vol_price_mult,
            vol_volume_mult=spec.vol_volume_mult,
            progress=log,
        )
        col = spec.column()
        if raw.height == 0:
            out = out.with_columns(pl.lit(0.0).alias(col))
            if log is not None:
                log.op(f"  → {col}: 0 إشارات")
        else:
            right = (
                raw.select(AVAILABILITY_TS, pl.col("fail_fvg").alias(col))
                .sort(AVAILABILITY_TS)
                .unique(subset=[AVAILABILITY_TS], keep="last")
            )
            if col in out.columns:
                out = out.drop(col)
            out = out.join(right, on=AVAILABILITY_TS, how="left").with_columns(
                pl.col(col).fill_null(0.0)
            )
            if log is not None:
                n_sig = int((raw["fail_fvg"] != 0).sum())
                log.op(f"  → {col}: {n_sig:,} إشارة / {raw.height:,} صف (pulse join)")
        if log is not None:
            log.heartbeat(i, n_specs, label="materialize_FVG", force=True)
    if log is not None:
        log.op(f"انتهى تجسيد {n_specs} فرضية FVG")
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

    ``z`` الناقص → بوابة مغلقة (ليس ``fill_null(0)`` الذي يفتح البوابة زيفًا).
    التداخل مع WF: العتبة عند ``t`` من الماضي فقط؛ الاختيار عبر
    ``walk_forward_select_hypotheses`` + selection-under-null.
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
    abs_z = pl.col(z_col).abs()
    past_q = abs_z.shift(1).rolling_quantile(
        quantile, window_size=_SSL_GATE_WINDOW, min_samples=_SSL_GATE_MIN_SAMPLES
    )
    # null z أو عتبة غير جاهزة → لا تمرّ البوابة
    gate = (abs_z.is_not_null() & past_q.is_not_null() & (abs_z >= past_q)).cast(pl.Float64)
    gated_exprs = [
        (pl.col(c).fill_null(0.0) * pl.col("_ssl_gate")).alias(f"{c}__ssl") for c in signal_columns
    ]
    with_gate = joined.with_columns(gate.alias("_ssl_gate")).with_columns(gated_exprs)
    return with_gate.drop("_ssl_gate"), gated


def _ic_on_slice(
    values: np.ndarray,
    forward: np.ndarray,
    idx: np.ndarray,
) -> float:
    """Spearman IC on a fold slice — ranking only (no permutation cost)."""
    if idx.size == 0:
        return 0.0
    v = np.asarray(values[idx], dtype=np.float64)
    f = np.asarray(forward[idx], dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(f)
    v, f = v[mask], f[mask]
    if v.shape[0] < _MIN_OOS_SAMPLES or float(np.std(v)) == 0.0:
        return 0.0
    return float(information_coefficient(v, f, method="spearman"))


def walk_forward_select_hypotheses(  # noqa: PLR0912, PLR0915
    features: pl.DataFrame,
    candidate_columns: Sequence[str],
    *,
    price_col: str = "nq_close",
    horizon: int = 1,
    n_splits: int = 3,
    embargo: int = 0,
    purge_samples: int = 0,
    n_permutations: int = SEARCH_N_PERMUTATIONS,
    selection_aware_null: bool = True,
    rng: np.random.Generator | None = None,
    progress: ProgressLike | None = None,
) -> tuple[pl.DataFrame, float, float, int, str | None]:
    """اختيار فرضية على التدريب فقط؛ قياس IC خارج العينة على الاختبار.

    ترتيب المرشّحين داخل كل طيّة = Spearman IC فقط (بلا تبديلات).
    الدلالة الإحصائية:

    * ``selection_aware_null=True`` (افتراضي): لكل تبديل يُعاد اختيار أفضل مرشّح
      على التدريب تحت العوائد المخلوطة ثم يُقاس IC خارج العينة — يحاسب على
      ميزانية المرشّحين (selection under the null).
    * وإلا: تبديل واحد على سلسلة الإشارة المختارة مجمّعة (أضعف عند الشبكات الكبيرة).

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
            log.op(f"walk-forward: تخطّي (cols={len(cols)} · rows={work.height})")
        return empty, 0.0, 1.0, 0, None

    # عزل أفق العائد الأمامي عن كتلة الاختبار
    effective_purge = max(int(purge_samples), int(horizon))
    folds = purged_walk_forward_split(
        times,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=effective_purge,
        min_train_size=max(10, work.height // (n_splits + 2)),
    )
    if log is not None:
        log.op(
            f"walk-forward: {len(folds)} طيّات · candidates={len(cols)} · "
            f"rank=IC · oos_perm={n_permutations} · "
            f"selection_null={selection_aware_null} · purge={effective_purge}"
        )

    col_mats = {c: work[c].to_numpy().astype(np.float64) for c in cols}
    rows: list[dict[str, float | int | str]] = []
    oos_values = np.full(work.height, np.nan, dtype=np.float64)
    oos_fwd = np.full(work.height, np.nan, dtype=np.float64)

    for fold_i, fold in enumerate(folds):
        if log is not None:
            log.op(
                f"WF fold {fold_i + 1}/{len(folds)} "
                f"(train={len(fold.train_idx):,} · test={len(fold.test_idx):,})"
            )
        best_name = cols[0]
        best_ic = -1e18
        for col_i, col in enumerate(cols, start=1):
            vals = col_mats[col]
            ic = _ic_on_slice(vals, forward, fold.train_idx)
            if abs(ic) > abs(best_ic) or (abs(ic) == abs(best_ic) and ic > best_ic):
                best_ic = ic
                best_name = col
            if log is not None:
                log.heartbeat(
                    col_i,
                    len(cols),
                    label=f"WF fold {fold_i + 1} candidates",
                )
        test_vals = col_mats[best_name]
        test_ic = _ic_on_slice(test_vals, forward, fold.test_idx)
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
            log.op(
                f"WF fold {fold_i + 1}: selected={best_name!r} · "
                f"train_ic={best_ic:.4g} · test_ic={test_ic:.4g}"
            )

    fold_df = pl.DataFrame(rows) if rows else empty
    mask = np.isfinite(oos_values) & np.isfinite(oos_fwd)
    oos_n = int(mask.sum())
    if oos_n >= _MIN_OOS_SAMPLES and float(np.std(oos_values[mask])) > 0:
        oos_ic = float(information_coefficient(oos_values[mask], oos_fwd[mask], method="spearman"))
        if selection_aware_null and n_permutations > 0 and folds:
            if log is not None:
                log.op(f"WF selection-under-null: {n_permutations} تبديل · candidates={len(cols)}")
            null_ics = np.empty(n_permutations, dtype=np.float64)
            for p_i in range(n_permutations):
                perm_fwd = generator.permutation(forward)
                null_oos = np.full(work.height, np.nan, dtype=np.float64)
                null_y = np.full(work.height, np.nan, dtype=np.float64)
                for fold in folds:
                    best_name = cols[0]
                    best_ic = -1e18
                    for col in cols:
                        ic = _ic_on_slice(col_mats[col], perm_fwd, fold.train_idx)
                        if abs(ic) > abs(best_ic) or (abs(ic) == abs(best_ic) and ic > best_ic):
                            best_ic = ic
                            best_name = col
                    null_oos[fold.test_idx] = col_mats[best_name][fold.test_idx]
                    null_y[fold.test_idx] = perm_fwd[fold.test_idx]
                nmask = np.isfinite(null_oos) & np.isfinite(null_y)
                if int(nmask.sum()) >= _MIN_OOS_SAMPLES and float(np.std(null_oos[nmask])) > 0:
                    null_ics[p_i] = float(
                        information_coefficient(null_oos[nmask], null_y[nmask], method="spearman")
                    )
                else:
                    null_ics[p_i] = 0.0
                if log is not None and ((p_i + 1) % max(1, n_permutations // 10) == 0):
                    log.heartbeat(p_i + 1, n_permutations, label="WF-selection-null")
            oos_p = float((int(np.sum(np.abs(null_ics) >= abs(oos_ic))) + 1) / (n_permutations + 1))
        else:
            oos_ev = evaluate_signal(
                "wf_selected",
                oos_values[mask],
                oos_fwd[mask],
                n_permutations=n_permutations,
                rng=generator,
                progress=log,
                progress_label="WF-oos-perm",
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
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """فرز BH استكشافي على المرشّحين (ليس أساس اختيار الإعداد على نفس العيّنة)."""
    generator = rng if rng is not None else np.random.default_rng(0)
    log = progress
    work = features.sort(AVAILABILITY_TS)
    forward = align_forward_returns(work[price_col].to_numpy().astype(np.float64), horizon=horizon)
    cols = [c for c in candidate_columns if c in work.columns]
    n_cols = len(cols)
    if log is not None:
        log.op(f"شاشة استكشافية: {n_cols} مرشّح · n_perm={n_permutations}")
    evaluations = []
    for i, col in enumerate(cols, start=1):
        if log is not None:
            log.op(f"استكشاف [{i}/{n_cols}]: {col!r}")
        evaluations.append(
            evaluate_signal(
                col,
                work[col].to_numpy().astype(np.float64),
                forward,
                n_permutations=n_permutations,
                rng=generator,
                progress=log,
                progress_label=f"perm:{col}",
            )
        )
        if log is not None:
            log.heartbeat(i, n_cols, label="exploratory", force=True)
    return screen_signals(evaluations, alpha=alpha)


def search_fail_fvg_hypotheses(  # noqa: PLR0912, PLR0915
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    specs: Sequence[FvgHypothesisSpec] | None = None,
    interval_ns: int = 1_000_000_000,
    horizon: int = 1,
    use_ssl_gate: bool = True,
    use_depth_filter: bool = True,
    ssl_window: int = 5,
    ssl_components: int = 4,
    n_splits: int = 3,
    alpha: float = 0.05,
    n_permutations: int = SEARCH_N_PERMUTATIONS,
    max_rows: int | None = None,
    global_seed: int = 0,
    output_dir: Path | str | None = None,
    rng: np.random.Generator | None = None,
    progress: PipelineProgress | bool | None = None,
    quiet: bool = False,
    understand: bool = False,
    full_grid: bool = False,
    lean_filters: bool = True,
    exploratory: bool = False,
    understand_n_permutations: int = UNDERSTAND_N_PERMUTATIONS,
) -> FvgHypothesisSearchResult:
    """يبحث أفضل إعداد/تايم فريم Failed FVG بـ walk-forward + بوابة SSL اختيارية.

    افتراضيًا: نواة مضغوطة (``core_fvg_grid``) + فلاتر عمق lean + بلا شاشة استكشافية.
    ``full_grid=True`` يستخدم الشبكة الكاملة (~84). ``lean_filters=False`` يوسّع
    كمّيات العمق. الدلالة permutation على OOS المجمّع فقط.

    ``understand``: طبقات فهم كمية بعد الاختيار — OOS فقط، بلا تغيير ``best_oos_spec``.
    """
    log = resolve_progress(progress, quiet=quiet)
    save_step = 1 if output_dir is not None else 0
    gate_step = 1 if use_ssl_gate else 0
    depth_step = 1 if use_depth_filter else 0
    understand_step = 1 if understand else 0
    explor_step = 1 if exploratory else 0
    if output_dir is not None:
        out_early = Path(output_dir)
        out_early.mkdir(parents=True, exist_ok=True)
        log.attach_log(out_early / "progress.log")
    log.begin(
        "بحث فرضيات Failed FVG (walk-forward)",
        total_steps=5 + gate_step + depth_step + save_step + understand_step + explor_step,
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

        if specs is not None:
            grid = tuple(specs)
        elif full_grid:
            grid = default_fvg_grid()
            log.note(f"شبكة FVG كاملة: {len(grid)}")
        else:
            grid = core_fvg_grid()
            log.note(f"نواة FVG مضغوطة: {len(grid)} (capacity-correct)")
        log.step("بناء ساعة البحث cross-market", f"interval_ns={interval_ns}")
        clock = cross_market_features(
            nq_frame,
            mnq_frame,
            interval_ns=interval_ns,
            lead_lag_window=2,
            latency_ns=0,
        )
        log.step("تجسيد شبكة فرضيات FVG", f"candidates={len(grid)}")
        features = materialize_fvg_hypotheses(nq_frame, grid, clock=clock, progress=log)
        hyp_cols = [s.column() for s in grid]
        for col in hyp_cols:
            if col in features.columns:
                features = features.with_columns(pl.col(col).fill_null(0.0))
        log.note(f"features={features.height:,} صف × {features.width} عمود")

        ctx_interval, mixed_tf = resolve_grid_context_interval(
            [s.signal_interval_ns for s in grid],
            default_ns=30 * NS_PER_MIN,
        )
        if mixed_tf:
            log.note(
                f"شبكة TF مختلطة — سياق العمق/الأفق على max={ctx_interval}ns "
                f"(مرشّحات أقصر تُقيَّم تحت أفق أطول)"
            )
        eval_horizon = align_horizon_to_context(
            horizon,
            research_interval_ns=interval_ns,
            context_interval_ns=ctx_interval,
        )
        if eval_horizon != int(horizon):
            log.note(
                f"محاذاة horizon: {horizon} → {eval_horizon} "
                f"(سياق {ctx_interval}ns / ساعة {interval_ns}ns)"
            )
        horizon = eval_horizon

        ssl_result: SSLPipelineResult | None = None
        candidate_list: list[str] = list(hyp_cols)
        depth_specs: tuple[DepthEntrySpec, ...] = ()
        if use_depth_filter:
            log.step(
                "مسار أحداث العمق داخل شمعة الإشارة → مرشّحي دخول",
                f"interval_ns={ctx_interval}",
            )
            features = attach_depth_path_to_features(
                features,
                nq_frame,
                interval_ns=ctx_interval,
                progress=log,
                signal_columns=hyp_cols,
            )
            depth_kwargs: dict[str, object] = {}
            if lean_filters:
                depth_kwargs["quantiles"] = LEAN_GATE_QUANTILES
            features, depth_cols, depth_specs = generate_depth_entry_candidates(
                features,
                hyp_cols,
                **depth_kwargs,  # type: ignore[arg-type]
            )
            candidate_list.extend(list(depth_cols))
            log.note(f"مرشّحو عمق: {len(depth_cols)} · lean={lean_filters}")

        if use_ssl_gate:
            log.step("تشغيل SSL tick + بوابة سببية", f"window={ssl_window}")
            min_ssl_hits = max(3, int(n_splits))
            base_hits = count_signal_hits(features, hyp_cols)
            if base_hits < min_ssl_hits:
                log.note(f"تخطي SSL — إشارات أساس غير كافية (hits={base_hits} < {min_ssl_hits})")
            else:
                log.note(f"إشارات أساس: hits={base_hits} ≥ {min_ssl_hits}")
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
                # احتفظ بالأساس + البوابة + العمق (لا تستبدل الأساس فقط بـ __ssl)
                candidate_list = list(dict.fromkeys([*hyp_cols, *gated, *candidate_list]))
                log.note(f"مرشّحون بعد الأساس+البوابة+عمق: {len(candidate_list)}")

        seen: set[str] = set()
        uniq: list[str] = []
        for c in candidate_list:
            if c in features.columns and c not in seen:
                seen.add(c)
                uniq.append(c)
        candidates = tuple(uniq)

        policy = TemporalPolicy.for_run(interval_ns=interval_ns, window=ssl_window, horizon=horizon)
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

        if exploratory:
            log.step("شاشة استكشافية للمرشّحين")
            explor = exploratory_screen_candidates(
                features,
                candidates,
                price_col="nq_close",
                horizon=horizon,
                alpha=alpha,
                n_permutations=n_permutations,
                rng=generator,
                progress=log,
            )
        else:
            explor = pl.DataFrame(
                schema={
                    "name": pl.Utf8(),
                    "n": pl.Int64(),
                    "ic": pl.Float64(),
                    "ic_pvalue": pl.Float64(),
                    "adjusted_pvalue": pl.Float64(),
                    "sharpe": pl.Float64(),
                    "selected": pl.Boolean(),
                }
            )
            log.note("تخطّي الشاشة الاستكشافية (ليست أساس الاختيار)")

        log.step("كتابة تقرير البحث الموثّق")
        assistant = ResearchAssistant(alpha=alpha)
        evidence = Evidence(
            id="fvg_search:oos_ic",
            source="fvg_hypothesis_search",
            metric="IC",
            value=oos_ic,
            pvalue=oos_p,
            sample_size=oos_n,
            detail=(
                f"best_oos_spec={best!r}; depth_filters={len(depth_specs)}; "
                f"lean_filters={lean_filters}; walk-forward nested selection"
            ),
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
            title="Failed FVG Hypothesis Search — Walk-Forward + SSL/Depth Gates",
        )

        understanding: UnderstandingReport | None = None
        if understand and best is not None and best in features.columns:
            log.step("طبقات الفهم الكمي (OOS فقط — بلا تغيير اختيار)")
            emb = ssl_result.embeddings if ssl_result is not None else None
            understanding = run_understanding_layers(
                features,
                selected_column=best,
                fold_selections=fold_df,
                embeddings=emb,
                price_col="nq_close",
                horizon=horizon,
                interval_ns=interval_ns,
                ssl_window=ssl_window,
                n_splits=n_splits,
                n_permutations=understand_n_permutations,
                seed=global_seed,
                progress=log,
            )
            log.note(
                f"understanding layers={list(understanding.layers)} · "
                f"findings={len(understanding.findings)}"
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
            understanding=understanding,
        )

        if output_dir is not None:
            out = Path(output_dir)
            log.step("حفظ المخرجات", str(out.resolve()))
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.md").write_text(report.to_markdown(), encoding="utf-8")
            features.write_parquet(out / "features.parquet")
            fold_df.write_parquet(out / "fold_selections.parquet")
            explor.write_parquet(out / "exploratory_screen.parquet")
            if depth_specs:
                pl.DataFrame(
                    {
                        "column": [s.column() for s in depth_specs],
                        "base": [s.base_column for s in depth_specs],
                        "name": [s.name for s in depth_specs],
                        "kind": [s.kind for s in depth_specs],
                    }
                ).write_parquet(out / "depth_entry_specs.parquet")
            if ssl_result is not None and ssl_result.metrics.height > 0:
                ssl_result.metrics.write_parquet(out / "ssl_metrics.parquet")
            if understanding is not None:
                write_understanding_outputs(understanding, out)
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
    "core_fvg_grid",
    "default_fvg_grid",
    "exploratory_screen_candidates",
    "materialize_fvg_hypotheses",
    "search_fail_fvg_hypotheses",
    "walk_forward_select_hypotheses",
]
