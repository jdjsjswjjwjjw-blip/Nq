"""Volume Profile / Auction — مسار واحد متصل (إشارة → تضليل → هولد → تنفيذ R:R).

ليست طبقة منفصلة عن المشروع: نفس الخط الموحّد ``run_research_pipeline`` ثم
امتداد تنفيذي داخل نفس الاستراتيجية:

1. تسجيل التضليل مرة واحدة من الخام → فلتر دفتر + براميل إدج.
2. رينج VP على **5 دقائق** (حدود علوي/متوسط/سفلي) + فعل على **30 ثانية**
   (ارتداد من متوازن · كسر+دخول من مختلّ) + SSL‖M9‖ألفا.
3. اختيار إشارة VP/FSM بـ walk-forward purged **قبل** التنفيذ.
4. حكم السوق بعد هولد (درجات من الخام المُعاد استخدامها) + بحث R:R معزول OOS.
5. دمج أعمدة التنفيذ للتقرير فقط — بلا تشعّب مسارات.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nq.alpha.discovery import AlphaDiscovery
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.temporal_policy import TemporalPolicy
from nq.ingestion.reader import load_mbo_frame
from nq.models.ssl_pipeline import SSLPipelineResult
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.orchestrator import (
    PipelineConfig,
    _attach_auction_vp_signals,
    run_research_pipeline,
)
from nq.research.progress import ProgressLike, resolve_progress
from nq.research.unified import UnifiedResearchReport
from nq.simulation.auction import (
    VP_PROFILE_INTERVAL_NS,
    VP_SIGNAL_INTERVAL_NS,
    auction_action_states,
    auction_signals_from_states,
)
from nq.simulation.deceptive_liquidity import (
    DECEPTIVE_FEATURE_COLUMNS,
    DeceptiveLiquidityConfig,
    deceptive_features_by_bucket,
    filter_deceptive_liquidity,
    score_deceptive_events,
)
from nq.simulation.edge_execution_plan import (
    EdgeSearchSpec,
    default_edge_search_grid,
    run_edge_plan,
    search_best_edge_spec,
    summarize_edge_trades,
)
from nq.simulation.market_truth import MARKET_TRUTH_COLUMNS
from nq.strategies.fvg_hypothesis import walk_forward_select_hypotheses

#: مرشّحو IC = متنبّئات اتجاهية موقّعة فقط.
#: مسافات/حدود VP (``vp_rel_*`` / excess / حدود مطلقة) وأعلام نظام
#: (balance/imbalance/session) تبقى في الإطار للسياق/FSM لكنها **ليست** ألفا IC.
_VP_AUCTION_FOCUS = (
    "vp_of_delta",
    "vp_absorb",
    "vp_look_fail",
    "vp_order_accel",
    "vp_early_imbalance",
    "vp_flip_to_imbalance",
    "vp_pullback_defense",
    "vp_fr_accepted_expansion",
    "vp_fr_exit",
    "vp_auction_setup",
    "vp_fsm_break",
    "vp_fsm_build",
    "vp_fsm_retest",
    "nq_delta",
)

#: ميزات مستوى/مسافة — وصفيّة للمزاد، ليست إشارة اتجاه لـ Spearman IC.
_VP_LEVEL_DISTANCE_FEATURES = (
    "vp_upper",
    "vp_mid",
    "vp_lower",
    "vp_rel_upper",
    "vp_rel_mid",
    "vp_rel_lower",
    "vp_excess_upper",
    "vp_excess_lower",
    "vp_fr_upper",
    "vp_fr_mid",
    "vp_fr_lower",
    "vp_fr_start_ts",
    "vp_fr_end_ts",
)

#: أعلام حالة/نظام — ليست متنبّئ اتجاه سعري.
_VP_REGIME_STATE_FEATURES = (
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_in_value_frac",
    "vp_liquidity_session",
    "vp_fr_active",
    "vp_fr_in_balance",
)

#: إشارات VP بعد بوابة الهولد/التضليل — جزء من نفس أنطولوجيا الفرز.
_VP_GATED_FOCUS = (
    "vp_flip_gated",
    "vp_imbalance_gated",
    "vp_expansion_gated",
    "entry_gate",
    "market_true",
)

_EXEC_JOIN_COLUMNS = (
    *DECEPTIVE_FEATURE_COLUMNS,
    *MARKET_TRUTH_COLUMNS,
    # أعمدة خطة فقط — بلا edge_pnl/edge_hit على إطار اختيار الإشارة
    "edge_signal",
    "edge_entry",
    "edge_stop",
    "edge_target",
    "edge_rr",
    "edge_risk",
    "edge_reward",
    "vp_flip_gated",
    "vp_imbalance_gated",
    "vp_expansion_gated",
)


@dataclass(frozen=True, slots=True)
class VpAuctionResearchResult:
    """نتيجة VP المتصلة: بحث إشارة + طبقة تنفيذ R:R على نفس المخرجات."""

    features: pl.DataFrame
    alpha: AlphaDiscovery
    ssl: SSLPipelineResult | None
    report: ResearchReport
    unified: UnifiedResearchReport
    signal_columns: tuple[str, ...]
    fold_df: pl.DataFrame
    oos_ic: float
    oos_pvalue: float
    oos_n: int
    best_signal: str | None
    exploratory_only: bool
    # طبقة التنفيذ (متصلة — ليست مسارًا بديلًا)
    with_execution: bool
    raw_mbo_rows: int
    cleaned_mbo_rows: int
    edge_search_table: pl.DataFrame
    best_edge_spec: EdgeSearchSpec | None
    best_edge_row: dict[str, float | str]
    edge_trades: pl.DataFrame
    edge_summary: dict[str, float]


def _load_nq(
    nq: pl.DataFrame | str | Path,
    *,
    max_rows: int | None,
    progress: ProgressLike | None,
) -> pl.DataFrame:
    return load_mbo_frame(nq, max_rows=max_rows, progress=progress)


def _attach_execution_layer(
    features: pl.DataFrame,
    edge_frame: pl.DataFrame,
) -> pl.DataFrame:
    """يلحق أعمدة التضليل/الحكم/التنفيذ على إطار البحث asof خلفي."""
    if features.height == 0:
        return features
    want = [c for c in _EXEC_JOIN_COLUMNS if c in edge_frame.columns]
    if not want or AVAILABILITY_TS not in edge_frame.columns:
        return features
    right = edge_frame.select([AVAILABILITY_TS, *want]).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop_existing = [c for c in want if c in left.columns]
    if drop_existing:
        left = left.drop(drop_existing)
    joined = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
    # بوابات صفر عند غياب الهولد
    fill_zero = [
        c
        for c in (
            "entry_gate",
            "hold_ok",
            "market_true",
            "market_false",
            "edge_signal",
            "vp_flip_gated",
            "vp_imbalance_gated",
            "vp_expansion_gated",
        )
        if c in joined.columns
    ]
    if fill_zero:
        joined = joined.with_columns([pl.col(c).fill_null(0.0) for c in fill_zero])
    return joined


def _with_gated_vp_columns(edge_frame: pl.DataFrame, features: pl.DataFrame) -> pl.DataFrame:
    """يبني إشارات VP مبوّبة بالهولد على إطار التنفيذ قبل الدمج."""
    # نحتاج vp_* من features asof على edge buckets
    vp_cols = [
        c for c in ("vp_flip_to_imbalance", "vp_imbalance", "vp_expansion") if c in features.columns
    ]
    work = edge_frame
    if vp_cols and AVAILABILITY_TS in features.columns:
        right = features.select([AVAILABILITY_TS, *vp_cols]).sort(AVAILABILITY_TS)
        left = work.sort(AVAILABILITY_TS)
        drop_existing = [c for c in vp_cols if c in left.columns]
        if drop_existing:
            left = left.drop(drop_existing)
        work = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
        work = work.with_columns([pl.col(c).fill_null(0.0) for c in vp_cols])
    gate = pl.col("entry_gate") if "entry_gate" in work.columns else pl.lit(0.0)
    exprs: list[pl.Expr] = []
    if "vp_flip_to_imbalance" in work.columns:
        exprs.append((pl.col("vp_flip_to_imbalance") * gate).alias("vp_flip_gated"))
    else:
        exprs.append(pl.lit(0.0).alias("vp_flip_gated"))
    if "vp_imbalance" in work.columns:
        exprs.append((pl.col("vp_imbalance") * gate).alias("vp_imbalance_gated"))
    else:
        exprs.append(pl.lit(0.0).alias("vp_imbalance_gated"))
    if "vp_expansion" in work.columns:
        exprs.append((pl.col("vp_expansion") * gate).alias("vp_expansion_gated"))
    else:
        exprs.append(pl.lit(0.0).alias("vp_expansion_gated"))
    return work.with_columns(exprs)


def run_vp_auction_research(  # noqa: PLR0912, PLR0915
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    ssl_window: int = 5,
    ssl_components: int = 4,
    horizon: int = 1,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    n_splits: int = 3,
    max_rows: int | None = None,
    rng: np.random.Generator | None = None,
    output_dir: Path | str | None = None,
    quiet: bool = False,
    exploratory_full_sample: bool = False,
    with_execution: bool = True,
    drop_deceptive: bool = True,
    deceptive: DeceptiveLiquidityConfig | None = None,
    edge_grid: tuple[EdgeSearchSpec, ...] | None = None,
    edge_train_frac: float = 0.6,
    min_oos_trades: int = 3,
    min_oos_rr: float = 2.0,
    interval_ns: int | None = None,
    profile_interval_ns: int | None = None,
    streaming_features: bool = False,
) -> VpAuctionResearchResult:
    """مسار VP المتصل بترتيب علمي آمن.

    1. تنظيف دفتر مضلل (بلا أشباح).
    2. خط موحّد → ``vp_*`` على ساعة فعل 30ث مع رينج VP 5د.
    3. Walk-forward على ``vp_*`` فقط (قبل أي أعمدة تنفيذ).
    4. ثم هولد/R:R؛ درجات التضليل من الخام؛ إلحاق التنفيذ للتقرير فقط.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    log = resolve_progress(None, quiet=quiet)
    deco_cfg = deceptive if deceptive is not None else DeceptiveLiquidityConfig()
    sig_iv = int(interval_ns) if interval_ns is not None else VP_SIGNAL_INTERVAL_NS
    prof_iv = (
        int(profile_interval_ns) if profile_interval_ns is not None else VP_PROFILE_INTERVAL_NS
    )

    log.step("VP: تحميل MBO")
    raw = _load_nq(nq, max_rows=max_rows, progress=log)
    raw_n = raw.height
    # تسجيل التضليل مرة واحدة لليوم — يُعاد استخدامه للفلتر + براميل الإدج.
    scored_raw: pl.DataFrame | None = None
    deco_by_bucket: pl.DataFrame | None = None
    if drop_deceptive or with_execution:
        log.step("VP: تسجيل التضليل مرة واحدة", f"events={raw_n:,}")
        scored_raw = score_deceptive_events(raw, config=deco_cfg, progress=log)
    if drop_deceptive:
        log.step("VP: فلتر التضليل العلمي", "إسقاط دورة الأمر الكاملة · reuse scored")
        cleaned = filter_deceptive_liquidity(raw, config=deco_cfg, progress=log, scored=scored_raw)
    else:
        cleaned = raw
    cleaned_n = cleaned.height
    if with_execution and scored_raw is not None:
        log.step("VP: براميل التضليل (مرة واحدة)", f"interval_ns={sig_iv}")
        deco_by_bucket = deceptive_features_by_bucket(
            raw,
            interval_ns=sig_iv,
            config=deco_cfg,
            progress=log,
            scored=scored_raw,
        )
    # حرّر إطار التسجيل الكامل قبل الخط الموحّد (M9 يعيد بناء الدفتر على كل الأحداث).
    # الإدج لاحقًا يكفيه ``deco_by_bucket`` إن وُجد — لا حاجة للإبقاء على scored لكل حدث.
    if scored_raw is not None:
        scored_raw = None
        log.op("تحرير scored_raw قبل SSL/M9 · الإبقاء على براميل التضليل فقط")

    partner: pl.DataFrame | str | Path
    if mnq is None:
        partner = cleaned
    elif isinstance(mnq, (str, Path)):
        partner = _load_nq(mnq, max_rows=max_rows, progress=log)
    else:
        partner = mnq.head(max_rows) if max_rows is not None else mnq

    cfg = PipelineConfig(
        include_auction_vp=False,
        include_failed_fvg=False,
        include_failed_breakout=False,
        cross_market_mode="nq_only" if mnq is None else "dual",
        max_rows=None,
        horizon=horizon,
        alpha=alpha,
        n_permutations=n_permutations,
        ssl_window=ssl_window,
        ssl_components=ssl_components,
        signal_columns=_VP_AUCTION_FOCUS,
        quiet=quiet,
        interval_ns=sig_iv,
        profile_interval_ns=prof_iv,
        embargo_ns=30_000_000_000,
        # افتراضي سريع: VP من شريط الصفقات لا يحتاج tick_stream حدث-بحدث
        feature_mode="batch" if not streaming_features else "streaming",
        ssl_mode="bucket" if not streaming_features else "tick",
        filter_depth_noise=streaming_features,
        include_bottom_book=streaming_features,
        # لا توازي SSL‖M9 داخل اليوم — مع توازي الأيام يضاعف ضغط الذاكرة
        parallel_coverage=False,
    )
    result = run_research_pipeline(
        cleaned,
        partner,
        config=cfg,
        signal_columns=_VP_AUCTION_FOCUS,
        output_dir=output_dir,
        rng=generator,
    )

    iv = int(cfg.interval_ns)
    log.step(
        "VP: حالات المزاد مرة واحدة",
        f"رينج={prof_iv // 1_000_000_000}s · فعل={iv // 1_000_000_000}s",
    )
    auction_day = auction_action_states(
        cleaned,
        profile_interval_ns=prof_iv,
        signal_interval_ns=iv,
        fixed_range=True,
        progress=log,
    )
    auction_signals = auction_signals_from_states(
        auction_day,
        fixed_range_decisions=True,
    )
    features = _attach_auction_vp_signals(result.features, auction_signals)

    policy = TemporalPolicy.for_run(
        interval_ns=iv,
        window=ssl_window,
        horizon=horizon,
        config_path=Path("configs/vp_auction.toml"),
    )
    feat_times = features["availability_ts"].to_numpy().astype(np.int64)
    embargo = policy.embargo_time_units(interval_ns=iv, times=feat_times)
    candidates = tuple(c for c in _VP_AUCTION_FOCUS if c in features.columns)
    log.step("VP walk-forward selection", f"candidates={len(candidates)} · pre-execution")
    fold_df, oos_ic, oos_p, oos_n, best = walk_forward_select_hypotheses(
        features,
        candidates,
        price_col="nq_close",
        horizon=horizon,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=policy.purge_samples(),
        n_permutations=n_permutations,
        selection_aware_null=True,
        rng=generator,
        progress=log,
    )

    edge_table = pl.DataFrame()
    best_edge: EdgeSearchSpec | None = None
    best_edge_row: dict[str, float | str] = {}
    edge_trades = pl.DataFrame()
    edge_summary: dict[str, float] = {
        "n_trades": 0.0,
        "win_rate": 0.0,
        "avg_rr_planned": 0.0,
        "expectancy": 0.0,
        "avg_pnl": 0.0,
        "profit_factor": 0.0,
    }

    if with_execution:
        log.step(
            "VP: حكم السوق + بحث R:R",
            f"بعد WF · رينج={prof_iv // 1_000_000_000}s · فعل={iv // 1_000_000_000}s",
        )
        specs = edge_grid if edge_grid is not None else default_edge_search_grid()
        edge_table, best_edge, best_edge_row = search_best_edge_spec(
            cleaned,
            interval_ns=iv,
            grid=specs,
            train_frac=edge_train_frac,
            deceptive=deco_cfg,
            score_mbo=raw,
            progress=log,
            min_oos_trades=min_oos_trades,
            min_oos_rr=min_oos_rr,
            auction=auction_day,
            deceptive_frame=deco_by_bucket,
            scored=scored_raw,
        )
        if best_edge is not None:
            edge_trades = run_edge_plan(
                cleaned,
                interval_ns=iv,
                truth_cfg=best_edge.truth_config(),
                exec_cfg=best_edge.exec_config(),
                deceptive=deco_cfg,
                score_mbo=raw,
                progress=log,
                auction=auction_day,
                deceptive_frame=deco_by_bucket,
            )
        else:
            edge_trades = run_edge_plan(
                cleaned,
                interval_ns=iv,
                deceptive=deco_cfg,
                score_mbo=raw,
                progress=log,
                auction=auction_day,
                deceptive_frame=deco_by_bucket,
            )
        edge_summary = summarize_edge_trades(edge_trades)
        edge_trades = _with_gated_vp_columns(edge_trades, features)
        if (
            "deceptive_score" not in edge_trades.columns
            and deco_by_bucket is not None
            and deco_by_bucket.height
            and AVAILABILITY_TS in deco_by_bucket.columns
        ):
            edge_trades = edge_trades.join_asof(
                deco_by_bucket.sort(AVAILABILITY_TS),
                on=AVAILABILITY_TS,
                strategy="backward",
            )
        features = _attach_execution_layer(features, edge_trades)

    assistant = ResearchAssistant(alpha=alpha)
    findings = [
        assistant.generate_hypothesis(
            (
                f"فرضية Volume Profile المختارة بـ walk-forward (best={best!r}) "
                f"تحقق IC خارج العينة = {oos_ic:.4g} (p={oos_p:.4g})."
            ),
            Evidence(
                id="vp_search:oos_ic",
                source="vp_auction_walk_forward",
                metric="IC",
                value=oos_ic,
                pvalue=oos_p,
                sample_size=oos_n,
                detail=(
                    f"best_oos={best!r}; oos_ic={oos_ic:.4g}; oos_p={oos_p:.4g}; "
                    f"n={oos_n}; with_execution={with_execution}; "
                    f"mbo={raw_n}→{cleaned_n}; wf_before_execution=True"
                ),
            ),
            requires_significance=True,
            category="vp_auction_search",
        )
    ]
    if with_execution:
        oos_exp = float(best_edge_row.get("oos_expectancy", 0.0) or 0.0)
        oos_rr = float(best_edge_row.get("oos_avg_rr", 0.0) or 0.0)
        oos_edge_n = float(best_edge_row.get("oos_n", 0.0) or 0.0)
        findings.append(
            assistant.generate_hypothesis(
                (
                    f"خطة التنفيذ المتصلة ({best_edge.name if best_edge else 'none'}) "
                    f"expectancy OOS={oos_exp:.4g} · R:R={oos_rr:.4g} · n={oos_edge_n:.0f}."
                ),
                Evidence(
                    id="vp_search:edge_oos_expectancy",
                    source="vp_auction_execution",
                    metric="expectancy",
                    value=oos_exp,
                    sample_size=int(oos_edge_n),
                    detail=(
                        f"best_edge={best_edge.name if best_edge else None}; "
                        f"avg_rr={oos_rr:.4g}; pf={best_edge_row.get('oos_profit_factor', 0)}; "
                        f"min_oos_rr={min_oos_rr}; score_mbo=raw"
                    ),
                ),
                requires_significance=False,
                category="vp_auction_execution",
            )
        )
    if exploratory_full_sample:
        findings.append(
            assistant.generate_hypothesis(
                "شاشة العيّنة الكاملة استكشافية فقط — ليست أساس الاختيار.",
                Evidence(
                    id="vp_search:exploratory_note",
                    source="vp_auction_walk_forward",
                    metric="note",
                    value=0.0,
                    detail="full-sample alpha screen is exploratory",
                ),
                requires_significance=False,
                category="vp_auction_search",
            )
        )
    report = assistant.write_report(
        findings,
        title="Volume Profile / Auction — Signal WF then Execution (Connected)",
    )
    if with_execution:
        report_md = report.to_markdown() + "\n".join(
            [
                "",
                "## طبقة التنفيذ المتصلة (بعد WF — ليست لاختيار الإشارة)",
                "- فلتر تضليل يسقط دورة الأمر الكاملة (بلا سيولة شبح).",
                "- درجات الهولد من MBO الخام؛ الدفتر المنظّف للمسارات الحسّاسة للعمق.",
                "- WF على `vp_*` فقط قبل إلحاق `entry_gate` / `edge_*`.",
                f"- MBO raw→cleaned: `{raw_n}` → `{cleaned_n}`",
                f"- edge n_trades: `{edge_summary['n_trades']:.0f}` · "
                f"win_rate=`{edge_summary['win_rate']:.4g}` · "
                f"avg_rr=`{edge_summary['avg_rr_planned']:.4g}` · "
                f"expectancy=`{edge_summary['expectancy']:.4g}`",
                "",
            ]
        )
    else:
        report_md = report.to_markdown()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fold_df.write_parquet(out / "vp_fold_selections.parquet")
        (out / "vp_walk_forward_report.md").write_text(report_md, encoding="utf-8")
        summary = pl.DataFrame(
            {
                "best_signal": [best],
                "oos_ic": [oos_ic],
                "oos_pvalue": [oos_p],
                "oos_n": [oos_n],
                "exploratory_full_sample": [exploratory_full_sample],
                "with_execution": [with_execution],
                "raw_mbo_rows": [raw_n],
                "cleaned_mbo_rows": [cleaned_n],
                "best_edge_spec": [best_edge.name if best_edge else None],
                "edge_oos_expectancy": [float(best_edge_row.get("oos_expectancy", 0.0) or 0.0)],
                "edge_oos_rr": [float(best_edge_row.get("oos_avg_rr", 0.0) or 0.0)],
                "edge_n_trades": [edge_summary["n_trades"]],
                "wf_before_execution": [True],
            }
        )
        summary.write_parquet(out / "vp_oos_summary.parquet")
        features.write_parquet(out / "features.parquet")
        if with_execution and edge_table.height:
            edge_table.write_parquet(out / "edge_search_grid.parquet")
        if with_execution and edge_trades.height:
            edge_trades.write_parquet(out / "edge_trades.parquet")
        log.note(f"مخرجات VP المتصلة محفوظة في {out.resolve()}")

    return VpAuctionResearchResult(
        features=features,
        alpha=result.alpha,
        ssl=result.ssl,
        report=report,
        unified=result.report,
        signal_columns=_VP_AUCTION_FOCUS,
        fold_df=fold_df,
        oos_ic=oos_ic,
        oos_pvalue=oos_p,
        oos_n=oos_n,
        best_signal=best,
        exploratory_only=exploratory_full_sample,
        with_execution=with_execution,
        raw_mbo_rows=raw_n,
        cleaned_mbo_rows=cleaned_n,
        edge_search_table=edge_table,
        best_edge_spec=best_edge,
        best_edge_row=best_edge_row,
        edge_trades=edge_trades,
        edge_summary=edge_summary,
    )


__all__ = [
    "VpAuctionResearchResult",
    "_VP_AUCTION_FOCUS",
    "_VP_LEVEL_DISTANCE_FEATURES",
    "_VP_REGIME_STATE_FEATURES",
    "run_vp_auction_research",
]
