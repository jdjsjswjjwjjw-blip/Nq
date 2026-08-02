"""استراتيجية إدج السيولة: تضليل + حكم سوق + دخول/خروج بـ R:R قوي.

تجمع الطبقات العلمية:

1. ``filter_deceptive_liquidity`` / درجات التضليل داخل الإشارة.
2. ``build_market_truth_frame`` — هل السوق صادق/كاذب بعد هولد؟
3. ``search_best_edge_spec`` — أفضل دخول/خروج بدون ملاحقة كل أمر.

المخرجات: تقرير Markdown + جداول parquet للشبكة والصفقات والخطة الفائزة.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from nq.ingestion.reader import load_mbo_frame
from nq.research.assistant import ResearchAssistant
from nq.research.evidence import Evidence
from nq.research.progress import resolve_progress
from nq.simulation.deceptive_liquidity import (
    DeceptiveLiquidityConfig,
    filter_deceptive_liquidity,
)
from nq.simulation.edge_execution_plan import (
    EdgeSearchSpec,
    default_edge_search_grid,
    run_edge_plan,
    search_best_edge_spec,
    summarize_edge_trades,
)


@dataclass(frozen=True, slots=True)
class LiquidityEdgeResult:
    """نتيجة بحث إدج السيولة التنفيذية."""

    cleaned_mbo_rows: int
    raw_mbo_rows: int
    search_table: pl.DataFrame
    best_spec: EdgeSearchSpec | None
    best_row: dict[str, float | str]
    trades: pl.DataFrame
    trade_summary: dict[str, float]
    report_md: str


def run_liquidity_edge_research(
    nq: pl.DataFrame | str | Path,
    *,
    interval_ns: int = 1_000_000_000,
    max_rows: int | None = None,
    train_frac: float = 0.6,
    min_oos_trades: int = 3,
    min_oos_rr: float = 2.0,
    drop_deceptive: bool = True,
    deceptive: DeceptiveLiquidityConfig | None = None,
    grid: tuple[EdgeSearchSpec, ...] | None = None,
    output_dir: Path | str | None = None,
    quiet: bool = False,
) -> LiquidityEdgeResult:
    """يشغّل البحث الكامل: تنظيف تضليل → شبكة R:R → خطة فائزة → صفقات."""
    log = resolve_progress(None, quiet=quiet)
    cfg = deceptive if deceptive is not None else DeceptiveLiquidityConfig()

    log.step("تحميل MBO")
    if isinstance(nq, (str, Path)):
        raw = load_mbo_frame(nq, max_rows=max_rows)
    else:
        raw = nq.head(max_rows) if max_rows is not None else nq
    raw_n = raw.height

    log.step("فلتر التضليل العلمي", f"drop={drop_deceptive}")
    cleaned = filter_deceptive_liquidity(raw, config=cfg) if drop_deceptive else raw
    cleaned_n = cleaned.height

    specs = grid if grid is not None else default_edge_search_grid()
    log.step("بحث أفضل دخول/خروج", f"grid={len(specs)}")
    table, best, best_row = search_best_edge_spec(
        cleaned,
        interval_ns=interval_ns,
        grid=specs,
        train_frac=train_frac,
        deceptive=cfg,
        progress=log,
        min_oos_trades=min_oos_trades,
        min_oos_rr=min_oos_rr,
    )

    if best is None:
        trades = pl.DataFrame()
        summary: dict[str, float] = {
            "n_trades": 0.0,
            "win_rate": 0.0,
            "avg_rr_planned": 0.0,
            "expectancy": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
        }
    else:
        log.step("محاكاة الخطة الفائزة", best.name)
        trades = run_edge_plan(
            cleaned,
            interval_ns=interval_ns,
            truth_cfg=best.truth_config(),
            exec_cfg=best.exec_config(),
            deceptive=cfg,
            progress=log,
        )
        summary = summarize_edge_trades(trades)

    assistant = ResearchAssistant(alpha=0.05)
    oos_exp = float(best_row.get("oos_expectancy", 0.0) or 0.0)
    oos_n = float(best_row.get("oos_n", 0.0) or 0.0)
    oos_rr = float(best_row.get("oos_avg_rr", 0.0) or 0.0)
    evidence = Evidence(
        id="liquidity_edge:oos_expectancy",
        source="liquidity_edge_search",
        metric="expectancy",
        value=oos_exp,
        sample_size=int(oos_n),
        detail=(
            f"best={best.name if best else None}; oos_rr={oos_rr:.4g}; "
            f"oos_pf={best_row.get('oos_profit_factor', 0)}; "
            f"dropped_events={raw_n - cleaned_n}; "
            f"min_rr_constraint={min_oos_rr}"
        ),
    )
    claim = (
        f"خطة الإدج المختارة ({best.name if best else 'none'}) "
        f"تحقق expectancy خارج العينة = {oos_exp:.4g} "
        f"مع متوسط R:R مخطط = {oos_rr:.4g} على {oos_n:.0f} صفقة."
    )
    findings = [
        assistant.generate_hypothesis(
            claim,
            evidence,
            requires_significance=False,
            category="liquidity_edge",
        )
    ]
    report = assistant.write_report(
        findings,
        title="Liquidity Edge — Deceptive Filter + Hold + R:R Execution",
    )
    extra = [
        "",
        "## مبادئ التنفيذ",
        "- لا ملاحقة لكل أمر: دخول فقط بعد هولد سيولة حقيقية + بوابة تضليل.",
        "- حكم السوق اللحظي: `market_true` / `market_false` بعد نافذة الهولد.",
        "- وقف/هدف هيكلي من VAL/VAH (أو مضاعف R) مع حد أدنى R:R قوي.",
        "- TRADE/FILL لا تُسقط أبدًا؛ الإسقاط على ADD/CANCEL/MODIFY المضلل فقط.",
        "",
        "## ملخص الصفقات (الخطة الفائزة على كامل العيّنة المنظّفة)",
        f"- n_trades: `{summary['n_trades']:.0f}`",
        f"- win_rate: `{summary['win_rate']:.4g}`",
        f"- avg_rr_planned: `{summary['avg_rr_planned']:.4g}`",
        f"- expectancy: `{summary['expectancy']:.4g}`",
        f"- profit_factor: `{summary['profit_factor']:.4g}`",
        f"- MBO raw→cleaned: `{raw_n}` → `{cleaned_n}`",
    ]
    report_md = report.to_markdown() + "\n".join(extra) + "\n"

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "liquidity_edge_report.md").write_text(report_md, encoding="utf-8")
        if table.height:
            table.write_parquet(out / "edge_search_grid.parquet")
        if trades.height:
            keep = [
                c
                for c in trades.columns
                if c
                in {
                    "availability_ts",
                    "bucket_start",
                    "close",
                    "poc",
                    "vah",
                    "val",
                    "thesis_dir",
                    "hold_ok",
                    "delta_instant",
                    "delta_cum",
                    "market_verdict",
                    "market_true",
                    "market_false",
                    "entry_gate",
                    "deceptive_score",
                    "real_liquidity_ratio",
                    "noise_instant",
                    "noise_cum",
                    "edge_signal",
                    "edge_entry",
                    "edge_stop",
                    "edge_target",
                    "edge_rr",
                    "edge_risk",
                    "edge_reward",
                    "edge_pnl",
                    "edge_hit",
                }
                or c.startswith("edge_")
                or c.startswith("vp_")
                or c.startswith("market_")
                or c.startswith("deceptive_")
                or c.startswith("noise_")
                or c.startswith("real_")
                or c in {"thesis_dir", "hold_ok", "delta_instant", "delta_cum", "entry_gate"}
            ]
            # أعمدة أساسية مضمونة الوجود
            cols = [c for c in trades.columns if c in set(keep) or c.startswith("edge_")]
            trades.select(cols).write_parquet(out / "edge_trades.parquet")
        pl.DataFrame(
            [
                {
                    "best_spec": best.name if best else None,
                    "oos_expectancy": oos_exp,
                    "oos_n": oos_n,
                    "oos_avg_rr": oos_rr,
                    "raw_rows": raw_n,
                    "cleaned_rows": cleaned_n,
                    **{f"trade_{k}": v for k, v in summary.items()},
                }
            ]
        ).write_parquet(out / "edge_oos_summary.parquet")
        log.note(f"مخرجات محفوظة في {out.resolve()}")

    return LiquidityEdgeResult(
        cleaned_mbo_rows=cleaned_n,
        raw_mbo_rows=raw_n,
        search_table=table,
        best_spec=best,
        best_row=best_row,
        trades=trades,
        trade_summary=summary,
        report_md=report_md,
    )


__all__ = [
    "LiquidityEdgeResult",
    "run_liquidity_edge_research",
]
