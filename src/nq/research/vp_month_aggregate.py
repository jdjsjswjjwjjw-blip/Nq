"""تجميع وصفي لنتيجة شهر VP من مخرجات الأيام المعزولة.

ليس اختيار فرضية عبر الأيام (لا selector موحّد). يقرأ ``vp_oos_summary`` /
``edge_trades`` لكل يوم ناجح ويُنتج:

* ``day_results.parquet`` — صف لكل يوم
* ``aggregate_edge_trades.parquet`` — صفقات كل الأيام مع ``day_id``
* ``FINAL_RESULT.md`` — الحكم النهائي من المجمل
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from nq.simulation.edge_execution_plan import summarize_edge_trades


def write_vp_month_aggregate(output_root: Path | str) -> Path:
    """يبني حكمًا نهائيًا وصفيًا من مجمل أيام ``output_root``."""
    root = Path(output_root)
    day_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "vp_oos_summary.parquet").exists()
    )
    if not day_dirs:
        raise FileNotFoundError(f"no per-day vp_oos_summary under {root}")

    day_rows: list[dict[str, object]] = []
    trade_frames: list[pl.DataFrame] = []

    for day_dir in day_dirs:
        day_id = day_dir.name
        summary = pl.read_parquet(day_dir / "vp_oos_summary.parquet").row(0, named=True)
        edge_path = day_dir / "edge_trades.parquet"
        n_trades = float(summary.get("edge_n_trades") or 0.0)
        edge_exp = float(summary.get("edge_oos_expectancy") or 0.0)
        edge_rr = float(summary.get("edge_oos_rr") or 0.0)
        oos_ic = float(summary.get("oos_ic") or 0.0)
        oos_p = float(summary.get("oos_pvalue") or 1.0)
        day_rows.append(
            {
                "day_id": day_id,
                "best_signal": summary.get("best_signal"),
                "oos_ic": oos_ic,
                "oos_pvalue": oos_p,
                "oos_n": int(summary.get("oos_n") or 0),
                "best_edge_spec": summary.get("best_edge_spec"),
                "edge_oos_expectancy": edge_exp,
                "edge_oos_rr": edge_rr,
                "edge_n_trades": n_trades,
                "raw_mbo_rows": int(summary.get("raw_mbo_rows") or 0),
                "cleaned_mbo_rows": int(summary.get("cleaned_mbo_rows") or 0),
                "signal_significant": bool(oos_p <= 0.05),
                "edge_positive": bool(edge_exp > 0.0 and n_trades >= 3),
            }
        )
        if edge_path.exists():
            trades = pl.read_parquet(edge_path)
            if trades.height and "edge_signal" in trades.columns:
                active = trades.filter(pl.col("edge_signal") != 0.0)
                if active.height:
                    trade_frames.append(active.with_columns(pl.lit(day_id).alias("day_id")))

    day_df = pl.DataFrame(day_rows).sort("day_id")
    day_df.write_parquet(root / "day_results.parquet")

    if trade_frames:
        pooled = pl.concat(trade_frames, how="diagonal_relaxed")
        pooled.write_parquet(root / "aggregate_edge_trades.parquet")
        pooled_summary = summarize_edge_trades(pooled)
    else:
        pooled = pl.DataFrame()
        pooled_summary = {
            "n_trades": 0.0,
            "win_rate": 0.0,
            "avg_rr_planned": 0.0,
            "expectancy": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
        }

    n_days = day_df.height
    n_sig = int(day_df["signal_significant"].sum())
    n_edge_pos = int(day_df["edge_positive"].sum())
    mean_ic = float(day_df["oos_ic"].mean()) if n_days else 0.0
    mean_edge_exp = float(day_df["edge_oos_expectancy"].mean()) if n_days else 0.0
    total_day_trades = float(day_df["edge_n_trades"].sum()) if n_days else 0.0

    signal_counts = Counter(
        str(s) for s in day_df["best_signal"].to_list() if s is not None and str(s) != "None"
    )
    edge_counts = Counter(
        str(s) for s in day_df["best_edge_spec"].to_list() if s is not None and str(s) != "None"
    )

    # حكم نهائي وصفي من المجمل (ليس selector عبر الأيام).
    pooled_n = float(pooled_summary["n_trades"])
    pooled_exp = float(pooled_summary["expectancy"])
    pooled_wr = float(pooled_summary["win_rate"])
    pooled_rr = float(pooled_summary["avg_rr_planned"])
    has_edge = pooled_n >= 30 and pooled_exp > 0.0 and pooled_rr >= 2.0
    verdict = (
        "EDGE_FOUND_POOLED"
        if has_edge
        else ("NO_EDGE_POOLED" if pooled_n > 0 else "INSUFFICIENT_POOLED_TRADES")
    )

    top_signal = signal_counts.most_common(1)[0][0] if signal_counts else "none"
    top_edge = edge_counts.most_common(1)[0][0] if edge_counts else "none"

    lines = [
        "# VP Month — FINAL RESULT (pooled across isolated days)",
        "",
        "## Verdict",
        "",
        f"**`{verdict}`**",
        "",
        "- Aggregation is **descriptive** over isolated day universes (no cross-day selector).",
        "- Pooled trades concatenate per-day OOS/execution fills; do not re-fit on the pool.",
        "",
        "## Pooled execution (all days)",
        "",
        f"- n_trades: `{pooled_n:.0f}`",
        f"- expectancy: `{pooled_exp:.6g}`",
        f"- win_rate: `{pooled_wr:.4g}`",
        f"- avg_rr: `{pooled_rr:.4g}`",
        f"- profit_factor: `{float(pooled_summary['profit_factor']):.4g}`",
        f"- sum of per-day edge_n_trades: `{total_day_trades:.0f}`",
        "",
        "## Per-day research summary",
        "",
        f"- days ok with summary: `{n_days}`",
        f"- days with significant WF signal (p≤0.05): `{n_sig}/{n_days}`",
        f"- days with positive edge_oos_expectancy (n≥3): `{n_edge_pos}/{n_days}`",
        f"- mean daily oos_ic: `{mean_ic:.4g}`",
        f"- mean daily edge_oos_expectancy: `{mean_edge_exp:.6g}`",
        f"- most frequent daily WF signal: `{top_signal}`",
        f"- most frequent daily edge spec: `{top_edge}`",
        "",
        "## Daily WF signal frequency",
        "",
    ]
    for name, count in signal_counts.most_common():
        lines.append(f"- `{name}`: {count}/{n_days}")
    lines.extend(["", "## Daily edge spec frequency", ""])
    for name, count in edge_counts.most_common():
        lines.append(f"- `{name}`: {count}/{n_days}")

    lines.extend(
        [
            "",
            "## Day table",
            "",
            "| day_id | best_signal | oos_ic | oos_p | edge_spec | edge_exp | edge_n |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in day_df.iter_rows(named=True):
        lines.append(
            f"| `{row['day_id']}` | `{row['best_signal']}` | {row['oos_ic']:.4g} | "
            f"{row['oos_pvalue']:.4g} | `{row['best_edge_spec']}` | "
            f"{row['edge_oos_expectancy']:.4g} | {row['edge_n_trades']:.0f} |"
        )

    # Stability check: fraction of days agreeing with pooled sign of expectancy.
    if n_days and np.isfinite(pooled_exp):
        if pooled_exp > 0:
            same_sign = int(day_df.filter(pl.col("edge_oos_expectancy") > 0).height)
        else:
            same_sign = int(day_df.filter(pl.col("edge_oos_expectancy") <= 0).height)
        lines.extend(
            [
                "",
                "## Stability",
                "",
                f"- days with edge expectancy sign matching pooled: `{same_sign}/{n_days}`",
            ]
        )

    final_path = root / "FINAL_RESULT.md"
    final_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = pl.DataFrame(
        {
            "verdict": [verdict],
            "n_days": [n_days],
            "pooled_n_trades": [pooled_n],
            "pooled_expectancy": [pooled_exp],
            "pooled_win_rate": [pooled_wr],
            "pooled_avg_rr": [pooled_rr],
            "n_signal_significant_days": [n_sig],
            "n_edge_positive_days": [n_edge_pos],
            "top_daily_signal": [top_signal],
            "top_daily_edge_spec": [top_edge],
        }
    )
    meta.write_parquet(root / "FINAL_RESULT.parquet")
    return final_path


__all__ = ["write_vp_month_aggregate"]
