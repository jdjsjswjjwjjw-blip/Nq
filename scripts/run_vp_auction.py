#!/usr/bin/env python3
"""تشغيل Volume Profile المتصل: إشارة مزاد + تضليل + هولد + تنفيذ R:R.

مسار واحد داخل الاستراتيجية — ليس تشعّبًا منفصلًا.

    python scripts/run_vp_auction.py --nq data/raw/nq.parquet --max-rows 500000
    python scripts/run_vp_auction.py --nq ... --no-execution   # IC فقط
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MIN_PYTHON = (3, 11)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.version_info < _MIN_PYTHON:
    sys.exit(
        f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ مطلوب؛ "
        f"الحالي {sys.version_info.major}.{sys.version_info.minor}"
    )

from nq.strategies.vp_auction import run_vp_auction_research  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Volume Profile / Auction — connected signal + deceptive filter + "
            "hold + structural R:R execution (single strategy path)"
        )
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--output", type=Path, default=Path("data/runs/vp_auction"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--no-execution",
        action="store_true",
        help="تعطيل طبقة التنفيذ (IC/WF فقط) — الافتراضي متصل كامل",
    )
    parser.add_argument(
        "--keep-deceptive",
        action="store_true",
        help="لا تسقط أحداث التضليل (درجة فقط)",
    )
    parser.add_argument("--min-oos-rr", type=float, default=2.0)
    parser.add_argument("--min-oos-trades", type=int, default=3)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="تفعيل tick_stream حدث-بحدث (أبطأ كثيرًا؛ الافتراضي batch سريع لـ VP)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="تعطيل طباعة تقدّم الخطوات على stderr",
    )
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")

    with_execution = not args.no_execution
    if not args.quiet:
        mode = "إشارة+تنفيذ متصل" if with_execution else "إشارة فقط"
        print(
            f"[nq] ========== بدء: run_vp_auction ({mode}) ==========",
            file=sys.stderr,
            flush=True,
        )

    result = run_vp_auction_research(
        args.nq,
        horizon=args.horizon,
        max_rows=args.max_rows,
        output_dir=args.output,
        quiet=args.quiet,
        with_execution=with_execution,
        drop_deceptive=not args.keep_deceptive,
        min_oos_rr=args.min_oos_rr,
        min_oos_trades=args.min_oos_trades,
        streaming_features=args.streaming,
    )
    print(result.report.to_markdown())
    print(
        f"\nWF best={result.best_signal!r} · oos_ic={result.oos_ic:.4g} · "
        f"p={result.oos_pvalue:.4g} · n={result.oos_n}"
    )
    if result.with_execution:
        edge = result.best_edge_spec.name if result.best_edge_spec else None
        print(
            f"EDGE best={edge!r} · oos_exp={result.best_edge_row.get('oos_expectancy', 0):.4g} · "
            f"rr={result.best_edge_row.get('oos_avg_rr', 0):.4g} · "
            f"MBO {result.raw_mbo_rows}→{result.cleaned_mbo_rows}"
        )
    print(f"signals: {result.signal_columns}")
    print(f"features: {result.features.height} rows")
    print(f"outputs: {args.output.resolve()}/")
    for name in (
        "vp_walk_forward_report.md",
        "vp_fold_selections.parquet",
        "vp_oos_summary.parquet",
        "edge_search_grid.parquet",
        "edge_trades.parquet",
        "report.md",
        "features.parquet",
        "ssl_metrics.parquet",
        "coverage_metrics.parquet",
        "alpha_evaluations.parquet",
    ):
        path = args.output / name
        if path.is_file():
            print(f"  - {name}")
    for col in (
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_flip_to_imbalance",
    ):
        assert col in result.features.columns, f"missing {col}"
    if not args.quiet:
        print(
            "\n[ملاحظة] مسار واحد: VP إشارة + تضليل + هولد + R:R داخل نفس الاستراتيجية.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
