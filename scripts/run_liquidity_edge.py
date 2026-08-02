#!/usr/bin/env python3
"""تشغيل بحث إدج السيولة: فلتر تضليل + حكم سوق + دخول/خروج بـ R:R قوي.

    python scripts/run_liquidity_edge.py \\
      --nq /path/to/nq.parquet \\
      --max-rows 500000 \\
      --output data/runs/liquidity_edge
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

from nq.strategies.liquidity_edge import run_liquidity_edge_research  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Liquidity edge research: deceptive-order filter + market truth hold + "
            "structural R:R entry/exit search (MBO-only, causal)"
        )
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runs/liquidity_edge"),
        help="مجلد المخرجات",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--interval-ns",
        type=int,
        default=1_000_000_000,
        help="طول برميل المزاد/الهولد (افتراضي 1s)",
    )
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--min-oos-trades", type=int, default=3)
    parser.add_argument(
        "--min-oos-rr",
        type=float,
        default=2.0,
        help="حد أدنى لمتوسط R:R المخطط خارج العينة",
    )
    parser.add_argument(
        "--keep-deceptive",
        action="store_true",
        help="لا تسقط أحداث التضليل (درجة فقط — للمقارنة A/B)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")

    if not args.quiet:
        print(
            "[nq] ========== بدء: run_liquidity_edge ==========",
            file=sys.stderr,
            flush=True,
        )

    result = run_liquidity_edge_research(
        args.nq,
        interval_ns=args.interval_ns,
        max_rows=args.max_rows,
        train_frac=args.train_frac,
        min_oos_trades=args.min_oos_trades,
        min_oos_rr=args.min_oos_rr,
        drop_deceptive=not args.keep_deceptive,
        output_dir=args.output,
        quiet=args.quiet,
    )
    print(result.report_md)
    best = result.best_spec.name if result.best_spec else None
    print(
        f"\nbest={best!r} · oos_exp={result.best_row.get('oos_expectancy', 0):.4g} · "
        f"oos_rr={result.best_row.get('oos_avg_rr', 0):.4g} · "
        f"oos_n={result.best_row.get('oos_n', 0)}"
    )
    print(f"MBO: {result.raw_mbo_rows} → cleaned {result.cleaned_mbo_rows}")
    print(f"outputs: {args.output.resolve()}/")


if __name__ == "__main__":
    main()
