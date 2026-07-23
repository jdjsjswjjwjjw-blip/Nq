#!/usr/bin/env python3
"""تشغيل بحث Failed Breakout — تركيز فوليوم فوق الخط الموحّد.

    # خط أساسي (إشارة + أعمدة فوليوم سببية)
    python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --max-rows 500000

    # بحث نواة فوليوم + تعزيزات SSL (تنخيل walk-forward)
    python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --search --max-rows 500000

    # شبكة فوليوم كاملة (~144 فرضية: bar/cum/delta/effort_result) بلا تعزيز SSL
    python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --search --no-enhance
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

from nq.strategies.breakout_hypothesis import search_fail_breakout_hypotheses  # noqa: E402
from nq.strategies.fail_breakout import run_fail_breakout_research  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Failed Breakout volume research — causal close entry; "
            "walk-forward volume hypotheses (bar/cum/delta/effort_result) + SSL sift"
        )
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--mnq", type=Path, default=None, help="مسار MNQ اختياري")
    parser.add_argument("--output", type=Path, default=Path("data/runs/fail_breakout"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--search",
        action="store_true",
        help="بحث فرضيات فوليوم بـ walk-forward + تعزيزات/بوابة SSL",
    )
    parser.add_argument(
        "--no-ssl-gate",
        action="store_true",
        help="مع --search: تعطيل بوابة SSL الكلاسيكية",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="مع --search: شبكة فوليوم كاملة (~144) بدل نواة+تعزيزات SSL",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")
    if args.mnq is not None and not args.mnq.is_file():
        raise FileNotFoundError(f"MNQ MBO not found: {args.mnq.resolve()}")

    if args.search:
        mode = (
            "شبكة فوليوم كاملة (--no-enhance)"
            if args.no_enhance
            else "نواة فوليوم + تعزيزات SSL"
        )
    else:
        mode = "خط Failed Breakout (فوليوم)"
    if not args.quiet:
        print(
            f"[nq] ========== بدء: run_fail_breakout · {mode} ==========",
            file=sys.stderr,
            flush=True,
        )

    if args.search:
        result = search_fail_breakout_hypotheses(
            args.nq,
            args.mnq,
            horizon=args.horizon,
            use_ssl_gate=not args.no_ssl_gate,
            enhance_with_ssl=not args.no_enhance,
            n_splits=args.n_splits,
            max_rows=args.max_rows,
            output_dir=args.output,
            quiet=args.quiet,
        )
        print(result.report.to_markdown())
        print(f"\nbest_oos_spec: {result.best_oos_spec}")
        print(f"oos_selected_ic: {result.oos_selected_ic}")
        print(f"candidates: {len(result.candidate_columns)}")
        print(f"volume_specs: {len(result.specs)}")
        print(f"enhancements: {len(result.enhancement_columns)}")
        print(f"features: {result.features.height} rows")
        print(f"outputs: {args.output.resolve()}/")
        return

    result = run_fail_breakout_research(
        args.nq,
        args.mnq,
        horizon=args.horizon,
        max_rows=args.max_rows,
        output_dir=args.output,
        quiet=args.quiet,
    )
    print(result.unified.to_markdown())
    print(f"\nsignals: {result.signal_columns}")
    print(f"features: {result.features.height} rows")
    print(f"outputs: {args.output.resolve()}/")
    for name in (
        "report.md",
        "features.parquet",
        "ssl_metrics.parquet",
        "coverage_metrics.parquet",
        "alpha_evaluations.parquet",
        "progress.log",
    ):
        path = args.output / name
        if path.is_file():
            print(f"  - {name}")
    assert "fail_breakout" in result.features.columns
    for col in (
        "fb_effort_volume_ratio",
        "fb_effort_result_ratio",
        "fb_cum_volume",
        "fb_delta",
        "fb_absorption",
    ):
        assert col in result.features.columns, f"missing volume column {col}"
    if "fb_entry_ref" in result.features.columns:
        print("entry_model: fb_entry_ref=signal_bar_close (executable); fb_break_level=analytic only")


if __name__ == "__main__":
    main()
