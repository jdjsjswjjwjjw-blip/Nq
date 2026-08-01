#!/usr/bin/env python3
"""تشغيل بحث Failed Breakout — تركيز فوليوم فوق الخط الموحّد.

    # استكشافي كامل العيّنة (BH screen — ليس اختيار WF/OOS)
    python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --max-rows 500000

    # بحث نواة فوليوم + تعزيزات SSL (تنخيل walk-forward)
    python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --search --max-rows 500000

    # شبكة فوليوم كاملة (~144 فرضية: bar/cum/delta/effort_result) بلا تعزيز SSL
    python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --search --no-enhance

    # تركيب volume-first + hold داخل الكسر (الفوليوم يولّد · البنية تؤكّد)
    python scripts/run_fail_breakout.py --nq data/raw/nq.parquet --search --compose-hold
    python scripts/run_fail_breakout.py --nq ... --search --compose-hold --no-enhance --horizon 2
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

from nq.research.capacity import RECOMMENDED_MAX_ROWS, SEARCH_N_PERMUTATIONS  # noqa: E402
from nq.strategies.breakout_hypothesis import search_fail_breakout_hypotheses  # noqa: E402
from nq.strategies.fail_breakout import run_fail_breakout_research  # noqa: E402

_EXPLORATORY_BANNER = (
    "[nq] NOTE: default mode (no --search) is an exploratory full-sample BH screen — "
    "not purged walk-forward hypothesis selection. Use --search for OOS selection."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Failed Breakout volume research — causal close entry; "
            "capacity-correct walk-forward volume hypotheses + SSL sift; "
            "optional volume-first hold composition"
        ),
        epilog=(
            "Without --search the pipeline runs an exploratory full-sample screen "
            "(not OOS/WF selection of the best volume core / enhancements)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--mnq", type=Path, default=None, help="مسار MNQ اختياري")
    parser.add_argument("--output", type=Path, default=Path("data/runs/fail_breakout"))
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=f"حد صفوف MBO (موصى به للبحث: {RECOMMENDED_MAX_ROWS:,})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="أفق hold التنفيذي عند التقييم (شموع ساعة البحث)",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="بحث فرضيات فوليوم بـ walk-forward + تعزيزات/بوابة SSL",
    )
    parser.add_argument(
        "--compose-hold",
        action="store_true",
        help=(
            "مع --search: يولّف استراتيجيات volume-first × hold "
            "(persist/absorption/imbalance) داخل الكسر"
        ),
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
    parser.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="مع --search: تعطيل فلتر مسار أحداث العمق داخل الشمعة",
    )
    parser.add_argument(
        "--no-lean-filters",
        action="store_true",
        help="مع --search: توسيع كمّيات العمق/التعزيز (أثقل)",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="مع --search: شاشة BH استكشافية (ليست أساس الاختيار)",
    )
    parser.add_argument(
        "--understand",
        action="store_true",
        help="مع --search: طبقات فهم كمية بعد الاختيار (OOS فقط، بلا تغيير best)",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=SEARCH_N_PERMUTATIONS,
        help=f"تبديلات دلالة OOS فقط (افتراضي {SEARCH_N_PERMUTATIONS})",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")
    if args.mnq is not None and not args.mnq.is_file():
        raise FileNotFoundError(f"MNQ MBO not found: {args.mnq.resolve()}")

    if args.search:
        if args.compose_hold:
            mode = (
                "تركيب volume-first+hold كامل (--no-enhance)"
                if args.no_enhance
                else "نواة volume-first+hold + تعزيزات SSL"
            )
        else:
            mode = (
                "شبكة فوليوم كاملة (--no-enhance)"
                if args.no_enhance
                else "نواة فوليوم + تعزيزات SSL (capacity-correct)"
            )
    else:
        mode = "استكشاف كامل العيّنة (بدون --search)"
    if not args.quiet:
        print(
            f"[nq] ========== بدء: run_fail_breakout · {mode} ==========",
            file=sys.stderr,
            flush=True,
        )
        if not args.search:
            print(_EXPLORATORY_BANNER, file=sys.stderr, flush=True)

    if args.search:
        result = search_fail_breakout_hypotheses(
            args.nq,
            args.mnq,
            horizon=args.horizon,
            use_ssl_gate=not args.no_ssl_gate,
            enhance_with_ssl=not args.no_enhance,
            use_depth_filter=not args.no_depth_filter,
            compose_hold=args.compose_hold,
            n_splits=args.n_splits,
            n_permutations=args.n_permutations,
            max_rows=args.max_rows,
            output_dir=args.output,
            quiet=args.quiet,
            understand=args.understand,
            lean_filters=not args.no_lean_filters,
            exploratory=args.exploratory,
        )
        print(result.report.to_markdown())
        print(f"\nbest_oos_spec: {result.best_oos_spec}")
        print(f"oos_selected_ic: {result.oos_selected_ic}")
        print(f"candidates: {len(result.candidate_columns)}")
        print(f"volume_specs: {len(result.specs)}")
        print(f"enhancements: {len(result.enhancement_columns)}")
        print(f"features: {result.features.height} rows")
        if result.understanding is not None:
            print(result.understanding.to_markdown())
            print(f"understanding: {args.output.resolve()}/understanding/")
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


if __name__ == "__main__":
    main()
