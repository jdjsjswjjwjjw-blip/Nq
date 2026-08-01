#!/usr/bin/env python3
"""تشغيل بحث Failed FVG — أمر منفصل فوق الخط الموحّد (ليس خارج المنظومة).

# استكشافي كامل العيّنة (BH screen — ليس اختيار WF/OOS)
python scripts/run_fail_fvg.py --nq data/raw/nq.parquet --max-rows 500000

# بحث تايم فريم/إعدادات + بوابة SSL سببية (walk-forward purged)
python scripts/run_fail_fvg.py --nq data/raw/nq.parquet --search --max-rows 500000
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
from nq.strategies.fail_fvg import run_fail_fvg_research  # noqa: E402
from nq.strategies.fvg_hypothesis import search_fail_fvg_hypotheses  # noqa: E402

_EXPLORATORY_BANNER = (
    "[nq] NOTE: default mode (no --search) is an exploratory full-sample BH screen — "
    "not purged walk-forward hypothesis selection. Use --search for OOS selection."
)


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description=(
            "Failed FVG research — capacity-correct walk-forward search + causal SSL gate"
        ),
        epilog=(
            "Without --search the pipeline runs an exploratory full-sample screen "
            "(not OOS/WF selection of the best TF/settings)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--mnq", type=Path, default=None, help="مسار MNQ اختياري (dual)")
    parser.add_argument("--output", type=Path, default=Path("data/runs/fail_fvg"))
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=f"حد صفوف MBO (موصى به للبحث: {RECOMMENDED_MAX_ROWS:,})",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--search",
        action="store_true",
        help="بحث نواة تايم فريم/إعدادات بـ walk-forward + بوابة SSL سببية",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="مع --search: شبكة FVG كاملة (~84) بدل النواة المضغوطة",
    )
    parser.add_argument(
        "--no-ssl-gate",
        action="store_true",
        help="مع --search: تعطيل بوابة SSL (الإشارات الخام فقط)",
    )
    parser.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="مع --search: تعطيل فلتر مسار أحداث العمق داخل الشمعة",
    )
    parser.add_argument(
        "--no-lean-filters",
        action="store_true",
        help="مع --search: توسيع كمّيات العمق (أثقل)",
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="تعطيل طباعة تقدّم الخطوات على stderr",
    )
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")
    if args.mnq is not None and not args.mnq.is_file():
        raise FileNotFoundError(f"MNQ MBO not found: {args.mnq.resolve()}")

    mode = "بحث فرضيات (--search)" if args.search else "استكشاف كامل العيّنة (بدون --search)"
    if not args.quiet:
        print(f"[nq] ========== بدء: run_fail_fvg · {mode} ==========", file=sys.stderr, flush=True)
        if not args.search:
            print(_EXPLORATORY_BANNER, file=sys.stderr, flush=True)

    if args.search:
        result = search_fail_fvg_hypotheses(
            args.nq,
            args.mnq,
            horizon=args.horizon,
            use_ssl_gate=not args.no_ssl_gate,
            use_depth_filter=not args.no_depth_filter,
            n_splits=args.n_splits,
            n_permutations=args.n_permutations,
            max_rows=args.max_rows,
            output_dir=args.output,
            quiet=args.quiet,
            understand=args.understand,
            full_grid=args.full_grid,
            lean_filters=not args.no_lean_filters,
            exploratory=args.exploratory,
        )
        print(result.report.to_markdown())
        print(f"\nbest_oos_spec: {result.best_oos_spec}")
        print(f"oos_selected_ic: {result.oos_selected_ic}")
        print(f"candidates: {len(result.candidate_columns)}")
        print(f"features: {result.features.height} rows")
        if result.fold_selections.height > 0:
            print(result.fold_selections)
        if result.exploratory_screen.height > 0:
            print(result.exploratory_screen.head(10))
        if result.understanding is not None:
            print(result.understanding.to_markdown())
            print(f"understanding: {args.output.resolve()}/understanding/")
        print(f"outputs: {args.output.resolve()}/")
        return

    result = run_fail_fvg_research(
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
    ):
        path = args.output / name
        if path.is_file():
            print(f"  - {name}")
    assert "fail_fvg" in result.features.columns, "missing fail_fvg"
    if result.alpha.evaluations.height > 0:
        print(result.alpha.evaluations)


if __name__ == "__main__":
    main()
