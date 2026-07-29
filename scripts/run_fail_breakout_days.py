#!/usr/bin/env python3
"""تشغيل Failed Breakout على ملفات يومية متوازية (بدون كسر المبادئ الأربعة).

كل يوم كون سببي مغلق → نفس محرّك ``--search`` / الخط الموحّد.
التوازي على مستوى الأيام فقط (ProcessPool). لا اختيار فرضية عبر الأيام.

    # 30 يومًا على 30 عملية (كل عامل خيطين لـ Polars)
    python scripts/run_fail_breakout_days.py \\
      --nq-glob '/data/nq/*.parquet' \\
      --mnq-dir /data/mnq \\
      --jobs 30 \\
      --threads-per-worker 2 \\
      --search \\
      --output data/runs/fail_breakout_month

    python scripts/run_fail_breakout_days.py \\
      --nq-dir /data/nq_days \\
      --jobs 30 \\
      --search \\
      --output data/runs/fail_breakout_month
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

from nq.research.capacity import SEARCH_N_PERMUTATIONS  # noqa: E402
from nq.research.day_parallel import (  # noqa: E402
    discover_day_inputs,
    run_fail_breakout_day_parallel,
)


def _collect_nq_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.nq_glob:
        # مسار واحد أو أكثر — يوسَّع من الشِل أو نستخدم glob يدويًا
        for pattern in args.nq_glob:
            p = Path(pattern)
            if p.is_file():
                paths.append(p)
            else:
                parent = p.parent if p.parent.parts else Path(".")
                paths.extend(sorted(parent.glob(p.name)))
    if args.nq_dir is not None:
        if not args.nq_dir.is_dir():
            raise FileNotFoundError(f"--nq-dir not found: {args.nq_dir.resolve()}")
        for p in sorted(args.nq_dir.iterdir()):
            if p.is_file() and (
                p.suffix.lower() in {".parquet", ".arrow", ".feather", ".csv"}
                or p.name.endswith(".parquet.zst")
                or p.suffix == ".zst"
            ):
                paths.append(p)
    # فريد مع الحفاظ على الترتيب
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    if not unique:
        raise FileNotFoundError(
            "لا ملفات NQ يومية — مرّر --nq-glob أو --nq-dir بملفات parquet/arrow"
        )
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Failed Breakout on daily MBO shards in parallel — "
            "each day isolated (zero cross-day leakage)"
        )
    )
    parser.add_argument(
        "--nq-glob",
        nargs="+",
        default=None,
        help="نمط/ملفات NQ يومية (مثال: /data/nq/*.parquet)",
    )
    parser.add_argument("--nq-dir", type=Path, default=None, help="مجلد ملفات NQ اليومية")
    parser.add_argument(
        "--mnq-dir",
        type=Path,
        default=None,
        help="مجلد MNQ يومي (مطابقة بالاسم أو day_id)",
    )
    parser.add_argument(
        "--mnq-glob",
        nargs="+",
        default=None,
        help="نمط/ملفات MNQ يومية اختيارية",
    )
    parser.add_argument("--output", type=Path, required=True, help="جذر مخرجات الشهر/الفترة")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="عدد العمليات المتوازية (= أيام متزامنة؛ مثال 30)",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=2,
        help="خيوط Polars/BLAS داخل كل يوم (افتراضي 2)",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="بحث فرضيات فوليوم + WF (موصى به)",
    )
    parser.add_argument(
        "--no-ssl-gate",
        action="store_true",
        help="مع --search: تعطيل بوابة SSL",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="مع --search: شبكة فوليوم كاملة بلا تعزيز SSL",
    )
    parser.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="مع --search: تعطيل فلتر مسار العمق",
    )
    parser.add_argument(
        "--no-lean-filters",
        action="store_true",
        help="مع --search: توسيع كمّيات العمق/التعزيز",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="مع --search: شاشة BH استكشافية",
    )
    parser.add_argument(
        "--understand",
        action="store_true",
        help="مع --search: طبقات فهم OOS بعد الاختيار (لكل يوم)",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=SEARCH_N_PERMUTATIONS,
        help=f"تبديلات OOS داخل كل يوم (افتراضي {SEARCH_N_PERMUTATIONS})",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="حد صفوف اختياري لكل يوم (عادة لا يُحتاج مع شريحة يومية)",
    )
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="إيقاف الجدولة بعد أول يوم فاشل",
    )
    parser.add_argument(
        "--verbose-workers",
        action="store_true",
        help="طباعة تقدّم كل عامل (أثقل على اللوج)",
    )
    args = parser.parse_args()

    if args.nq_glob is None and args.nq_dir is None:
        parser.error("مطلوب --nq-glob أو --nq-dir")

    nq_paths = _collect_nq_paths(args)
    mnq_paths: list[Path] | None = None
    if args.mnq_glob:
        mnq_paths = []
        for pattern in args.mnq_glob:
            p = Path(pattern)
            if p.is_file():
                mnq_paths.append(p)
            else:
                parent = p.parent if p.parent.parts else Path(".")
                mnq_paths.extend(sorted(parent.glob(p.name)))

    days = discover_day_inputs(
        nq_paths=nq_paths,
        mnq_dir=args.mnq_dir,
        mnq_paths=mnq_paths,
    )
    mode = "search" if args.search else "unified"
    print(
        f"[nq] day-parallel FB · days={len(days)} · jobs={args.jobs} · "
        f"mode={mode} · threads/worker={args.threads_per_worker}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[nq] isolation: each day causal-closed · no cross-day selection",
        file=sys.stderr,
        flush=True,
    )

    manifest = run_fail_breakout_day_parallel(
        days,
        output_root=args.output,
        mode=mode,  # type: ignore[arg-type]
        jobs=args.jobs,
        threads_per_worker=args.threads_per_worker,
        global_seed=args.global_seed,
        horizon=args.horizon,
        max_rows=args.max_rows,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        use_ssl_gate=not args.no_ssl_gate,
        enhance_with_ssl=not args.no_enhance,
        use_depth_filter=not args.no_depth_filter,
        lean_filters=not args.no_lean_filters,
        exploratory=args.exploratory,
        understand=args.understand,
        quiet_workers=not args.verbose_workers,
        fail_fast=args.fail_fast,
    )
    print(manifest.to_markdown())
    print(f"manifest: {(args.output / 'manifest.json').resolve()}")
    print(f"summary:  {(args.output / 'summary.md').resolve()}")
    if manifest.n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
