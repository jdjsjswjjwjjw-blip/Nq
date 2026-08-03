#!/usr/bin/env python3
"""تشغيل Volume Profile على أيام متوازية — مقياس شهر عملي.

كل يوم كون سببي مغلق. التوازي = ProcessPool على الأيام.
داخل اليوم: مسار batch السريع افتراضيًا (stream snapshots مع --streaming).

    python scripts/run_vp_auction_days.py \\
      --nq-dir /data/mnq_days \\
      --jobs 20 \\
      --threads-per-worker 4 \\
      --output data/runs/vp_month
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

from nq.research.day_parallel import discover_day_inputs  # noqa: E402
from nq.research.vp_day_parallel import run_vp_auction_day_parallel  # noqa: E402


def _collect_nq_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.nq_glob:
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
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    if not unique:
        raise FileNotFoundError("لا ملفات يومية — مرّر --nq-glob أو --nq-dir")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VP auction day-parallel month runner (isolated days, ProcessPool)"
    )
    parser.add_argument("--nq-glob", nargs="+", default=None)
    parser.add_argument("--nq-dir", type=Path, default=None)
    parser.add_argument("--mnq-dir", type=Path, default=None)
    parser.add_argument("--mnq-glob", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--min-oos-rr", type=float, default=2.0)
    parser.add_argument("--no-execution", action="store_true")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="tick_stream snapshots داخل كل يوم (أبطأ من batch)",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    nq_paths = _collect_nq_paths(args)
    mnq_paths: list[Path] = []
    if args.mnq_glob:
        for pattern in args.mnq_glob:
            p = Path(pattern)
            if p.is_file():
                mnq_paths.append(p)
            else:
                parent = p.parent if p.parent.parts else Path(".")
                mnq_paths.extend(sorted(parent.glob(p.name)))
    days = discover_day_inputs(
        nq_paths=nq_paths,
        mnq_paths=mnq_paths or None,
        mnq_dir=args.mnq_dir,
    )

    print(
        f"[nq] VP day-parallel: {len(days)} يوم · jobs={args.jobs} · "
        f"threads/worker={args.threads_per_worker} · "
        f"mode={'streaming-snapshots' if args.streaming else 'batch'}",
        flush=True,
    )
    print(
        f"[nq] live progress per day → {args.output}/<YYYY-MM-DD>/progress.log",
        flush=True,
    )
    manifest = run_vp_auction_day_parallel(
        days,
        output_root=args.output,
        jobs=args.jobs,
        threads_per_worker=args.threads_per_worker,
        global_seed=args.seed,
        max_rows=args.max_rows,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        with_execution=not args.no_execution,
        streaming_features=args.streaming,
        min_oos_rr=args.min_oos_rr,
        quiet_workers=False,
        fail_fast=args.fail_fast,
    )
    print(manifest.to_markdown())
    final = Path(args.output) / "FINAL_RESULT.md"
    if final.exists():
        print(final.read_text(encoding="utf-8"))
    print(f"outputs: {args.output.resolve()}/")
    if final.exists():
        print(f"FINAL: {final.resolve()}")


if __name__ == "__main__":
    main()
