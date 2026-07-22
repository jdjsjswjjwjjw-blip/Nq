#!/usr/bin/env python3
"""تشغيل بحث Volume Profile + التوازن/الاختلال عبر الخط الموحّد (NQ فقط).

    python scripts/run_vp_auction.py --nq data/raw/nq.parquet --max-rows 500000
    python scripts/run_week.py --nq data/raw/nq.parquet --nq-only \\
      --config configs/vp_auction.toml
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
        description="Volume Profile / Auction balance-imbalance research (NQ-only)"
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--output", type=Path, default=Path("data/runs/vp_auction"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="تعطيل طباعة تقدّم الخطوات على stderr",
    )
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")

    if not args.quiet:
        print(
            "[nq] ========== بدء: run_vp_auction (Volume Profile) ==========",
            file=sys.stderr,
            flush=True,
        )

    result = run_vp_auction_research(
        args.nq,
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
    for col in (
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_flip_to_imbalance",
    ):
        assert col in result.features.columns, f"missing {col}"
    if result.alpha.evaluations.height > 0:
        print(result.alpha.evaluations)


if __name__ == "__main__":
    main()
