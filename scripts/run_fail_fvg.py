#!/usr/bin/env python3
"""تشغيل بحث Failed FVG عبر الخط الموحّد (نفس طبقة trap_setup / lead_lag).

    python scripts/run_fail_fvg.py --nq data/raw/nq.parquet --max-rows 500000
    python scripts/run_week.py --nq data/raw/nq.parquet --nq-only
    # ↑ run_week يُفرز fail_fvg تلقائيًا مع باقي الإشارات
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

from nq.strategies.fail_fvg import run_fail_fvg_research  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Failed FVG strategy research pipeline")
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--output", type=Path, default=Path("data/runs/fail_fvg"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-ssl-gate", action="store_true", help="تعطيل بوابة SSL latent")
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")

    result = run_fail_fvg_research(
        args.nq,
        use_ssl_gate=not args.no_ssl_gate,
        horizon=args.horizon,
        max_rows=args.max_rows,
        output_dir=args.output,
    )
    print(result.report.to_markdown())
    print(f"\nsignals: {result.signal_columns}")
    print(f"features: {result.features.height} rows")
    if result.alpha.evaluations.height > 0:
        print(result.alpha.evaluations)


if __name__ == "__main__":
    main()
