#!/usr/bin/env python3
"""تشغيل بحث Failed FVG — أمر منفصل فوق الخط الموحّد (ليس خارج المنظومة).

يمرّ بنفس مرّات المشروع كاملة: تحميل MBO → ميزات → SSL ‖ M9 ‖ ألفا → تقرير + ملفات.

    python scripts/run_fail_fvg.py --nq data/raw/nq.parquet --max-rows 500000
    python scripts/run_week.py --nq data/raw/nq.parquet --nq-only \\
      --config configs/fail_fvg.toml
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
    parser = argparse.ArgumentParser(
        description=(
            "Failed FVG research — separate run command on the unified pipeline "
            "(full SSL‖M9‖alpha outputs)"
        )
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--mnq", type=Path, default=None, help="مسار MNQ اختياري (dual)")
    parser.add_argument("--output", type=Path, default=Path("data/runs/fail_fvg"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")
    if args.mnq is not None and not args.mnq.is_file():
        raise FileNotFoundError(f"MNQ MBO not found: {args.mnq.resolve()}")

    result = run_fail_fvg_research(
        args.nq,
        args.mnq,
        horizon=args.horizon,
        max_rows=args.max_rows,
        output_dir=args.output,
    )
    # التقرير الموحّد الكامل (قناة SSL + M9 + ألفا) — ليس ملخّص الاستراتيجية فقط
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
