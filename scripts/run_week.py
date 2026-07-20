#!/usr/bin/env python3
"""تشغيل أسبوعي — خط واحد من MBO إلى التقرير.

    python scripts/run_week.py
    python scripts/run_week.py --config configs/research.toml --output data/runs/w29
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# يضمن استيراد الحزمة من جذر المستودع عند التشغيل المباشر
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.orchestrator import PipelineConfig, run_research_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Nq unified research pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research.toml"),
        help="ملف إعدادات TOML",
    )
    parser.add_argument("--nq", type=Path, default=None, help="مسار NQ MBO (يتجاوز config)")
    parser.add_argument("--mnq", type=Path, default=None, help="مسار MNQ MBO (يتجاوز config)")
    parser.add_argument("--output", type=Path, default=None, help="مجلد مخرجات التقرير")
    args = parser.parse_args()

    cfg = PipelineConfig.from_toml(args.config)
    data = {}
    if args.config.is_file():
        import tomllib

        with args.config.open("rb") as handle:
            data = tomllib.load(handle).get("data", {})

    nq_path = args.nq or Path(data.get("nq_path", "data/raw/nq.parquet"))
    mnq_path = args.mnq or Path(data.get("mnq_path", "data/raw/mnq.parquet"))
    output_dir = args.output or Path(data.get("output_dir", "data/runs/latest"))

    result = run_research_pipeline(
        nq_path,
        mnq_path,
        config=cfg,
        output_dir=output_dir,
    )
    print(result.report.to_markdown())


if __name__ == "__main__":
    main()
