#!/usr/bin/env python3
"""تشغيل أسبوعي — خط واحد من MBO إلى التقرير.

    python scripts/run_week.py
    python scripts/run_week.py --config configs/research.toml --output data/runs/w29
    python scripts/run_week.py --nq data/raw/nq.parquet --nq-only --max-rows 500000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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

from nq.research.orchestrator import PipelineConfig, run_research_pipeline  # noqa: E402


def _require_path(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path.resolve()}")
    return path


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
    parser.add_argument(
        "--nq-only",
        action="store_true",
        help="استخدم NQ فقط (بدون ملف MNQ منفصل)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="حد أقصى لصفوف MBO (للتجارب أو الذاكرة المحدودة)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="تعطيل طباعة تقدّم الخطوات على stderr",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_toml(args.config) if args.config.is_file() else PipelineConfig()
    data: dict[str, object] = {}
    if args.config.is_file():
        import tomllib

        with args.config.open("rb") as handle:
            data = tomllib.load(handle).get("data", {})

    nq_path = Path(args.nq or data.get("nq_path", "data/raw/nq.parquet"))
    mnq_path = Path(args.mnq or data.get("mnq_path", "data/raw/mnq.parquet"))
    output_dir = Path(args.output or data.get("output_dir", "data/runs/latest"))

    nq_only = args.nq_only or str(data.get("cross_market_mode", "")) == "nq_only"
    max_rows = args.max_rows if args.max_rows is not None else cfg.max_rows

    _require_path(nq_path, "NQ MBO")
    if not nq_only:
        _require_path(mnq_path, "MNQ MBO")

    if nq_only:
        cfg = replace(cfg, cross_market_mode="nq_only", max_rows=max_rows, quiet=args.quiet)
    else:
        cfg = replace(cfg, max_rows=max_rows, quiet=args.quiet)

    result = run_research_pipeline(
        nq_path,
        mnq_path if not nq_only else nq_path,
        config=cfg,
        output_dir=output_dir,
    )
    print(result.report.to_markdown())


if __name__ == "__main__":
    main()
