#!/usr/bin/env python3
"""طبقة ب: خروج فيتشز بعد إطلاق OOF — قابلة للخلع.

  .venv/bin/python scripts/run_feature_exit.py \\
      --period-dir data/runs/auction_behavior_year/period_realized_path \\
      --output data/runs/auction_behavior_year/period_realized_path
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

from nq.research.feature_exit import (  # noqa: E402
    FeatureExitConfig,
    run_feature_exit_from_period_dir,
    write_feature_exit_report,
)
from nq.research.progress import PipelineProgress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="removable layer B: feature-based exit after OOF fire"
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hold-bars", type=int, default=30)
    parser.add_argument("--round-trip-cost-pts", type=float, default=0.75)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = FeatureExitConfig(
        max_hold_bars=int(args.max_hold_bars),
        round_trip_cost_pts=float(args.round_trip_cost_pts),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("feature_exit", total_steps=2)
    report = run_feature_exit_from_period_dir(args.period_dir, config=cfg, progress=log)
    written = write_feature_exit_report(report, args.output)
    log.done(f"layer=feature_exit trades={report.diagnostics.get('n_trades')}")
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "FEATURE_EXIT.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
