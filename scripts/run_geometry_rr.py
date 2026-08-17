#!/usr/bin/env python3
"""طبقة د: هندسة 1:4 إلى مستوى مجمّد عند t — قابلة للخلع.

  .venv/bin/python scripts/run_geometry_rr.py \\
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

from nq.research.geometry_rr import (  # noqa: E402
    GeometryRRConfig,
    run_geometry_rr_from_period_dir,
    write_geometry_rr_report,
)
from nq.research.progress import PipelineProgress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="removable layer D: 1:4 to a level frozen at t (not a timeframe change)"
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-p", type=float, default=0.5)
    parser.add_argument("--reward-multiple", type=float, default=4.0)
    parser.add_argument("--min-ahead-ticks", type=float, default=16.0)
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--round-trip-cost-pts", type=float, default=0.75)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = GeometryRRConfig(
        min_p=float(args.min_p),
        reward_multiple=float(args.reward_multiple),
        min_ahead_ticks=float(args.min_ahead_ticks),
        max_hold_bars=int(args.max_hold_bars),
        round_trip_cost_pts=float(args.round_trip_cost_pts),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("geometry_rr", total_steps=2)
    report = run_geometry_rr_from_period_dir(args.period_dir, config=cfg, progress=log)
    written = write_geometry_rr_report(report, args.output)
    log.done(
        f"layer=geometry_rr traded={report.diagnostics.get('n_traded')} "
        f"skipped={report.diagnostics.get('n_skipped')}"
    )
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "GEOMETRY.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
