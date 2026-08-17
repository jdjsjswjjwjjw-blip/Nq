#!/usr/bin/env python3
"""طبقة ح: فلتر زخم مبكر على OOF — ليست أداة تنفيذ حيّة.

  .venv/bin/python scripts/run_early_momentum.py \\
      --period-dir data/runs/auction_behavior_year/period_realized_path \\
      --output data/runs/auction_behavior_year/period_realized_path
"""

from __future__ import annotations

import argparse
import json
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

from nq.research.early_momentum_filter import (  # noqa: E402
    EarlyMomentumConfig,
    run_early_momentum,
    run_early_momentum_grid,
    write_early_momentum_report,
)
from nq.research.hold_horizon import (  # noqa: E402
    jsonable,
    load_overlay_period_inputs,
    oof_timestamps,
)
from nq.research.progress import PipelineProgress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF test: early momentum/volume/break entry filter (not live)"
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-p", type=float, default=0.5)
    parser.add_argument("--atr-days", type=int, default=14)
    parser.add_argument("--momentum-bars", type=int, default=5)
    parser.add_argument("--momentum-atr-frac", type=float, default=0.15)
    parser.add_argument("--volume-bars", type=int, default=20)
    parser.add_argument("--volume-multiple", type=float, default=1.5)
    parser.add_argument("--break-pts", type=float, default=2.0)
    parser.add_argument("--no-momentum", action="store_true")
    parser.add_argument("--no-volume", action="store_true")
    parser.add_argument("--no-break", action="store_true")
    parser.add_argument("--max-hold-bars", type=int, default=780)
    parser.add_argument("--round-trip-cost-pts", type=float, default=0.75)
    parser.add_argument(
        "--grid",
        action="store_true",
        help="drop volume and/or break (pre-declared; not a threshold search)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = EarlyMomentumConfig(
        min_p=float(args.min_p),
        atr_days=int(args.atr_days),
        momentum_bars=int(args.momentum_bars),
        momentum_atr_frac=float(args.momentum_atr_frac),
        volume_bars=int(args.volume_bars),
        volume_multiple=float(args.volume_multiple),
        break_pts=float(args.break_pts),
        require_momentum=not bool(args.no_momentum),
        require_volume=not bool(args.no_volume),
        require_break=not bool(args.no_break),
        max_hold_bars=int(args.max_hold_bars),
        round_trip_cost_pts=float(args.round_trip_cost_pts),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("early_momentum_filter", total_steps=2)
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(args.period_dir)
    oof_ts = oof_timestamps(oof)
    report = run_early_momentum(
        labeled,
        blended,
        config=cfg,
        oof_availability_ts=oof_ts,
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=log,
    )
    written = write_early_momentum_report(report, args.output)
    if args.grid:
        grid = run_early_momentum_grid(
            labeled,
            blended,
            config=cfg,
            oof_availability_ts=oof_ts,
            holdout_cut_ts=cut_ts,
            predictions=oof,
            progress=log,
        )
        if grid.height:
            grid.write_parquet(written / "early_momentum_grid.parquet")
            (written / "early_momentum_grid.json").write_text(
                json.dumps(jsonable(grid.to_dicts()), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("robustness grid:", flush=True)
            print(grid, flush=True)
    log.done(
        f"layer=early_momentum_filter traded={report.diagnostics.get('n_traded')} "
        f"skipped={report.diagnostics.get('n_skipped')}"
    )
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "EARLY_MOMENTUM.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
