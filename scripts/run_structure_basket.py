#!/usr/bin/env python3
"""طبقة هـ: باسكت هيكل على OOF — ليست أداة تنفيذ حيّة.

  .venv/bin/python scripts/run_structure_basket.py \\
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

from nq.research.hold_horizon import (  # noqa: E402
    jsonable,
    load_overlay_period_inputs,
    oof_timestamps,
)
from nq.research.progress import PipelineProgress  # noqa: E402
from nq.research.structure_basket import (  # noqa: E402
    DEFAULT_LOOKBACKS,
    StructureBasketConfig,
    run_structure_basket,
    run_structure_lookback_grid,
    write_structure_basket_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF test: structure stop/target basket (not live 1:4 execution)"
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-p", type=float, default=0.5)
    parser.add_argument("--lookback-bars", type=int, default=8)
    parser.add_argument(
        "--lookbacks",
        type=str,
        default="5,8,10",
        help="comma-separated robustness lookbacks; empty = primary only",
    )
    parser.add_argument("--stop-buffer-ticks", type=float, default=1.0)
    parser.add_argument("--min-ahead-ticks", type=float, default=16.0)
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--round-trip-cost-pts", type=float, default=0.75)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = StructureBasketConfig(
        min_p=float(args.min_p),
        lookback_bars=int(args.lookback_bars),
        stop_buffer_ticks=float(args.stop_buffer_ticks),
        min_ahead_ticks=float(args.min_ahead_ticks),
        max_hold_bars=int(args.max_hold_bars),
        round_trip_cost_pts=float(args.round_trip_cost_pts),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("structure_basket", total_steps=2)
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(args.period_dir)
    oof_ts = oof_timestamps(oof)
    report = run_structure_basket(
        labeled,
        blended,
        config=cfg,
        oof_availability_ts=oof_ts,
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=log,
    )
    written = write_structure_basket_report(report, args.output)
    lookbacks_raw = str(args.lookbacks).strip()
    if lookbacks_raw:
        lookbacks = tuple(int(x) for x in lookbacks_raw.split(",") if x.strip())
        if not lookbacks:
            lookbacks = DEFAULT_LOOKBACKS
        grid = run_structure_lookback_grid(
            labeled,
            blended,
            lookbacks=lookbacks,
            config=cfg,
            oof_availability_ts=oof_ts,
            holdout_cut_ts=cut_ts,
            predictions=oof,
            progress=log,
        )
        if grid.height:
            grid.write_parquet(written / "structure_basket_lookbacks.parquet")
            (written / "structure_basket_lookbacks.json").write_text(
                json.dumps(jsonable(grid.to_dicts()), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("lookback grid:", flush=True)
            print(grid, flush=True)
    log.done(
        f"layer=structure_basket traded={report.diagnostics.get('n_traded')} "
        f"skipped={report.diagnostics.get('n_skipped')}"
    )
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "STRUCTURE.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
