#!/usr/bin/env python3
"""طبقة و: هدف/وقف ATR + هاي آسيا على OOF — ليست أداة تنفيذ حيّة.

  .venv/bin/python scripts/run_volatility_target.py \\
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
from nq.research.volatility_adjusted_target import (  # noqa: E402
    VolatilityTargetConfig,
    run_volatility_target,
    run_volatility_target_grid,
    write_volatility_target_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF test: ATR-scaled Asia target/stop (not live execution)"
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-p", type=float, default=0.5)
    parser.add_argument("--atr-days", type=int, default=14)
    parser.add_argument("--min-atr-pts", type=float, default=60.0)
    parser.add_argument("--target-atr-frac", type=float, default=0.4)
    parser.add_argument("--stop-atr-frac", type=float, default=0.2)
    parser.add_argument("--min-rr", type=float, default=2.0)
    parser.add_argument("--no-asia-extreme", action="store_true")
    parser.add_argument("--max-hold-bars", type=int, default=120)
    parser.add_argument("--round-trip-cost-pts", type=float, default=0.75)
    parser.add_argument("--grid", action="store_true", help="asia on/off × min_rr 2.0/1.5")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = VolatilityTargetConfig(
        min_p=float(args.min_p),
        atr_days=int(args.atr_days),
        min_atr_pts=float(args.min_atr_pts),
        target_atr_frac=float(args.target_atr_frac),
        stop_atr_frac=float(args.stop_atr_frac),
        min_rr=float(args.min_rr),
        use_asia_extreme=not bool(args.no_asia_extreme),
        max_hold_bars=int(args.max_hold_bars),
        round_trip_cost_pts=float(args.round_trip_cost_pts),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("volatility_target", total_steps=2)
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(args.period_dir)
    oof_ts = oof_timestamps(oof)
    report = run_volatility_target(
        labeled,
        blended,
        config=cfg,
        oof_availability_ts=oof_ts,
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=log,
    )
    written = write_volatility_target_report(report, args.output)
    if args.grid:
        grid = run_volatility_target_grid(
            labeled,
            blended,
            config=cfg,
            oof_availability_ts=oof_ts,
            holdout_cut_ts=cut_ts,
            predictions=oof,
            progress=log,
        )
        if grid.height:
            grid.write_parquet(written / "volatility_target_grid.parquet")
            (written / "volatility_target_grid.json").write_text(
                json.dumps(jsonable(grid.to_dicts()), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("robustness grid:", flush=True)
            print(grid, flush=True)
    log.done(
        f"layer=volatility_target traded={report.diagnostics.get('n_traded')} "
        f"skipped={report.diagnostics.get('n_skipped')}"
    )
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "VOLATILITY.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
