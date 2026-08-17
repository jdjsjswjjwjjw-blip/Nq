#!/usr/bin/env python3
"""ميكانيكا الامتداد من science_labeled.parquet — بلا MBO وبلا holdout.

  .venv/bin/python scripts/run_expansion_mechanics.py \\
      --period-dir data/runs/auction_behavior_year/period_realized_path

أو مباشرة:

  .venv/bin/python scripts/run_expansion_mechanics.py \\
      --labeled path/to/science_labeled.parquet \\
      --output data/runs/expansion_mechanics
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

import polars as pl  # noqa: E402

from nq.research.expansion_mechanics import (  # noqa: E402
    ExpansionMechanicsConfig,
    run_expansion_mechanics,
    run_expansion_mechanics_from_period_dir,
    write_expansion_mechanics_report,
)
from nq.research.progress import PipelineProgress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "expansion mechanics on labeled parquet: volume vs price lead-lag, "
            "balance→imbalance→expansion, protection — never reconstruct, never score holdout"
        )
    )
    parser.add_argument("--period-dir", type=Path, default=None)
    parser.add_argument("--labeled", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-months", type=int, default=4)
    parser.add_argument("--holdout-cut-ts", type=int, default=None)
    parser.add_argument("--n-permutations", type=int, default=199)
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.period_dir is None and args.labeled is None:
        parser.error("pass --period-dir or --labeled")

    cfg = ExpansionMechanicsConfig(
        holdout_months=int(args.holdout_months),
        n_permutations=int(args.n_permutations),
        lag=int(args.lag),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("expansion_mechanics", total_steps=2)
    if args.period_dir is not None:
        report = run_expansion_mechanics_from_period_dir(args.period_dir, config=cfg, progress=log)
    else:
        labeled = pl.read_parquet(args.labeled)
        report = run_expansion_mechanics(
            labeled,
            config=cfg,
            holdout_cut_ts=args.holdout_cut_ts,
            progress=log,
        )
    written = write_expansion_mechanics_report(report, args.output)
    log.done(f"scope={report.diagnostics.get('primary_scope')} holdout_scored=false")
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "EXPANSION.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
