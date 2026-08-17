#!/usr/bin/env python3
"""مسار 40 برميلًا: لقطة مقابل خطي مقابل لفّ سببي — بلا إعادة تدريب الرؤوس.

    .venv/bin/python scripts/run_path40_sequence.py \\
      --blended data/runs/auction_behavior_year/period_clean_trade/period_blended.parquet \\
      --output data/runs/auction_behavior_year/path40_sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.contracts.temporal import AVAILABILITY_TS  # noqa: E402
from nq.research.path40_sequence import (  # noqa: E402
    run_path40_sequence,
    write_path40_report,
)

_BLENDED_COLS = (
    AVAILABILITY_TS,
    "close",
    "high",
    "low",
    "path_beyond_asia_ticks",
    "vp_fsm_break",
    "vp_fsm_retest",
    "proj_break_direction",
    "asia_vah",
    "asia_val",
    "_behavior_story_run",
    "vp_liquidity_session",
)


def _read_columns(path: Path, wanted: tuple[str, ...]) -> pl.DataFrame:
    names = set(pl.read_parquet_schema(path))
    cols = [c for c in wanted if c in names]
    return pl.read_parquet(path, columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="40-bar path: last bar vs linear vs conv")
    parser.add_argument("--blended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if "live" in args.blended.name.lower():
        raise SystemExit("refuse live files")
    blended = _read_columns(args.blended, _BLENDED_COLS)
    scored, diagnostics = run_path40_sequence(blended, seed=args.seed)
    written = write_path40_report(scored, diagnostics, args.output)
    print((written / "PATH40.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
