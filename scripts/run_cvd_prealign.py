#!/usr/bin/env python3
"""مسار 60ث قبل توافق MNQ/NQ بعد تعاكس قوي.

    .venv/bin/python scripts/run_cvd_prealign.py \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --mnq-trades path/to/mnq.trades.clean.parquet \\
      --nq-trades path/to/nq.trades.clean.parquet \\
      --output data/runs/cvd_prealign
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.clock_flow import scan_cvd_prealign, write_cvd_prealign_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="60s path into MNQ/NQ CVD alignment")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tz", type=str, default="America/New_York")
    args = parser.parse_args()
    joined = f"{args.mnq_mbo} {args.mnq_trades} {args.nq_trades} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    for path, label in (
        (args.mnq_mbo, "MNQ MBO"),
        (args.mnq_trades, "MNQ trades"),
        (args.nq_trades, "NQ trades"),
    ):
        if not path.is_file():
            raise SystemExit(f"no {label} file {path}")
    print(f"load mnq mbo {args.mnq_mbo}", flush=True)
    mnq_mbo = pl.read_parquet(args.mnq_mbo)
    print(f"load mnq trades {args.mnq_trades}", flush=True)
    mnq_trades = pl.read_parquet(args.mnq_trades)
    print(f"load nq trades {args.nq_trades}", flush=True)
    nq_trades = pl.read_parquet(args.nq_trades)
    minutes, summaries, diagnostics = scan_cvd_prealign(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        tz_name=args.tz,
    )
    diagnostics["mnq_mbo_path"] = str(args.mnq_mbo)
    diagnostics["mnq_trades_path"] = str(args.mnq_trades)
    diagnostics["nq_trades_path"] = str(args.nq_trades)
    written = write_cvd_prealign_report(minutes, summaries, diagnostics, args.output)
    print(
        "episodes",
        diagnostics.get("n_episodes"),
        "mnq_joins_nq",
        diagnostics.get("n_mnq_joins_nq"),
        "slowing",
        diagnostics.get("n_cvd_slowing"),
        "imb0",
        diagnostics.get("n_imb_near_zero"),
        "t_drop",
        diagnostics.get("n_t_drop"),
        "t_jump",
        diagnostics.get("n_t_jump"),
        "cvd500",
        diagnostics.get("n_cvd_jump_500"),
        flush=True,
    )
    if summaries.height:
        print(summaries, flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
