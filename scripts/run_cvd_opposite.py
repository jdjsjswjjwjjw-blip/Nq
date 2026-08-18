#!/usr/bin/env python3
"""شرائح CVD المتعاكسة بين MNQ MBO وNQ.

    .venv/bin/python scripts/run_cvd_opposite.py \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --mnq-trades path/to/mnq.trades.clean.parquet \\
      --nq-trades path/to/nq.trades.clean.parquet \\
      --output data/runs/cvd_opposite --bin-s 300 --tz America/New_York
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.clock_flow import scan_cvd_opposite, write_cvd_opposite_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MNQ vs NQ opposite CVD bins")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-s", type=int, default=300)
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
    table, diagnostics = scan_cvd_opposite(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        bin_s=args.bin_s,
        tz_name=args.tz,
    )
    diagnostics["mnq_mbo_path"] = str(args.mnq_mbo)
    diagnostics["mnq_trades_path"] = str(args.mnq_trades)
    diagnostics["nq_trades_path"] = str(args.nq_trades)
    written = write_cvd_opposite_report(table, diagnostics, args.output)
    print(
        "bins",
        diagnostics.get("n_bins"),
        "delta_opposite",
        diagnostics.get("n_delta_opposite"),
        "end_opposite",
        diagnostics.get("n_end_opposite"),
        flush=True,
    )
    if table.height:
        print(table.filter(pl.col("delta_opposite")), flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
