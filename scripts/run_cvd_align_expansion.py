#!/usr/bin/env python3
"""تعاكس CVD قوي ثم أول توافق MNQ/NQ ومدى السعر.

    .venv/bin/python scripts/run_cvd_align_expansion.py \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --mnq-trades path/to/mnq.trades.clean.parquet \\
      --nq-trades path/to/nq.trades.clean.parquet \\
      --output data/runs/cvd_align_expansion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.clock_flow import (  # noqa: E402
    scan_cvd_align_expansion,
    write_cvd_align_expansion_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong opposite CVD then align expansion")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-s", type=int, default=300)
    parser.add_argument("--tz", type=str, default="America/New_York")
    parser.add_argument("--strong-mnq", type=int, default=500)
    parser.add_argument("--strong-nq", type=int, default=80)
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
    table, diagnostics = scan_cvd_align_expansion(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        bin_s=args.bin_s,
        tz_name=args.tz,
        strong_mnq=args.strong_mnq,
        strong_nq=args.strong_nq,
    )
    diagnostics["mnq_mbo_path"] = str(args.mnq_mbo)
    diagnostics["mnq_trades_path"] = str(args.mnq_trades)
    diagnostics["nq_trades_path"] = str(args.nq_trades)
    written = write_cvd_align_expansion_report(table, diagnostics, args.output)
    print(
        "strong",
        diagnostics.get("n_strong_episodes"),
        "aligned",
        diagnostics.get("n_aligned"),
        "wide",
        diagnostics.get("n_wide_vs_median"),
        "with_align",
        diagnostics.get("n_moved_with_align"),
        "with_nq_opp",
        diagnostics.get("n_moved_with_nq_opp"),
        flush=True,
    )
    if table.height:
        print(table, flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
