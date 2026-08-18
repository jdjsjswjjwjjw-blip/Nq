#!/usr/bin/env python3
"""NQ trades مقابل MNQ MBO على نوافذ القمة المقفلة.

    .venv/bin/python scripts/run_cross_nq_mnq.py \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --nq-trades path/to/nq.trades.clean.parquet \\
      --price-hi 30339 --output data/runs/cross_nq_mnq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.cross_nq_mnq import compare_nq_mnq_windows, write_cross_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="NQ trades vs MNQ MBO on locked peak windows")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--price-hi", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = f"{args.mnq_mbo} {args.nq_trades} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    if not args.mnq_mbo.is_file():
        raise SystemExit(f"no MNQ MBO file {args.mnq_mbo}")
    if not args.nq_trades.is_file():
        raise SystemExit(f"no NQ trades file {args.nq_trades}")
    print(f"load mnq {args.mnq_mbo}", flush=True)
    print(f"load nq {args.nq_trades}", flush=True)
    mnq = pl.read_parquet(args.mnq_mbo)
    nq = pl.read_parquet(args.nq_trades)
    stacked, diffs, diagnostics = compare_nq_mnq_windows(mnq, nq, price_hi=args.price_hi)
    diagnostics["mnq_mbo_path"] = str(args.mnq_mbo)
    diagnostics["nq_trades_path"] = str(args.nq_trades)
    written = write_cross_report(stacked, diffs, diagnostics, args.output)
    print(diffs, flush=True)
    print(
        "high_leader",
        diagnostics.get("high_leader"),
        "lag_ns",
        diagnostics.get("high_lag_ns"),
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
