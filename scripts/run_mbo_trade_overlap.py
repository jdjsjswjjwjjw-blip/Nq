#!/usr/bin/env python3
"""نسب تنفيذ شريط TRADE مقابل MBO ليوم جلسة واحد.

    .venv/bin/python scripts/run_mbo_trade_overlap.py \\
      --mbo path/to/mbo.clean.parquet \\
      --trades path/to/trades.clean.parquet \\
      --output data/runs/mbo_trade_overlap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.mbo_trade_overlap import (  # noqa: E402
    compare_mbo_trades,
    overlap_diagnostics,
    overlap_table,
    write_overlap_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MBO vs TRADE overlap ratios, one session day")
    parser.add_argument("--mbo", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="optional tag stored in summary.json (e.g. clean or raw)",
    )
    args = parser.parse_args()
    joined = f"{args.mbo} {args.trades} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    if not args.mbo.is_file():
        raise SystemExit(f"no MBO file {args.mbo}")
    if not args.trades.is_file():
        raise SystemExit(f"no trades file {args.trades}")
    print(f"load mbo {args.mbo}", flush=True)
    mbo = pl.read_parquet(args.mbo)
    print(f"load trades {args.trades} mbo_rows={mbo.height}", flush=True)
    trades = pl.read_parquet(args.trades)
    print(f"compare trades_rows={trades.height}", flush=True)
    result = compare_mbo_trades(mbo, trades)
    table = overlap_table(result)
    diagnostics = overlap_diagnostics(
        result,
        mbo_path=str(args.mbo),
        trades_path=str(args.trades),
        label=args.label,
        mbo_rows_raw=mbo.height,
        trades_rows_raw=trades.height,
    )
    written = write_overlap_report(table, diagnostics, args.output)
    print(table, flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
