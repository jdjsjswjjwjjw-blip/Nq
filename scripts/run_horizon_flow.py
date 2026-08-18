#!/usr/bin/env python3
"""مسار 5 دقائق بعد القمة + مسح اليوم مع ضابط T_rate>50.

    .venv/bin/python scripts/run_horizon_flow.py \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --mnq-trades path/to/mnq.trades.clean.parquet \\
      --nq-trades path/to/nq.trades.clean.parquet \\
      --price-hi 30339 --output data/runs/horizon_flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.horizon_flow import compare_post_peak_horizon, write_horizon_report  # noqa: E402
from nq.research.triple_tape import scan_triple_pattern, write_triple_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="5-minute post-peak NQ path + busy control scan")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--price-hi", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-scan", action="store_true")
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
    table, diagnostics = compare_post_peak_horizon(
        mnq_mbo, mnq_trades, nq_trades, price_hi=args.price_hi
    )
    written = write_horizon_report(table, diagnostics, args.output)
    print(table, flush=True)
    print(
        "5m NQ imb",
        diagnostics.get("horizon_5m_nq_imbalance"),
        "faded_nonpos",
        diagnostics.get("nq_faded_nonpos"),
        "old_drop",
        diagnostics.get("old_drop_30_60s_nq_imbalance"),
        flush=True,
    )
    if not args.skip_scan:
        scored, scan_diag = scan_triple_pattern(mnq_mbo, mnq_trades, nq_trades, seed=args.seed)
        scan_diag["mnq_mbo_path"] = str(args.mnq_mbo)
        scan_diag["mnq_trades_path"] = str(args.mnq_trades)
        scan_diag["nq_trades_path"] = str(args.nq_trades)
        write_triple_report(table.head(0), scan_diag, Path(args.output) / "scan", scored=scored)
        print(scan_diag.get("summary"), flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
