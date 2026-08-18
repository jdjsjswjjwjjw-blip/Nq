#!/usr/bin/env python3
"""MNQ MBO + MNQ Trades + NQ Trades على نوافذ القمة ثم مسح اليوم.

    .venv/bin/python scripts/run_triple_tape.py \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --mnq-trades path/to/mnq.trades.clean.parquet \\
      --nq-trades path/to/nq.trades.clean.parquet \\
      --price-hi 30339 --output data/runs/triple_tape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.triple_tape import (  # noqa: E402
    compare_triple_windows,
    scan_triple_pattern,
    write_triple_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple tape named windows + day scan")
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
    print(f"load mnq trades {args.mnq_trades}", flush=True)
    print(f"load nq trades {args.nq_trades}", flush=True)
    mnq_mbo = pl.read_parquet(args.mnq_mbo)
    mnq_trades = pl.read_parquet(args.mnq_trades)
    nq_trades = pl.read_parquet(args.nq_trades)
    named, diagnostics = compare_triple_windows(
        mnq_mbo, mnq_trades, nq_trades, price_hi=args.price_hi
    )
    scored = None
    if not args.skip_scan:
        scored, scan_diag = scan_triple_pattern(mnq_mbo, mnq_trades, nq_trades, seed=args.seed)
        diagnostics.update({k: v for k, v in scan_diag.items() if k not in {"layer", "window_s"}})
        diagnostics["named_and_scan"] = True
    diagnostics["mnq_mbo_path"] = str(args.mnq_mbo)
    diagnostics["mnq_trades_path"] = str(args.mnq_trades)
    diagnostics["nq_trades_path"] = str(args.nq_trades)
    written = write_triple_report(named, diagnostics, args.output, scored=scored)
    print(named, flush=True)
    print(
        "peak_hypothesis",
        diagnostics.get("peak_hypothesis_holds"),
        "drop_hypothesis",
        diagnostics.get("drop_hypothesis_holds"),
        flush=True,
    )
    print(diagnostics.get("summary"), flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
