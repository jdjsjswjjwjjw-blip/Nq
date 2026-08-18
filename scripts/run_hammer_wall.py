#!/usr/bin/env python3
"""مطرقة MNQ Trades × جدار NQ MBP-10. بلا إعادة بناء دفتر.

    .venv/bin/python scripts/run_hammer_wall.py \\
      --mnq-mbo path --mnq-trades path --nq-trades path --nq-mbp10 path \\
      --output data/runs/hammer_wall
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.hammer_wall import scan_hammer_wall, write_hammer_wall_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Hammer vs wall, not a lock")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--nq-mbp10", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = (
        f"{args.mnq_mbo} {args.mnq_trades} {args.nq_trades} {args.nq_mbp10} {args.output}"
    ).lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    for path, name in (
        (args.mnq_mbo, "MNQ MBO"),
        (args.mnq_trades, "MNQ trades"),
        (args.nq_trades, "NQ trades"),
        (args.nq_mbp10, "NQ MBP-10"),
    ):
        if not path.is_file():
            raise SystemExit(f"no {name} file {path}")
    print("load frames", flush=True)
    mnq_mbo = pl.read_parquet(args.mnq_mbo)
    mnq_trades = pl.read_parquet(args.mnq_trades)
    nq_trades = pl.read_parquet(args.nq_trades)
    nq_mbp = pl.read_parquet(
        args.nq_mbp10,
        columns=[
            "ts_event",
            "ts_recv",
            "sequence",
            "ask_sz_00",
            "bid_sz_00",
            "ask_sz_01",
            "ask_sz_02",
            "bid_sz_01",
            "bid_sz_02",
        ],
    )
    table, diag = scan_hammer_wall(mnq_mbo, mnq_trades, nq_trades, nq_mbp)
    written = write_hammer_wall_report(table, diag, args.output)
    print(
        "bins",
        diag.get("n_bins"),
        "median_avg",
        diag.get("median_avg_trade_size"),
        "median_ask_l1",
        diag.get("median_ask_l1"),
        "thin",
        diag.get("n_ask_thin"),
        "thick",
        diag.get("n_ask_thick"),
        "hammer",
        diag.get("n_hammer"),
        "pass",
        diag.get("n_hyp_pass"),
        "pass_up_5m",
        diag.get("pass_up_rate_5m"),
        "pass_med5",
        diag.get("pass_median_next5m"),
        "pass_med15",
        diag.get("pass_median_next15m"),
        "fail_recipe",
        diag.get("n_hyp_fail_recipe"),
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
