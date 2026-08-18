#!/usr/bin/env python3
"""تدفق MNQ/NQ على شريحة ساعة مسمّاة.

    .venv/bin/python scripts/run_clock_flow.py \\
      --day 2026-08-17 --start 09:25:00 --end 09:40:00 \\
      --tz America/New_York --bin-s 300 --stack-bins \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --mnq-trades path/to/mnq.trades.clean.parquet \\
      --nq-trades path/to/nq.trades.clean.parquet \\
      --output data/runs/clock_flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.clock_flow import compare_clock_range, write_clock_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Named-clock MNQ/NQ three-tape flow")
    parser.add_argument("--day", type=str, required=True)
    parser.add_argument("--start", type=str, required=True, help="HH:MM:SS in --tz")
    parser.add_argument("--end", type=str, required=True, help="HH:MM:SS in --tz")
    parser.add_argument("--tz", type=str, default="America/New_York")
    parser.add_argument("--price-lo", type=float, default=None, help="named trough/level")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-s", type=int, default=60)
    parser.add_argument(
        "--stack-bins",
        action="store_true",
        help="include inner bins in the three-tape table (use with --bin-s 300)",
    )
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
    table, diagnostics = compare_clock_range(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        day=args.day,
        start_clock=args.start,
        end_clock=args.end,
        bin_s=args.bin_s,
        tz_name=args.tz,
        price_lo=args.price_lo,
        stack_bins=args.stack_bins,
    )
    diagnostics["mnq_mbo_path"] = str(args.mnq_mbo)
    diagnostics["mnq_trades_path"] = str(args.mnq_trades)
    diagnostics["nq_trades_path"] = str(args.nq_trades)
    written = write_clock_report(table, diagnostics, args.output)
    sources = diagnostics.get("sources") or []
    if sources:
        print(
            pl.DataFrame(sources).select(
                "name",
                "source",
                "n_t",
                "t_buy_size",
                "t_sell_size",
                "t_imbalance",
                "t_imbalance_early",
                "t_imbalance_late",
                "f_ask_size",
                "c_ask_size",
                "ask_hit_share",
                "t_per_s",
                "t_notional",
                "cvd_before",
                "cvd_end",
                "cvd_delta",
                "cvd_notional_end",
            ),
            flush=True,
        )
    print(table.filter(pl.col("name").is_in(["range", "after-0-300s"])), flush=True)
    print(
        "range NQ imb",
        diagnostics.get("range_nq_imbalance"),
        "NQ CVD",
        diagnostics.get("range_nq_cvd_end"),
        "$NQ CVD",
        diagnostics.get("range_nq_cvd_notional_end"),
        "MNQ fill",
        diagnostics.get("range_mnq_fill_ratio"),
        "low",
        diagnostics.get("low_clock"),
        diagnostics.get("low_px"),
        "level",
        diagnostics.get("price_lo"),
        diagnostics.get("level_clock"),
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
