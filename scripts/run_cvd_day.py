#!/usr/bin/env python3
"""مسح يوم واحد: تغطية + تعاكس 5د + توافق/توسّع + 60ث. بلا ساعات منسوخة.

    .venv/bin/python scripts/run_cvd_day.py \\
      --mnq-mbo path --mnq-trades path --nq-trades path \\
      --output data/runs/cvd_day --label 2026-08-16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.clock_flow import clock_to_ns  # noqa: E402
from nq.research.cvd_day_compare import scan_cvd_day, write_day_scan_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="CVD day scan, no copied clocks")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--tz", type=str, default="America/New_York")
    parser.add_argument("--bin-s", type=int, default=300)
    parser.add_argument("--day", type=str, default=None)
    parser.add_argument("--start-clock", type=str, default=None)
    parser.add_argument("--end-clock", type=str, default=None)
    args = parser.parse_args()
    joined = f"{args.mnq_mbo} {args.mnq_trades} {args.nq_trades} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    for path, name in (
        (args.mnq_mbo, "MNQ MBO"),
        (args.mnq_trades, "MNQ trades"),
        (args.nq_trades, "NQ trades"),
    ):
        if not path.is_file():
            raise SystemExit(f"no {name} file {path}")
    start_ts = end_ts = None
    if args.start_clock or args.end_clock:
        if not args.day:
            raise SystemExit("--day required with --start-clock/--end-clock")
        if args.start_clock:
            start_ts = clock_to_ns(args.day, args.start_clock, args.tz)
        if args.end_clock:
            end_ts = clock_to_ns(args.day, args.end_clock, args.tz)
    print(f"load {args.label}", flush=True)
    mnq_mbo = pl.read_parquet(args.mnq_mbo)
    mnq_trades = pl.read_parquet(args.mnq_trades)
    nq_trades = pl.read_parquet(args.nq_trades)
    scan = scan_cvd_day(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        label=args.label,
        tz_name=args.tz,
        bin_s=args.bin_s,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    written = write_day_scan_report(scan, args.output)
    print(
        "coverage",
        scan.coverage.get("coverage_class"),
        "hours",
        scan.coverage.get("t_hours"),
        "has_rth",
        scan.coverage.get("has_rth"),
        "bins5",
        scan.summary.get("n_bins_5m"),
        "opp",
        scan.summary.get("n_delta_opposite"),
        "strong",
        scan.summary.get("n_strong"),
        "wide",
        scan.summary.get("n_wide"),
        "rth_joins",
        scan.summary.get("n_rth_mnq_joins_nq"),
        "hyp1000",
        scan.summary.get("n_hyp_cvd"),
        "all3",
        scan.summary.get("n_hyp_all_three"),
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
