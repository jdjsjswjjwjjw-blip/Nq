#!/usr/bin/env python3
"""شرائح 60ث لليوم و30ث/5ث حول دقائق عنيفة. بلا قفل عتبة.

    .venv/bin/python scripts/run_cvd_burst.py \\
      --mnq-mbo path --mnq-trades path --nq-trades path \\
      --output data/runs/cvd_burst --day 2026-08-17
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
    clock_to_ns,
    scan_tape_bins,
    write_tape_bins_report,
)

_SPANS_30S = (
    ("violent_1024", "10:23:00", "10:27:00"),
    ("burst_1151", "11:50:00", "11:53:00"),
    ("fail_1245", "12:44:00", "12:47:00"),
    ("fail_1328", "13:27:00", "13:31:00"),
)
_SPANS_5S = (
    ("inner_1024", "10:24:00", "10:25:00"),
    ("inner_1151", "11:51:00", "11:52:00"),
    ("inner_1245", "12:45:00", "12:46:00"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tape burst bins, not a lock")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--nq-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day", type=str, default="2026-08-17")
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
    print("load frames", flush=True)
    mnq_mbo = pl.read_parquet(args.mnq_mbo)
    mnq_trades = pl.read_parquet(args.mnq_trades)
    nq_trades = pl.read_parquet(args.nq_trades)
    day60, day_diag = scan_tape_bins(
        mnq_mbo, mnq_trades, nq_trades, bin_s=60, tz_name=args.tz, label="day60", inner_s=5
    )
    tables: dict[str, pl.DataFrame] = {"day60": day60}
    diagnostics: dict[str, object] = {
        "day60": day_diag,
        "not_pattern": True,
        "not_burst_lock": True,
        "mnq_mbo_path": str(args.mnq_mbo),
        "mnq_trades_path": str(args.mnq_trades),
        "nq_trades_path": str(args.nq_trades),
    }
    print(
        "day60",
        day_diag.get("n_bins"),
        "hyp_cvd",
        day_diag.get("n_hyp_cvd"),
        "all_three",
        day_diag.get("n_hyp_all_three"),
        flush=True,
    )
    for name, start, end in _SPANS_30S:
        table, diag = scan_tape_bins(
            mnq_mbo,
            mnq_trades,
            nq_trades,
            bin_s=30,
            tz_name=args.tz,
            start_ts=clock_to_ns(args.day, start, args.tz),
            end_ts=clock_to_ns(args.day, end, args.tz),
            inner_s=5,
            label=name,
        )
        tables[name] = table
        diagnostics[name] = diag
        print(name, diag.get("n_bins"), flush=True)
    for name, start, end in _SPANS_5S:
        table, diag = scan_tape_bins(
            mnq_mbo,
            mnq_trades,
            nq_trades,
            bin_s=5,
            tz_name=args.tz,
            start_ts=clock_to_ns(args.day, start, args.tz),
            end_ts=clock_to_ns(args.day, end, args.tz),
            inner_s=1,
            label=name,
        )
        tables[name] = table
        diagnostics[name] = diag
        print(name, diag.get("n_bins"), flush=True)
    written = write_tape_bins_report(tables, diagnostics, args.output)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
