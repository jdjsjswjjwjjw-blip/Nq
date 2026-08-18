#!/usr/bin/env python3
"""عمر أوامر MNQ MBO حول حزم T. وصف لا سبب.

    .venv/bin/python scripts/run_burst_lifetime.py \\
      --mnq-mbo path/to/mnq.mbo.clean.parquet \\
      --output data/runs/burst_lifetime --day 2026-08-17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.burst_lifetime import (  # noqa: E402
    BurstWindow,
    score_burst_lifetimes,
    write_burst_lifetime_report,
)
from nq.research.clock_flow import clock_to_ns  # noqa: E402

_WINDOWS = (
    ("pkt_1024", "10:24:00", "10:25:00", "10:24:20"),
    ("pkt_1151", "11:51:00", "11:52:00", "11:51:25"),
    ("pkt_1245", "12:45:00", "12:46:00", "12:45:40"),
    ("control_1200", "12:00:00", "12:01:00", "12:00:30"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MNQ MBO lifetimes around T packets")
    parser.add_argument("--mnq-mbo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day", type=str, default="2026-08-17")
    parser.add_argument("--tz", type=str, default="America/New_York")
    args = parser.parse_args()
    joined = f"{args.mnq_mbo} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    if not args.mnq_mbo.is_file():
        raise SystemExit(f"no MNQ MBO file {args.mnq_mbo}")
    print(f"load {args.mnq_mbo}", flush=True)
    mbo = pl.read_parquet(args.mnq_mbo)
    windows = [
        BurstWindow(
            name,
            clock_to_ns(args.day, start, args.tz),
            clock_to_ns(args.day, end, args.tz),
            clock_to_ns(args.day, burst, args.tz),
        )
        for name, start, end, burst in _WINDOWS
    ]
    table, diagnostics = score_burst_lifetimes(mbo, windows)
    diagnostics["mnq_mbo_path"] = str(args.mnq_mbo)
    diagnostics["day"] = args.day
    written = write_burst_lifetime_report(table, diagnostics, args.output)
    print(table, flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
