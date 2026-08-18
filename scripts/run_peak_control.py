#!/usr/bin/env python3
"""قمة مقابل صعود/هبوط على نفس اليوم. أرقام فقط.

    .venv/bin/python scripts/run_peak_control.py \\
      --mbo path/to/mbo.clean.parquet \\
      --price-hi 30339 \\
      --output data/runs/peak_control
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.peak_control import compare_peak_controls, write_control_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Peak vs climb vs drop 30s diagnostic (not a model)"
    )
    parser.add_argument("--mbo", type=Path, required=True)
    parser.add_argument("--price-hi", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = f"{args.mbo} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    if not args.mbo.is_file():
        raise SystemExit(f"no MBO file {args.mbo}")
    print(f"load mbo {args.mbo}", flush=True)
    mbo = pl.read_parquet(args.mbo)
    table, diagnostics = compare_peak_controls(mbo, price_hi=args.price_hi)
    diagnostics["mbo_path"] = str(args.mbo)
    written = write_control_report(table, diagnostics, args.output)
    print(table, flush=True)
    print(diagnostics.get("day_30s"), flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
