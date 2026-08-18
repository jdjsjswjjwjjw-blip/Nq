#!/usr/bin/env python3
"""اختلال T/F في نوافذ القمة الضابطة. أرقام فقط.

    .venv/bin/python scripts/run_peak_flow.py \\
      --mbo path/to/mbo.clean.parquet --price-hi 30339 \\
      --output data/runs/peak_flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.peak_flow import compare_peak_flow, write_flow_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Peak T/F flow diagnostic (not a model)")
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
    table, diagnostics = compare_peak_flow(mbo, price_hi=args.price_hi)
    diagnostics["mbo_path"] = str(args.mbo)
    written = write_flow_report(table, diagnostics, args.output)
    print(table, flush=True)
    print(diagnostics.get("day_30s"), flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
