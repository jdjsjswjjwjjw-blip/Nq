#!/usr/bin/env python3
"""مسح نمط القمة على كل شرائح اليوم مقابل ضابط.

    .venv/bin/python scripts/run_peak_pattern.py \\
      --mbo path/to/mbo.clean.parquet --price-hi 30339 \\
      --output data/runs/peak_pattern
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.peak_pattern import scan_peak_pattern, write_pattern_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan locked peak pattern vs control")
    parser.add_argument("--mbo", type=Path, required=True)
    parser.add_argument("--price-hi", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    joined = f"{args.mbo} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    if not args.mbo.is_file():
        raise SystemExit(f"no MBO file {args.mbo}")
    print(f"load mbo {args.mbo}", flush=True)
    mbo = pl.read_parquet(args.mbo)
    scored, diagnostics = scan_peak_pattern(mbo, price_hi=args.price_hi, seed=args.seed)
    diagnostics["mbo_path"] = str(args.mbo)
    written = write_pattern_report(scored, diagnostics, args.output)
    print(diagnostics.get("summary"), flush=True)
    print("named_peak", diagnostics.get("named_peak"), flush=True)
    print(
        f"n_pattern_windows={diagnostics.get('n_pattern_windows')} "
        f"episodes={diagnostics.get('n_pattern_episodes')}",
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
