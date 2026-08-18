#!/usr/bin/env python3
"""ضوضاء عكسية قبل T العدواني داخل نطاق سعري. يوم جلسة واحد.

    .venv/bin/python scripts/run_opposite_phantom.py \\
      --mbo path/to/mbo.clean.parquet \\
      --price-lo 30331 --price-hi 30339 \\
      --output data/runs/opposite_phantom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.opposite_phantom import (  # noqa: E402
    DEFAULT_SIZE_MULT,
    DEFAULT_TICK_BAND,
    DEFAULT_WINDOWS_S,
    opposite_phantom,
    write_phantom_report,
)


def _parse_windows(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(int(p) for p in parts) if parts else DEFAULT_WINDOWS_S


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opposite unfilled cancels before aggressive T (heuristic, not spoofing)"
    )
    parser.add_argument("--mbo", type=Path, required=True)
    parser.add_argument("--price-lo", type=float, required=True)
    parser.add_argument("--price-hi", type=float, required=True)
    parser.add_argument("--windows", type=str, default="1,5,15,30")
    parser.add_argument("--tick-band", type=int, default=DEFAULT_TICK_BAND)
    parser.add_argument("--size-mult", type=float, default=DEFAULT_SIZE_MULT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = f"{args.mbo} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    if not args.mbo.is_file():
        raise SystemExit(f"no MBO file {args.mbo}")
    print(f"load mbo {args.mbo}", flush=True)
    mbo = pl.read_parquet(args.mbo)
    print(f"score rows={mbo.height} band=[{args.price_lo}, {args.price_hi}]", flush=True)
    windows, per_t, diagnostics = opposite_phantom(
        mbo,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
        windows_s=_parse_windows(args.windows),
        tick_band=args.tick_band,
        size_mult=args.size_mult,
    )
    diagnostics["mbo_path"] = str(args.mbo)
    written = write_phantom_report(windows, diagnostics, args.output, per_print=per_t)
    print(windows, flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
