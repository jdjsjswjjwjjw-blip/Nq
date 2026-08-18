#!/usr/bin/env python3
"""كسر فاشل سببي: ملء عند افتتاح الشمعة التالية. بلا أمر معلّق على range_high.

    .venv/bin/python scripts/run_failed_breakout.py \\
      --mnq-trades path --output data/runs/failed_breakout_causal

    # تشخيص بلا SMA50 (ليس الاستراتيجية الأصلية)
    .venv/bin/python scripts/run_failed_breakout.py \\
      --mnq-trades path --sma-bars 0 --output data/runs/failed_breakout_nosma
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.failed_breakout import (  # noqa: E402
    SMA_PERIOD,
    scan_failed_breakout,
    write_failed_breakout_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal failed breakout, next-open fill")
    parser.add_argument("--mnq-trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sma-bars",
        type=int,
        default=SMA_PERIOD,
        help="Hourly SMA period. 50 is the original filter; 0 disables it (diagnostic only)",
    )
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--hold-bars", type=int, default=None)
    parser.add_argument("--rr", type=float, default=None)
    args = parser.parse_args()
    joined = f"{args.mnq_trades} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    if not args.mnq_trades.is_file():
        raise SystemExit(f"no MNQ trades file {args.mnq_trades}")
    kwargs: dict[str, int | float] = {"sma_period": int(args.sma_bars)}
    if args.lookback is not None:
        kwargs["lookback"] = int(args.lookback)
    if args.hold_bars is not None:
        kwargs["hold_bars"] = int(args.hold_bars)
    if args.rr is not None:
        kwargs["reward_ratio"] = float(args.rr)
    print("load trades", flush=True)
    trades = pl.read_parquet(args.mnq_trades)
    table, diag = scan_failed_breakout(trades, **kwargs)
    written = write_failed_breakout_report(table, diag, args.output)
    print(
        "sma_filter",
        diag.get("sma_filter"),
        "bars30",
        diag.get("n_30m_bars"),
        "pattern",
        diag.get("n_fb_pattern"),
        "skipped_sma",
        diag.get("n_skipped_sma"),
        "skipped_gap",
        diag.get("n_skipped_gap_sl"),
        "trades",
        diag.get("n_trades"),
        "win",
        diag.get("n_win"),
        "gross_usd",
        diag.get("gross_pnl_usd"),
        "median_pnl",
        diag.get("median_pnl_pts"),
        "median_leak",
        diag.get("median_leak_pts"),
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
