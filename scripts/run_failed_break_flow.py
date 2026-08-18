#!/usr/bin/env python3
"""فشل كسر مبكر: تيك يوم واحد أو براميل سنة قبل holdout.

    .venv/bin/python scripts/run_failed_break_flow.py tick \\
      --mnq-trades path [--mnq-mbo path] --output data/runs/fb_flow_tick

    .venv/bin/python scripts/run_failed_break_flow.py year \\
      --year-dir data/runs/auction_behavior_year --output data/runs/fb_flow_year
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.failed_break_flow import (  # noqa: E402
    scan_tick_early_fail,
    scan_year_blended,
    write_failed_break_flow_report,
)


def _refuse_live(*parts: Path | str) -> None:
    joined = " ".join(str(p) for p in parts).lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")


def main() -> None:
    parser = argparse.ArgumentParser(description="In-bar failed break, not a lock")
    sub = parser.add_subparsers(dest="mode", required=True)
    tick = sub.add_parser("tick")
    tick.add_argument("--mnq-trades", type=Path, required=True)
    tick.add_argument("--mnq-mbo", type=Path, default=None)
    tick.add_argument("--output", type=Path, required=True)
    tick.add_argument("--point-value", type=float, default=2.0)
    year = sub.add_parser("year")
    year.add_argument("--year-dir", type=Path, required=True)
    year.add_argument("--output", type=Path, required=True)
    year.add_argument("--point-value", type=float, default=2.0)
    args = parser.parse_args()
    if args.mode == "tick":
        _refuse_live(args.mnq_trades, args.output, args.mnq_mbo or "")
        if not args.mnq_trades.is_file():
            raise SystemExit(f"no trades {args.mnq_trades}")
        mbo = None
        if args.mnq_mbo is not None:
            if not args.mnq_mbo.is_file():
                raise SystemExit(f"no MBO {args.mnq_mbo}")
            print("load mbo", flush=True)
            mbo = pl.read_parquet(args.mnq_mbo)
        print("load trades", flush=True)
        trades = pl.read_parquet(args.mnq_trades)
        table, diag = scan_tick_early_fail(trades, mbo, point_value=float(args.point_value))
    else:
        _refuse_live(args.year_dir, args.output)
        if not args.year_dir.is_dir():
            raise SystemExit(f"no year dir {args.year_dir}")
        print("scan year blended (holdout skipped)", flush=True)
        table, diag = scan_year_blended(args.year_dir, point_value=float(args.point_value))
    written = write_failed_break_flow_report(table, diag, args.output)
    print(
        "source",
        diag.get("source"),
        "days",
        diag.get("n_days"),
        "holdout_skipped",
        diag.get("n_skipped_holdout"),
        "breaks",
        diag.get("n_breaks"),
        "early",
        diag.get("n_early_fail"),
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
