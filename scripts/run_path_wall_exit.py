#!/usr/bin/env python3
"""وقف مسار خلف أقوى جدار: وصف عند إعداد المزاد، ليست overlay.

    .venv/bin/python scripts/run_path_wall_exit.py scan \\
      --year-dir data/runs/auction_behavior_year \\
      --mbo-root Restore_Data/2025 \\
      --output /tmp/path_wall_exit_2025

    .venv/bin/python scripts/run_path_wall_exit.py join \\
      --wall-parquet /tmp/path_wall_exit_2025/path_wall_exit.parquet \\
      --oof-parquet data/runs/auction_behavior_year/period_realized_path/oof_predictions.parquet \\
      --output /tmp/path_wall_exit_2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.path_wall_exit import (  # noqa: E402
    attach_period_path_oof,
    scan_year_path_wall,
    write_path_wall_exit_report,
)


def _refuse_live(*parts: Path | str) -> None:
    joined = " ".join(str(p) for p in parts).lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")


def _print_diag(diag: dict) -> None:
    wall = diag.get("with_wall") or {}
    oof = diag.get("oof_ge_half") or {}
    print(
        "days",
        diag.get("n_days"),
        "holdout_skipped",
        diag.get("n_skipped_holdout"),
        "onsets",
        diag.get("n_onsets"),
        "directed",
        diag.get("n_directed"),
        "with_wall",
        diag.get("n_with_wall"),
        "median_mae",
        wall.get("median_mae_pts") if isinstance(wall, dict) else None,
        "median_risk",
        wall.get("median_risk_pts") if isinstance(wall, dict) else None,
        "share_hit_sl",
        wall.get("share_hit_sl") if isinstance(wall, dict) else None,
        "oof_scored",
        diag.get("n_oof_scored"),
        "oof_ge_half",
        diag.get("n_oof_ge_half"),
        "oof_mae",
        oof.get("median_mae_pts") if isinstance(oof, dict) else None,
        "oof_hit",
        oof.get("share_hit_sl") if isinstance(oof, dict) else None,
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-setup wall stop, not a lock")
    sub = parser.add_subparsers(dest="mode", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--year-dir", type=Path, required=True)
    scan.add_argument("--mbo-root", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--point-value", type=float, default=2.0)
    join = sub.add_parser("join")
    join.add_argument("--wall-parquet", type=Path, required=True)
    join.add_argument("--oof-parquet", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "scan":
        _refuse_live(args.year_dir, args.mbo_root, args.output)
        if not args.year_dir.is_dir():
            raise SystemExit(f"no year dir {args.year_dir}")
        if not args.mbo_root.is_dir():
            raise SystemExit(f"no IDrive root {args.mbo_root}")
        print("scan path onsets + wall stop (holdout skipped, exits stay manual)", flush=True)
        table, diag = scan_year_path_wall(
            args.year_dir,
            args.mbo_root,
            point_value=float(args.point_value),
            log=lambda msg: print(msg, flush=True),
        )
    else:
        _refuse_live(args.wall_parquet, args.oof_parquet, args.output)
        if not args.wall_parquet.is_file():
            raise SystemExit(f"no wall parquet {args.wall_parquet}")
        if not args.oof_parquet.is_file():
            raise SystemExit(f"no OOF parquet {args.oof_parquet}")
        print("join period path OOF onto existing wall rows (not a new backtest)", flush=True)
        table = pl.read_parquet(args.wall_parquet)
        oof = pl.read_parquet(args.oof_parquet)
        prior = args.wall_parquet.parent / "summary.json"
        diagnostics = None
        if prior.is_file():
            diagnostics = json.loads(prior.read_text(encoding="utf-8"))
        table, diag = attach_period_path_oof(table, oof, diagnostics=diagnostics)
    written = write_path_wall_exit_report(table, diag, args.output)
    _print_diag(diag)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
