#!/usr/bin/env python3
"""وقف مسار خلف أقوى جدار: وصف عند إعداد المزاد، ليست overlay.

    .venv/bin/python scripts/run_path_wall_exit.py \\
      --year-dir data/runs/auction_behavior_year \\
      --mbo-root Restore_Data/2025 \\
      --output /tmp/path_wall_exit_2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.path_wall_exit import (  # noqa: E402
    scan_year_path_wall,
    write_path_wall_exit_report,
)


def _refuse_live(*parts: Path | str) -> None:
    joined = " ".join(str(p) for p in parts).lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-setup wall stop, not a lock")
    parser.add_argument("--year-dir", type=Path, required=True)
    parser.add_argument("--mbo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-value", type=float, default=2.0)
    args = parser.parse_args()
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
    written = write_path_wall_exit_report(table, diag, args.output)
    wall = diag.get("with_wall") or {}
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
        "oof_ge_half",
        diag.get("n_oof_ge_half"),
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
