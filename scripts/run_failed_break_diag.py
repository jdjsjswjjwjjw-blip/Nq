#!/usr/bin/env python3
"""تشخيص كسر فاشل: Fill_Ratio عند الكسر + هندسة الوقف. ليست قفلًا.

    .venv/bin/python scripts/run_failed_break_diag.py idrive \\
      --mbo-root Restore_Data/2025 --output /tmp/fb_flow_diag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.failed_break_diag import (  # noqa: E402
    scan_year_idrive_diag,
    write_failed_break_diag_report,
)


def _refuse_live(*parts: Path | str) -> None:
    joined = " ".join(str(p) for p in parts).lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")


def main() -> None:
    parser = argparse.ArgumentParser(description="Failed-break diagnostics, not a lock")
    sub = parser.add_subparsers(dest="mode", required=True)
    idrive = sub.add_parser("idrive")
    idrive.add_argument("--mbo-root", type=Path, required=True)
    idrive.add_argument("--output", type=Path, required=True)
    idrive.add_argument("--point-value", type=float, default=2.0)
    args = parser.parse_args()
    _refuse_live(args.mbo_root, args.output)
    if not args.mbo_root.is_dir():
        raise SystemExit(f"no IDrive root {args.mbo_root}")
    print("scan IDrive Fill_Ratio + stop geometry (holdout skipped)", flush=True)
    breaks, stops, diag = scan_year_idrive_diag(
        args.mbo_root,
        point_value=float(args.point_value),
        log=lambda msg: print(msg, flush=True),
    )
    written = write_failed_break_diag_report(breaks, stops, diag, args.output)
    print(
        "days",
        diag.get("n_days"),
        "holdout_skipped",
        diag.get("n_skipped_holdout"),
        "breaks",
        diag.get("n_breaks"),
        "failed",
        diag.get("n_failed"),
        "held",
        diag.get("n_held"),
        "fill_failed",
        diag.get("median_fill_5_failed"),
        "fill_held",
        diag.get("median_fill_5_held"),
        "share_lt020_failed",
        diag.get("share_lt_020_failed"),
        "share_lt020_held",
        diag.get("share_lt_020_held"),
        "base_trades",
        diag.get("n_base_trades"),
        "stops",
        diag.get("stops"),
        flush=True,
    )
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
