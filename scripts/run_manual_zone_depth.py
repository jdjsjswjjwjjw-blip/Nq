#!/usr/bin/env python3
"""عمق دفتر عند مستويات يحدّدها العين. يوم جلسة واحد، بلا overlay.

    .venv/bin/python scripts/run_manual_zone_depth.py \\
      --day 2025-05-01 --clock 10:35:00 --levels 21450.25,21448.00 \\
      --mbo-root /opt/IDriveForLinux/.../Restore_Data/2025 \\
      --output data/runs/auction_behavior_year/manual_zone_depth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.ingestion.reader import load_mbo_frame  # noqa: E402
from nq.research.manual_zone_depth import (  # noqa: E402
    parse_zone,
    write_zone_depth_report,
    zone_depth,
)
from nq.research.mbo_sequence_mlp import resolve_idrive_mbo  # noqa: E402


def _parse_levels(raw: str) -> tuple[float, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("need at least one level in --levels")
    return tuple(float(p) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eye-specified zone → causal book depth")
    parser.add_argument("--day", type=str, required=True, help="session day YYYY-MM-DD")
    parser.add_argument("--clock", type=str, required=True, help="America/New_York HH:MM:SS")
    parser.add_argument(
        "--levels",
        type=str,
        required=True,
        help="NQ points, comma-separated (tick 0.25)",
    )
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--band-ticks", type=int, default=4)
    parser.add_argument("--mbo-root", type=Path, default=None)
    parser.add_argument("--mbo", type=Path, default=None, help="explicit day parquet")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = f"{args.mbo_root} {args.mbo} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    mbo_path = args.mbo
    if mbo_path is None:
        if args.mbo_root is None:
            raise SystemExit("need --mbo or --mbo-root")
        mbo_path = resolve_idrive_mbo(args.mbo_root, args.day)
    if mbo_path is None or not mbo_path.is_file():
        raise SystemExit(f"no MBO for {args.day}")
    zone = parse_zone(
        day=args.day,
        clock=args.clock,
        levels_points=_parse_levels(args.levels),
        label=args.label,
        band_ticks=args.band_ticks,
    )
    print(f"load {args.day} t={zone.availability_ts} from {mbo_path}", flush=True)
    depth = zone_depth(load_mbo_frame(mbo_path), zone)
    written = write_zone_depth_report(
        depth,
        {
            "layer_id": "manual_zone_depth",
            "day": args.day,
            "clock_et": args.clock,
            "levels_points": list(zone.levels_points),
            "reconstructed_order_book": True,
            "concatenated_raw_mbo": False,
            "live_overlay": False,
            "holdout_touched": False,
            "mbo_path": str(mbo_path),
        },
        args.output,
    )
    print((written / "MANUAL_ZONE_DEPTH.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
