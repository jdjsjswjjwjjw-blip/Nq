#!/usr/bin/env python3
"""دورة حياة MBO قبل امتداد تحدّده العين. يوم واحد، بلا ملف Trades منفصل.

    .venv/bin/python scripts/run_order_lifecycle.py \\
      --day 2025-05-01 --clock 10:35:00 \\
      --mbo-root /opt/IDriveForLinux/.../Restore_Data/2025 \\
      --output data/runs/auction_behavior_year/order_lifecycle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.ingestion.reader import load_mbo_frame  # noqa: E402
from nq.research.manual_zone_depth import et_clock_to_ns, points_to_fixed  # noqa: E402
from nq.research.mbo_sequence_mlp import resolve_idrive_mbo  # noqa: E402
from nq.research.order_lifecycle import (  # noqa: E402
    DEFAULT_WINDOWS_S,
    metrics_frame,
    window_metrics,
    write_lifecycle_report,
)


def _parse_windows(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(int(p) for p in parts) if parts else DEFAULT_WINDOWS_S


def _parse_levels(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    return tuple(points_to_fixed(float(p.strip())) for p in raw.split(",") if p.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MBO order lifecycle in pre-extension windows (eye clock)"
    )
    parser.add_argument("--day", type=str, required=True)
    parser.add_argument("--clock", type=str, required=True, help="America/New_York HH:MM:SS")
    parser.add_argument("--windows", type=str, default="10,20,30")
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="optional NQ-point filter for closed-order stats",
    )
    parser.add_argument("--mbo-root", type=Path, default=None)
    parser.add_argument("--mbo", type=Path, default=None)
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
    ts = et_clock_to_ns(args.day, args.clock)
    print(f"load {args.day} t={ts} from {mbo_path}", flush=True)
    mbo = load_mbo_frame(mbo_path)
    rows = window_metrics(
        mbo,
        ts,
        windows_s=_parse_windows(args.windows),
        prices=_parse_levels(args.levels),
    )
    frame = metrics_frame(rows)
    written = write_lifecycle_report(
        frame,
        {
            "layer_id": "order_lifecycle",
            "day": args.day,
            "clock_et": args.clock,
            "availability_ts": ts,
            "windows_s": list(_parse_windows(args.windows)),
            "concatenated_raw_mbo": False,
            "live_overlay": False,
            "separate_trades_file": False,
            "fleeting_is_legal_spoofing": False,
            "mbo_path": str(mbo_path),
        },
        args.output,
    )
    print((written / "ORDER_LIFECYCLE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
