#!/usr/bin/env python3
"""تسلسل MBO داخل البرميل → MLP. يوم بيوم، بلا لصق تدفق خام وبلا دفتر.

يتطلب ``YYYY-MM-DD/mbo.parquet`` إلى جانب ``blended.parquet``.
تشغيل السنة الحالي على Vast فيه blended فقط — بدون MBO لا يُقاس OOF.

    .venv/bin/python scripts/run_mbo_sequence_mlp.py \\
      --days-root data/runs/auction_behavior_year \\
      --output data/runs/auction_behavior_year/mbo_sequence_mlp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.mbo_sequence_mlp import (  # noqa: E402
    run_mbo_sequence_mlp,
    write_mbo_sequence_report,
)

_DAY_DIR_NAME_LEN = 10


def _day_pairs(root: Path) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    days: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    missing_mbo = 0
    for path in sorted(root.iterdir()):
        if not path.is_dir() or len(path.name) != _DAY_DIR_NAME_LEN:
            continue
        blended_path = path / "blended.parquet"
        mbo_path = path / "mbo.parquet"
        if not blended_path.is_file():
            continue
        if not mbo_path.is_file():
            missing_mbo += 1
            continue
        days.append((pl.read_parquet(mbo_path), pl.read_parquet(blended_path)))
    if not days:
        raise SystemExit(
            f"no per-day mbo.parquet under {root} "
            f"(blended-only days={missing_mbo}). "
            "This layer cannot run on period_blended.parquet; it needs daily MBO."
        )
    return days


def main() -> None:
    parser = argparse.ArgumentParser(description="Intra-bar MBO sequence MLP vs 30s aggregates")
    parser.add_argument("--days-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if "live" in str(args.days_root).lower():
        raise SystemExit("refuse live paths")
    days = _day_pairs(args.days_root)
    scored, diagnostics = run_mbo_sequence_mlp(days, seed=args.seed)
    written = write_mbo_sequence_report(scored, diagnostics, args.output)
    print((written / "MBO_SEQUENCE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
