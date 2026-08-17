#!/usr/bin/env python3
"""تسلسل MBO داخل البرميل → MLP. يوم بيوم، بلا لصق تدفق خام وبلا دفتر.

يقرأ ``blended.parquet`` من تشغيل السنة، وMBO ذلك اليوم من IDrive
(``MES_MBO_YYYY_MM/glbx-mdp3-YYYYMMDD.continuous.clean.parquet``) أو من
``YYYY-MM-DD/mbo.parquet``. يحمّل يوماً واحداً في الذاكرة ثم يسقطه.

    .venv/bin/python scripts/run_mbo_sequence_mlp.py \\
      --days-root data/runs/auction_behavior_year \\
      --mbo-root /opt/IDriveForLinux/.../Restore_Data/2025 \\
      --output data/runs/auction_behavior_year/mbo_sequence_mlp
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.ingestion.reader import load_mbo_frame  # noqa: E402
from nq.research.mbo_sequence_mlp import (  # noqa: E402
    HOLDOUT_START_DATE,
    labels_from_blended,
    resolve_idrive_mbo,
    run_mbo_sequence_mlp,
    write_mbo_sequence_report,
)

_DAY_DIR_NAME_LEN = 10


def iter_day_pairs(
    days_root: Path,
    *,
    mbo_root: Path | None,
    holdout_start: str,
    start_date: str,
    max_days: int | None,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    n = 0
    missing_mbo = 0
    skipped_empty = 0
    for path in sorted(days_root.iterdir()):
        if not path.is_dir() or len(path.name) != _DAY_DIR_NAME_LEN:
            continue
        if path.name < start_date or path.name >= holdout_start:
            continue
        blended_path = path / "blended.parquet"
        if not blended_path.is_file():
            continue
        blended = pl.read_parquet(blended_path)
        if labels_from_blended(blended).height == 0:
            skipped_empty += 1
            continue
        mbo_path = path / "mbo.parquet"
        if mbo_root is not None:
            resolved = resolve_idrive_mbo(mbo_root, path.name)
            if resolved is None:
                missing_mbo += 1
                continue
            mbo_path = resolved
        elif not mbo_path.is_file():
            missing_mbo += 1
            continue
        print(f"load {path.name} from {mbo_path}", flush=True)
        mbo = load_mbo_frame(mbo_path)
        yield mbo, blended
        n += 1
        if max_days is not None and n >= max_days:
            return
    if n == 0:
        raise SystemExit(
            f"no per-day MBO under days-root={days_root} mbo-root={mbo_root} "
            f"(missing_mbo={missing_mbo} empty_labels={skipped_empty}). "
            "This layer cannot run on period_blended.parquet."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Intra-bar MBO sequence MLP vs 30s aggregates")
    parser.add_argument("--days-root", type=Path, required=True)
    parser.add_argument(
        "--mbo-root",
        type=Path,
        default=None,
        help="IDrive Restore_Data/2025 root with MES_MBO_YYYY_MM day files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--holdout-start", type=str, default=HOLDOUT_START_DATE)
    args = parser.parse_args()
    joined = f"{args.days_root} {args.mbo_root} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    days = iter_day_pairs(
        args.days_root,
        mbo_root=args.mbo_root,
        holdout_start=args.holdout_start,
        start_date=args.start_date,
        max_days=args.max_days,
    )
    scored, diagnostics = run_mbo_sequence_mlp(
        days, seed=args.seed, holdout_start=args.holdout_start
    )
    diagnostics["mbo_root"] = None if args.mbo_root is None else str(args.mbo_root)
    diagnostics["max_days"] = args.max_days
    written = write_mbo_sequence_report(scored, diagnostics, args.output)
    print((written / "MBO_SEQUENCE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
