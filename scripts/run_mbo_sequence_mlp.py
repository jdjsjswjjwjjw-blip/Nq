#!/usr/bin/env python3
"""تسلسل MBO داخل البرميل → MLP. يوم بيوم، بلا لصق تدفق خام وبلا دفتر.

التسميات من ``period_*/fold_scores.parquet`` (أو period_blended) حتى ATR لندن
يرى الأيام السابقة. MBO يُحمَّل من IDrive يوماً واحداً ثم يُسقط.

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

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS  # noqa: E402
from nq.core.session import session_date_from_ns  # noqa: E402
from nq.ingestion.reader import load_mbo_frame  # noqa: E402
from nq.research.mbo_sequence_mlp import (  # noqa: E402
    HOLDOUT_START_DATE,
    prepare_labels,
    resolve_idrive_mbo,
    run_mbo_sequence_mlp,
    write_mbo_sequence_report,
)

_DAY_DIR_NAME_LEN = 10


def _default_labels_path(days_root: Path) -> Path | None:
    for name in ("period_clean_trade", "period_realized_path"):
        for fname in ("fold_scores.parquet", "period_blended.parquet"):
            path = days_root / name / fname
            if path.is_file():
                return path
    return None


def _day_mbo_path(days_root: Path, day: str, mbo_root: Path | None) -> Path | None:
    if mbo_root is not None:
        return resolve_idrive_mbo(mbo_root, day)
    path = days_root / day / "mbo.parquet"
    return path if path.is_file() else None


def _iter_period_days(
    days_root: Path,
    *,
    mbo_root: Path | None,
    label_source: Path,
    holdout_start: str,
    start_date: str,
    max_days: int | None,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    print(f"labels from {label_source}", flush=True)
    labels = prepare_labels(pl.read_parquet(label_source))
    if labels.height == 0:
        raise SystemExit(f"no resolved {label_source} labels")
    dates = [session_date_from_ns(int(t)) for t in labels[SETUP_AVAILABILITY_TS].to_list()]
    labeled = labels.with_columns(pl.Series("_day", dates))
    n = 0
    missing_mbo = 0
    for day in sorted(set(dates)):
        if day < start_date or day >= holdout_start:
            continue
        part = labeled.filter(pl.col("_day") == day).drop("_day")
        if part.height == 0:
            continue
        mbo_path = _day_mbo_path(days_root, day, mbo_root)
        if mbo_path is None:
            missing_mbo += 1
            continue
        print(f"load {day} n={part.height} from {mbo_path}", flush=True)
        yield load_mbo_frame(mbo_path), part
        n += 1
        if max_days is not None and n >= max_days:
            return
    if n == 0:
        raise SystemExit(f"no per-day MBO matched to period labels (missing_mbo={missing_mbo})")


def iter_day_pairs(
    days_root: Path,
    *,
    mbo_root: Path | None,
    labels_path: Path | None,
    holdout_start: str,
    start_date: str,
    max_days: int | None,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    label_source = labels_path if labels_path is not None else _default_labels_path(days_root)
    if label_source is not None:
        yield from _iter_period_days(
            days_root,
            mbo_root=mbo_root,
            label_source=label_source,
            holdout_start=holdout_start,
            start_date=start_date,
            max_days=max_days,
        )
        return
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
        ready = prepare_labels(pl.read_parquet(blended_path))
        if ready.height == 0:
            skipped_empty += 1
            continue
        mbo_path = _day_mbo_path(days_root, path.name, mbo_root)
        if mbo_path is None:
            missing_mbo += 1
            continue
        print(f"load {path.name} from {mbo_path}", flush=True)
        yield load_mbo_frame(mbo_path), ready
        n += 1
        if max_days is not None and n >= max_days:
            return
    if n == 0:
        raise SystemExit(
            f"no per-day MBO under days-root={days_root} mbo-root={mbo_root} "
            f"(missing_mbo={missing_mbo} empty_labels={skipped_empty}). "
            "This layer cannot run on concatenated raw MBO."
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
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="period fold_scores or period_blended (London ATR needs prior days)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--holdout-start", type=str, default=HOLDOUT_START_DATE)
    args = parser.parse_args()
    joined = f"{args.days_root} {args.mbo_root} {args.output} {args.labels}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    days = iter_day_pairs(
        args.days_root,
        mbo_root=args.mbo_root,
        labels_path=args.labels,
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
