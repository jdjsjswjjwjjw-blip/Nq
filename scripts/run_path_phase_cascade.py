#!/usr/bin/env python3
"""فلتر التيك ثم الطور على fold_scores OOF — بلا إعادة تدريب وبلا live.

يجب أن يحتوي fold_scores على ``y_phase_extend`` (موجود في period_clean_trade؛
غير موجود في period_realized_path).

    .venv/bin/python scripts/run_path_phase_cascade.py \\
      --blended data/runs/auction_behavior_year/period_clean_trade/period_blended.parquet \\
      --fold-scores data/runs/auction_behavior_year/period_clean_trade/fold_scores.parquet \\
      --output data/runs/auction_behavior_year/path_phase_cascade
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS  # noqa: E402
from nq.contracts.temporal import AVAILABILITY_TS  # noqa: E402
from nq.research.path_phase_cascade import (  # noqa: E402
    run_path_phase_cascade,
    write_cascade_report,
)

_SCORE_COLS = (
    SETUP_AVAILABILITY_TS,
    "outcome_name",
    "p_cal",
    "p_hat",
    "prediction_is_oof",
    "eligible_for_backtest",
)
_BLENDED_COLS = (
    AVAILABILITY_TS,
    "close",
    "high",
    "low",
    "proj_break_direction",
    "asia_vah",
    "asia_val",
    "_behavior_story_run",
    "_period_day_id",
)


def _read_columns(path: Path, wanted: tuple[str, ...]) -> pl.DataFrame:
    names = set(pl.read_parquet_schema(path))
    cols = [c for c in wanted if c in names]
    return pl.read_parquet(path, columns=cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Path then phase cascade on OOF fold_scores")
    parser.add_argument("--blended", type=Path, required=True)
    parser.add_argument("--fold-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name = args.fold_scores.name.lower()
    if "live" in name:
        raise SystemExit("refuse live_predictions; pass fold_scores.parquet")
    blended = _read_columns(args.blended, _BLENDED_COLS)
    scores = _read_columns(args.fold_scores, _SCORE_COLS)
    quality, diagnostics = run_path_phase_cascade(blended=blended, fold_scores=scores)
    written = write_cascade_report(quality, diagnostics, args.output)
    print((written / "CASCADE.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
