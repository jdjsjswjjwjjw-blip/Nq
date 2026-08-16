#!/usr/bin/env python3
"""مرحلة 2: علم الفترة على حالات الأيام المجمّعة — ليس متوسط احتمالات يومية.

يقرأ ``<days-root>/YYYY-MM-DD/blended.parquet`` فقط.
لا تحميل MBO، لا إعادة بناء دفتر، لا إعادة حساب ميزات المستوى.

الافتراضي لسنة: 4 أشهر تدريب · 4 walk-forward · 4 holdout مجمّد.

    .venv/bin/python scripts/run_auction_behavior_period.py \\
      --days-root data/runs/auction_behavior_year \\
      --output data/runs/auction_behavior_year/period

لا تلمّس الـholdout في أول تشغيل لسنة (الافتراضي). بعد قفل التطوير:
      --evaluate-holdout
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_MIN_PYTHON = (3, 11)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.version_info < _MIN_PYTHON:
    sys.exit(
        f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ مطلوب؛ "
        f"الحالي {sys.version_info.major}.{sys.version_info.minor}"
    )

from nq.auction_behavior.science import ScienceConfig  # noqa: E402
from nq.research.behavior_period import (  # noqa: E402
    YEAR_HOLDOUT_MONTHS,
    YEAR_TRAIN_MONTHS,
    YEAR_WALK_FORWARD_MONTHS,
    default_period_science_config,
    run_behavior_period_science,
    write_behavior_period_report,
)
from nq.research.progress import PipelineProgress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "auction_behavior period science (phase 2): "
            "4/4/4 train/walk-forward/holdout on blended.parquet — never reconstruct"
        )
    )
    parser.add_argument("--days-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-months", type=int, default=YEAR_TRAIN_MONTHS)
    parser.add_argument("--walk-forward-months", type=int, default=YEAR_WALK_FORWARD_MONTHS)
    parser.add_argument("--holdout-months", type=int, default=YEAR_HOLDOUT_MONTHS)
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=None,
        help="override: fraction holdout instead of calendar-month blocks",
    )
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--min-train-size", type=int, default=16)
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-month-folds", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.holdout_frac is not None:
        cfg = ScienceConfig(
            n_splits=args.n_splits,
            holdout_frac=args.holdout_frac,
            holdout_months=None,
            walk_forward_months=None,
            min_train_months=1,
            min_train_size=args.min_train_size,
            evaluate_holdout=bool(args.evaluate_holdout),
            use_month_folds=not bool(args.no_month_folds),
        )
    else:
        cfg = replace(
            default_period_science_config(),
            min_train_months=int(args.train_months),
            walk_forward_months=int(args.walk_forward_months),
            holdout_months=int(args.holdout_months),
            n_splits=int(args.n_splits),
            min_train_size=int(args.min_train_size),
            evaluate_holdout=bool(args.evaluate_holdout),
            use_month_folds=not bool(args.no_month_folds),
        )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("behavior_period", total_steps=4)
    report = run_behavior_period_science(
        args.days_root,
        config=cfg,
        include_ablation=not args.no_ablation,
        progress=log,
    )
    written = write_behavior_period_report(report, args.output)
    log.done(f"days={len(report.day_ids)} oof={report.science.conditional_oof_predictions.height}")
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "PERIOD.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
