#!/usr/bin/env python3
"""التقاط سببي بعد إطلاق النموذج — بلا MBO وبلا holdout وبلا ذروة موجة.

  .venv/bin/python scripts/run_causal_entry.py \\
      --period-dir data/runs/auction_behavior_year/period_realized_path \\
      --output data/runs/auction_behavior_year/period_realized_path
"""

from __future__ import annotations

import argparse
import sys
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

from nq.research.causal_entry import (  # noqa: E402
    CausalEntryConfig,
    run_causal_entry_from_period_dir,
    write_causal_entry_report,
)
from nq.research.progress import PipelineProgress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "causal entry: MFE/MAE after a live model fire inside the labeled "
            "window — never reconstruct, never score holdout, never use peak"
        )
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-p", type=float, default=0.5)
    parser.add_argument("--expansion-start-ticks", type=float, default=16.0)
    parser.add_argument("--min-printed-late-ticks", type=float, default=80.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = CausalEntryConfig(
        min_p=float(args.min_p),
        expansion_start_ticks=float(args.expansion_start_ticks),
        min_printed_late_ticks=float(args.min_printed_late_ticks),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("causal_entry", total_steps=2)
    report = run_causal_entry_from_period_dir(args.period_dir, config=cfg, progress=log)
    written = write_causal_entry_report(report, args.output)
    log.done(
        f"scope={report.diagnostics.get('primary_scope')} "
        f"late={report.diagnostics.get('n_late_confirmed')} "
        f"holdout_scored=false"
    )
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "CAUSAL.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
