#!/usr/bin/env python3
"""موقع أول إشارة على الموجة المكتملة — بلا MBO وبلا holdout.

  .venv/bin/python scripts/run_wave_position.py \\
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

from nq.research.progress import PipelineProgress  # noqa: E402
from nq.research.wave_position import (  # noqa: E402
    WavePositionConfig,
    run_wave_position_from_period_dir,
    write_wave_position_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "wave position: first signal vs completed wave "
            "(0-20/20-40/40-60/60+) — never reconstruct, never score holdout"
        )
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-peak-ticks", type=float, default=8.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = WavePositionConfig(min_peak_ticks=float(args.min_peak_ticks))
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("wave_position", total_steps=2)
    report = run_wave_position_from_period_dir(args.period_dir, config=cfg, progress=log)
    written = write_wave_position_report(report, args.output)
    log.done(f"scope={report.diagnostics.get('primary_scope')} holdout_scored=false")
    print(f"outputs: {written.resolve()}/", flush=True)
    print((written / "WAVE.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
