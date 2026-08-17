#!/usr/bin/env python3
"""أداة عين: كل إطلاقات النموذج، 10د قبل و15د بعد، فريم 30ث.

  .venv/bin/python scripts/run_entry_replay.py \\
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

from nq.research.entry_replay import (  # noqa: E402
    EntryReplayConfig,
    run_entry_replay_from_period_dir,
    write_entry_replay_report,
)
from nq.research.progress import PipelineProgress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eye tool: every OOF fire, 10m before to 15m after on 30s bars"
    )
    parser.add_argument("--period-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-p", type=float, default=0.5)
    parser.add_argument("--lookback-minutes", type=float, default=10.0)
    parser.add_argument("--lookahead-minutes", type=float, default=15.0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    ns_min = 60 * 1_000_000_000
    cfg = EntryReplayConfig(
        min_p=float(args.min_p),
        lookback_ns=int(float(args.lookback_minutes) * ns_min),
        lookahead_ns=int(float(args.lookahead_minutes) * ns_min),
    )
    log = PipelineProgress(enabled=not args.quiet)
    log.begin("entry_replay", total_steps=2)
    report = run_entry_replay_from_period_dir(args.period_dir, config=cfg, progress=log)
    written = write_entry_replay_report(report, args.output)
    log.done(f"layer=entry_replay trades={report.diagnostics.get('n_trades')}")
    html = written / "ENTRY_REPLAY.html"
    print(f"outputs: {written.resolve()}/", flush=True)
    print(f"open: {html.resolve()}", flush=True)
    print((written / "ENTRY_REPLAY.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
