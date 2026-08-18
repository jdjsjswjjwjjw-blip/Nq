#!/usr/bin/env python3
"""قارن ملخصَيْ يومين كُتبا بـ run_cvd_day.py. بلا قفل.

    .venv/bin/python scripts/run_cvd_day_compare.py \\
      --a-dir data/runs/cvd_0816 --b-dir data/runs/cvd_0817 \\
      --output data/runs/cvd_day_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nq.research.cvd_day_compare import (  # noqa: E402
    compare_day_metrics,
    write_cvd_day_compare_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two CVD day summaries")
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a-label", type=str, default="a")
    parser.add_argument("--b-label", type=str, default="b")
    args = parser.parse_args()
    joined = f"{args.a_dir} {args.b_dir} {args.output}".lower()
    if "live" in joined:
        raise SystemExit("refuse live paths")
    a_path = args.a_dir / "summary.json"
    b_path = args.b_dir / "summary.json"
    if not a_path.is_file():
        raise SystemExit(f"no {a_path}")
    if not b_path.is_file():
        raise SystemExit(f"no {b_path}")
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    a_label = str(a.get("label") or args.a_label)
    b_label = str(b.get("label") or args.b_label)
    table = compare_day_metrics(a, b, a_label=a_label, b_label=b_label)
    written = write_cvd_day_compare_report(
        table,
        a_summary=a,
        b_summary=b,
        output_dir=args.output,
        a_label=a_label,
        b_label=b_label,
    )
    print(table, flush=True)
    print(f"wrote {written}", flush=True)


if __name__ == "__main__":
    main()
