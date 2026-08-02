#!/usr/bin/env python3
"""غلاف توافق → نفس ``run_vp_auction`` المتصل (ليست استراتيجية منفصلة).

    python scripts/run_liquidity_edge.py --nq ...   # يفوّض لـ vp_auction
    # المفضّل:
    python scripts/run_vp_auction.py --nq ...
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

from nq.strategies.vp_auction import run_vp_auction_research  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper → run_vp_auction (connected VP strategy). "
            "Prefer: python scripts/run_vp_auction.py"
        )
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runs/vp_auction"),
        help="نفس مخرجات vp_auction (مسار واحد)",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--min-oos-trades", type=int, default=3)
    parser.add_argument("--min-oos-rr", type=float, default=2.0)
    parser.add_argument("--keep-deceptive", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")

    if not args.quiet:
        print(
            "[nq] run_liquidity_edge → تفويض لـ run_vp_auction (مسار متصل)",
            file=sys.stderr,
            flush=True,
        )

    result = run_vp_auction_research(
        args.nq,
        max_rows=args.max_rows,
        output_dir=args.output,
        quiet=args.quiet,
        with_execution=True,
        drop_deceptive=not args.keep_deceptive,
        edge_train_frac=args.train_frac,
        min_oos_trades=args.min_oos_trades,
        min_oos_rr=args.min_oos_rr,
    )
    print(result.report.to_markdown())
    edge = result.best_edge_spec.name if result.best_edge_spec else None
    print(
        f"\nVP best={result.best_signal!r} · EDGE best={edge!r} · "
        f"oos_ic={result.oos_ic:.4g} · "
        f"oos_exp={result.best_edge_row.get('oos_expectancy', 0):.4g}"
    )
    print(f"outputs: {args.output.resolve()}/")


if __name__ == "__main__":
    main()
