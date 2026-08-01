#!/usr/bin/env python3
"""بحث رمزي بلا if — DEAP (GP) + gplearn فوق ميزات الخط الموحّد.

    pip install 'nq[gp]'
    python scripts/run_symbolic_search.py --nq data/raw/nq.parquet --max-rows 200000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

_MIN_PYTHON = (3, 11)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

if sys.version_info < _MIN_PYTHON:
    sys.exit(
        f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ مطلوب؛ "
        f"الحالي {sys.version_info.major}.{sys.version_info.minor}"
    )

from nq.alpha.symbolic_gp import (  # noqa: E402
    default_symbolic_feature_columns,
    require_gp_deps,
    search_symbolic_hypotheses,
)
from nq.contracts.temporal import AVAILABILITY_TS  # noqa: E402
from nq.core.temporal_policy import TemporalPolicy  # noqa: E402
from nq.research.orchestrator import PipelineConfig, run_research_pipeline  # noqa: E402
from nq.research.progress import PipelineProgress  # noqa: E402

_MIN_SYMBOLIC_FEATURES = 2


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Symbolic GP search (DEAP + gplearn) with purged walk-forward"
    )
    parser.add_argument("--nq", type=Path, required=True, help="مسار NQ MBO")
    parser.add_argument("--mnq", type=Path, default=None, help="مسار MNQ اختياري")
    parser.add_argument("--config", type=Path, default=Path("configs/research.toml"))
    parser.add_argument("--output", type=Path, default=Path("data/runs/symbolic_gp"))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=("deap", "gplearn", "both"),
        default="both",
        help="محرّك الاكتشاف (افتراضي: الاثنان)",
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--n-programs", type=int, default=2)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.nq.is_file():
        raise FileNotFoundError(f"NQ MBO not found: {args.nq.resolve()}")
    if args.mnq is not None and not args.mnq.is_file():
        raise FileNotFoundError(f"MNQ MBO not found: {args.mnq.resolve()}")

    require_gp_deps()
    args.output.mkdir(parents=True, exist_ok=True)
    progress = None if args.quiet else PipelineProgress(enabled=True)
    if progress is not None:
        progress.op("بناء إطار الميزات عبر الخط الموحّد…")

    cfg = (
        PipelineConfig.from_toml(args.config) if args.config.is_file() else PipelineConfig()
    )
    cfg = replace(
        cfg,
        horizon=args.horizon,
        max_rows=args.max_rows if args.max_rows is not None else cfg.max_rows,
        quiet=args.quiet,
        cross_market_mode="nq_only" if args.mnq is None else cfg.cross_market_mode,
        n_permutations=min(cfg.n_permutations, 200),
        parallel_coverage=False,
    )

    mnq_path = args.mnq if args.mnq is not None else args.nq
    pipe = run_research_pipeline(
        args.nq,
        mnq_path,
        config=cfg,
        output_dir=args.output / "pipeline",
    )
    frame = pipe.features
    if frame.height == 0:
        raise RuntimeError("pipeline produced empty feature frame")

    feats = [c for c in default_symbolic_feature_columns() if c in frame.columns]
    if len(feats) < _MIN_SYMBOLIC_FEATURES:
        skip = {"availability_ts", "nq_close", "mnq_close"}
        feats = [
            c
            for c, dtype in zip(frame.columns, frame.dtypes, strict=True)
            if c not in skip and getattr(dtype, "is_numeric", lambda: False)()
        ][:12]
    if progress is not None:
        progress.op(f"symbolic features ({len(feats)}): {feats}")

    price_col = "nq_close" if "nq_close" in frame.columns else str(frame.columns[-1])
    times = frame[AVAILABILITY_TS].to_numpy()
    policy = TemporalPolicy.for_run(
        interval_ns=cfg.interval_ns,
        horizon=args.horizon,
        config_path=args.config if args.config.is_file() else None,
    )
    embargo = policy.embargo_time_units(interval_ns=cfg.interval_ns, times=times)
    purge_samples = policy.purge_samples()
    if progress is not None:
        progress.op(
            f"temporal: embargo={embargo} · purge={purge_samples} · "
            f"interval_ns={cfg.interval_ns} · horizon={args.horizon}"
        )
    result = search_symbolic_hypotheses(
        frame,
        feats,
        price_col=price_col,
        horizon=args.horizon,
        backend=args.backend,
        n_splits=args.n_splits,
        embargo=embargo,
        purge_samples=purge_samples,
        population_size=args.population,
        generations=args.generations,
        max_depth=args.max_depth,
        n_programs=args.n_programs,
        n_permutations=args.n_permutations,
        selection_aware_null=True,
        seed=args.seed,
        progress=progress,
    )

    result.fold_selections.write_parquet(args.output / "fold_selections.parquet")
    programs_meta = [
        {
            "name": p.name,
            "backend": p.backend,
            "expression": p.expression,
            "train_ic": p.train_ic,
        }
        for p in result.programs
    ]
    (args.output / "programs.json").write_text(
        json.dumps(
            {
                "backend": args.backend,
                "best_name": result.best_name,
                "oos_ic": result.oos_ic,
                "oos_pvalue": result.oos_pvalue,
                "oos_n": result.oos_n,
                "programs": programs_meta,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prog_cols = [p.name for p in result.programs]
    keep = [c for c in ("availability_ts", price_col, *prog_cols) if c in result.frame.columns]
    if keep:
        result.frame.select(keep).write_parquet(args.output / "symbolic_signals.parquet")

    if not args.quiet:
        print(
            f"[nq] symbolic done · best={result.best_name!r} · "
            f"oos_ic={result.oos_ic:.4g} · p={result.oos_pvalue:.4g} · "
            f"n={result.oos_n} · programs={len(result.programs)}",
            file=sys.stderr,
            flush=True,
        )
        print(f"[nq] outputs → {args.output.resolve()}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
