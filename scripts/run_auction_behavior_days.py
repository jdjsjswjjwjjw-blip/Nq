#!/usr/bin/env python3
"""مرحلة 1: auction_behavior يوم-بيوم متوازٍ — كل يوم كون سببي مغلق.

لا يخلط MBO عبر الأيام. مخرج كل يوم: blended + labels + OOF داخل اليوم.
مرحلة 2 (علم الفترة) منفصلة: scripts/run_auction_behavior_period.py

    .venv/bin/python scripts/run_auction_behavior_days.py \\
      --nq-glob /data/glbx-mdp3-*.mbo.continuous.clean.parquet \\
      --jobs 20 --threads-per-worker 4 \\
      --output data/runs/auction_behavior_year
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
from glob import glob as glob_files
from pathlib import Path
from typing import Any

_MIN_PYTHON = (3, 11)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if sys.version_info < _MIN_PYTHON:
    sys.exit(
        f"Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ مطلوب؛ "
        f"الحالي {sys.version_info.major}.{sys.version_info.minor}"
    )

from nq.research.day_parallel import day_id_from_path  # noqa: E402


def _jsonable(obj: Any) -> Any:  # noqa: PLR0911
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _configure_worker_threads(n: int) -> None:
    n = max(1, int(n))
    for key in (
        "POLARS_MAX_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(n)


def _expand_nq_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_file():
            paths.append(p.resolve())
            continue
        matched = [Path(x) for x in glob_files(pattern, recursive=True)]
        if not matched:
            parent = p.parent if p.parent.parts else Path(".")
            matched = list(parent.glob(p.name))
        paths.extend(x.resolve() for x in matched if x.is_file())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _run_one_day(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    _configure_worker_threads(int(payload.get("threads_per_worker", 4)))
    day_id = str(payload["day_id"])
    nq_path = Path(payload["nq_path"])
    out = Path(payload["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "progress.log"
    t0 = time.perf_counter()

    def _fail(exc: BaseException) -> dict[str, Any]:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=20)}"
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n[FAIL] {err}\n")
            (out / "error.txt").write_text(err, encoding="utf-8")
        except OSError:
            pass
        return {
            "day_id": day_id,
            "ok": False,
            "nq_path": str(nq_path),
            "output_dir": str(out),
            "error": err,
            "elapsed_sec": time.perf_counter() - t0,
        }

    try:
        from nq.auction_behavior import (  # noqa: PLC0415
            BehaviorConfig,
            run_auction_behavior_analysis,
        )
        from nq.ingestion.reader import load_mbo_frame  # noqa: PLC0415

        with log_path.open("w", encoding="utf-8", buffering=1) as log_f:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = log_f
            sys.stderr = log_f
            try:
                print(f"[nq] day={day_id} START path={nq_path}", flush=True)
                max_rows = payload.get("max_rows")
                mbo = load_mbo_frame(nq_path, max_rows=max_rows)
                print(
                    f"[nq] day={day_id} loaded rows={mbo.height:,} cols={len(mbo.columns)}",
                    flush=True,
                )
                heavy = bool(payload.get("full_mbo_layers", True))
                cfg = BehaviorConfig(
                    quiet=False,
                    progress_log_path=str(out / "pipeline_progress.log"),
                    include_science=True,
                    include_deceptive_scores=heavy,
                    include_level_flow=heavy,
                    include_reliability_evidence=heavy,
                    include_asia_london_projection=True,
                    n_splits=int(payload.get("n_splits", 3)),
                    evaluate_holdout=False,
                )
                result = run_auction_behavior_analysis(mbo, config=cfg)
                print(
                    f"[nq] day={day_id} DONE ok={result.validation.ok} "
                    f"bars={result.blended.height:,} events={result.events.height:,}",
                    flush=True,
                )
                if result.blended.height:
                    result.blended.write_parquet(out / "blended.parquet")
                if result.events.height:
                    result.events.write_parquet(out / "events.parquet")
                if result.projection.height:
                    result.projection.write_parquet(out / "projection.parquet")
                if result.fold_metrics.height:
                    result.fold_metrics.write_parquet(out / "fold_metrics.parquet")
                if result.oof_predictions.height:
                    result.oof_predictions.write_parquet(out / "oof_predictions.parquet")
                science_diag: dict[str, Any] = {}
                if result.science is not None:
                    science_diag = dict(result.science.diagnostics)
                    if result.science.labeled.height:
                        result.science.labeled.write_parquet(out / "science_labeled.parquet")
                summary = {
                    "day_id": day_id,
                    "ok": bool(result.validation.ok),
                    "nq_path": str(nq_path),
                    "elapsed_sec": time.perf_counter() - t0,
                    "validation": {
                        "ok": bool(result.validation.ok),
                        "n_rows": int(result.validation.n_rows),
                        "n_folds": int(result.validation.n_folds),
                        "causal_ok": bool(result.validation.causal_ok),
                        "detail": str(result.validation.detail),
                    },
                    "diagnostics": _jsonable(result.diagnostics),
                    "science_diagnostics": _jsonable(science_diag),
                    "heights": {
                        "mbo": int(mbo.height),
                        "blended": int(result.blended.height),
                        "events": int(result.events.height),
                        "projection": int(result.projection.height),
                    },
                }
                (out / "summary.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                return {
                    "day_id": day_id,
                    "ok": bool(result.validation.ok),
                    "nq_path": str(nq_path),
                    "output_dir": str(out.resolve()),
                    "elapsed_sec": summary["elapsed_sec"],
                    "heights": summary["heights"],
                    "science_n_labeled": science_diag.get("n_labeled"),
                    "error": None,
                }
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
    except Exception as exc:
        return _fail(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="auction_behavior day-parallel (phase 1)")
    parser.add_argument("--nq-glob", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="disable deceptive + level_flow + reliability (ops smoke only)",
    )
    args = parser.parse_args()

    paths = _expand_nq_globs(args.nq_glob)
    paths = [p for p in paths if "Copy" not in p.name]
    if not paths:
        raise SystemExit("no input parquet files")

    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(args.jobs, len(paths)))
    heavy = not bool(args.fast)
    payloads = []
    seen_days: dict[str, str] = {}
    for p in sorted(paths, key=day_id_from_path):
        day_id = day_id_from_path(p)
        if day_id in seen_days:
            raise SystemExit(
                f"duplicate day_id {day_id}: {seen_days[day_id]} and {p} — refusing overwrite"
            )
        seen_days[day_id] = str(p)
        payloads.append(
            {
                "day_id": day_id,
                "nq_path": str(p),
                "output_dir": str((out_root / day_id).resolve()),
                "threads_per_worker": args.threads_per_worker,
                "max_rows": args.max_rows,
                "n_splits": args.n_splits,
                "full_mbo_layers": heavy,
            }
        )
    print(
        f"[nq] phase-1 day-parallel: {len(payloads)} days · jobs={jobs} · "
        f"threads/worker={args.threads_per_worker} · full_mbo_layers={heavy}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_run_one_day, p): p["day_id"] for p in payloads}
        for fut in as_completed(futs):
            day = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"day_id": day, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(r)
            print(
                f"[nq] DONE {r.get('day_id')} ok={r.get('ok')} "
                f"elapsed={r.get('elapsed_sec')} labeled={r.get('science_n_labeled')}",
                flush=True,
            )
    results_sorted = sorted(results, key=lambda r: str(r.get("day_id")))
    manifest = {
        "phase": 1,
        "jobs": jobs,
        "threads_per_worker": args.threads_per_worker,
        "n_days": len(payloads),
        "n_ok": sum(1 for r in results_sorted if r.get("ok")),
        "n_failed": sum(1 for r in results_sorted if not r.get("ok")),
        "elapsed_sec": time.perf_counter() - t0,
        "full_mbo_layers": heavy,
        "results": results_sorted,
        "next": (
            "phase 2: python scripts/run_auction_behavior_period.py "
            f"--days-root {out_root} --output {out_root}/period"
        ),
        "principles": [
            "zero_temporal_leakage: each day is an isolated causal universe",
            "do not average per-day probabilities; run phase-2 period science",
        ],
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"[nq] phase-1 manifest: {out_root.resolve()}/manifest.json", flush=True)
    print(f"[nq] next: {manifest['next']}", flush=True)


if __name__ == "__main__":
    main()
