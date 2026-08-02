"""تشغيل Volume Profile يومي متوازٍ — شهر بمقياس عملي (كل يوم كون سببي مغلق).

يستعيد ``discover_day_inputs`` من ``day_parallel``. داخل كل يوم:
``run_vp_auction_research`` (batch افتراضيًا؛ stream snapshots إن ``--streaming``).
"""

from __future__ import annotations

import json
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nq.research.day_parallel import (
    DayInput,
    DayJobResult,
    DayParallelManifest,
    _configure_worker_threads,
    stable_day_seed,
)


@dataclass(frozen=True, slots=True)
class VpDayParallelManifest(DayParallelManifest):
    """نفس بيان العزل — محرّك VP."""

    principles: tuple[str, ...] = (
        "zero_temporal_leakage: each day is an isolated causal universe",
        "no_cross_day_selection: summary is descriptive only",
        "no_cross_day_cache: never share MBO/state across day processes",
        "same_engine: run_vp_auction_research",
        "mbo_only: daily parquet/arrow MBO shards",
        "performance: ProcessPool over days; tick_stream emits interval snapshots",
    )
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "Do not treat mode(best_signal) across days as a single OOS selector.",
            "Prefer batch VP path; use streaming only when tick SSL is required.",
            "FINAL_RESULT.md pools per-day fills for a month-level descriptive verdict.",
        )
    )

    def to_markdown(self) -> str:
        text = super().to_markdown()
        return text.replace(
            "# Failed Breakout — Day-Parallel Manifest",
            "# Volume Profile — Day-Parallel Manifest",
            1,
        )


def _run_one_vp_day(payload: dict[str, Any]) -> DayJobResult:
    _configure_worker_threads(int(payload.get("threads_per_worker", 4)))
    day_id = str(payload["day_id"])
    nq_path = Path(payload["nq_path"])
    mnq_raw = payload.get("mnq_path")
    mnq_path = Path(mnq_raw) if mnq_raw else None
    out = Path(payload["output_dir"])
    seed = int(payload["seed"])
    # Always emit live progress into per-day progress.log (parent console stays clean).
    quiet = False

    try:
        import sys  # noqa: PLC0415

        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "progress.log"
        log_f = log_path.open("w", encoding="utf-8", buffering=1)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = log_f
        sys.stderr = log_f
        try:
            from nq.core.determinism import make_generator  # noqa: PLC0415
            from nq.strategies.vp_auction import run_vp_auction_research  # noqa: PLC0415

            print(f"[nq] day={day_id} START nq={nq_path}", flush=True)
            result = run_vp_auction_research(
                nq_path,
                mnq_path,
                horizon=int(payload.get("horizon", 1)),
                n_permutations=int(payload.get("n_permutations", 200)),
                n_splits=int(payload.get("n_splits", 3)),
                max_rows=payload.get("max_rows"),
                output_dir=out,
                quiet=quiet,
                rng=make_generator(seed),
                with_execution=bool(payload.get("with_execution", True)),
                drop_deceptive=bool(payload.get("drop_deceptive", True)),
                streaming_features=bool(payload.get("streaming_features", False)),
                min_oos_rr=float(payload.get("min_oos_rr", 2.0)),
                min_oos_trades=int(payload.get("min_oos_trades", 3)),
            )
            print(
                f"[nq] day={day_id} DONE features={result.features.height} "
                f"best={result.best_signal!r} oos_ic={result.oos_ic:.4g}",
                flush=True,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            log_f.close()

        return DayJobResult(
            day_id=day_id,
            ok=True,
            nq_path=str(nq_path),
            mnq_path=str(mnq_path) if mnq_path else None,
            output_dir=str(out.resolve()),
            mode="search",
            best_oos_spec=result.best_signal,
            oos_selected_ic=float(result.oos_ic),
            n_candidates=len(result.signal_columns),
            n_features=int(result.features.height),
            seed=seed,
        )
    except Exception as exc:
        try:
            err_path = out / "progress.log"
            with err_path.open("a", encoding="utf-8") as ef:
                ef.write(f"\n[nq] day={day_id} FAILED: {type(exc).__name__}: {exc}\n")
                ef.write(traceback.format_exc(limit=12))
        except OSError:
            pass
        return DayJobResult(
            day_id=day_id,
            ok=False,
            nq_path=str(nq_path),
            mnq_path=str(mnq_path) if mnq_path else None,
            output_dir=str(out.resolve()),
            mode="search",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            seed=seed,
        )


def run_vp_auction_day_parallel(
    days: Sequence[DayInput],
    *,
    output_root: Path | str,
    jobs: int = 1,
    threads_per_worker: int = 4,
    global_seed: int = 0,
    horizon: int = 1,
    max_rows: int | None = None,
    n_splits: int = 3,
    n_permutations: int = 200,
    with_execution: bool = True,
    drop_deceptive: bool = True,
    streaming_features: bool = False,
    min_oos_rr: float = 2.0,
    min_oos_trades: int = 3,
    quiet_workers: bool = True,
    fail_fast: bool = False,
) -> VpDayParallelManifest:
    """يشغّل VP على كل يوم في عملية منفصلة (مثالي لشهر على 20–30 كور)."""
    if not days:
        raise ValueError("days must be non-empty")
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if threads_per_worker < 1:
        raise ValueError(f"threads_per_worker must be >= 1, got {threads_per_worker}")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    for day in days:
        day_out = root / day.day_id
        payloads.append(
            {
                "day_id": day.day_id,
                "nq_path": str(day.nq_path),
                "mnq_path": str(day.mnq_path) if day.mnq_path else None,
                "output_dir": str(day_out),
                "seed": stable_day_seed(global_seed, day.day_id),
                "threads_per_worker": threads_per_worker,
                "horizon": horizon,
                "max_rows": max_rows,
                "n_splits": n_splits,
                "n_permutations": n_permutations,
                "with_execution": with_execution,
                "drop_deceptive": drop_deceptive,
                "streaming_features": streaming_features,
                "min_oos_rr": min_oos_rr,
                "min_oos_trades": min_oos_trades,
                "quiet": quiet_workers,
            }
        )

    workers = min(jobs, len(payloads))
    results: list[DayJobResult] = []
    print(
        f"[nq] VP day-parallel: writing live logs to {root}/<day_id>/progress.log",
        flush=True,
    )
    if workers == 1:
        for payload in payloads:
            results.append(_run_one_vp_day(payload))
            r = results[-1]
            print(
                f"[nq] day {r.day_id}: {'OK' if r.ok else 'FAIL'} "
                f"({len(results)}/{len(payloads)})",
                flush=True,
            )
            if fail_fast and not r.ok:
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one_vp_day, p): p["day_id"] for p in payloads}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                print(
                    f"[nq] day {res.day_id}: {'OK' if res.ok else 'FAIL'} "
                    f"({sum(1 for x in results if x.ok)} ok / "
                    f"{len(results)} done / {len(payloads)} total) "
                    f"log={root / res.day_id / 'progress.log'}",
                    flush=True,
                )
                status = root / "STATUS.txt"
                status.write_text(
                    "\n".join(
                        [
                            f"done={len(results)}/{len(payloads)}",
                            f"ok={sum(1 for x in results if x.ok)}",
                            f"failed={sum(1 for x in results if not x.ok)}",
                            "",
                            *[
                                f"{'OK' if r.ok else 'FAIL'} {r.day_id}"
                                for r in sorted(results, key=lambda x: x.day_id)
                            ],
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if fail_fast and not res.ok:
                    for other in futures:
                        other.cancel()
                    break

    results_sorted = tuple(sorted(results, key=lambda r: r.day_id))
    n_ok = sum(1 for r in results_sorted if r.ok)
    manifest = VpDayParallelManifest(
        jobs=workers,
        mode="search",
        output_root=str(root.resolve()),
        n_days=len(payloads),
        n_ok=n_ok,
        n_failed=len(results_sorted) - n_ok,
        results=results_sorted,
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "summary.md").write_text(manifest.to_markdown(), encoding="utf-8")
    if n_ok > 0:
        from nq.research.vp_month_aggregate import write_vp_month_aggregate  # noqa: PLC0415

        final_path = write_vp_month_aggregate(root)
        notes = list(manifest.notes) + (
            f"FINAL_RESULT pooled descriptive verdict: `{final_path.name}`",
            "Pooled expectancy uses concatenated per-day fills — not a cross-day re-fit.",
        )
        manifest = VpDayParallelManifest(
            jobs=manifest.jobs,
            mode=manifest.mode,
            output_root=manifest.output_root,
            n_days=manifest.n_days,
            n_ok=manifest.n_ok,
            n_failed=manifest.n_failed,
            results=manifest.results,
            principles=manifest.principles,
            notes=tuple(notes),
        )
        (root / "summary.md").write_text(manifest.to_markdown(), encoding="utf-8")
    return manifest


__all__ = [
    "VpDayParallelManifest",
    "run_vp_auction_day_parallel",
]
