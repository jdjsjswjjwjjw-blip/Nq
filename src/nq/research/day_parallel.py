"""تشغيل يومي متوازٍ — يستغل الشرائح اليومية دون كسر المبادئ الأربعة.

المبادئ (Non‑Negotiable)
------------------------
1. **Zero Temporal Leakage:** كل يوم = كون سببي مغلق. لا دمج MBO عبر الأيام،
   لا fit عالمي عبر الأيام، لا اختيار فرضية يستخدم نتائج يوم لاحق على يوم سابق.
2. **صرامة كمية:** نفس محرّك البحث/الخط الموحّد؛ بذرة حتمية لكل يوم؛
   مخرجات كاملة + بيان Manifest يصف العزل صراحةً.
3. **أداء:** ``ProcessPool`` على مستوى الأيام (لا توازي داخل دفتر واحد)؛
   حد خيوط Polars/BLAS لكل عامل لتفادي التنازع.
4. **MBO فقط:** كل يوم يمر عبر ``load_mbo_frame`` / محرّك Failed Breakout القائم.

التجميع عبر الأيام = **وصف إحصائي فقط** (تكرار best_oos، متوسط IC يومي).
ليس اختيارًا موحّدًا خارج العيّنة عبر الشهر.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any, Literal, cast

FbDayMode = Literal["search", "unified"]

_YMD_DIGIT_LEN = 8
_DAY_STEM_RE = re.compile(r"(?P<date>\d{4}[-_]?\d{2}[-_]?\d{2})|(?P<ymd>\d{8})")


@dataclass(frozen=True, slots=True)
class DayInput:
    """زوج ملفات يوم واحد (NQ إلزامي، MNQ اختياري)."""

    day_id: str
    nq_path: Path
    mnq_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DayJobResult:
    """نتيجة عامل واحد — خفيفة وقابلة للـ pickle (الميزات على القرص فقط)."""

    day_id: str
    ok: bool
    nq_path: str
    mnq_path: str | None
    output_dir: str
    mode: FbDayMode
    best_oos_spec: str | None = None
    oos_selected_ic: float | None = None
    n_candidates: int | None = None
    n_features: int | None = None
    error: str | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class DayParallelManifest:
    """ملخص تشغيل متعدد الأيام مع إقرار صريح بعدم الاختيار عبر الأيام."""

    jobs: int
    mode: FbDayMode
    output_root: str
    n_days: int
    n_ok: int
    n_failed: int
    results: tuple[DayJobResult, ...]
    principles: tuple[str, ...] = (
        "zero_temporal_leakage: each day is an isolated causal universe",
        "no_cross_day_selection: summary is descriptive only",
        "same_engine: search_fail_breakout_hypotheses / run_fail_breakout_research",
        "mbo_only: daily parquet/arrow MBO shards",
    )
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "Do not treat mode(best_oos_spec) across days as a single OOS selector.",
            "Walk-forward / purged splits run inside each day only.",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "jobs": self.jobs,
            "mode": self.mode,
            "output_root": self.output_root,
            "n_days": self.n_days,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "principles": list(self.principles),
            "notes": list(self.notes),
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Failed Breakout — Day-Parallel Manifest",
            "",
            "## Isolation (four principles)",
            "",
        ]
        for p in self.principles:
            lines.append(f"- `{p}`")
        lines.extend(
            [
                "",
                f"- mode: `{self.mode}`",
                f"- jobs: `{self.jobs}`",
                f"- days: `{self.n_ok}/{self.n_days}` ok · failed `{self.n_failed}`",
                f"- output: `{self.output_root}`",
                "",
                "## Per-day results (isolated)",
                "",
                "| day_id | ok | best_oos_spec | oos_ic | candidates | features |",
                "|---|---|---|---|---|---|",
            ]
        )
        for r in self.results:
            ic = "" if r.oos_selected_ic is None else f"{r.oos_selected_ic:.4g}"
            lines.append(
                f"| `{r.day_id}` | {r.ok} | `{r.best_oos_spec}` | {ic} | "
                f"{r.n_candidates} | {r.n_features} |"
            )
        ok_specs = [r.best_oos_spec for r in self.results if r.ok and r.best_oos_spec]
        if ok_specs:
            counts = Counter(ok_specs)
            lines.extend(["", "## Descriptive frequency of daily winners (not a selector)", ""])
            for spec, count in counts.most_common():
                lines.append(f"- `{spec}`: {count}/{len(ok_specs)}")
        fails = [r for r in self.results if not r.ok]
        if fails:
            lines.extend(["", "## Failures", ""])
            for r in fails:
                lines.append(f"- `{r.day_id}`: {r.error}")
        lines.extend(["", "## Notes", ""])
        for note in self.notes:
            lines.append(f"- {note}")
        return "\n".join(lines) + "\n"


def day_id_from_path(path: Path) -> str:
    """يستخرج معرّف يوم مستقر من اسم الملف إن أمكن، وإلا من الـ stem كاملاً."""
    stem = path.stem
    match = _DAY_STEM_RE.search(stem)
    if match is None:
        return stem
    raw = match.group("date") or match.group("ymd") or stem
    digits = re.sub(r"\D", "", raw)
    if len(digits) == _YMD_DIGIT_LEN:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return raw.replace("_", "-")


def discover_day_inputs(
    *,
    nq_paths: Sequence[Path],
    mnq_dir: Path | None = None,
    mnq_paths: Sequence[Path] | None = None,
) -> tuple[DayInput, ...]:
    """يبني قائمة أيام مرتّبة؛ يطابق MNQ بالـ ``day_id`` أو نفس اسم الملف."""
    if not nq_paths:
        raise ValueError("nq_paths must be non-empty")

    mnq_by_day: dict[str, Path] = {}
    mnq_by_name: dict[str, Path] = {}
    if mnq_paths:
        for p in mnq_paths:
            mnq_by_day[day_id_from_path(p)] = p
            mnq_by_name[p.name] = p
    elif mnq_dir is not None:
        if not mnq_dir.is_dir():
            raise FileNotFoundError(f"mnq_dir not found: {mnq_dir.resolve()}")
        for p in sorted(mnq_dir.iterdir()):
            if p.suffix.lower() in {".parquet", ".arrow", ".feather", ".csv"} or p.suffix == ".zst":
                mnq_by_day[day_id_from_path(p)] = p
                mnq_by_name[p.name] = p

    days: list[DayInput] = []
    seen: set[str] = set()
    for nq in sorted(nq_paths, key=lambda p: (day_id_from_path(p), str(p))):
        if not nq.is_file():
            raise FileNotFoundError(f"NQ day file not found: {nq.resolve()}")
        day = day_id_from_path(nq)
        if day in seen:
            raise ValueError(f"duplicate day_id={day!r} from {nq}")
        seen.add(day)
        mnq = mnq_by_name.get(nq.name) or mnq_by_day.get(day)
        days.append(DayInput(day_id=day, nq_path=nq.resolve(), mnq_path=mnq))
    return tuple(days)


def stable_day_seed(global_seed: int, day_id: str) -> int:
    """بذرة حتمية لكل يوم (مستقلة، قابلة لإعادة الإنتاج)."""
    digest = blake2b(f"{global_seed}:{day_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _configure_worker_threads(threads_per_worker: int) -> None:
    n = max(1, int(threads_per_worker))
    for key in (
        "POLARS_MAX_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(n)


def _run_one_day(payload: dict[str, Any]) -> DayJobResult:
    """عامل عملية واحدة — يستورد المحرّك داخل العملية الفرعية."""
    _configure_worker_threads(int(payload.get("threads_per_worker", 2)))
    day_id = str(payload["day_id"])
    nq_path = Path(payload["nq_path"])
    mnq_raw = payload.get("mnq_path")
    mnq_path = Path(mnq_raw) if mnq_raw else None
    out = Path(payload["output_dir"])
    mode = cast(FbDayMode, payload["mode"])
    seed = int(payload["seed"])
    quiet = bool(payload.get("quiet", True))

    try:
        out.mkdir(parents=True, exist_ok=True)
        if mode == "search":
            from nq.strategies.breakout_hypothesis import (  # noqa: PLC0415
                search_fail_breakout_hypotheses,
            )

            search_result = search_fail_breakout_hypotheses(
                nq_path,
                mnq_path,
                horizon=int(payload.get("horizon", 1)),
                use_ssl_gate=bool(payload.get("use_ssl_gate", True)),
                enhance_with_ssl=bool(payload.get("enhance_with_ssl", True)),
                use_depth_filter=bool(payload.get("use_depth_filter", True)),
                compose_hold=bool(payload.get("compose_hold", False)),
                n_splits=int(payload.get("n_splits", 3)),
                n_permutations=int(payload.get("n_permutations", 100)),
                max_rows=payload.get("max_rows"),
                global_seed=seed,
                output_dir=out,
                quiet=quiet,
                understand=bool(payload.get("understand", False)),
                lean_filters=bool(payload.get("lean_filters", True)),
                exploratory=bool(payload.get("exploratory", False)),
            )
            return DayJobResult(
                day_id=day_id,
                ok=True,
                nq_path=str(nq_path),
                mnq_path=str(mnq_path) if mnq_path else None,
                output_dir=str(out.resolve()),
                mode=mode,
                best_oos_spec=search_result.best_oos_spec,
                oos_selected_ic=float(search_result.oos_selected_ic),
                n_candidates=len(search_result.candidate_columns),
                n_features=int(search_result.features.height),
                seed=seed,
            )

        from nq.strategies.fail_breakout import run_fail_breakout_research  # noqa: PLC0415

        unified_result = run_fail_breakout_research(
            nq_path,
            mnq_path,
            horizon=int(payload.get("horizon", 1)),
            max_rows=payload.get("max_rows"),
            output_dir=out,
            quiet=quiet,
        )
        return DayJobResult(
            day_id=day_id,
            ok=True,
            nq_path=str(nq_path),
            mnq_path=str(mnq_path) if mnq_path else None,
            output_dir=str(out.resolve()),
            mode=mode,
            best_oos_spec=None,
            oos_selected_ic=None,
            n_candidates=len(unified_result.signal_columns),
            n_features=int(unified_result.features.height),
            seed=seed,
        )
    except Exception as exc:
        # نُبلّغ اليوم الفاشل دون إسقاط الشهر كله افتراضيًا
        return DayJobResult(
            day_id=day_id,
            ok=False,
            nq_path=str(nq_path),
            mnq_path=str(mnq_path) if mnq_path else None,
            output_dir=str(out.resolve()),
            mode=mode,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            seed=seed,
        )


def run_fail_breakout_day_parallel(
    days: Sequence[DayInput],
    *,
    output_root: Path | str,
    mode: FbDayMode = "search",
    jobs: int = 1,
    threads_per_worker: int = 2,
    global_seed: int = 0,
    horizon: int = 1,
    max_rows: int | None = None,
    n_splits: int = 3,
    n_permutations: int = 100,
    use_ssl_gate: bool = True,
    enhance_with_ssl: bool = True,
    use_depth_filter: bool = True,
    compose_hold: bool = False,
    lean_filters: bool = True,
    exploratory: bool = False,
    understand: bool = False,
    quiet_workers: bool = True,
    fail_fast: bool = False,
) -> DayParallelManifest:
    """يشغّل Failed Breakout على كل يوم في عمليات منفصلة.

    Parameters
    ----------
    jobs:
        عدد العمليات المتوازية (≤ عدد الأيام). كل عملية = يوم واحد كامل.
    threads_per_worker:
        خيوط Polars/BLAS داخل كل عملية (افتراضي 2 لتفادي فرط الاكتتاب).
    fail_fast:
        إن ``True`` يوقف الجدولة بعد أول فشل (النتائج الجزئية تُحفظ).
    """
    if not days:
        raise ValueError("days must be non-empty")
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if threads_per_worker < 1:
        raise ValueError(f"threads_per_worker must be >= 1, got {threads_per_worker}")
    if mode not in ("search", "unified"):
        raise ValueError(f"mode must be 'search' or 'unified', got {mode!r}")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    for day in days:
        seed = stable_day_seed(global_seed, day.day_id)
        payloads.append(
            {
                "day_id": day.day_id,
                "nq_path": str(day.nq_path),
                "mnq_path": str(day.mnq_path) if day.mnq_path else None,
                "output_dir": str((root / day.day_id).resolve()),
                "mode": mode,
                "seed": seed,
                "horizon": horizon,
                "max_rows": max_rows,
                "n_splits": n_splits,
                "n_permutations": n_permutations,
                "use_ssl_gate": use_ssl_gate,
                "enhance_with_ssl": enhance_with_ssl,
                "use_depth_filter": use_depth_filter,
                "compose_hold": compose_hold,
                "lean_filters": lean_filters,
                "exploratory": exploratory,
                "understand": understand,
                "quiet": quiet_workers,
                "threads_per_worker": threads_per_worker,
            }
        )

    workers = min(jobs, len(payloads))
    results_map: dict[str, DayJobResult] = {}

    if workers == 1:
        for payload in payloads:
            res = _run_one_day(payload)
            results_map[res.day_id] = res
            if fail_fast and not res.ok:
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one_day, p): p["day_id"] for p in payloads}
            for fut in as_completed(futures):
                res = fut.result()
                results_map[res.day_id] = res
                if fail_fast and not res.ok:
                    for pending in futures:
                        pending.cancel()
                    break

    ordered = tuple(results_map[d.day_id] for d in days if d.day_id in results_map)
    # أيام لم تُشغَّل بسبب fail_fast
    missing = [
        DayJobResult(
            day_id=d.day_id,
            ok=False,
            nq_path=str(d.nq_path),
            mnq_path=str(d.mnq_path) if d.mnq_path else None,
            output_dir=str((root / d.day_id).resolve()),
            mode=mode,
            error="skipped: fail_fast after earlier day failure",
            seed=stable_day_seed(global_seed, d.day_id),
        )
        for d in days
        if d.day_id not in results_map
    ]
    all_results = ordered + tuple(missing)
    n_ok = sum(1 for r in all_results if r.ok)
    manifest = DayParallelManifest(
        jobs=workers,
        mode=mode,
        output_root=str(root.resolve()),
        n_days=len(days),
        n_ok=n_ok,
        n_failed=len(all_results) - n_ok,
        results=all_results,
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "summary.md").write_text(manifest.to_markdown(), encoding="utf-8")
    return manifest


__all__ = [
    "DayInput",
    "DayJobResult",
    "DayParallelManifest",
    "day_id_from_path",
    "discover_day_inputs",
    "run_fail_breakout_day_parallel",
    "stable_day_seed",
]
