"""مرحلة 2: علم الفترة — تجميع حالات الأيام ثم walk-forward واحد.

المرحلة 1 (يوم بيوم) تُنتج ``blended.parquet`` داخل كون سببي مغلق. هذه الوحدة
**لا** تعيد بناء الدفتر ولا تخلط MBO عبر الأيام، ولا تحسب متوسط احتمالات يومية.

المرحلة 2:
  concat(blended الأيام) → إعادة ترقيم قصص الجلسة عالميًا →
  ``run_behavior_science`` + ``run_behavior_ablation`` على كل الحالات →
  OOF الفترة + معايرة + holdout مجمّد.

خطر يُعالَج صراحة: ``_behavior_story_run`` عدّاد محلي لكل يوم (1، 2، 3…).
بدون إعادة الترقيم، آخر قصة يوم 1 وأول قصة يوم 2 قد تشتركان في نفس المعرّف
فتمتد نافذة التسمية عبر منتصف الليل.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.ablation import AblationReport, run_behavior_ablation
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
)
from nq.auction_behavior.science import BehaviorScienceReport, ScienceConfig, run_behavior_science
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession, session_date_from_ns
from nq.research.progress import ProgressLike

_DAY_DIR_NAME_LEN = 10  # YYYY-MM-DD
_PERIOD_DAY_ID = "_period_day_id"
_ASIA_LONDON = {int(VpLiquiditySession.ASIA), int(VpLiquiditySession.LONDON)}


def discover_day_blended(output_root: Path | str) -> tuple[Path, ...]:
    """مجلدات أيام ناجحة تحت جذر التشغيل المتوازي (فيها ``blended.parquet``)."""
    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"period root not found: {root.resolve()}")
    days = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and len(p.name) == _DAY_DIR_NAME_LEN and (p / "blended.parquet").is_file()
    )
    if not days:
        raise FileNotFoundError(f"no per-day blended.parquet under {root.resolve()}")
    return tuple(days)


def _run_ids_from_keys(keys: Sequence[str]) -> list[int]:
    runs: list[int] = []
    run = 0
    previous: str | None = None
    for key in keys:
        if key != previous:
            run += 1
        runs.append(run)
        previous = key
    return runs


def remint_period_story_runs(
    frame: pl.DataFrame,
    *,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يعيد ترقيم ``_liquidity_run`` و``_behavior_story_run`` على الإطار المجمّع.

    المعرّفات المحلية لكل يوم تُحذف أولًا. مفتاح القصة يشمل ``_period_day_id``
    حتى لا تُعاد عبر منتصف الليل ولا تُدمَج شرائح نفس جلسة CME من ملفين يوميين
    (كل ملف مرحلة-1 كون دفتري مغلق).
    """
    if frame.height == 0:
        return frame
    drop = [c for c in ("_liquidity_run", "_behavior_story_run") if c in frame.columns]
    work = frame.drop(drop) if drop else frame
    ordered = work.sort(AVAILABILITY_TS)
    times = [int(x) for x in ordered[AVAILABILITY_TS].to_list()]
    n = len(times)
    if progress is not None:
        progress.op(f"period remint story runs n={n:,}")
    if VP_LIQUIDITY_SESSION in ordered.columns:
        sessions = [int(x) for x in ordered[VP_LIQUIDITY_SESSION].fill_null(-1).to_list()]
    else:
        sessions = [-1] * n
    if _PERIOD_DAY_ID in ordered.columns:
        day_ids = [str(x) for x in ordered[_PERIOD_DAY_ID].fill_null("").to_list()]
    else:
        day_ids = [session_date_from_ns(ts) for ts in times]
        ordered = ordered.with_columns(pl.Series(_PERIOD_DAY_ID, day_ids, dtype=pl.Utf8))
    liq_keys: list[str] = []
    story_keys: list[str] = []
    for i, (day_id, ts, sess) in enumerate(zip(day_ids, times, sessions, strict=True), start=1):
        if progress is not None:
            progress.heartbeat(i, n, label="period-story-keys")
        liq_keys.append(f"{day_id}:{sess}")
        bucket = "asia_london" if sess in _ASIA_LONDON else str(sess)
        story_keys.append(f"{day_id}:{session_date_from_ns(ts)}:{bucket}")
    return ordered.with_columns(
        pl.Series("_liquidity_run", _run_ids_from_keys(liq_keys), dtype=pl.Int64),
        pl.Series("_behavior_story_run", _run_ids_from_keys(story_keys), dtype=pl.Int64),
    )


def load_period_blended(
    day_dirs: Sequence[Path],
    *,
    progress: ProgressLike | None = None,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """يحمّل ``blended`` الأيام، يوحّد الأعمدة، يرتّب سببيًا، ويعيد ترقيم القصص."""
    if not day_dirs:
        raise ValueError("day_dirs must be non-empty")
    frames: list[pl.DataFrame] = []
    day_ids: list[str] = []
    n_days = len(day_dirs)
    for i, day_dir in enumerate(day_dirs, start=1):
        path = Path(day_dir) / "blended.parquet"
        if progress is not None:
            progress.heartbeat(i, n_days, label="load-day-blended", force=True)
        frame = pl.read_parquet(path)
        if AVAILABILITY_TS not in frame.columns:
            raise ValueError(f"{path} missing {AVAILABILITY_TS}")
        if frame.height == 0:
            continue
        day_id = Path(day_dir).name
        frames.append(frame.with_columns(pl.lit(day_id).alias(_PERIOD_DAY_ID)))
        day_ids.append(day_id)
    if not frames:
        raise ValueError("all discovered blended frames were empty")
    if progress is not None:
        progress.op(f"concat {len(frames)} day blended frames")
    pooled = pl.concat(frames, how="diagonal_relaxed").sort(AVAILABILITY_TS)
    n_unique = int(pooled[AVAILABILITY_TS].n_unique())
    if n_unique != pooled.height:
        raise ValueError(
            "duplicate availability_ts across pooled days "
            f"(rows={pooled.height}, unique={n_unique}); refusing silent fan-out"
        )
    pooled = remint_period_story_runs(pooled, progress=progress)
    return pooled, tuple(day_ids)


def _session_dates(ts: Sequence[int]) -> list[str]:
    return [session_date_from_ns(int(t)) for t in ts]


def assert_labels_do_not_cross_session_dates(labeled: pl.DataFrame) -> None:
    """نافذة التسمية يجب أن تبقى داخل تاريخ تداول واحد بعد إعادة الترقيم."""
    if labeled.height == 0:
        return
    need = (SETUP_AVAILABILITY_TS, OUTCOME_AVAILABLE_TS)
    missing = [c for c in need if c not in labeled.columns]
    if missing:
        raise ValueError(f"labeled frame missing {missing}")
    setup_dates = _session_dates(labeled[SETUP_AVAILABILITY_TS].to_list())
    outcome_dates = _session_dates(labeled[OUTCOME_AVAILABLE_TS].to_list())
    crossed = [
        (setup_dates[i], outcome_dates[i])
        for i in range(len(setup_dates))
        if setup_dates[i] != outcome_dates[i]
    ]
    if crossed:
        raise AssertionError(
            f"label window crossed session dates after period concat (examples={crossed[:5]})"
        )


def assert_labels_do_not_cross_period_days(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
) -> None:
    """نافذة التسمية يجب ألا تخرج من ملف يوم المرحلة-1 (كون دفتري مغلق)."""
    if labeled.height == 0 or _PERIOD_DAY_ID not in blended.columns:
        return
    need = (SETUP_AVAILABILITY_TS, OUTCOME_AVAILABLE_TS)
    missing = [c for c in need if c not in labeled.columns]
    if missing:
        raise ValueError(f"labeled frame missing {missing}")
    ts_to_day = {
        int(ts): str(day)
        for ts, day in zip(
            blended[AVAILABILITY_TS].to_list(),
            blended[_PERIOD_DAY_ID].to_list(),
            strict=True,
        )
    }
    crossed: list[tuple[str, str, int, int]] = []
    for setup_ts, outcome_ts in zip(
        labeled[SETUP_AVAILABILITY_TS].to_list(),
        labeled[OUTCOME_AVAILABLE_TS].to_list(),
        strict=True,
    ):
        setup_day = ts_to_day.get(int(setup_ts))
        outcome_day = ts_to_day.get(int(outcome_ts))
        if setup_day is None or outcome_day is None:
            raise AssertionError(
                "label timestamps missing from period blended "
                f"(setup={int(setup_ts)}, outcome={int(outcome_ts)})"
            )
        if setup_day != outcome_day:
            crossed.append((setup_day, outcome_day, int(setup_ts), int(outcome_ts)))
    if crossed:
        raise AssertionError(
            f"label window crossed phase-1 day files after period concat (examples={crossed[:5]})"
        )


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


@dataclass(frozen=True, slots=True)
class BehaviorPeriodReport:
    """نتيجة علم الفترة — ليست متوسط احتمالات الأيام."""

    blended: pl.DataFrame
    day_ids: tuple[str, ...]
    science: BehaviorScienceReport
    ablation: AblationReport | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def run_behavior_period_science(
    output_root: Path | str | None = None,
    *,
    day_dirs: Sequence[Path] | None = None,
    blended: pl.DataFrame | None = None,
    day_ids: Sequence[str] | None = None,
    config: ScienceConfig | None = None,
    include_ablation: bool = True,
    progress: ProgressLike | None = None,
) -> BehaviorPeriodReport:
    """يجمّع حالات الأيام ثم يدرّب علمًا واحدًا (OOF الفترة + holdout).

    مرِّر ``output_root`` (جذر تشغيل يومي) أو ``blended`` جاهزًا للاختبارات.
    """
    cfg = config or ScienceConfig()
    if blended is None:
        dirs = tuple(day_dirs) if day_dirs is not None else discover_day_blended(output_root or ".")
        pooled, ids = load_period_blended(dirs, progress=progress)
    else:
        pooled = remint_period_story_runs(blended, progress=progress)
        ids = tuple(day_ids) if day_ids is not None else ()
        if AVAILABILITY_TS in pooled.columns and pooled.height:
            n_unique = int(pooled[AVAILABILITY_TS].n_unique())
            if n_unique != pooled.height:
                raise ValueError(
                    "duplicate availability_ts in provided blended "
                    f"(rows={pooled.height}, unique={n_unique})"
                )
    if progress is not None:
        progress.op(f"period science bars={pooled.height:,} days={len(ids)}")
    science = run_behavior_science(pooled, config=cfg, progress=progress)
    assert_labels_do_not_cross_session_dates(science.labeled)
    assert_labels_do_not_cross_period_days(science.labeled, pooled)
    ablation: AblationReport | None = None
    if include_ablation:
        if progress is not None:
            progress.op("period ablation")
        ablation = run_behavior_ablation(pooled, config=cfg, progress=progress)
    diagnostics: dict[str, Any] = {
        "n_days": len(ids),
        "day_ids": list(ids),
        "n_bars": int(pooled.height),
        "pooled_not_averaged_daily_probabilities": True,
        "raw_mbo_not_concatenated": True,
        "story_runs_reminted_globally": True,
        "phase1_day_files_not_rejoined": True,
        "oof_predictions_eligible_for_backtest": True,
        "live_predictions_eligible_for_backtest": False,
        "science": science.diagnostics,
        "ablation": None if ablation is None else ablation.diagnostics,
        "principles": (
            "phase1: each day is an isolated causal universe (MBO/book never pooled)",
            "phase2: pool completed blended states + labels; one walk-forward",
            "not a mean of per-day p_*; OOF is from period folds on unique setups",
            "story runs reminted so label windows cannot cross session dates or phase-1 day files",
            "holdout is the frozen temporal tail of the period, single-touch",
        ),
    }
    return BehaviorPeriodReport(
        blended=pooled,
        day_ids=tuple(ids),
        science=science,
        ablation=ablation,
        diagnostics=diagnostics,
    )


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    if frame.height:
        frame.write_parquet(path)


def write_behavior_period_report(
    report: BehaviorPeriodReport,
    output_dir: Path | str,
) -> Path:
    """يكتب مخرجات الفترة (باركيه + ملخص JSON + بيان Markdown)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    science = report.science
    _write_parquet(report.blended, out / "period_blended.parquet")
    _write_parquet(science.labeled, out / "science_labeled.parquet")
    _write_parquet(science.fold_frame, out / "fold_metrics.parquet")
    _write_parquet(science.fold_scores, out / "fold_scores.parquet")
    _write_parquet(science.conditional_oof_predictions, out / "oof_predictions.parquet")
    _write_parquet(science.live_model_predictions, out / "live_predictions.parquet")
    _write_parquet(science.competing_labeled, out / "competing_labeled.parquet")
    _write_parquet(science.competing_oof_predictions, out / "competing_oof.parquet")
    _write_parquet(science.competing_live_predictions, out / "competing_live.parquet")
    _write_parquet(science.calibration_by_outcome, out / "calibration_by_outcome.parquet")
    if report.ablation is not None:
        _write_parquet(report.ablation.frame, out / "ablation.parquet")
        _write_parquet(report.ablation.competing_frame, out / "ablation_competing.parquet")

    summary = {
        "day_ids": list(report.day_ids),
        "n_days": len(report.day_ids),
        "n_bars": int(report.blended.height),
        "n_oof": int(science.conditional_oof_predictions.height),
        "n_live": int(science.live_model_predictions.height),
        "n_labeled": science.diagnostics.get("n_labeled"),
        "n_resolved": science.diagnostics.get("n_resolved"),
        "n_develop": science.diagnostics.get("n_develop"),
        "n_holdout": science.diagnostics.get("n_holdout"),
        "holdout_touched": science.diagnostics.get("holdout_touched"),
        "n_folds": science.diagnostics.get("n_folds"),
        "n_features": science.diagnostics.get("n_features"),
        "n_level_flow_features": science.diagnostics.get("n_level_flow_features"),
        "n_reliability_features": science.diagnostics.get("n_reliability_features"),
        "n_path_features": science.diagnostics.get("n_path_features"),
        "n_memory_features": science.diagnostics.get("n_memory_features"),
        "competing_class_counts": science.diagnostics.get("competing_class_counts"),
        "n_competing_develop": science.diagnostics.get("n_competing_develop"),
        "pooled_not_averaged_daily_probabilities": True,
        "oof_predictions_eligible_for_backtest": True,
        "live_predictions_eligible_for_backtest": False,
        "diagnostics": _jsonable(report.diagnostics),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "n_days": len(report.day_ids),
                "day_ids": list(report.day_ids),
                "principles": report.diagnostics.get("principles"),
                "pooled_not_averaged_daily_probabilities": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lines = [
        "# auction_behavior — period science (phase 2)",
        "",
        "Not a mean of per-day probabilities. One walk-forward on pooled setups.",
        "",
        f"- days: {len(report.day_ids)}",
        f"- bars: {report.blended.height:,}",
        f"- labeled: {science.diagnostics.get('n_labeled')} · "
        f"resolved: {science.diagnostics.get('n_resolved')}",
        f"- develop: {science.diagnostics.get('n_develop')} · "
        f"holdout: {science.diagnostics.get('n_holdout')} "
        f"(touched={science.diagnostics.get('holdout_touched')})",
        f"- folds: {science.diagnostics.get('n_folds')}",
        f"- OOF rows: {science.conditional_oof_predictions.height:,} (eligible_for_backtest=true)",
        f"- live rows: {science.live_model_predictions.height:,} (eligible_for_backtest=false)",
        f"- competing develop: {science.diagnostics.get('n_competing_develop')} · "
        f"counts={science.diagnostics.get('competing_class_counts')}",
        f"- features: total={science.diagnostics.get('n_features')} "
        f"lf={science.diagnostics.get('n_level_flow_features')} "
        f"rel={science.diagnostics.get('n_reliability_features')} "
        f"path={science.diagnostics.get('n_path_features')}",
        "",
        "## Isolation",
        "",
    ]
    for p in report.diagnostics.get("principles", ()):
        lines.append(f"- {p}")
    (out / "PERIOD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


__all__ = [
    "BehaviorPeriodReport",
    "assert_labels_do_not_cross_period_days",
    "assert_labels_do_not_cross_session_dates",
    "discover_day_blended",
    "load_period_blended",
    "remint_period_story_runs",
    "run_behavior_period_science",
    "write_behavior_period_report",
]
