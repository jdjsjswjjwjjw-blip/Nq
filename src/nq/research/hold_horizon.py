"""مسار إمساك بعد إطلاق سببي — N بارميل، ليست نافذة تسمية التيك الواحد.

بنية تحتية للطبقات القابلة للخلع. ليست وقف سعر حيًا، وليست ذروة موجة.
لا تحميل MBO، لا تنبؤ حي في الباك تست.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.causal_entry import (
    CausalEntryConfig,
    assert_not_raw_mbo_stream,
    run_causal_entry,
    ticks_to_nq_points,
)
from nq.research.progress import ProgressLike

_GROUP = "_behavior_story_run"
_LOOKAHEAD = frozenset(
    {
        "wave_frac",
        "wave_peak_ticks",
        "expansion_frac",
        "ticks_remaining_to_peak",
        "wave_bin",
    }
)

HoldRow = dict[str, Any]
ExitFn = Callable[[Mapping[str, np.ndarray], Mapping[str, float]], tuple[int, str]]


def _f64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    if frame.height == 0 or name not in frame.columns:
        return np.zeros(int(frame.height), dtype=np.float64)
    return frame[name].cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64, copy=False)


def _i64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame[name].cast(pl.Int64).to_numpy().astype(np.int64, copy=False)


def _path_by_story(path: pl.DataFrame, group_col: str) -> dict[Any, pl.DataFrame]:
    grouped: dict[Any, pl.DataFrame] = {}
    for key, group in path.group_by(group_col, maintain_order=True):
        story = key[0] if isinstance(key, tuple) else key
        grouped[story] = group
    return grouped


def attach_oof_p(
    blended: pl.DataFrame,
    predictions: pl.DataFrame | None,
    *,
    score_col: str = "p_y_path_further_beyond",
) -> pl.DataFrame:
    """يلحق ``model_p`` من OOF فقط. لا forward-fill — البارميل بلا درجة يبقى فارغًا."""
    if predictions is None or predictions.height == 0:
        return blended.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("model_p"),
            pl.lit(False).alias("oof_p_at_bar"),
        )
    src = score_col if score_col in predictions.columns else "model_p"
    ts_col = (
        SETUP_AVAILABILITY_TS if SETUP_AVAILABILITY_TS in predictions.columns else AVAILABILITY_TS
    )
    if src not in predictions.columns or ts_col not in predictions.columns:
        return blended.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("model_p"),
            pl.lit(False).alias("oof_p_at_bar"),
        )
    scores = (
        predictions.select(
            pl.col(ts_col).cast(pl.Int64).alias(AVAILABILITY_TS),
            pl.col(src).cast(pl.Float64).alias("model_p"),
        )
        .filter(pl.col("model_p").is_finite())
        .unique(subset=[AVAILABILITY_TS], keep="first")
    )
    joined = blended.join(scores, on=AVAILABILITY_TS, how="left")
    return joined.with_columns(pl.col("model_p").is_not_null().alias("oof_p_at_bar"))


def walk_hold_windows(
    entries: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    max_hold_bars: int,
    group_col: str = _GROUP,
    path_cols: Sequence[str] = (),
    decide_exit: ExitFn,
    round_trip_cost_pts: float = 0.0,
    size_of: Callable[[Mapping[str, Any]], float] | None = None,
) -> pl.DataFrame:
    """يمسك N بارميل بعد ``t`` على المسار الممزوج — ليس ``outcome_available_ts``.

    ``decide_exit(window_cols, entry_scalars) -> (offset, reason)``.
    الذروة المكتملة لا تُقرأ.
    """
    if max_hold_bars < 1:
        raise ValueError("max_hold_bars must be >= 1")
    if entries.height == 0:
        return entries.head(0)
    drop_look = [c for c in _LOOKAHEAD if c in blended.columns]
    work_b = blended.drop(drop_look) if drop_look else blended
    cols = [
        group_col,
        AVAILABILITY_TS,
        "path_beyond_asia_ticks",
        "path_extreme_ticks",
        *list(path_cols),
    ]
    present = [c for c in cols if c in work_b.columns]
    stories = entries[group_col].unique().to_list()
    path = (
        work_b.filter(pl.col(group_col).is_in(stories))
        .select(present)
        .sort([group_col, AVAILABILITY_TS])
    )
    grouped = _path_by_story(path, group_col)
    rows: list[HoldRow] = []
    t0 = _i64_arr(entries, SETUP_AVAILABILITY_TS)
    beyond0 = _f64_arr(entries, "path_beyond_asia_ticks")
    story_vals = entries[group_col].to_list()
    for i, story in enumerate(story_vals):
        story_path = grouped.get(story)
        scalars: dict[str, float] = {"beyond0": float(beyond0[i])}
        if "model_p" in entries.columns:
            raw_p = entries["model_p"][i]
            scalars["p_entry"] = float(raw_p) if raw_p is not None else float("nan")
        for name in path_cols:
            if name in entries.columns:
                val = entries[name][i]
                scalars[f"{name}_0"] = 0.0 if val is None else float(val)
        size = 1.0
        if size_of is not None:
            size = float(size_of(entries.row(i, named=True)))
        if story_path is None or story_path.height == 0:
            rows.append(_empty_trade(i, t0[i], size, round_trip_cost_pts, reason="no_path"))
            continue
        ts = _i64_arr(story_path, AVAILABILITY_TS)
        lo = int(np.searchsorted(ts, t0[i], side="right"))
        hi = min(int(lo + max_hold_bars), int(story_path.height))
        if lo >= hi:
            rows.append(_empty_trade(i, t0[i], size, round_trip_cost_pts, reason="empty"))
            continue
        window = story_path[lo:hi]
        col_map: dict[str, np.ndarray] = {
            "availability_ts": _i64_arr(window, AVAILABILITY_TS),
            "path_beyond_asia_ticks": _f64_arr(window, "path_beyond_asia_ticks"),
        }
        n_win = int(window.height)
        for name in path_cols:
            if name in window.columns:
                col_map[name] = _f64_arr(window, name)
            else:
                col_map[name] = np.zeros(n_win, dtype=np.float64)
        offset, reason = decide_exit(col_map, scalars)
        rows.append(
            _scored_trade(
                i,
                t0[i],
                beyond0[i],
                size,
                round_trip_cost_pts,
                col_map,
                offset,
                reason,
                p_entry=scalars.get("p_entry", float("nan")),
            )
        )
    return pl.DataFrame(rows)


def _scored_trade(
    i: int,
    t0: int,
    beyond0: float,
    size: float,
    cost: float,
    col_map: Mapping[str, np.ndarray],
    offset: int,
    reason: str,
    *,
    p_entry: float,
) -> HoldRow:
    offset = max(0, min(int(offset), int(col_map["path_beyond_asia_ticks"].size) - 1))
    beyond = col_map["path_beyond_asia_ticks"]
    used = beyond[: offset + 1]
    mfe = float(max(0.0, float(np.nanmax(used)) - beyond0))
    mae = float(max(0.0, beyond0 - float(np.nanmin(used))))
    realized_ticks = float(used[-1] - beyond0)
    realized_pts = ticks_to_nq_points(realized_ticks)
    return {
        "entry_i": i,
        SETUP_AVAILABILITY_TS: int(t0),
        "exit_availability_ts": int(col_map["availability_ts"][offset]),
        "exit_reason": reason,
        "hold_bars": int(offset + 1),
        "size": float(size),
        "p_entry": p_entry,
        "mfe_beyond_ticks": mfe,
        "mae_beyond_ticks": mae,
        "realized_beyond_ticks": realized_ticks,
        "mfe_beyond_pts": ticks_to_nq_points(mfe),
        "mae_beyond_pts": ticks_to_nq_points(mae),
        "realized_beyond_pts": realized_pts,
        "round_trip_cost_pts": float(cost),
        "net_pts": float(size) * (realized_pts - float(cost)),
    }


def _empty_trade(
    i: int,
    t0: int,
    size: float,
    cost: float,
    *,
    reason: str,
) -> HoldRow:
    return {
        "entry_i": i,
        SETUP_AVAILABILITY_TS: int(t0),
        "exit_availability_ts": int(t0),
        "exit_reason": reason,
        "hold_bars": 0,
        "size": float(size),
        "p_entry": float("nan"),
        "mfe_beyond_ticks": 0.0,
        "mae_beyond_ticks": 0.0,
        "realized_beyond_ticks": 0.0,
        "mfe_beyond_pts": 0.0,
        "mae_beyond_pts": 0.0,
        "realized_beyond_pts": 0.0,
        "round_trip_cost_pts": float(cost),
        "net_pts": float(size) * (0.0 - float(cost)),
    }


def causal_fires(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    predictions: pl.DataFrame | None,
    oof_availability_ts: Sequence[int] | None,
    holdout_cut_ts: int | None,
    min_p: float,
    expansion_start_ticks: float,
    holdout_months: int | None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """إطلاقات OOF السببية (p + امتداد ظاهر) — نفس بوابة الطبقة الأساسية."""
    report = run_causal_entry(
        labeled,
        blended,
        config=CausalEntryConfig(
            min_p=min_p,
            expansion_start_ticks=expansion_start_ticks,
            holdout_months=holdout_months,
        ),
        oof_availability_ts=oof_availability_ts,
        holdout_cut_ts=holdout_cut_ts,
        predictions=predictions,
        progress=progress,
    )
    return report.all_entries


def load_overlay_period_inputs(
    period_dir: Path | str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, int | None]:
    """باركيه المرحلة 2 + OOF فقط. يرفض ``live_predictions`` للباك تست."""
    root = Path(period_dir)
    labeled_path = root / "science_labeled.parquet"
    blended_path = root / "period_blended.parquet"
    if not labeled_path.is_file():
        raise FileNotFoundError(f"science_labeled.parquet not found under {root.resolve()}")
    if not blended_path.is_file():
        raise FileNotFoundError(f"period_blended.parquet not found under {root.resolve()}")
    labeled = pl.read_parquet(labeled_path)
    blended = pl.read_parquet(blended_path)
    assert_not_raw_mbo_stream(labeled, source=str(labeled_path))
    assert_not_raw_mbo_stream(blended, source=str(blended_path))
    oof: pl.DataFrame | None = None
    oof_path = root / "oof_predictions.parquet"
    if oof_path.is_file():
        oof = pl.read_parquet(oof_path)
    cut_ts: int | None = None
    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        diag = summary.get("diagnostics", {})
        science = diag.get("science", diag)
        raw_cut = science.get("holdout_cut_ts", summary.get("holdout_cut_ts"))
        if raw_cut is not None and int(raw_cut) >= 0:
            cut_ts = int(raw_cut)
    return labeled, blended, oof, cut_ts


def oof_timestamps(predictions: pl.DataFrame | None) -> list[int] | None:
    if predictions is None or predictions.height == 0:
        return None
    ts_col = (
        SETUP_AVAILABILITY_TS if SETUP_AVAILABILITY_TS in predictions.columns else AVAILABILITY_TS
    )
    if ts_col not in predictions.columns:
        return None
    return [int(t) for t in predictions[ts_col].to_list()]


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if obj != obj else obj  # noqa: PLR0124
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    return str(obj)


def median_mean(frame: pl.DataFrame, col: str) -> tuple[float | None, float | None]:
    if frame.height == 0 or col not in frame.columns:
        return None, None
    arr = _f64_arr(frame, col)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return float(np.median(finite)), float(np.mean(finite))


__all__ = [
    "attach_oof_p",
    "causal_fires",
    "jsonable",
    "load_overlay_period_inputs",
    "median_mean",
    "oof_timestamps",
    "ticks_to_nq_points",
    "walk_hold_windows",
]
