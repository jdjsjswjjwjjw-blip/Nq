"""التقاط سببي بعد إطلاق النموذج — بلا ذروة موجة مكتملة.

تشخيص موقع الموجة خلط ساعتين:

- ``p`` عند ``t`` سببي (OOF، الميزات معروفة عند ``setup_availability_ts``).
- وسم الإطلاق «متأخر» لأن الذروة *المكتملة* كانت قد طُبعت بنسبة 80% نظرة
  أمامية، ولا يجوز أن يكون فلتر دخول.

هذه الوحدة تجيب سؤالًا آخر: إذا سُمح بالإطلاق عند ``t`` (النموذج + امتداد
ظاهر، واختياريًا عتبة طباعة *ثابتة*)، كم من نافذة المخاطر المتنافسة بعد
``t`` يُلتقط، وكم الانخفاض السببي داخل النافذة نفسها.

الالتقاط هو MFE / MAE / المحقّق على المسار الممزوج حيث

    setup_availability_ts < availability_ts <= outcome_available_ts

نفس أفق ``y_path_further_beyond``. الذروة و``wave_frac`` لا تُقرآن أبدًا.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.holdout import carve_frozen_holdout
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    filter_resolved_outcomes,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_availability_not_before_event, assert_causal_order

_RAW_MBO_SIGNATURE = frozenset({"order_id", "action"})
_GROUP = "_behavior_story_run"
_ONSET = 0.5
TICKS_PER_NQ_POINT = 4
DEFAULT_MIN_PRINTED_LATE_TICKS = 80.0  # 20 نقطة NQ ظاهرة عند t
DEFAULT_EXPANSION_START_TICKS = 16.0
DEFAULT_MIN_P = 0.5
FURTHER_BEYOND = "y_path_further_beyond"
SCORE_COL = "p_y_path_further_beyond"

# أعمدة الموجة المكتملة نظرة أمامية — تُتجاهل إن وُجدت ولا تُستخدم للدخول.
_LOOKAHEAD_WAVE_COLS = frozenset(
    {
        "wave_frac",
        "wave_peak_ticks",
        "expansion_frac",
        "expansion_start_ticks",
        "ticks_printed_of_wave",
        "ticks_remaining_to_peak",
        "wave_bin",
        "expansion_bin",
        "full_wave_frac",
        "wave_peak_ts",
        "expansion_start_ts",
        "expansion_run_ticks",
        "expansion_start_extent",
    }
)


@dataclass(frozen=True, slots=True)
class CausalEntryConfig:
    """بوابات دخول سببية ثابتة — غير مُقدَّرة على العينة."""

    outcome_further: str = FURTHER_BEYOND
    group_col: str = _GROUP
    score_col: str = SCORE_COL
    min_p: float = DEFAULT_MIN_P
    expansion_start_ticks: float = DEFAULT_EXPANSION_START_TICKS
    min_printed_late_ticks: float = DEFAULT_MIN_PRINTED_LATE_TICKS
    require_outside_volume: bool = False
    require_depth_confirm: bool = False
    min_outside_volume_share: float = 0.5
    min_depth_confirm: float = 0.5
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class CausalEntryReport:
    """التقاط كل الإطلاقات مقابل الدخول المتأخر المؤكد (عتبة ثابتة)."""

    all_entries: pl.DataFrame
    late_entries: pl.DataFrame
    summaries: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def ticks_to_nq_points(ticks: float) -> float:
    return float(ticks) / float(TICKS_PER_NQ_POINT)


def assert_not_raw_mbo_stream(frame: pl.DataFrame, *, source: str = "") -> None:
    """يرفض تدفق MBO خام — البحث يقرأ حالات مكتملة فقط."""
    present = _RAW_MBO_SIGNATURE.intersection(frame.columns)
    if present != _RAW_MBO_SIGNATURE:
        return
    where = f" in {source}" if source else ""
    raise ValueError(
        "causal entry refuses raw MBO streams "
        f"(found {sorted(present)}{where}); "
        "it reads completed blended states only — no book reconstruction"
    )


def _f64(frame: pl.DataFrame, name: str, default: float = 0.0) -> pl.Expr:
    if name in frame.columns:
        return pl.col(name).cast(pl.Float64).fill_null(default)
    return pl.lit(default, dtype=pl.Float64)


def _f64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    if frame.height == 0 or name not in frame.columns:
        return np.zeros(int(frame.height), dtype=np.float64)
    return frame[name].cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64, copy=False)


def _i64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame[name].cast(pl.Int64).to_numpy().astype(np.int64, copy=False)


def _exclude_holdout(
    labeled: pl.DataFrame,
    *,
    holdout_cut_ts: int | None,
    holdout_months: int | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "holdout_scored": False,
        "holdout_excluded": False,
        "holdout_cut_ts": -1,
        "holdout_n_rows": 0,
    }
    if labeled.height == 0 or SETUP_AVAILABILITY_TS not in labeled.columns:
        return labeled, meta
    if holdout_cut_ts is not None and int(holdout_cut_ts) >= 0:
        cut = int(holdout_cut_ts)
        develop = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= cut)
        held = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) > cut)
        meta.update(
            holdout_excluded=held.height > 0,
            holdout_cut_ts=cut,
            holdout_n_rows=int(held.height),
        )
        return develop, meta
    if holdout_months is None:
        return labeled, meta
    try:
        pack = carve_frozen_holdout(
            labeled, holdout_months=int(holdout_months), ts_col=SETUP_AVAILABILITY_TS
        )
    except ValueError:
        meta["holdout_carve"] = "insufficient_months_used_all_labeled_as_develop"
        return labeled, meta
    meta.update(
        holdout_excluded=pack.holdout.height > 0,
        holdout_cut_ts=int(pack.cut_ts),
        holdout_n_rows=int(pack.holdout.height),
        holdout_months=int(holdout_months),
    )
    return pack.develop, meta


def _outcome_frame(labeled: pl.DataFrame, name: str) -> pl.DataFrame:
    if labeled.height == 0 or "outcome_name" not in labeled.columns:
        return labeled.head(0)
    work = labeled.filter(pl.col("outcome_name") == name)
    if "label_status" in work.columns:
        work = filter_resolved_outcomes(work)
    return work


def _prediction_scores(predictions: pl.DataFrame | None, score_col: str) -> pl.DataFrame:
    empty = pl.DataFrame(schema={SETUP_AVAILABILITY_TS: pl.Int64(), "model_p": pl.Float64()})
    if predictions is None or predictions.height == 0:
        return empty
    src = score_col if score_col in predictions.columns else "model_p"
    if src not in predictions.columns:
        return empty
    ts_col = (
        SETUP_AVAILABILITY_TS if SETUP_AVAILABILITY_TS in predictions.columns else AVAILABILITY_TS
    )
    if ts_col not in predictions.columns:
        return empty
    return (
        predictions.select(
            pl.col(ts_col).cast(pl.Int64).alias(SETUP_AVAILABILITY_TS),
            pl.col(src).cast(pl.Float64).alias("model_p"),
        )
        .filter(pl.col("model_p").is_finite())
        .unique(subset=[SETUP_AVAILABILITY_TS], keep="first")
    )


def _join_model_p(setups: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    if setups.height == 0:
        return setups
    if "model_p" in setups.columns and scores.height == 0:
        return setups
    if scores.height == 0:
        if "model_p" in setups.columns:
            return setups
        return setups.with_columns(pl.lit(None, dtype=pl.Float64).alias("model_p"))
    work = setups.drop("model_p") if "model_p" in setups.columns else setups
    return work.join(scores, on=SETUP_AVAILABILITY_TS, how="left")


def _scope_slice(
    frame: pl.DataFrame,
    *,
    oof_availability_ts: Sequence[int] | None,
) -> tuple[pl.DataFrame, str]:
    if not oof_availability_ts or SETUP_AVAILABILITY_TS not in frame.columns:
        return frame, "develop"
    oof_set = {int(t) for t in oof_availability_ts}
    oof = frame.filter(pl.col(SETUP_AVAILABILITY_TS).is_in(list(oof_set)))
    if oof.height == 0:
        return frame, "develop"
    return oof, "oof_develop"


def _printed_expr(frame: pl.DataFrame) -> pl.Expr:
    return pl.max_horizontal(
        _f64(frame, "path_beyond_asia_ticks"),
        _f64(frame, "path_extreme_ticks"),
    )


def _entry_mask(frame: pl.DataFrame, *, cfg: CausalEntryConfig, late: bool) -> pl.Expr:
    printed = _printed_expr(frame)
    fired = pl.col("model_p").is_finite() & (pl.col("model_p") >= float(cfg.min_p))
    fired = fired & (printed >= float(cfg.expansion_start_ticks))
    if late:
        fired = fired & (printed >= float(cfg.min_printed_late_ticks))
    if cfg.require_depth_confirm:
        fired = fired & (_f64(frame, "path_depth_confirm") >= float(cfg.min_depth_confirm))
    if cfg.require_outside_volume:
        fired = fired & (
            _f64(frame, "proj_outside_volume_share") >= float(cfg.min_outside_volume_share)
        )
    return fired.fill_null(False)


def _path_by_story(path: pl.DataFrame, group_col: str) -> dict[Any, pl.DataFrame]:
    grouped: dict[Any, pl.DataFrame] = {}
    for key, group in path.group_by(group_col, maintain_order=True):
        story = key[0] if isinstance(key, tuple) else key
        grouped[story] = group
    return grouped


def _fill_capture_windows(
    base: pl.DataFrame,
    grouped: dict[Any, pl.DataFrame],
    *,
    group_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(base.height)
    mfe_b = np.zeros(n, dtype=np.float64)
    mae_b = np.zeros(n, dtype=np.float64)
    real_b = np.zeros(n, dtype=np.float64)
    mfe_e = np.zeros(n, dtype=np.float64)
    empty = np.ones(n, dtype=np.bool_)
    n_bars = np.zeros(n, dtype=np.int64)
    t0 = _i64_arr(base, SETUP_AVAILABILITY_TS)
    t1 = _i64_arr(base, OUTCOME_AVAILABLE_TS)
    beyond0 = _f64_arr(base, "path_beyond_asia_ticks")
    extreme0 = _f64_arr(base, "path_extreme_ticks")
    by_story: dict[Any, list[int]] = {}
    for i, story in enumerate(base[group_col].to_list()):
        by_story.setdefault(story, []).append(i)
    for story, idxs in by_story.items():
        story_path = grouped.get(story)
        if story_path is None or story_path.height == 0:
            continue
        ts = _i64_arr(story_path, AVAILABILITY_TS)
        beyond = _f64_arr(story_path, "path_beyond_asia_ticks")
        extreme = _f64_arr(story_path, "path_extreme_ticks")
        for i in idxs:
            lo = int(np.searchsorted(ts, t0[i], side="right"))
            hi = int(np.searchsorted(ts, t1[i], side="right"))
            if lo >= hi:
                continue
            window_b = beyond[lo:hi]
            window_e = extreme[lo:hi]
            empty[i] = False
            n_bars[i] = int(hi - lo)
            mfe_b[i] = float(max(0.0, float(np.nanmax(window_b)) - beyond0[i]))
            mae_b[i] = float(max(0.0, beyond0[i] - float(np.nanmin(window_b))))
            real_b[i] = float(window_b[-1] - beyond0[i])
            mfe_e[i] = float(max(0.0, float(np.nanmax(window_e)) - extreme0[i]))
    return mfe_b, mae_b, real_b, mfe_e, empty, n_bars


def compute_forward_capture(
    entries: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    group_col: str = _GROUP,
) -> pl.DataFrame:
    """MFE/MAE/المحقّق على نافذة النتيجة بعد كل دخول.

    براميل المسار: ``setup_availability_ts < availability_ts <= outcome_available_ts``.
    أعمدة الذروة المكتملة لا تدخل الحساب حتى إن وُجدت.
    """
    schema_extra = {
        "printed_at_entry_ticks": pl.Float64(),
        "mfe_beyond_ticks": pl.Float64(),
        "mae_beyond_ticks": pl.Float64(),
        "realized_beyond_ticks": pl.Float64(),
        "mfe_extreme_ticks": pl.Float64(),
        "window_empty": pl.Boolean(),
        "window_n_bars": pl.Int64(),
    }
    empty_cols = [pl.lit(None, dtype=dtype).alias(name) for name, dtype in schema_extra.items()]
    if entries.height == 0:
        return entries.with_columns(empty_cols)
    needed = (
        group_col,
        SETUP_AVAILABILITY_TS,
        OUTCOME_AVAILABLE_TS,
        "path_beyond_asia_ticks",
        "path_extreme_ticks",
    )
    missing = [c for c in needed if c not in entries.columns]
    if missing:
        raise ValueError(f"causal capture entries missing columns: {missing}")
    if group_col not in blended.columns or AVAILABILITY_TS not in blended.columns:
        raise ValueError("blended path requires story id and availability_ts")
    drop_look = [c for c in _LOOKAHEAD_WAVE_COLS if c in entries.columns]
    work = entries.drop(drop_look) if drop_look else entries
    stories = work[group_col].unique().to_list()
    path = (
        blended.filter(pl.col(group_col).is_in(stories))
        .select(
            group_col,
            AVAILABILITY_TS,
            _f64(blended, "path_beyond_asia_ticks").alias("path_beyond_asia_ticks"),
            _f64(blended, "path_extreme_ticks").alias("path_extreme_ticks"),
        )
        .sort([group_col, AVAILABILITY_TS])
    )
    grouped = _path_by_story(path, group_col)
    printed0 = _printed_expr(work)
    base = work.with_row_index("_eid").with_columns(
        printed0.alias("printed_at_entry_ticks"),
        pl.lit(0.0).alias("mfe_beyond_ticks"),
        pl.lit(0.0).alias("mae_beyond_ticks"),
        pl.lit(0.0).alias("realized_beyond_ticks"),
        pl.lit(0.0).alias("mfe_extreme_ticks"),
        pl.lit(True).alias("window_empty"),
        pl.lit(0, dtype=pl.Int64).alias("window_n_bars"),
    )
    mfe_b, mae_b, real_b, mfe_e, empty, n_bars = _fill_capture_windows(
        base, grouped, group_col=group_col
    )
    captured = base.with_columns(
        pl.Series("mfe_beyond_ticks", mfe_b),
        pl.Series("mae_beyond_ticks", mae_b),
        pl.Series("realized_beyond_ticks", real_b),
        pl.Series("mfe_extreme_ticks", mfe_e),
        pl.Series("window_empty", empty),
        pl.Series("window_n_bars", n_bars),
    ).drop("_eid")
    return captured.with_columns(
        (pl.col("printed_at_entry_ticks") / TICKS_PER_NQ_POINT).alias("printed_at_entry_pts"),
        (pl.col("mfe_beyond_ticks") / TICKS_PER_NQ_POINT).alias("mfe_beyond_pts"),
        (pl.col("mae_beyond_ticks") / TICKS_PER_NQ_POINT).alias("mae_beyond_pts"),
        (pl.col("realized_beyond_ticks") / TICKS_PER_NQ_POINT).alias("realized_beyond_pts"),
        (pl.col("mfe_extreme_ticks") / TICKS_PER_NQ_POINT).alias("mfe_extreme_pts"),
    )


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _bucket_summary(entries: pl.DataFrame, *, bucket: str, scope: str) -> dict[str, Any]:
    if entries.height == 0:
        return {
            "bucket": bucket,
            "scope": scope,
            "n_entries": 0,
            "n_y_pos": 0,
            "y_rate": None,
            "n_window_empty": 0,
            "mfe_beyond_ticks_median": None,
            "mfe_beyond_ticks_mean": None,
            "mfe_beyond_pts_median": None,
            "mae_beyond_ticks_median": None,
            "mae_beyond_ticks_mean": None,
            "mae_beyond_pts_median": None,
            "realized_beyond_ticks_median": None,
            "realized_beyond_pts_median": None,
            "mfe_extreme_ticks_median": None,
            "mfe_extreme_pts_median": None,
            "printed_at_entry_ticks_median": None,
            "printed_at_entry_pts_median": None,
        }
    y = _f64_arr(entries, "y") if "y" in entries.columns else np.zeros(0)
    n_y = int(np.isfinite(y).sum()) if y.size else 0
    n_pos = int((y > _ONSET).sum()) if y.size else 0
    mfe = _f64_arr(entries, "mfe_beyond_ticks")
    mae = _f64_arr(entries, "mae_beyond_ticks")
    realized = _f64_arr(entries, "realized_beyond_ticks")
    mfe_e = _f64_arr(entries, "mfe_extreme_ticks")
    printed = _f64_arr(entries, "printed_at_entry_ticks")
    mfe_med = _median(mfe)
    mae_med = _median(mae)
    real_med = _median(realized)
    mfe_e_med = _median(mfe_e)
    printed_med = _median(printed)
    return {
        "bucket": bucket,
        "scope": scope,
        "n_entries": int(entries.height),
        "n_y_pos": n_pos,
        "y_rate": float(n_pos / n_y) if n_y else None,
        "n_window_empty": int(entries["window_empty"].sum())
        if "window_empty" in entries.columns
        else 0,
        "mfe_beyond_ticks_median": mfe_med,
        "mfe_beyond_ticks_mean": _mean(mfe),
        "mfe_beyond_pts_median": None if mfe_med is None else ticks_to_nq_points(mfe_med),
        "mae_beyond_ticks_median": mae_med,
        "mae_beyond_ticks_mean": _mean(mae),
        "mae_beyond_pts_median": None if mae_med is None else ticks_to_nq_points(mae_med),
        "realized_beyond_ticks_median": real_med,
        "realized_beyond_pts_median": None if real_med is None else ticks_to_nq_points(real_med),
        "mfe_extreme_ticks_median": mfe_e_med,
        "mfe_extreme_pts_median": None if mfe_e_med is None else ticks_to_nq_points(mfe_e_med),
        "printed_at_entry_ticks_median": printed_med,
        "printed_at_entry_pts_median": None
        if printed_med is None
        else ticks_to_nq_points(printed_med),
    }


def _summaries_frame(rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def run_causal_entry(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: CausalEntryConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> CausalEntryReport:
    """يلتقط MFE/MAE بعد إطلاق سببي داخل نافذة التسمية فقط."""
    cfg = config or CausalEntryConfig()
    if cfg.min_p < 0.0 or cfg.min_p > 1.0:
        raise ValueError("min_p must be in [0, 1]")
    if cfg.expansion_start_ticks < 1.0 or cfg.min_printed_late_ticks < 1.0:
        raise ValueError("tick thresholds must be >= 1")
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    empty = CausalEntryReport(
        all_entries=pl.DataFrame(),
        late_entries=pl.DataFrame(),
        summaries=pl.DataFrame(),
        diagnostics={"empty": True, "holdout_scored": False},
    )
    if labeled.height == 0 or blended.height == 0:
        return empty
    if SETUP_AVAILABILITY_TS in labeled.columns:
        assert_causal_order(labeled.sort(SETUP_AVAILABILITY_TS)[SETUP_AVAILABILITY_TS].to_list())
    if SETUP_AVAILABILITY_TS in labeled.columns and OUTCOME_AVAILABLE_TS in labeled.columns:
        assert_availability_not_before_event(
            labeled[SETUP_AVAILABILITY_TS].to_list(),
            labeled[OUTCOME_AVAILABLE_TS].to_list(),
        )
    develop, holdout_meta = _exclude_holdout(
        labeled, holdout_cut_ts=holdout_cut_ts, holdout_months=cfg.holdout_months
    )
    if progress is not None:
        progress.op(f"causal_entry labeled_develop={develop.height:,}")
    further = _outcome_frame(develop, cfg.outcome_further)
    scores = _prediction_scores(predictions, cfg.score_col)
    if scores.height == 0 and cfg.score_col in further.columns:
        scores = _prediction_scores(further, cfg.score_col)
    scored = _join_model_p(further, scores)
    primary, scope = _scope_slice(scored, oof_availability_ts=oof_availability_ts)
    all_mask = _entry_mask(primary, cfg=cfg, late=False)
    late_mask = _entry_mask(primary, cfg=cfg, late=True)
    all_src = primary.filter(all_mask)
    late_src = primary.filter(late_mask)
    if progress is not None:
        progress.op(
            f"causal_entry fires={all_src.height:,} "
            f"late_confirmed={late_src.height:,} scope={scope}"
        )
    all_entries = compute_forward_capture(all_src, blended, group_col=cfg.group_col)
    late_entries = compute_forward_capture(late_src, blended, group_col=cfg.group_col)
    all_sum = _bucket_summary(all_entries, bucket="all_fires", scope=scope)
    late_sum = _bucket_summary(late_entries, bucket="late_confirmed", scope=scope)
    summaries = _summaries_frame((all_sum, late_sum))
    lookahead_present = sorted(
        col for col in _LOOKAHEAD_WAVE_COLS if col in labeled.columns or col in blended.columns
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "holdout_scored": False,
        "raw_mbo_not_loaded": True,
        "book_not_reconstructed": True,
        "features_not_recomputed_from_mbo": True,
        "completed_wave_peak_not_used": True,
        "wave_frac_not_used_as_entry_filter": True,
        "completed_wave_60pct_is_not_a_live_filter": True,
        "wave_bin_late_prediction_is_not_a_live_filter": True,
        "live_entry_is_p_at_t": True,
        "lookahead_wave_columns_ignored": lookahead_present,
        "primary_scope": scope,
        "n_further_develop": int(further.height),
        "n_further_primary": int(primary.height),
        "n_all_fires": int(all_entries.height),
        "n_late_confirmed": int(late_entries.height),
        "min_p": float(cfg.min_p),
        "expansion_start_ticks": float(cfg.expansion_start_ticks),
        "min_printed_late_ticks": float(cfg.min_printed_late_ticks),
        "ticks_per_nq_point": TICKS_PER_NQ_POINT,
        "all_fires": all_sum,
        "late_confirmed": late_sum,
        "principles": (
            "live entry is OOF p at t plus live volume/depth — that is causal",
            "completed-wave 60%+ / late_prediction is look-ahead; never a live filter",
            "the 60%+ share is a reassurance statistic after the fact, not a gate",
            "already-printed is a fixed tick threshold at t, not a peak fraction",
            "MFE/MAE/realized are inside the labeled competing-risk window only",
            "remaining-to-peak and completed-wave take-profits are the same leak",
            "holdout rows after holdout_cut_ts are excluded and never scored",
        ),
        **holdout_meta,
    }
    return CausalEntryReport(
        all_entries=all_entries,
        late_entries=late_entries,
        summaries=summaries,
        diagnostics=diagnostics,
    )


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if obj != obj else obj  # noqa: PLR0124  # NaN
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    return str(obj)


def _fmt_pts(ticks: float | None) -> str:
    if ticks is None:
        return "n/a"
    return f"{ticks_to_nq_points(ticks):.2f} pts ({ticks:.1f} ticks)"


def _as_summary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reading(report: CausalEntryReport) -> tuple[str, ...]:
    d = report.diagnostics
    late = _as_summary(d.get("late_confirmed"))
    all_f = _as_summary(d.get("all_fires"))
    lines = [
        (
            f"Late confirmed (printed ≥ {d.get('min_printed_late_ticks')} ticks at t): "
            f"n={late.get('n_entries')} · "
            f"MFE {_fmt_pts(late.get('mfe_beyond_ticks_median'))} · "
            f"drawdown {_fmt_pts(late.get('mae_beyond_ticks_median'))} · "
            f"realized {_fmt_pts(late.get('realized_beyond_ticks_median'))}."
        ),
        (
            f"All causal fires (p≥{d.get('min_p')} and expansion visible): "
            f"n={all_f.get('n_entries')} · "
            f"MFE {_fmt_pts(all_f.get('mfe_beyond_ticks_median'))} · "
            f"drawdown {_fmt_pts(all_f.get('mae_beyond_ticks_median'))} · "
            f"realized {_fmt_pts(all_f.get('realized_beyond_ticks_median'))}."
        ),
        "Capture is inside the labeled outcome window; completed-wave peak is not used.",
        (
            "Before live: enter on p at t. Do not wait to confirm a completed-wave "
            "60%+ / late_prediction bin — that bin is look-ahead."
        ),
    ]
    return tuple(lines)


def render_causal_entry_markdown(report: CausalEntryReport) -> str:
    d = report.diagnostics
    lines = [
        "# Causal entry capture (no completed-wave leakage)",
        "",
        "Live entry = OOF `p` at `t` plus live volume/depth. That is causal.",
        "Already-printed = that fire after a **fixed** tick threshold at `t` "
        f"(default {DEFAULT_MIN_PRINTED_LATE_TICKS:.0f} ticks / "
        f"{ticks_to_nq_points(DEFAULT_MIN_PRINTED_LATE_TICKS):.0f} NQ points). "
        "It is **not** completed-wave 60%+ / `late_prediction`.",
        "MFE / MAE / realized use only bars with "
        "`setup_availability_ts < availability_ts <= outcome_available_ts`.",
        "",
        "## Before live operation",
        "",
        "- Enter when `p` fires with live confirms. Do **not** wait until the wave",
        "  is known to be in the last 60%. At `t` you cannot know whether this fire",
        "  will sit at 30% or 80% of a wave that has not finished.",
        "- Completed-wave `late_prediction` / 60%+ is a reassurance statistic after",
        "  the fact, not a live filter and not a backtest entry gate.",
        "- Remaining-to-peak is the same leak. Size the stop from the causal pullback",
        "  already printed at `t`, not from the future high.",
        "",
        f"- scope={d.get('primary_scope')} · holdout excluded={d.get('holdout_excluded')} · "
        f"scored={d.get('holdout_scored')}",
        f"- all fires={d.get('n_all_fires')} · already printed={d.get('n_late_confirmed')}",
        f"- live_entry_is_p_at_t={d.get('live_entry_is_p_at_t')}",
        f"- completed_wave_60pct_is_not_a_live_filter="
        f"{d.get('completed_wave_60pct_is_not_a_live_filter')}",
        "",
        "## Reading",
        "",
    ]
    for claim in _reading(report):
        lines.append(f"- {claim}")
    lines.extend(["", "## Summaries", ""])
    if report.summaries.height:
        cols = [
            "bucket",
            "n_entries",
            "y_rate",
            "mfe_beyond_pts_median",
            "mae_beyond_pts_median",
            "realized_beyond_pts_median",
            "printed_at_entry_pts_median",
        ]
        present = [c for c in cols if c in report.summaries.columns]
        header = "| " + " | ".join(present) + " |"
        sep = "| " + " | ".join("---" for _ in present) + " |"
        lines.extend([header, sep])
        for row in report.summaries.select(present).iter_rows(named=True):
            cells: list[str] = []
            for col in present:
                val = row[col]
                if isinstance(val, float):
                    cells.append(f"{val:.3f}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["## Principles", ""])
    for item in d.get("principles", ()):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_causal_entry_report(report: CausalEntryReport, output_dir: Path | str) -> Path:
    """يكتب باركيه + JSON + CAUSAL.md — بدون لمس holdout."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.all_entries.height:
        report.all_entries.write_parquet(out / "causal_entry_all.parquet")
    if report.late_entries.height:
        report.late_entries.write_parquet(out / "causal_entry_late.parquet")
    if report.summaries.height:
        report.summaries.write_parquet(out / "causal_entry_summaries.parquet")
    payload = {
        "diagnostics": _jsonable(report.diagnostics),
        "reading": list(_reading(report)),
        "holdout_scored": False,
    }
    (out / "causal_entry.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "CAUSAL.md").write_text(render_causal_entry_markdown(report), encoding="utf-8")
    return out


def run_causal_entry_from_period_dir(
    period_dir: Path | str,
    *,
    config: CausalEntryConfig | None = None,
    progress: ProgressLike | None = None,
) -> CausalEntryReport:
    """يشغّل الالتقاط من مخرجات المرحلة 2 الجاهزة (باركيه فقط)."""
    root = Path(period_dir)
    labeled_path = root / "science_labeled.parquet"
    blended_path = root / "period_blended.parquet"
    if not labeled_path.is_file():
        raise FileNotFoundError(f"science_labeled.parquet not found under {root.resolve()}")
    if not blended_path.is_file():
        raise FileNotFoundError(
            f"period_blended.parquet not found under {root.resolve()} "
            "(forward capture needs the story path; no MBO reload)"
        )
    labeled = pl.read_parquet(labeled_path)
    blended = pl.read_parquet(blended_path)
    assert_not_raw_mbo_stream(labeled, source=str(labeled_path))
    assert_not_raw_mbo_stream(blended, source=str(blended_path))
    predictions: pl.DataFrame | None = None
    oof_ts: list[int] | None = None
    oof_path = root / "oof_predictions.parquet"
    if oof_path.is_file():
        oof = pl.read_parquet(oof_path)
        predictions = oof
        ts_col = SETUP_AVAILABILITY_TS if SETUP_AVAILABILITY_TS in oof.columns else AVAILABILITY_TS
        if ts_col in oof.columns:
            oof_ts = [int(t) for t in oof[ts_col].to_list()]
    cut_ts: int | None = None
    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        diag = summary.get("diagnostics", {})
        science = diag.get("science", diag)
        raw_cut = science.get("holdout_cut_ts", summary.get("holdout_cut_ts"))
        if raw_cut is not None and int(raw_cut) >= 0:
            cut_ts = int(raw_cut)
    if progress is not None:
        progress.op(f"causal_entry from {root}")
    return run_causal_entry(
        labeled,
        blended,
        config=config or CausalEntryConfig(),
        oof_availability_ts=oof_ts,
        holdout_cut_ts=cut_ts,
        predictions=predictions,
        progress=progress,
    )


__all__ = [
    "TICKS_PER_NQ_POINT",
    "CausalEntryConfig",
    "CausalEntryReport",
    "assert_not_raw_mbo_stream",
    "compute_forward_capture",
    "render_causal_entry_markdown",
    "run_causal_entry",
    "run_causal_entry_from_period_dir",
    "ticks_to_nq_points",
    "write_causal_entry_report",
]
