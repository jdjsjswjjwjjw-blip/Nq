"""موقع الإشارة على مسار الامتداد بعد أن يبدأ — لا أول كسر، ولا الموجات القصيرة.

الدخول المستهدف ليس أول تكة من الموجة، وليس الحركة القصيرة. الساعة تبدأ عندما
يصير الامتداد ظاهرًا، ثم تُقاس 20% من مسار الامتداد المتبقي (الموجة الثانية):

  expansion_frac = (extent_at_t - expansion_start) / (peak - expansion_start)

  pre_expansion     أول دفعة، قبل أن يُحسب الامتداد ظاهرًا
  0–20%             early_prediction  ← نافذة الموجة الثانية
  20–40%            early_continuation
  40–60%            mid_wave
  60%+              late_prediction

الذروة وبداية الامتداد نظرة أمامية **للتشخيص فقط**. القرار عند t يستخدم المدى
السببي فقط. موجات قصيرة تُستبعد. لا MBO، لا إعادة بناء، لا لمس holdout.
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
_ONSET = 0.5
_MAJORITY_SHARE = 0.5
_LOW_EARLY_SHARE = 0.2
_GROUP = "_behavior_story_run"
_EPS = 1e-12

WAVE_BIN_COL = "wave_bin"
PRE_EXPANSION_BIN = "pre_expansion"
WAVE_BINS = (
    PRE_EXPANSION_BIN,
    "early_prediction",
    "early_continuation",
    "mid_wave",
    "late_prediction",
)
WAVE_BIN_EDGES = (0.20, 0.40, 0.60)
_DEFAULT_MIN_PEAK = 80.0
_DEFAULT_EXPANSION_START = 16.0
_DEFAULT_MIN_RUN = 32.0


@dataclass(frozen=True, slots=True)
class WavePositionConfig:
    """عتبات موقع الموجة — تشخيص، ليست بوابة تداول."""

    outcome_further: str = "y_path_further_beyond"
    group_col: str = _GROUP
    min_peak_ticks: float = _DEFAULT_MIN_PEAK
    expansion_start_ticks: float = _DEFAULT_EXPANSION_START
    min_expansion_run_ticks: float = _DEFAULT_MIN_RUN
    holdout_months: int | None = 4
    early_prediction: float = WAVE_BIN_EDGES[0]
    early_continuation: float = WAVE_BIN_EDGES[1]
    mid_wave: float = WAVE_BIN_EDGES[2]
    score_col: str = "p_y_path_further_beyond"
    score_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class WavePositionReport:
    """توزيع موقع الإشارة على الموجة المكتملة (تطوير/OOF فقط)."""

    waves: pl.DataFrame
    first_signals: pl.DataFrame
    all_setups: pl.DataFrame
    first_summary: pl.DataFrame
    all_summary: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def assert_not_raw_mbo_stream(frame: pl.DataFrame, *, source: str = "") -> None:
    """يرفض تدفق MBO خام — البحث يقرأ حالات مكتملة فقط."""
    present = _RAW_MBO_SIGNATURE.intersection(frame.columns)
    if present != _RAW_MBO_SIGNATURE:
        return
    where = f" in {source}" if source else ""
    raise ValueError(
        "wave position refuses raw MBO streams "
        f"(found {sorted(present)}{where}); "
        "it reads completed blended states only — no book reconstruction"
    )


def _f64(frame: pl.DataFrame, name: str, default: float = 0.0) -> pl.Expr:
    if name in frame.columns:
        return pl.col(name).cast(pl.Float64).fill_null(default)
    return pl.lit(default, dtype=pl.Float64)


def classify_wave_bin(
    frac: pl.Expr,
    *,
    early: float = 0.20,
    early_cont: float = 0.40,
    mid: float = 0.60,
) -> pl.Expr:
    """يصنّف كسر الموجة المحققة عند الإشارة."""
    return (
        pl.when(frac.is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(frac < float(early))
        .then(pl.lit("early_prediction"))
        .when(frac < float(early_cont))
        .then(pl.lit("early_continuation"))
        .when(frac < float(mid))
        .then(pl.lit("mid_wave"))
        .otherwise(pl.lit("late_prediction"))
    )


def build_wave_geometry(
    blended: pl.DataFrame,
    *,
    group_col: str = _GROUP,
    holdout_cut_ts: int | None = None,
    expansion_start_ticks: float = _DEFAULT_EXPANSION_START,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """ذروة الموجة المكتملة لكل قصة — نظرة أمامية تشخيصية، ليست ميزة.

    الذروة تُحسب من براميل التطوير فقط إذا وُجد ``holdout_cut_ts`` حتى لا
    تُستعار نهاية موجة من الـholdout.
    """
    schema = {
        group_col: pl.Int64(),
        "wave_onset_ts": pl.Int64(),
        "wave_peak_ts": pl.Int64(),
        "wave_end_ts": pl.Int64(),
        "onset_beyond": pl.Float64(),
        "wave_peak_ticks": pl.Float64(),
        "expansion_start_ts": pl.Int64(),
        "expansion_start_extent": pl.Float64(),
        "expansion_run_ticks": pl.Float64(),
        "n_bars": pl.Int64(),
        "n_bars_after_onset": pl.Int64(),
        "wave_censored_by_holdout": pl.Boolean(),
    }
    if blended.height == 0:
        return pl.DataFrame(schema=schema)
    assert_not_raw_mbo_stream(blended, source="blended")
    if AVAILABILITY_TS not in blended.columns:
        raise ValueError("blended requires availability_ts")
    if group_col not in blended.columns:
        raise ValueError(f"blended requires {group_col}")
    work = blended.sort([group_col, AVAILABILITY_TS])
    if holdout_cut_ts is not None and int(holdout_cut_ts) >= 0:
        cut = int(holdout_cut_ts)
        work = work.filter(pl.col(AVAILABILITY_TS) <= cut)
    if work.height == 0:
        return pl.DataFrame(schema=schema)
    assert_causal_order(work[AVAILABILITY_TS].to_list())
    if progress is not None:
        progress.op(f"wave_geometry bars={work.height:,}")
    beyond = _f64(work, "path_beyond_asia_ticks")
    extreme = _f64(work, "path_extreme_ticks")
    extent = pl.max_horizontal(beyond, extreme)
    onset = (beyond > _ONSET) | (_f64(work, "vp_fsm_break") > _ONSET)
    tagged = work.with_columns(
        extent.alias("_extent"),
        onset.alias("_onset"),
    )
    onset_ts = (
        pl.when(pl.col("_onset"))
        .then(pl.col(AVAILABILITY_TS))
        .otherwise(pl.lit(None, dtype=pl.Int64))
        .min()
        .over(group_col)
    )
    tagged = tagged.with_columns(onset_ts.alias("wave_onset_ts"))
    after = tagged.filter(
        pl.col("wave_onset_ts").is_not_null() & (pl.col(AVAILABILITY_TS) >= pl.col("wave_onset_ts"))
    )
    if after.height == 0:
        return pl.DataFrame(schema=schema)
    waves = after.group_by(group_col).agg(
        pl.col("wave_onset_ts").first().alias("wave_onset_ts"),
        pl.col(AVAILABILITY_TS).max().alias("wave_end_ts"),
        pl.col("_extent")
        .filter(pl.col(AVAILABILITY_TS) == pl.col("wave_onset_ts"))
        .first()
        .alias("onset_beyond"),
        pl.col("_extent").max().alias("wave_peak_ticks"),
        pl.col(AVAILABILITY_TS)
        .filter(pl.col("_extent") == pl.col("_extent").max())
        .min()
        .alias("wave_peak_ts"),
        pl.len().cast(pl.Int64).alias("n_bars_after_onset"),
    )
    n_bars = work.group_by(group_col).agg(pl.len().cast(pl.Int64).alias("n_bars"))
    waves = waves.join(n_bars, on=group_col, how="left")
    start_thr = float(expansion_start_ticks)
    expansion = (
        after.filter(pl.col("_extent") >= start_thr)
        .sort([group_col, AVAILABILITY_TS])
        .group_by(group_col, maintain_order=True)
        .agg(
            pl.col(AVAILABILITY_TS).first().alias("expansion_start_ts"),
            pl.col("_extent").first().alias("expansion_start_extent"),
        )
    )
    waves = waves.join(expansion, on=group_col, how="left")
    waves = waves.with_columns(
        (
            pl.col("wave_peak_ticks").cast(pl.Float64)
            - pl.col("expansion_start_extent").cast(pl.Float64).fill_null(0.0)
        )
        .clip(lower_bound=0.0)
        .alias("expansion_run_ticks"),
        pl.lit(False).alias("wave_censored_by_holdout"),
    )
    return waves.select(list(schema.keys()))


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


def _attach_frac(
    setups: pl.DataFrame,
    waves: pl.DataFrame,
    *,
    cfg: WavePositionConfig,
    scope: str,
    role: str,
) -> pl.DataFrame:
    empty_schema = {
        "scope": pl.Utf8(),
        "role": pl.Utf8(),
        cfg.group_col: pl.Int64(),
        SETUP_AVAILABILITY_TS: pl.Int64(),
        "y": pl.Float64(),
        "beyond_at_t": pl.Float64(),
        "extreme_at_t": pl.Float64(),
        "wave_peak_ticks": pl.Float64(),
        "full_wave_frac": pl.Float64(),
        "expansion_start_extent": pl.Float64(),
        "expansion_run_ticks": pl.Float64(),
        "wave_frac": pl.Float64(),
        WAVE_BIN_COL: pl.Utf8(),
        "wave_too_small": pl.Boolean(),
        "pre_expansion": pl.Boolean(),
    }
    if setups.height == 0 or waves.height == 0 or cfg.group_col not in setups.columns:
        return pl.DataFrame(schema=empty_schema)
    beyond = _f64(setups, "path_beyond_asia_ticks")
    extreme = _f64(setups, "path_extreme_ticks")
    extent_t = pl.max_horizontal(beyond, extreme)
    wave_cols = [
        cfg.group_col,
        "wave_onset_ts",
        "wave_peak_ts",
        "wave_peak_ticks",
        "onset_beyond",
        "wave_censored_by_holdout",
    ]
    for extra in ("expansion_start_ts", "expansion_start_extent", "expansion_run_ticks"):
        if extra in waves.columns:
            wave_cols.append(extra)
    joined = setups.with_columns(
        beyond.alias("beyond_at_t"),
        extreme.alias("extreme_at_t"),
        extent_t.alias("_extent_t"),
    ).join(waves.select(wave_cols), on=cfg.group_col, how="inner")
    if "expansion_start_ts" not in joined.columns:
        joined = joined.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("expansion_start_ts"),
            pl.lit(None, dtype=pl.Float64).alias("expansion_start_extent"),
            pl.col("wave_peak_ticks").cast(pl.Float64).alias("expansion_run_ticks"),
        )
    peak = pl.col("wave_peak_ticks").cast(pl.Float64)
    start_ext = pl.col("expansion_start_extent").cast(pl.Float64).fill_null(0.0)
    run = pl.col("expansion_run_ticks").cast(pl.Float64).fill_null(0.0)
    full_frac = (pl.col("_extent_t") / (peak + _EPS)).clip(0.0, 1.0)
    exp_frac = ((pl.col("_extent_t") - start_ext) / (run + _EPS)).clip(0.0, 1.0)
    start_thr = float(cfg.expansion_start_ticks)
    pre = (
        pl.col("expansion_start_ts").is_null()
        | (pl.col("_extent_t") < start_thr)
        | (pl.col(SETUP_AVAILABILITY_TS) < pl.col("expansion_start_ts"))
    )
    too_small = (
        (peak < float(cfg.min_peak_ticks))
        | pl.col("expansion_start_ts").is_null()
        | (run < float(cfg.min_expansion_run_ticks))
    )
    bin_expr = (
        pl.when(too_small)
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pre)
        .then(pl.lit(PRE_EXPANSION_BIN))
        .otherwise(
            classify_wave_bin(
                exp_frac,
                early=cfg.early_prediction,
                early_cont=cfg.early_continuation,
                mid=cfg.mid_wave,
            )
        )
    )
    y_col = pl.col("y") if "y" in joined.columns else pl.lit(None, dtype=pl.Float64)
    return joined.with_columns(
        pl.lit(scope).alias("scope"),
        pl.lit(role).alias("role"),
        full_frac.alias("full_wave_frac"),
        start_ext.alias("expansion_start_extent"),
        run.alias("expansion_run_ticks"),
        pl.when(pre).then(pl.lit(0.0)).otherwise(exp_frac).alias("wave_frac"),
        too_small.alias("wave_too_small"),
        pre.alias("pre_expansion"),
        bin_expr.alias(WAVE_BIN_COL),
        y_col.alias("y"),
    ).select(list(empty_schema.keys()))


def _bin_summary(frame: pl.DataFrame, *, scope: str, role: str, success_only: bool) -> pl.DataFrame:
    schema = {
        "scope": pl.Utf8(),
        "role": pl.Utf8(),
        "success_only": pl.Boolean(),
        WAVE_BIN_COL: pl.Utf8(),
        "n": pl.Int64(),
        "share": pl.Float64(),
        "mean_wave_frac": pl.Float64(),
        "median_wave_frac": pl.Float64(),
        "mean_beyond_at_t": pl.Float64(),
        "mean_peak": pl.Float64(),
        "pos_rate": pl.Float64(),
    }
    work = frame
    if "scope" in work.columns:
        work = work.filter(pl.col("scope") == scope)
    if "role" in work.columns:
        work = work.filter(pl.col("role") == role)
    work = work.filter(~pl.col("wave_too_small") & pl.col(WAVE_BIN_COL).is_not_null())
    if success_only and "y" in work.columns:
        work = work.filter(pl.col("y") > _ONSET)
    if work.height == 0:
        n_bins = len(WAVE_BINS)
        return pl.DataFrame(
            {
                "scope": [scope] * n_bins,
                "role": [role] * n_bins,
                "success_only": [success_only] * n_bins,
                WAVE_BIN_COL: list(WAVE_BINS),
                "n": [0] * n_bins,
                "share": [0.0] * n_bins,
                "mean_wave_frac": [0.0] * n_bins,
                "median_wave_frac": [0.0] * n_bins,
                "mean_beyond_at_t": [0.0] * n_bins,
                "mean_peak": [0.0] * n_bins,
                "pos_rate": [0.0] * n_bins,
            }
        ).select(list(schema.keys()))
    total = float(work.height)
    stats = work.group_by(WAVE_BIN_COL).agg(
        pl.len().cast(pl.Int64).alias("n"),
        pl.col("wave_frac").mean().alias("mean_wave_frac"),
        pl.col("wave_frac").median().alias("median_wave_frac"),
        pl.col("beyond_at_t").mean().alias("mean_beyond_at_t"),
        pl.col("wave_peak_ticks").mean().alias("mean_peak"),
        pl.col("y").mean().alias("pos_rate"),
    )
    stats = stats.with_columns(
        pl.lit(scope).alias("scope"),
        pl.lit(role).alias("role"),
        pl.lit(success_only).alias("success_only"),
        (pl.col("n") / total).alias("share"),
    )
    present = set(stats[WAVE_BIN_COL].to_list())
    missing = [b for b in WAVE_BINS if b not in present]
    if missing:
        zeros = pl.DataFrame(
            {
                WAVE_BIN_COL: missing,
                "n": [0] * len(missing),
                "mean_wave_frac": [0.0] * len(missing),
                "median_wave_frac": [0.0] * len(missing),
                "mean_beyond_at_t": [0.0] * len(missing),
                "mean_peak": [0.0] * len(missing),
                "pos_rate": [0.0] * len(missing),
                "scope": [scope] * len(missing),
                "role": [role] * len(missing),
                "success_only": [success_only] * len(missing),
                "share": [0.0] * len(missing),
            }
        )
        stats = pl.concat([stats, zeros], how="diagonal_relaxed")
    return stats.select(list(schema.keys())).sort(WAVE_BIN_COL)


def _first_per_story(setups: pl.DataFrame, group_col: str) -> pl.DataFrame:
    if setups.height == 0 or group_col not in setups.columns:
        return setups
    return setups.sort(SETUP_AVAILABILITY_TS).group_by(group_col, maintain_order=True).first()


def _first_success_per_story(setups: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """أول امتداد ناجح في القصة — ليس أول صف مُسمّى ثم تصفيته."""
    if setups.height == 0 or group_col not in setups.columns or "y" not in setups.columns:
        return setups.head(0)
    ok = setups.filter(pl.col("y") > _ONSET)
    if ok.height == 0:
        return ok
    return ok.sort(SETUP_AVAILABILITY_TS).group_by(group_col, maintain_order=True).first()


def _prediction_scores(predictions: pl.DataFrame | None, score_col: str) -> pl.DataFrame:
    empty = pl.DataFrame(schema={SETUP_AVAILABILITY_TS: pl.Int64(), "model_p": pl.Float64()})
    if predictions is None or predictions.height == 0 or score_col not in predictions.columns:
        return empty
    ts_col = (
        SETUP_AVAILABILITY_TS if SETUP_AVAILABILITY_TS in predictions.columns else AVAILABILITY_TS
    )
    if ts_col not in predictions.columns:
        return empty
    return (
        predictions.select(
            pl.col(ts_col).cast(pl.Int64).alias(SETUP_AVAILABILITY_TS),
            pl.col(score_col).cast(pl.Float64).alias("model_p"),
        )
        .filter(pl.col("model_p").is_finite())
        .unique(subset=[SETUP_AVAILABILITY_TS], keep="first")
    )


def _join_model_p(setups: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    if setups.height == 0:
        return setups
    if scores.height == 0:
        return setups.with_columns(pl.lit(None, dtype=pl.Float64).alias("model_p"))
    return setups.join(scores, on=SETUP_AVAILABILITY_TS, how="left")


def _first_model_per_story(
    setups: pl.DataFrame,
    *,
    scope: str,
    group_col: str,
    threshold: float,
) -> pl.DataFrame:
    if setups.height == 0 or "model_p" not in setups.columns or group_col not in setups.columns:
        return setups.head(0)
    work = setups.filter(
        (pl.col("scope") == scope)
        & (~pl.col("wave_too_small"))
        & pl.col("model_p").is_finite()
        & (pl.col("model_p") >= float(threshold))
    )
    if work.height == 0:
        return work
    return (
        work.sort(SETUP_AVAILABILITY_TS)
        .group_by(group_col, maintain_order=True)
        .first()
        .with_columns(pl.lit("first_model").alias("role"))
    )


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


def run_wave_position(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: WavePositionConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> WavePositionReport:
    """يحسب كسر الموجة المحققة عند أول إشارة وعند كل إعداد ناجح."""
    cfg = config or WavePositionConfig()
    assert_not_raw_mbo_stream(labeled, source="labeled")
    empty = WavePositionReport(
        waves=pl.DataFrame(),
        first_signals=pl.DataFrame(),
        all_setups=pl.DataFrame(),
        first_summary=pl.DataFrame(),
        all_summary=pl.DataFrame(),
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
    cut = holdout_meta.get("holdout_cut_ts", holdout_cut_ts)
    cut_i = int(cut) if cut is not None and int(cut) >= 0 else None
    if progress is not None:
        progress.op(f"wave_position labeled_develop={develop.height:,}")
    waves = build_wave_geometry(
        blended,
        group_col=cfg.group_col,
        holdout_cut_ts=cut_i,
        expansion_start_ticks=cfg.expansion_start_ticks,
        progress=progress,
    )
    further = _outcome_frame(develop, cfg.outcome_further)
    primary, scope = _scope_slice(further, oof_availability_ts=oof_availability_ts)
    first_dev = _first_per_story(further, cfg.group_col)
    first_pri = _first_per_story(primary, cfg.group_col)
    first_ok_dev = _first_success_per_story(further, cfg.group_col)
    first_ok_pri = _first_success_per_story(primary, cfg.group_col)
    first_parts = [
        _attach_frac(first_dev, waves, cfg=cfg, scope="develop", role="first_signal"),
        _attach_frac(first_ok_dev, waves, cfg=cfg, scope="develop", role="first_success"),
    ]
    all_parts = [_attach_frac(further, waves, cfg=cfg, scope="develop", role="all_setups")]
    if scope != "develop":
        first_parts.extend(
            [
                _attach_frac(first_pri, waves, cfg=cfg, scope=scope, role="first_signal"),
                _attach_frac(first_ok_pri, waves, cfg=cfg, scope=scope, role="first_success"),
            ]
        )
        all_parts.append(_attach_frac(primary, waves, cfg=cfg, scope=scope, role="all_setups"))
    first_signals = pl.concat(first_parts, how="diagonal_relaxed")
    all_setups = _join_model_p(
        pl.concat(all_parts, how="diagonal_relaxed"),
        _prediction_scores(predictions, cfg.score_col),
    )
    model_first_parts = [
        _first_model_per_story(
            all_setups,
            scope="develop",
            group_col=cfg.group_col,
            threshold=cfg.score_threshold,
        )
    ]
    if scope != "develop":
        model_first_parts.append(
            _first_model_per_story(
                all_setups,
                scope=scope,
                group_col=cfg.group_col,
                threshold=cfg.score_threshold,
            )
        )
    model_first = pl.concat(model_first_parts, how="diagonal_relaxed")
    if model_first.height:
        first_signals = pl.concat([first_signals, model_first], how="diagonal_relaxed")
    first_summary = pl.concat(
        _first_summary_parts(first_signals, scope=scope), how="diagonal_relaxed"
    )
    all_model = _all_model_rows(all_setups, threshold=cfg.score_threshold)
    all_summary = pl.concat(
        [
            _bin_summary(all_setups, scope="develop", role="all_setups", success_only=False),
            _bin_summary(all_setups, scope=scope, role="all_setups", success_only=True),
            _bin_summary(all_model, scope=scope, role="all_model", success_only=False),
        ],
        how="diagonal_relaxed",
    )
    first_all = _role_rows(first_signals, scope=scope, role="first_signal")
    first_ok = _role_rows(first_signals, scope=scope, role="first_success")
    first_model = _role_rows(first_signals, scope=scope, role="first_model")
    all_ok = _role_rows(all_setups, scope=scope, role="all_setups").filter(pl.col("y") > _ONSET)
    all_model_ok = all_model.filter((pl.col("scope") == scope) & (~pl.col("wave_too_small")))
    first_y = _mean_y(first_all)
    diagnostics: dict[str, Any] = {
        "empty": False,
        "holdout_scored": False,
        "raw_mbo_not_loaded": True,
        "book_not_reconstructed": True,
        "features_not_recomputed_from_mbo": True,
        "wave_peak_is_diagnostic_lookahead_not_a_feature": True,
        "primary_scope": scope,
        "n_waves": int(waves.height),
        "n_large_waves": int(
            waves.filter(
                (pl.col("wave_peak_ticks") >= float(cfg.min_peak_ticks))
                & pl.col("expansion_start_ts").is_not_null()
                & (pl.col("expansion_run_ticks") >= float(cfg.min_expansion_run_ticks))
            ).height
            if waves.height and "expansion_start_ts" in waves.columns
            else 0
        ),
        "n_further_develop": int(further.height),
        "n_further_primary": int(primary.height),
        "n_first_primary": int(first_all.height),
        "n_first_success_primary": int(first_ok.height),
        "n_all_success_primary": int(all_ok.height),
        "n_first_model_primary": int(first_model.height),
        "n_all_model_primary": int(all_model_ok.height),
        "min_peak_ticks": float(cfg.min_peak_ticks),
        "expansion_start_ticks": float(cfg.expansion_start_ticks),
        "min_expansion_run_ticks": float(cfg.min_expansion_run_ticks),
        "score_threshold": float(cfg.score_threshold),
        "mean_y_first_primary": first_y,
        "median_first_frac": _median_frac(first_all),
        "median_first_success_frac": _median_frac(first_ok),
        "median_first_model_frac": _median_frac(first_model),
        "median_all_success_frac": _median_frac(all_ok),
        "median_all_model_frac": _median_frac(all_model_ok),
        "share_first_early_prediction": _share(first_all, "early_prediction"),
        "share_first_late_prediction": _share(first_all, "late_prediction"),
        "share_first_pre_expansion": _share(first_all, PRE_EXPANSION_BIN),
        "share_first_success_early_prediction": _share(first_ok, "early_prediction"),
        "share_first_success_late_prediction": _share(first_ok, "late_prediction"),
        "share_first_success_pre_expansion": _share(first_ok, PRE_EXPANSION_BIN),
        "share_first_model_early_prediction": _share(first_model, "early_prediction"),
        "share_first_model_late_prediction": _share(first_model, "late_prediction"),
        "share_first_model_pre_expansion": _share(first_model, PRE_EXPANSION_BIN),
        "share_all_success_early_prediction": _share(all_ok, "early_prediction"),
        "share_all_success_late_prediction": _share(all_ok, "late_prediction"),
        "share_all_success_pre_expansion": _share(all_ok, PRE_EXPANSION_BIN),
        "share_all_model_early_prediction": _share(all_model_ok, "early_prediction"),
        "share_all_model_late_prediction": _share(all_model_ok, "late_prediction"),
        "share_all_model_pre_expansion": _share(all_model_ok, PRE_EXPANSION_BIN),
        "principles": (
            "clock starts at visible expansion, not the first poke",
            "early_prediction is the first 20% of the expansion run — the second-leg window",
            "pre_expansion is the first impulse; short waves (peak < min_peak) are dropped",
            "wave peak and expansion start are look-ahead used only to score where the signal sat",
            "extent at t is causal (path_beyond / path_extreme known at setup_availability_ts)",
            "holdout bars never enter the peak and holdout setups are never scored",
            "first_signal is the earliest labeled setup in the story (any y)",
            "first_success is the earliest successful further-beyond (y=1) in the story",
            "first_model is the earliest OOF p>=threshold in the story (the predictive signal)",
            "all_setups are the further-beyond labels that produced the continuation skill",
            "all_model are OOF rows with p>=threshold — where the model actually fires",
            "0-20 early_prediction · 20-40 early_continuation · 40-60 mid_wave · 60+ late",
        ),
        **holdout_meta,
    }
    return WavePositionReport(
        waves=waves,
        first_signals=first_signals,
        all_setups=all_setups,
        first_summary=first_summary,
        all_summary=all_summary,
        diagnostics=diagnostics,
    )


def _role_rows(frame: pl.DataFrame, *, scope: str, role: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col("scope") == scope) & (~pl.col("wave_too_small")) & (pl.col("role") == role)
    )


def _mean_y(frame: pl.DataFrame) -> float | None:
    if frame.height == 0 or "y" not in frame.columns:
        return None
    y_arr = frame["y"].drop_nulls().to_numpy().astype(np.float64)
    if y_arr.size == 0:
        return None
    return float(np.mean(y_arr))


def _concat_scope(parts: list[pl.DataFrame], extra: list[pl.DataFrame], *, scope: str) -> None:
    if scope != "develop":
        parts.extend(extra)


def _first_summary_parts(first_signals: pl.DataFrame, *, scope: str) -> list[pl.DataFrame]:
    parts = [
        _bin_summary(first_signals, scope="develop", role="first_signal", success_only=False),
        _bin_summary(first_signals, scope="develop", role="first_signal", success_only=True),
        _bin_summary(first_signals, scope="develop", role="first_success", success_only=True),
        _bin_summary(first_signals, scope="develop", role="first_model", success_only=False),
    ]
    _concat_scope(
        parts,
        [
            _bin_summary(first_signals, scope=scope, role="first_signal", success_only=False),
            _bin_summary(first_signals, scope=scope, role="first_signal", success_only=True),
            _bin_summary(first_signals, scope=scope, role="first_success", success_only=True),
            _bin_summary(first_signals, scope=scope, role="first_model", success_only=False),
        ],
        scope=scope,
    )
    return parts


def _all_model_rows(all_setups: pl.DataFrame, *, threshold: float) -> pl.DataFrame:
    if "model_p" not in all_setups.columns:
        return all_setups.head(0)
    return all_setups.filter(
        pl.col("model_p").is_finite() & (pl.col("model_p") >= float(threshold))
    ).with_columns(pl.lit("all_model").alias("role"))


def _share(frame: pl.DataFrame, bin_name: str) -> float:
    if frame.height == 0 or WAVE_BIN_COL not in frame.columns:
        return 0.0
    return float(frame.filter(pl.col(WAVE_BIN_COL) == bin_name).height) / float(frame.height)


def _median_frac(frame: pl.DataFrame) -> float | None:
    if frame.height == 0 or "wave_frac" not in frame.columns:
        return None
    values = frame["wave_frac"].drop_nulls().to_numpy().astype(np.float64)
    if values.size == 0:
        return None
    return float(np.median(values))


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


def _md_table(frame: pl.DataFrame, columns: Sequence[str]) -> list[str]:
    cols = [c for c in columns if c in frame.columns]
    if not cols or frame.height == 0:
        return ["_(empty)_", ""]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in frame.select(cols).iter_rows(named=True):
        cells: list[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _share_claim(early: float, late: float, *, early_msg: str, late_msg: str) -> str | None:
    if early >= _MAJORITY_SHARE:
        return early_msg.format(p=f"{early:.0%}")
    if late >= _MAJORITY_SHARE:
        return late_msg.format(p=f"{late:.0%}")
    return None


def _median_claim(value: Any, template: str) -> str | None:
    if value is None:
        return None
    return template.format(p=f"{float(value):.0%}")


def _reading(report: WavePositionReport) -> tuple[str, ...]:
    d = report.diagnostics
    scope = str(d.get("primary_scope", "develop"))
    first_y = d.get("mean_y_first_primary")
    claims: list[str] = []
    for claim in (
        _median_claim(
            d.get("median_first_frac"),
            "first labeled setup sits at median {p} of the expansion run "
            f"(after visible expansion; scope={scope}).",
        ),
        _median_claim(
            d.get("median_first_success_frac"),
            "first successful further-beyond sits at median {p} of the expansion run.",
        ),
        _median_claim(
            d.get("median_first_model_frac"),
            "first OOF model fire (p>=threshold) sits at median {p} of the expansion run.",
        ),
        _median_claim(
            d.get("median_all_success_frac"),
            "successful further-beyond setups sit at median {p} of the expansion run.",
        ),
        _median_claim(
            d.get("median_all_model_frac"),
            "all OOF model fires sit at median {p} of the expansion run.",
        ),
        _share_claim(
            float(d.get("share_first_early_prediction") or 0.0),
            float(d.get("share_first_late_prediction") or 0.0),
            early_msg=(
                "first setups are mostly early_prediction ({p}) — "
                "the second-leg 20% window after expansion has started."
            ),
            late_msg=(
                "first setups are mostly late_prediction ({p}) — "
                "already past 60% of the expansion run."
            ),
        ),
        _share_claim(
            float(d.get("share_first_success_early_prediction") or 0.0),
            float(d.get("share_first_success_late_prediction") or 0.0),
            early_msg="first successful labels are mostly early_prediction ({p}).",
            late_msg="first successful labels are mostly late_prediction ({p}).",
        ),
        _share_claim(
            float(d.get("share_first_model_early_prediction") or 0.0),
            float(d.get("share_first_model_late_prediction") or 0.0),
            early_msg=(
                "first model fires are mostly early_prediction ({p}) — "
                "the signal sits in the first 20% of the expansion run."
            ),
            late_msg="first model fires are mostly late_prediction ({p}).",
        ),
    ):
        if claim:
            claims.append(claim)
    pre_model = float(d.get("share_first_model_pre_expansion") or 0.0)
    if pre_model >= _MAJORITY_SHARE:
        claims.append(
            f"first model fires are mostly pre_expansion ({pre_model:.0%}) — "
            "still the first impulse, not the second-leg 20% after expansion."
        )
    if first_y is not None:
        y_txt = f"those first setups have mean y_path_further_beyond={float(first_y):.2f}"
        if float(first_y) < _ONSET:
            claims.insert(
                1,
                y_txt
                + " (the first labeled further-beyond is usually not a successful extension).",
            )
        else:
            claims.insert(1, y_txt + ".")
    all_early = float(d.get("share_all_success_early_prediction") or 0.0)
    all_late = float(d.get("share_all_success_late_prediction") or 0.0)
    all_model_early = float(d.get("share_all_model_early_prediction") or 0.0)
    all_model_late = float(d.get("share_all_model_late_prediction") or 0.0)
    if all_late >= _MAJORITY_SHARE and all_early < _LOW_EARLY_SHARE:
        claims.append(
            f"the continuation-skill rows are late ({all_late:.0%} at 60%+ of the expansion run; "
            f"second-leg 20%={all_early:.0%})."
        )
    if all_model_late >= _MAJORITY_SHARE and all_model_early < _LOW_EARLY_SHARE:
        claims.append(
            f"model fires overall are late ({all_model_late:.0%} at 60%+ of the expansion run; "
            f"second-leg 20%={all_model_early:.0%})."
        )
    if not claims:
        claims.append("insufficient completed waves to locate the signal on the path.")
    return tuple(claims)


def render_wave_position_markdown(report: WavePositionReport) -> str:
    d = report.diagnostics
    scope = str(d.get("primary_scope", "develop"))
    lines = [
        "# wave position — second-leg entry after expansion (large waves)",
        "",
        "Clock starts when expansion is visible, not at the first poke.",
        "early_prediction = first 20% of the expansion run (the second leg).",
        "pre_expansion = first impulse. Short waves are dropped.",
        "Diagnostic look-ahead for peak and expansion start only. Features at t stay causal.",
        "Holdout never scored. No MBO reload. No book reconstruction.",
        "",
        "Bins after expansion start: 0–20% early_prediction · 20–40% early_continuation · "
        "40–60% mid_wave · 60%+ late_prediction · plus pre_expansion.",
        "",
        f"- primary scope: `{scope}`",
        f"- waves: {d.get('n_waves')} · large waves: {d.get('n_large_waves')}",
        f"- expansion start ticks: {d.get('expansion_start_ticks')} · "
        f"min peak: {d.get('min_peak_ticks')} · min run: {d.get('min_expansion_run_ticks')}",
        f"- first labeled setups: {d.get('n_first_primary')} "
        f"(mean y={d.get('mean_y_first_primary')})",
        f"- first successful further-beyond: {d.get('n_first_success_primary')}",
        f"- first model fire p>={d.get('score_threshold')}: {d.get('n_first_model_primary')}",
        f"- successful further-beyond setups: {d.get('n_all_success_primary')}",
        f"- model fires p>={d.get('score_threshold')}: {d.get('n_all_model_primary')}",
        f"- holdout excluded={d.get('holdout_excluded')} · scored={d.get('holdout_scored')}",
        "",
        "## Isolation",
        "",
    ]
    for p in d.get("principles", ()):
        lines.append(f"- {p}")
    lines.extend(["", "## Reading", ""])
    for claim in _reading(report):
        lines.append(f"- {claim}")
    first = report.first_summary
    if first.height and "success_only" in first.columns:
        first_all_tbl = first.filter(
            (pl.col("scope") == scope)
            & (~pl.col("success_only"))
            & (pl.col("role") == "first_signal")
        )
        first_ok_tbl = first.filter(
            (pl.col("scope") == scope)
            & (pl.col("success_only"))
            & (pl.col("role") == "first_success")
        )
        first_model_tbl = first.filter(
            (pl.col("scope") == scope)
            & (~pl.col("success_only"))
            & (pl.col("role") == "first_model")
        )
    else:
        first_all_tbl = first
        first_ok_tbl = first
        first_model_tbl = first
    lines.extend(["", "## First labeled setup in the story (any y)", ""])
    lines.extend(
        _md_table(
            first_all_tbl,
            (
                WAVE_BIN_COL,
                "n",
                "share",
                "mean_wave_frac",
                "median_wave_frac",
                "mean_beyond_at_t",
                "mean_peak",
                "pos_rate",
            ),
        )
    )
    lines.extend(["## First successful further-beyond in the story", ""])
    lines.extend(
        _md_table(
            first_ok_tbl,
            (
                WAVE_BIN_COL,
                "n",
                "share",
                "mean_wave_frac",
                "median_wave_frac",
                "mean_beyond_at_t",
                "mean_peak",
            ),
        )
    )
    lines.extend(["## First OOF model fire in the story (p>=threshold)", ""])
    lines.extend(
        _md_table(
            first_model_tbl,
            (
                WAVE_BIN_COL,
                "n",
                "share",
                "mean_wave_frac",
                "median_wave_frac",
                "mean_beyond_at_t",
                "mean_peak",
                "pos_rate",
            ),
        )
    )
    all_s = report.all_summary
    all_model_tbl = all_s
    if all_s.height and "success_only" in all_s.columns:
        if "role" in all_s.columns:
            all_model_tbl = all_s.filter(
                (pl.col("scope") == scope) & (pl.col("role") == "all_model")
            )
            all_s = all_s.filter(
                (pl.col("scope") == scope)
                & (pl.col("success_only"))
                & (pl.col("role") == "all_setups")
            )
        else:
            all_s = all_s.filter((pl.col("scope") == scope) & (pl.col("success_only")))
    lines.extend(["## All successful further-beyond setups (continuation skill rows)", ""])
    lines.extend(
        _md_table(
            all_s,
            (
                WAVE_BIN_COL,
                "n",
                "share",
                "mean_wave_frac",
                "median_wave_frac",
                "mean_beyond_at_t",
                "mean_peak",
            ),
        )
    )
    lines.extend(["## All OOF model fires (p>=threshold)", ""])
    lines.extend(
        _md_table(
            all_model_tbl,
            (
                WAVE_BIN_COL,
                "n",
                "share",
                "mean_wave_frac",
                "median_wave_frac",
                "mean_beyond_at_t",
                "mean_peak",
                "pos_rate",
            ),
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def write_wave_position_report(report: WavePositionReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.waves.height:
        report.waves.write_parquet(out / "wave_geometry.parquet")
    if report.first_signals.height:
        report.first_signals.write_parquet(out / "wave_first_signals.parquet")
    if report.all_setups.height:
        report.all_setups.write_parquet(out / "wave_all_setups.parquet")
    if report.first_summary.height:
        report.first_summary.write_parquet(out / "wave_first_summary.parquet")
    if report.all_summary.height:
        report.all_summary.write_parquet(out / "wave_all_summary.parquet")
    payload = {
        "diagnostics": _jsonable(report.diagnostics),
        "reading": list(_reading(report)),
        "holdout_scored": False,
    }
    (out / "wave_position.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "WAVE.md").write_text(render_wave_position_markdown(report), encoding="utf-8")
    return out


def run_wave_position_from_period_dir(
    period_dir: Path | str,
    *,
    config: WavePositionConfig | None = None,
    progress: ProgressLike | None = None,
) -> WavePositionReport:
    """يشغّل التشخيص من مخرجات المرحلة 2 (باركيه فقط)."""
    root = Path(period_dir)
    labeled_path = root / "science_labeled.parquet"
    blended_path = root / "period_blended.parquet"
    if not labeled_path.is_file():
        raise FileNotFoundError(f"science_labeled.parquet not found under {root.resolve()}")
    if not blended_path.is_file():
        raise FileNotFoundError(
            f"period_blended.parquet not found under {root.resolve()} "
            "(wave peak needs the full story path; no MBO reload)"
        )
    labeled = pl.read_parquet(labeled_path)
    blended = pl.read_parquet(blended_path)
    oof_ts: list[int] | None = None
    predictions: pl.DataFrame | None = None
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
    return run_wave_position(
        labeled,
        blended,
        config=config or WavePositionConfig(),
        oof_availability_ts=oof_ts,
        holdout_cut_ts=cut_ts,
        predictions=predictions,
        progress=progress,
    )


__all__ = [
    "PRE_EXPANSION_BIN",
    "WAVE_BINS",
    "WAVE_BIN_COL",
    "WavePositionConfig",
    "WavePositionReport",
    "assert_not_raw_mbo_stream",
    "build_wave_geometry",
    "classify_wave_bin",
    "render_wave_position_markdown",
    "run_wave_position",
    "run_wave_position_from_period_dir",
    "write_wave_position_report",
]
