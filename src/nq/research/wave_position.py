"""موقع أول إشارة تنبؤية نسبةً إلى الموجة المكتملة.

نتائج الامتداد قالت إن المهارة الحالية استمرار حركة بدأت، لا إثبات أنها تمسك
أول 20% من الموجة. هذه الطبقة تقيس ذلك صراحة:

  wave_frac = extent_at_signal / completed_wave_peak

  0–20%  early_prediction
  20–40% early_continuation
  40–60% mid_wave
  60%+   late_prediction

``completed_wave_peak`` نظرة إلى الأمام **للتشخيص فقط** (مثل Y)، ليست ميزة.
القرار عند ``setup_availability_ts`` يستخدم فقط المدى السببي حتى t.
لا MBO، لا إعادة بناء، لا لمس holdout.
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
WAVE_BINS = (
    "early_prediction",
    "early_continuation",
    "mid_wave",
    "late_prediction",
)
WAVE_BIN_EDGES = (0.20, 0.40, 0.60)


@dataclass(frozen=True, slots=True)
class WavePositionConfig:
    """عتبات موقع الموجة — تشخيص، ليست بوابة تداول."""

    outcome_further: str = "y_path_further_beyond"
    group_col: str = _GROUP
    min_peak_ticks: float = 8.0
    holdout_months: int | None = 4
    early_prediction: float = WAVE_BIN_EDGES[0]
    early_continuation: float = WAVE_BIN_EDGES[1]
    mid_wave: float = WAVE_BIN_EDGES[2]


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
    return waves.with_columns(pl.lit(False).alias("wave_censored_by_holdout")).select(
        list(schema.keys())
    )


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
    if setups.height == 0 or waves.height == 0 or cfg.group_col not in setups.columns:
        return pl.DataFrame(
            schema={
                "scope": pl.Utf8(),
                "role": pl.Utf8(),
                cfg.group_col: pl.Int64(),
                SETUP_AVAILABILITY_TS: pl.Int64(),
                "y": pl.Float64(),
                "beyond_at_t": pl.Float64(),
                "extreme_at_t": pl.Float64(),
                "wave_peak_ticks": pl.Float64(),
                "wave_frac": pl.Float64(),
                WAVE_BIN_COL: pl.Utf8(),
                "wave_too_small": pl.Boolean(),
            }
        )
    beyond = _f64(setups, "path_beyond_asia_ticks")
    extreme = _f64(setups, "path_extreme_ticks")
    extent_t = pl.max_horizontal(beyond, extreme)
    joined = setups.with_columns(
        beyond.alias("beyond_at_t"),
        extreme.alias("extreme_at_t"),
        extent_t.alias("_extent_t"),
    ).join(
        waves.select(
            cfg.group_col,
            "wave_onset_ts",
            "wave_peak_ts",
            "wave_peak_ticks",
            "onset_beyond",
            "wave_censored_by_holdout",
        ),
        on=cfg.group_col,
        how="inner",
    )
    peak = pl.col("wave_peak_ticks").cast(pl.Float64)
    frac = (pl.col("_extent_t") / (peak + _EPS)).clip(0.0, 1.0)
    too_small = peak < float(cfg.min_peak_ticks)
    bin_expr = (
        pl.when(too_small)
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(
            classify_wave_bin(
                frac,
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
        frac.alias("wave_frac"),
        too_small.alias("wave_too_small"),
        bin_expr.alias(WAVE_BIN_COL),
        y_col.alias("y"),
    ).select(
        "scope",
        "role",
        cfg.group_col,
        SETUP_AVAILABILITY_TS,
        "y",
        "beyond_at_t",
        "extreme_at_t",
        "wave_peak_ticks",
        "wave_frac",
        WAVE_BIN_COL,
        "wave_too_small",
    )


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
        return pl.DataFrame(schema=schema)
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
        progress=progress,
    )
    further = _outcome_frame(develop, cfg.outcome_further)
    primary, scope = _scope_slice(further, oof_availability_ts=oof_availability_ts)
    first_dev = _first_per_story(further, cfg.group_col)
    first_pri = _first_per_story(primary, cfg.group_col)
    first_parts = [_attach_frac(first_dev, waves, cfg=cfg, scope="develop", role="first_signal")]
    all_parts = [_attach_frac(further, waves, cfg=cfg, scope="develop", role="all_setups")]
    if scope != "develop":
        first_parts.append(
            _attach_frac(first_pri, waves, cfg=cfg, scope=scope, role="first_signal")
        )
        all_parts.append(_attach_frac(primary, waves, cfg=cfg, scope=scope, role="all_setups"))
    first_signals = pl.concat(first_parts, how="diagonal_relaxed")
    all_setups = pl.concat(all_parts, how="diagonal_relaxed")
    first_summary = pl.concat(
        [
            _bin_summary(first_signals, scope="develop", role="first_signal", success_only=False),
            _bin_summary(first_signals, scope=scope, role="first_signal", success_only=True),
        ],
        how="diagonal_relaxed",
    )
    all_summary = pl.concat(
        [
            _bin_summary(all_setups, scope="develop", role="all_setups", success_only=False),
            _bin_summary(all_setups, scope=scope, role="all_setups", success_only=True),
        ],
        how="diagonal_relaxed",
    )
    first_ok = first_signals.filter(
        (pl.col("scope") == scope) & (~pl.col("wave_too_small")) & (pl.col("y") > _ONSET)
    )
    all_ok = all_setups.filter(
        (pl.col("scope") == scope) & (~pl.col("wave_too_small")) & (pl.col("y") > _ONSET)
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "holdout_scored": False,
        "raw_mbo_not_loaded": True,
        "book_not_reconstructed": True,
        "features_not_recomputed_from_mbo": True,
        "wave_peak_is_diagnostic_lookahead_not_a_feature": True,
        "primary_scope": scope,
        "n_waves": int(waves.height),
        "n_further_develop": int(further.height),
        "n_further_primary": int(primary.height),
        "n_first_success_primary": int(first_ok.height),
        "n_all_success_primary": int(all_ok.height),
        "min_peak_ticks": float(cfg.min_peak_ticks),
        "median_first_success_frac": _median_frac(first_ok),
        "median_all_success_frac": _median_frac(all_ok),
        "share_first_success_early_prediction": _share(first_ok, "early_prediction"),
        "share_first_success_late_prediction": _share(first_ok, "late_prediction"),
        "share_all_success_early_prediction": _share(all_ok, "early_prediction"),
        "share_all_success_late_prediction": _share(all_ok, "late_prediction"),
        "principles": (
            "wave peak is look-ahead used only to score where the signal sat; not a feature",
            "extent at t is causal (path_beyond / path_extreme known at setup_availability_ts)",
            "holdout bars never enter the peak and holdout setups are never scored",
            "first_signal is the earliest labeled setup in the story",
            "all_setups are the further-beyond labels that produced the continuation skill",
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


def _reading(report: WavePositionReport) -> tuple[str, ...]:
    d = report.diagnostics
    scope = str(d.get("primary_scope", "develop"))
    first_early = float(d.get("share_first_success_early_prediction") or 0.0)
    first_late = float(d.get("share_first_success_late_prediction") or 0.0)
    all_early = float(d.get("share_all_success_early_prediction") or 0.0)
    all_late = float(d.get("share_all_success_late_prediction") or 0.0)
    med_first = d.get("median_first_success_frac")
    med_all = d.get("median_all_success_frac")
    claims: list[str] = []
    if med_first is not None:
        claims.append(
            f"first successful signal sits at median {float(med_first):.0%} of the completed wave "
            f"(scope={scope})."
        )
    if med_all is not None:
        claims.append(
            "successful further-beyond setups sit at median "
            f"{float(med_all):.0%} of the completed wave."
        )
    if first_early >= _MAJORITY_SHARE:
        claims.append(
            f"first signals are mostly early_prediction ({first_early:.0%}) — "
            "the system can catch the start of the wave."
        )
    elif first_late >= _MAJORITY_SHARE:
        claims.append(
            f"first signals are mostly late_prediction ({first_late:.0%}) — "
            "the first labeled setup is already past 60% of the eventual wave."
        )
    else:
        claims.append(
            f"first-signal mass is mixed (early={first_early:.0%}, late={first_late:.0%}); "
            "this does not prove a 0–20% catch."
        )
    if all_late >= _MAJORITY_SHARE and all_early < _LOW_EARLY_SHARE:
        claims.append(
            f"the continuation-skill rows are late ({all_late:.0%} at 60%+; "
            f"early_prediction={all_early:.0%}), matching 'the wave already started'."
        )
    if not claims:
        claims.append("insufficient completed waves to locate the signal on the path.")
    return tuple(claims)


def render_wave_position_markdown(report: WavePositionReport) -> str:
    d = report.diagnostics
    scope = str(d.get("primary_scope", "develop"))
    lines = [
        "# wave position — first signal vs completed wave",
        "",
        "Diagnostic look-ahead for the completed peak only. Features at t stay causal.",
        "Holdout never scored. No MBO reload. No book reconstruction.",
        "",
        "Bins: 0–20% early_prediction · 20–40% early_continuation · "
        "40–60% mid_wave · 60%+ late_prediction.",
        "",
        f"- primary scope: `{scope}`",
        f"- waves: {d.get('n_waves')}",
        f"- first successful signals: {d.get('n_first_success_primary')}",
        f"- successful further-beyond setups: {d.get('n_all_success_primary')}",
        f"- min peak ticks: {d.get('min_peak_ticks')}",
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
        first = first.filter(
            (pl.col("scope") == scope)
            & (pl.col("success_only"))
            & (pl.col("role") == "first_signal")
        )
    lines.extend(["", "## First signal in the story (successful further-beyond)", ""])
    lines.extend(
        _md_table(
            first,
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
    all_s = report.all_summary
    if all_s.height and "success_only" in all_s.columns:
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
    oof_path = root / "oof_predictions.parquet"
    if oof_path.is_file():
        oof = pl.read_parquet(oof_path)
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
        progress=progress,
    )


__all__ = [
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
