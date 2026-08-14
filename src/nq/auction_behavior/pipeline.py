"""End-to-end Phase-1 auction behavior analysis (no trading decisions).

Causality contract (must match auction / volume-profile research fixes):
  - decisions use decision_* VP levels only (via auction_action_states)
  - asof / profile→signal joins stay backward inside auction builders
  - deceptive liquidity is scored, never deleted on this path
  - OOS validation is purged walk-forward
  - outputs are behavior probabilities, not entries/exits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.events import BEHAVIOR_EVENT_COLUMNS, build_behavior_events
from nq.auction_behavior.memory import attach_causal_memory
from nq.auction_behavior.model import estimate_behavior_probabilities
from nq.auction_behavior.projection import (
    PROJECTION_NUMERIC_COLUMNS,
    AsiaLondonProjectionConfig,
    build_asia_london_projection,
)
from nq.auction_behavior.quality import attach_signal_quality
from nq.auction_behavior.state import STATE_FEATURE_COLUMNS, attach_state_vector
from nq.auction_behavior.types import BehaviorProbabilities
from nq.auction_behavior.validate import (
    TRADE_FORBIDDEN_COLUMNS,
    BehaviorValidationReport,
    validate_behavior_frame,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import (
    VP_LIQUIDITY_SESSION,
    VpLiquiditySession,
    session_date_from_ns,
    vp_liquidity_session_label,
)
from nq.simulation.auction import (
    VP_PROFILE_INTERVAL_NS,
    VP_SIGNAL_INTERVAL_NS,
    auction_action_states,
    auction_signals_from_states,
)
from nq.simulation.common import BUCKET_START
from nq.simulation.deceptive_liquidity import (
    deceptive_features_by_bucket,
    score_deceptive_events,
)

_MEMORY_BASE_COLS = (
    "vp_fsm_break",
    "vp_fsm_retest",
    "vp_fsm_expand",
    "vp_absorb",
    "vp_look_fail",
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_early_imbalance",
    "signal_quality",
    "deceptive_score",
    "real_liquidity_ratio",
    *PROJECTION_NUMERIC_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class BehaviorConfig:
    """إعدادات تشغيل المرحلة الأولى (فهم سلوك — بلا تداول)."""

    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS
    signal_interval_ns: int = VP_SIGNAL_INTERVAL_NS
    fixed_range: bool = True
    include_deceptive_scores: bool = True
    n_splits: int = 3
    embargo: int = 0
    purge_samples: int = 1
    min_train_size: int = 8
    memory_lags: tuple[int, ...] = (1, 2, 3, 5)
    outcome_window: int = 8
    include_asia_london_projection: bool = True
    projection_config: AsiaLondonProjectionConfig = field(
        default_factory=AsiaLondonProjectionConfig
    )

    def __post_init__(self) -> None:
        if self.profile_interval_ns < 1 or self.signal_interval_ns < 1:
            raise ValueError("profile_interval_ns and signal_interval_ns must be positive")
        if self.n_splits < 1 or self.min_train_size < 1:
            raise ValueError("n_splits and min_train_size must be >= 1")
        if self.embargo < 0 or self.purge_samples < 0:
            raise ValueError("embargo and purge_samples must be non-negative")
        if self.outcome_window < 1:
            raise ValueError("outcome_window must be >= 1")
        if any(lag < 1 for lag in self.memory_lags):
            raise ValueError("all memory_lags must be >= 1")
        if len(set(self.memory_lags)) != len(self.memory_lags):
            raise ValueError("memory_lags must be unique")


@dataclass(frozen=True, slots=True)
class AuctionBehaviorResult:
    """تقرير المرحلة الأولى: احتمالات + إطارات وصفية (بدون بوابات صفقة)."""

    probabilities: BehaviorProbabilities
    validation: BehaviorValidationReport
    blended: pl.DataFrame
    events: pl.DataFrame
    fold_metrics: pl.DataFrame
    session_profiles: pl.DataFrame
    london_scenarios: pl.DataFrame
    projection: pl.DataFrame = field(default_factory=pl.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        return float(value.item())
    return float(str(value))


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _as_float(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    if isinstance(value, np.generic):
        return int(value.item())
    return int(str(value))


def _require_decision_columns(states: pl.DataFrame) -> None:
    need = ("decision_poc", "decision_vah", "decision_val")
    missing = [c for c in need if c not in states.columns]
    if missing:
        raise ValueError(
            "auction_behavior requires lagged decision_* columns "
            f"(missing={missing}). Never use current-bar vah/val/poc for decisions."
        )


def _assert_no_trade_columns(frame: pl.DataFrame) -> None:
    hit = [c for c in TRADE_FORBIDDEN_COLUMNS if c in frame.columns]
    if hit:
        raise ValueError(f"auction_behavior must not emit trade columns; found {sorted(hit)}")


def _with_session_runs(frame: pl.DataFrame) -> pl.DataFrame:
    """يرقّم كل مقطع جلسة متصل؛ لا يجمع جلسات متكررة عبر الأيام."""
    if frame.height == 0 or VP_LIQUIDITY_SESSION not in frame.columns:
        return frame
    session = pl.col(VP_LIQUIDITY_SESSION).cast(pl.Int64).fill_null(-1)
    return frame.sort(AVAILABILITY_TS).with_columns(
        (session != session.shift(1).fill_null(-2)).cast(pl.Int64).cum_sum().alias("_liquidity_run")
    )


def _with_behavior_story_runs(frame: pl.DataFrame) -> pl.DataFrame:
    """يجمع آسيا+لندن في قصة واحدة، ويعيد الضبط عند نيويورك/يوم جديد."""
    if frame.height == 0 or VP_LIQUIDITY_SESSION not in frame.columns:
        return frame
    ordered = frame.sort(AVAILABILITY_TS)
    times = [int(x) for x in ordered[AVAILABILITY_TS].to_list()]
    sessions = [int(x) for x in ordered[VP_LIQUIDITY_SESSION].fill_null(-1).to_list()]
    asia_london = {int(VpLiquiditySession.ASIA), int(VpLiquiditySession.LONDON)}
    keys = [
        f"{session_date_from_ns(ts)}:{'asia_london' if sess in asia_london else sess}"
        for ts, sess in zip(times, sessions, strict=True)
    ]
    runs: list[int] = []
    run = 0
    previous: str | None = None
    for key in keys:
        if key != previous:
            run += 1
        runs.append(run)
        previous = key
    return ordered.with_columns(pl.Series("_behavior_story_run", runs, dtype=pl.Int64))


def _attach_projection(blended: pl.DataFrame, projection: pl.DataFrame) -> pl.DataFrame:
    """يلحق آخر إسقاط لندن المكتمل داخل يوم التداول نفسه (asof خلفي)."""
    if blended.height == 0 or projection.height == 0:
        return blended.with_columns(pl.lit(0.0).alias(c) for c in PROJECTION_NUMERIC_COLUMNS)
    ordered = blended.sort(AVAILABILITY_TS)
    story_dates = [session_date_from_ns(int(ts)) for ts in ordered[AVAILABILITY_TS].to_list()]
    left = ordered.with_columns(pl.Series("_projection_story_date", story_dates, dtype=pl.Utf8))
    projection_cols = [
        AVAILABILITY_TS,
        "projection_story_date",
        "auction_phase",
        "asia_poc",
        "asia_vah",
        "asia_val",
        "asia_primary_hvn",
        "composite_poc",
        "composite_vah",
        "composite_val",
        "composite_primary_hvn",
        *PROJECTION_NUMERIC_COLUMNS,
    ]
    right = (
        projection.filter(pl.col("projection_stage") == "london_extend")
        .select(c for c in projection_cols if c in projection.columns)
        .sort(AVAILABILITY_TS)
    )
    if right.height == 0:
        return ordered.with_columns(pl.lit(0.0).alias(c) for c in PROJECTION_NUMERIC_COLUMNS)
    joined = left.join_asof(
        right,
        left_on=AVAILABILITY_TS,
        right_on=AVAILABILITY_TS,
        by_left="_projection_story_date",
        by_right="projection_story_date",
        strategy="backward",
        check_sortedness=False,
    )
    london = pl.col(VP_LIQUIDITY_SESSION).cast(pl.Int64) == int(VpLiquiditySession.LONDON)
    numeric = [c for c in PROJECTION_NUMERIC_COLUMNS if c in joined.columns]
    joined = joined.with_columns(
        pl.when(london).then(pl.col(c).fill_null(0.0)).otherwise(0.0).alias(c) for c in numeric
    )
    contextual = [
        c
        for c in (
            "auction_phase",
            "asia_poc",
            "asia_vah",
            "asia_val",
            "asia_primary_hvn",
            "composite_poc",
            "composite_vah",
            "composite_val",
            "composite_primary_hvn",
        )
        if c in joined.columns
    ]
    return joined.with_columns(
        pl.when(london).then(pl.col(c)).otherwise(None).alias(c) for c in contextual
    ).drop("_projection_story_date")


def _session_vp_summary(states: pl.DataFrame) -> pl.DataFrame:
    """ملخص لكل دورة جلسة متصلة، مع فصل حدود القرار عن الملف المكتمل."""
    if states.height == 0 or VP_LIQUIDITY_SESSION not in states.columns:
        return pl.DataFrame(
            schema={
                "liquidity_session": pl.Int64(),
                "session_run": pl.Int64(),
                "session_name": pl.Utf8(),
                "session_start_ts": pl.Int64(),
                "session_end_ts": pl.Int64(),
                "n_bars": pl.Int64(),
                "decision_poc": pl.Float64(),
                "decision_vah": pl.Float64(),
                "decision_val": pl.Float64(),
                "completed_poc": pl.Float64(),
                "completed_vah": pl.Float64(),
                "completed_val": pl.Float64(),
                "va_width": pl.Float64(),
            }
        )

    work = _with_session_runs(states)
    rows: list[dict[str, Any]] = []
    for run_id, g in work.group_by("_liquidity_run", maintain_order=True):
        rid = _as_int(run_id[0] if isinstance(run_id, tuple) else run_id)
        sid = _as_int(g[VP_LIQUIDITY_SESSION][0])
        last = g.tail(1)
        vah = last["decision_vah"][0] if "decision_vah" in last.columns else None
        val = last["decision_val"][0] if "decision_val" in last.columns else None
        poc = last["decision_poc"][0] if "decision_poc" in last.columns else None
        completed_vah = last["vah"][0] if "vah" in last.columns else None
        completed_val = last["val"][0] if "val" in last.columns else None
        completed_poc = last["poc"][0] if "poc" in last.columns else None
        va_w = (
            _as_float(completed_vah) - _as_float(completed_val)
            if completed_vah is not None and completed_val is not None
            else None
        )
        rows.append(
            {
                "liquidity_session": sid,
                "session_run": rid,
                "session_name": vp_liquidity_session_label(sid),
                "session_start_ts": _as_int(g[AVAILABILITY_TS][0]),
                "session_end_ts": _as_int(last[AVAILABILITY_TS][0]),
                "n_bars": int(g.height),
                "decision_poc": _as_optional_float(poc),
                "decision_vah": _as_optional_float(vah),
                "decision_val": _as_optional_float(val),
                "completed_poc": _as_optional_float(completed_poc),
                "completed_vah": _as_optional_float(completed_vah),
                "completed_val": _as_optional_float(completed_val),
                "va_width": va_w,
            }
        )
    return pl.DataFrame(rows)


def _london_scenario_summary(states: pl.DataFrame) -> pl.DataFrame:
    """سيناريو لندن مقابل قيمة آسيا المكتملة (وصفي؛ بلا توصية صفقة).

    يستخدم الملف النهائي لآسيا عند انتقال الجلسة. نتائج مدى لندن تحمل
    ``outcome_available_ts`` عند نهاية دورة لندن حتى لا تُعامل كمعلومة افتتاح.
    """
    schema = {
        "asia_end_ts": pl.Int64(),
        "london_open_ts": pl.Int64(),
        "outcome_available_ts": pl.Int64(),
        "scenario": pl.Utf8(),
        "london_open": pl.Float64(),
        "asia_completed_poc": pl.Float64(),
        "asia_completed_vah": pl.Float64(),
        "asia_completed_val": pl.Float64(),
        "open_inside_asia_va": pl.Boolean(),
        "london_broke_asia_vah": pl.Boolean(),
        "london_broke_asia_val": pl.Boolean(),
    }
    need = {VP_LIQUIDITY_SESSION, "vah", "val", "close", AVAILABILITY_TS}
    if states.height == 0 or not need.issubset(states.columns):
        return pl.DataFrame(schema=schema)

    ordered = states.sort(AVAILABILITY_TS)
    asia_code = int(VpLiquiditySession.ASIA)
    london_code = int(VpLiquiditySession.LONDON)

    # تقسيم إلى شرائح جلسة متصلة (آسيا→لندن→نيويورك→آسيا…)
    runs = _with_session_runs(ordered)

    rows: list[dict[str, Any]] = []
    # اجمع أزواج: آخر آسيا قبل أول لندن التالية
    asia_last: pl.DataFrame | None = None
    for _run_id, g in runs.group_by("_liquidity_run", maintain_order=True):
        sess = int(g[VP_LIQUIDITY_SESSION][0])
        if sess == asia_code:
            asia_last = g.tail(1)
            continue
        if sess != london_code:
            asia_last = None
            continue
        if asia_last is None:
            continue

        a_vah = asia_last["vah"][0]
        a_val = asia_last["val"][0]
        a_poc = asia_last["poc"][0] if "poc" in asia_last.columns else None
        if a_vah is None or a_val is None:
            asia_last = None
            continue

        lon0 = g.head(1)
        if "open" in lon0.columns and lon0["open"][0] is not None:
            open_px = lon0["open"][0]
        else:
            open_px = lon0["close"][0]
        if open_px is None:
            asia_last = None
            continue

        open_f = _as_float(open_px)
        vah_f = _as_float(a_vah)
        val_f = _as_float(a_val)
        inside = bool(val_f <= open_f <= vah_f)
        if open_f > vah_f:
            scenario = "open_above_asia_vah"
        elif open_f < val_f:
            scenario = "open_below_asia_val"
        else:
            scenario = "inside_asia_va"

        hi = g["high"].max() if "high" in g.columns else None
        lo = g["low"].min() if "low" in g.columns else None
        broke_vah = bool(hi is not None and _as_float(hi) > vah_f)
        broke_val = bool(lo is not None and _as_float(lo) < val_f)

        rows.append(
            {
                "asia_end_ts": _as_int(asia_last[AVAILABILITY_TS][0]),
                "london_open_ts": _as_int(lon0[AVAILABILITY_TS][0]),
                "outcome_available_ts": _as_int(g[AVAILABILITY_TS][-1]),
                "scenario": scenario,
                "london_open": open_f,
                "asia_completed_poc": _as_optional_float(a_poc),
                "asia_completed_vah": vah_f,
                "asia_completed_val": val_f,
                "open_inside_asia_va": inside,
                "london_broke_asia_vah": broke_vah,
                "london_broke_asia_val": broke_val,
            }
        )
        asia_last = None

    return pl.DataFrame(rows) if rows else pl.DataFrame(schema=schema)


def run_auction_behavior_analysis(
    mbo: pl.DataFrame,
    *,
    config: BehaviorConfig | None = None,
    score_mbo: pl.DataFrame | None = None,
) -> AuctionBehaviorResult:
    """يشغّل طبقات المرحلة 1–11 ويعيد تقرير سلوك (بدون بوابات تداول).

    ``mbo``: تدفّق MBO لبناء حالات المزاد.
    ``score_mbo``: اختياري — خام قبل أي تنظيف لتسجيل نية التضليل (درجات فقط).
    """
    cfg = config or BehaviorConfig()
    projection = (
        build_asia_london_projection(mbo, config=cfg.projection_config)
        if mbo is not None and mbo.height > 0 and cfg.include_asia_london_projection
        else pl.DataFrame()
    )
    if mbo is None or mbo.height == 0:
        empty_probs = BehaviorProbabilities(
            p_balanced=0.0,
            p_imbalanced=0.0,
            p_true_break=0.0,
            p_false_break=0.0,
            p_retest_success=0.0,
            p_retest_fail=0.0,
            p_expansion_continue=0.0,
            p_return_to_value=0.0,
            confidence=0.0,
            n_samples=0,
            detail="empty mbo",
        )
        empty_val = validate_behavior_frame(pl.DataFrame())
        return AuctionBehaviorResult(
            probabilities=empty_probs,
            validation=empty_val,
            blended=pl.DataFrame(),
            events=pl.DataFrame(),
            fold_metrics=pl.DataFrame(),
            session_profiles=pl.DataFrame(),
            london_scenarios=pl.DataFrame(),
            projection=projection,
            diagnostics={"empty": True, "deceptive_filtered": False},
        )

    # Layers 1–3: session-aware auction states (decision_* lagged inside builder)
    states = auction_action_states(
        mbo,
        profile_interval_ns=cfg.profile_interval_ns,
        signal_interval_ns=cfg.signal_interval_ns,
        fixed_range=cfg.fixed_range,
    )
    if states.height == 0:
        empty_probs = BehaviorProbabilities(
            p_balanced=0.0,
            p_imbalanced=0.0,
            p_true_break=0.0,
            p_false_break=0.0,
            p_retest_success=0.0,
            p_retest_fail=0.0,
            p_expansion_continue=0.0,
            p_return_to_value=0.0,
            confidence=0.0,
            n_samples=0,
            detail="no trade-derived auction bars",
        )
        return AuctionBehaviorResult(
            probabilities=empty_probs,
            validation=validate_behavior_frame(pl.DataFrame()),
            blended=pl.DataFrame(),
            events=pl.DataFrame(),
            fold_metrics=pl.DataFrame(),
            session_profiles=pl.DataFrame(),
            london_scenarios=pl.DataFrame(),
            projection=projection,
            diagnostics={
                "empty": True,
                "reason": "no_trade_derived_auction_bars",
                "n_mbo_rows": int(mbo.height),
                "deceptive_filtered": False,
            },
        )
    _require_decision_columns(states)
    session_profiles = _session_vp_summary(states)
    london_scenarios = _london_scenario_summary(states)

    # Layer 4: order-flow intention via deceptive scores — raw path, no deletion
    deceptive_bucket = pl.DataFrame()
    scored_rows = 0
    if cfg.include_deceptive_scores:
        score_src = score_mbo if score_mbo is not None else mbo
        scored = score_deceptive_events(score_src)
        scored_rows = int(scored.height)
        # Intentionally do NOT call filter_deceptive_liquidity.
        deceptive_bucket = deceptive_features_by_bucket(
            score_src,
            interval_ns=cfg.signal_interval_ns,
            scored=scored,
        )

    # Layers 5–6: VP/FSM signals + behavior events
    signals = auction_signals_from_states(
        states,
        fixed_range_decisions=cfg.fixed_range,
    )
    blended = signals
    if deceptive_bucket.height > 0 and AVAILABILITY_TS in deceptive_bucket.columns:
        deco_cols = [
            c
            for c in (
                AVAILABILITY_TS,
                "deceptive_score",
                "real_liquidity_ratio",
                "noise_instant",
                "noise_cum",
                "deceptive_volume_share",
                "deceptive_cancel_rate",
            )
            if c in deceptive_bucket.columns
        ]
        # Exact availability join (same signal cadence); no forward-fill of future scores.
        blended = blended.join(
            deceptive_bucket.select(deco_cols),
            on=AVAILABILITY_TS,
            how="left",
        )
    if "deceptive_score" not in blended.columns:
        blended = blended.with_columns(pl.lit(0.0).alias("deceptive_score"))
    else:
        blended = blended.with_columns(pl.col("deceptive_score").fill_null(0.0))
    if "real_liquidity_ratio" not in blended.columns:
        blended = blended.with_columns(pl.lit(1.0).alias("real_liquidity_ratio"))
    else:
        blended = blended.with_columns(pl.col("real_liquidity_ratio").fill_null(1.0))

    # Attach decision_* for validation visibility (from states; already lagged)
    decision_cols = [
        c
        for c in ("decision_poc", "decision_vah", "decision_val", BUCKET_START)
        if c in states.columns
    ]
    state_decision = states.select(AVAILABILITY_TS, *decision_cols)
    blended = blended.join(state_decision, on=AVAILABILITY_TS, how="left")
    blended = _with_session_runs(blended)
    blended = _attach_projection(blended, projection)
    blended = _with_behavior_story_runs(blended)

    events = build_behavior_events(
        blended,
        outcome_window=cfg.outcome_window,
        group_col="_behavior_story_run",
    )

    # Layers 7–9: quality → memory → state vector
    blended = attach_signal_quality(blended)
    mem_cols = [c for c in _MEMORY_BASE_COLS if c in blended.columns]
    blended = attach_causal_memory(
        blended,
        columns=mem_cols,
        lags=cfg.memory_lags,
        group_col="_behavior_story_run",
    )
    blended = attach_state_vector(blended)
    _assert_no_trade_columns(blended)
    _assert_no_trade_columns(events)

    # Layers 10–11: probabilistic model + purged OOS + validation
    probs, fold_metrics = estimate_behavior_probabilities(
        blended,
        events,
        n_splits=cfg.n_splits,
        embargo=cfg.embargo,
        purge_samples=cfg.purge_samples,
        min_train_size=cfg.min_train_size,
    )
    validation = validate_behavior_frame(blended, fold_df=fold_metrics)

    diagnostics: dict[str, Any] = {
        "n_mbo_rows": int(mbo.height),
        "n_state_bars": int(states.height),
        "n_signal_bars": int(signals.height),
        "n_events": int(events.height),
        "n_behavior_event_cols": len(BEHAVIOR_EVENT_COLUMNS),
        "n_state_feature_cols": len(STATE_FEATURE_COLUMNS),
        "n_projection_bars": int(projection.height),
        "deceptive_scored_rows": scored_rows,
        "deceptive_filtered": False,
        "causality": {
            "decision_columns_required": True,
            "profile_to_signal": "backward_asof_inside_auction_action_states",
            "deceptive_deletion": False,
            "validation": "purged_walk_forward",
            "trade_outputs": False,
        },
        "config": {
            "profile_interval_ns": cfg.profile_interval_ns,
            "signal_interval_ns": cfg.signal_interval_ns,
            "fixed_range": cfg.fixed_range,
            "include_deceptive_scores": cfg.include_deceptive_scores,
            "n_splits": cfg.n_splits,
            "embargo": cfg.embargo,
            "purge_samples": cfg.purge_samples,
            "outcome_window": cfg.outcome_window,
            "include_asia_london_projection": cfg.include_asia_london_projection,
            "projection_interval_ns": cfg.projection_config.interval_ns,
            "memory_scope": "asia_london_story_then_reset_at_new_york_or_new_day",
        },
    }

    return AuctionBehaviorResult(
        probabilities=probs,
        validation=validation,
        blended=blended,
        events=events,
        fold_metrics=fold_metrics,
        session_profiles=session_profiles,
        london_scenarios=london_scenarios,
        projection=projection,
        diagnostics=diagnostics,
    )


def behavior_state_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """إطار زمني خفيف للحالة/الجودة (ليس جدول احتمالات)."""
    cols = [
        AVAILABILITY_TS,
        "signal_quality",
        "vp_balance",
        "vp_imbalance",
        "vp_fsm_break",
        "vp_fsm_retest",
        "vp_fsm_expand",
        "vp_absorb",
        "vp_look_fail",
        "deceptive_score",
        "real_liquidity_ratio",
        "decision_poc",
        "decision_vah",
        "decision_val",
        "auction_phase",
        "asia_poc",
        "asia_vah",
        "asia_val",
        "composite_poc",
        "composite_vah",
        "composite_val",
        *PROJECTION_NUMERIC_COLUMNS,
    ]
    frame = result.blended
    keep = [c for c in cols if c in frame.columns]
    return frame.select(keep) if keep else pl.DataFrame()


def behavior_probabilities_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """اسم توافق قديم لـ :func:`behavior_state_frame`؛ لا يحتوي تنبؤًا لكل صف."""
    return behavior_state_frame(result)


def behavior_probability_summary(result: AuctionBehaviorResult) -> pl.DataFrame:
    """صف واحد للتوقعات المجمعة مع وقت اكتمال التحقق الذي أتاحها."""
    probs = result.probabilities
    available_after_ts: int | None = None
    if result.fold_metrics.height and "test_end_ts" in result.fold_metrics.columns:
        value = result.fold_metrics["test_end_ts"].max()
        available_after_ts = None if value is None else _as_int(value)
    elif result.blended.height and AVAILABILITY_TS in result.blended.columns:
        value = result.blended[AVAILABILITY_TS].max()
        available_after_ts = None if value is None else _as_int(value)
    return pl.DataFrame(
        {
            "available_after_ts": [available_after_ts],
            "p_balanced": [probs.p_balanced],
            "p_imbalanced": [probs.p_imbalanced],
            "p_true_break": [probs.p_true_break],
            "p_false_break": [probs.p_false_break],
            "p_retest_success": [probs.p_retest_success],
            "p_retest_fail": [probs.p_retest_fail],
            "p_expansion_continue": [probs.p_expansion_continue],
            "p_return_to_value": [probs.p_return_to_value],
            "confidence": [probs.confidence],
            "n_samples": [probs.n_samples],
            "detail": [probs.detail],
        }
    )


__all__ = [
    "AuctionBehaviorResult",
    "BehaviorConfig",
    "behavior_probabilities_frame",
    "behavior_probability_summary",
    "behavior_state_frame",
    "run_auction_behavior_analysis",
]
