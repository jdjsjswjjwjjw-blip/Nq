"""End-to-end Phase-1 auction behavior analysis (no trading decisions).

Causality contract (must match auction / volume-profile research fixes):
  - decisions use decision_* VP levels only (via auction_action_states)
  - asof / profile→signal joins stay backward inside auction builders
  - deceptive liquidity is scored, never deleted on this path
  - OOS validation is purged walk-forward
  - outputs are behavior probabilities, not entries/exits
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.events import BEHAVIOR_EVENT_COLUMNS, build_behavior_events
from nq.auction_behavior.level_flow import (
    LEVEL_FLOW_COLUMNS,
    LevelFlowConfig,
    attach_level_flow_features,
)
from nq.auction_behavior.memory import attach_market_memory, attach_sequence_memory
from nq.auction_behavior.model import estimate_behavior_probabilities
from nq.auction_behavior.path_confirm import PATH_CONFIRM_COLUMNS, attach_path_depth_confirmation
from nq.auction_behavior.projection import (
    PROJECTION_NUMERIC_COLUMNS,
    AsiaLondonProjectionConfig,
    build_asia_london_projection,
)
from nq.auction_behavior.quality import attach_signal_quality
from nq.auction_behavior.reliability import RELIABILITY_COLUMNS, attach_reliability_evidence
from nq.auction_behavior.science import BehaviorScienceReport, ScienceConfig, run_behavior_science
from nq.auction_behavior.state import STATE_FEATURE_COLUMNS, attach_state_vector
from nq.auction_behavior.structure import STRUCTURE_FEATURE_COLUMNS, attach_structure_features
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
from nq.research.progress import PipelineProgress, ProgressLike, resolve_progress
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
    "rel_credibility",
    "rel_evidence_strength",
    "lf_absorption_proxy",
    "lf_liquidity_withdrawal",
    "lf_arrival_intensity",
    *PROJECTION_NUMERIC_COLUMNS,
    *PATH_CONFIRM_COLUMNS,
)

_HOLDOUT_FRAC_MIN = 0.05
_HOLDOUT_FRAC_MAX = 0.5


@dataclass(frozen=True, slots=True)
class BehaviorConfig:
    """إعدادات تشغيل المرحلة الأولى (فهم سلوك — بلا تداول)."""

    profile_interval_ns: int = VP_PROFILE_INTERVAL_NS
    signal_interval_ns: int = VP_SIGNAL_INTERVAL_NS
    fixed_range: bool = True
    include_deceptive_scores: bool = True
    include_level_flow: bool = True
    include_reliability_evidence: bool = True
    n_splits: int = 3
    embargo: int = 0
    purge_samples: int = 1
    min_train_size: int = 8
    memory_lags: tuple[int, ...] = (1, 2, 3, 5)
    outcome_window: int = 8
    include_asia_london_projection: bool = True
    include_science: bool = True
    holdout_frac: float = 0.2
    # الـholdout لا يُلمس تلقائيًا؛ فعّله صراحة بعد قفل التطوير.
    evaluate_holdout: bool = False
    quiet: bool = False
    progress_log_path: str | None = None
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
        if not _HOLDOUT_FRAC_MIN <= self.holdout_frac <= _HOLDOUT_FRAC_MAX:
            raise ValueError(f"holdout_frac must be in [{_HOLDOUT_FRAC_MIN}, {_HOLDOUT_FRAC_MAX}]")


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
    science: BehaviorScienceReport | None = None
    #: إطار تنبؤ منفصل: State(t)→p(outcome|state) — ليس behavior_state_frame.
    predictions: pl.DataFrame = field(default_factory=pl.DataFrame)
    base_rate_fold_metrics: pl.DataFrame = field(default_factory=pl.DataFrame)
    conditional_fold_metrics: pl.DataFrame = field(default_factory=pl.DataFrame)
    oof_predictions: pl.DataFrame = field(default_factory=pl.DataFrame)
    live_predictions: pl.DataFrame = field(default_factory=pl.DataFrame)
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


def _with_behavior_story_runs(
    frame: pl.DataFrame,
    *,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يجمع آسيا+لندن في قصة واحدة، ويعيد الضبط عند نيويورك/يوم جديد."""
    if frame.height == 0 or VP_LIQUIDITY_SESSION not in frame.columns:
        return frame
    ordered = frame.sort(AVAILABILITY_TS)
    times = [int(x) for x in ordered[AVAILABILITY_TS].to_list()]
    sessions = [int(x) for x in ordered[VP_LIQUIDITY_SESSION].fill_null(-1).to_list()]
    asia_london = {int(VpLiquiditySession.ASIA), int(VpLiquiditySession.LONDON)}
    n = len(times)
    if progress is not None:
        progress.op(f"behavior story runs n={n:,}")
    keys = []
    for i, (ts, sess) in enumerate(zip(times, sessions, strict=True), start=1):
        if progress is not None:
            progress.heartbeat(i, n, label="behavior-story-keys")
        keys.append(f"{session_date_from_ns(ts)}:{'asia_london' if sess in asia_london else sess}")
    runs: list[int] = []
    run = 0
    previous: str | None = None
    for i, key in enumerate(keys, start=1):
        if progress is not None:
            progress.heartbeat(i, n, label="behavior-story-runs")
        if key != previous:
            run += 1
        runs.append(run)
        previous = key
    return ordered.with_columns(pl.Series("_behavior_story_run", runs, dtype=pl.Int64))


def _attach_projection(
    blended: pl.DataFrame,
    projection: pl.DataFrame,
    *,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يلحق آخر إسقاط لندن المكتمل داخل يوم التداول نفسه (asof خلفي)."""
    if progress is not None:
        progress.op(f"attach projection asof bars={blended.height:,} proj={projection.height:,}")
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


def _session_vp_summary(
    states: pl.DataFrame,
    *,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
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
    groups = list(work.group_by("_liquidity_run", maintain_order=True))
    n_groups = len(groups)
    if progress is not None:
        progress.op(f"session_vp_summary runs={n_groups}")
    rows: list[dict[str, Any]] = []
    for i, (run_id, g) in enumerate(groups, start=1):
        if progress is not None:
            progress.heartbeat(i, n_groups, label="session-vp-summary")
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


def _london_scenario_summary(  # noqa: PLR0912, PLR0915
    states: pl.DataFrame,
    *,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """سيناريو لندن مقابل قيمة آسيا المكتملة (وصفي؛ بلا توصية صفقة).

    يستخدم الملف النهائي لآسيا عند انتقال الجلسة. نتائج مدى لندن تحمل
    ``outcome_available_ts`` عند نهاية دورة لندن في العينة — إذا الملف مقصوص
    قبل 09:30 ET تُقفَل لندن عند آخر بار، وكذلك أي زون أخرى.
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
    run_groups = list(runs.group_by("_liquidity_run", maintain_order=True))
    n_runs = len(run_groups)
    if progress is not None:
        progress.op(f"london_scenario_summary runs={n_runs}")
    for i, (_run_id, g) in enumerate(run_groups, start=1):
        if progress is not None:
            progress.heartbeat(i, n_runs, label="london-scenario")
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


_BEHAVIOR_PIPELINE_STEPS = 12


def run_auction_behavior_analysis(
    mbo: pl.DataFrame,
    *,
    config: BehaviorConfig | None = None,
    score_mbo: pl.DataFrame | None = None,
    progress: PipelineProgress | bool | None = None,
    quiet: bool | None = None,
) -> AuctionBehaviorResult:
    """يشغّل طبقات المرحلة 1–11 ويعيد تقرير سلوك (بدون بوابات تداول).

    ``mbo``: تدفّق MBO لبناء حالات المزاد.
    ``score_mbo``: اختياري — خام قبل أي تنظيف لتسجيل نية التضليل (درجات فقط).
    كل خطوة/عملية تُطبع فورًا عبر ``PipelineProgress`` (stderr + progress.log).
    """
    cfg = config or BehaviorConfig()
    resolved_quiet = cfg.quiet if quiet is None else bool(quiet)
    if quiet is None and progress is None and os.environ.get("PYTEST_CURRENT_TEST"):
        resolved_quiet = True
    log = resolve_progress(progress, quiet=resolved_quiet)
    if cfg.progress_log_path and log.enabled:
        log.attach_log(cfg.progress_log_path)
    log.begin("auction_behavior", total_steps=_BEHAVIOR_PIPELINE_STEPS)
    try:
        return _run_auction_behavior_analysis(
            mbo,
            cfg=cfg,
            score_mbo=score_mbo,
            log=log,
        )
    except Exception as exc:
        log.fail(exc)
        raise


def _run_auction_behavior_analysis(  # noqa: PLR0912, PLR0915
    mbo: pl.DataFrame,
    *,
    cfg: BehaviorConfig,
    score_mbo: pl.DataFrame | None,
    log: PipelineProgress,
) -> AuctionBehaviorResult:
    n_mbo = 0 if mbo is None else int(mbo.height)
    log.step("asia_london_projection", f"mbo_rows={n_mbo:,}")
    projection = (
        build_asia_london_projection(mbo, config=cfg.projection_config, progress=log)
        if mbo is not None and mbo.height > 0 and cfg.include_asia_london_projection
        else pl.DataFrame()
    )
    log.op(f"projection bars={projection.height:,}")
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
        empty = AuctionBehaviorResult(
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
        log.done("empty mbo")
        return empty

    log.step(
        "auction_action_states",
        f"profile={cfg.profile_interval_ns} · signal={cfg.signal_interval_ns}",
    )
    states = auction_action_states(
        mbo,
        profile_interval_ns=cfg.profile_interval_ns,
        signal_interval_ns=cfg.signal_interval_ns,
        fixed_range=cfg.fixed_range,
        progress=log,
    )
    log.op(f"state bars={states.height:,}")
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
        empty = AuctionBehaviorResult(
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
        log.done("no auction bars")
        return empty
    _require_decision_columns(states)
    log.step("session_summaries")
    log.op("session_vp_summary")
    session_profiles = _session_vp_summary(states, progress=log)
    log.op(f"session_profiles={session_profiles.height:,}")
    log.op("london_scenario_summary")
    london_scenarios = _london_scenario_summary(states, progress=log)
    log.op(f"london_scenarios={london_scenarios.height:,}")

    # Layer 4: order-flow intention via deceptive scores — raw path, no deletion
    deceptive_bucket = pl.DataFrame()
    scored_rows = 0
    scored: pl.DataFrame | None = None
    reliability = pl.DataFrame()
    level_flow = pl.DataFrame()
    log.step("order_flow_scores", "raw MBO · no deletion")
    if cfg.include_deceptive_scores or cfg.include_reliability_evidence:
        score_src = score_mbo if score_mbo is not None else mbo
        log.op(f"score_deceptive_events events={score_src.height:,}")
        scored = score_deceptive_events(score_src, progress=log)
        scored_rows = int(scored.height)
        log.op(f"scored_rows={scored_rows:,}")
        # Intentionally do NOT call filter_deceptive_liquidity.
        if cfg.include_deceptive_scores:
            log.op("deceptive_features_by_bucket")
            deceptive_bucket = deceptive_features_by_bucket(
                score_src,
                interval_ns=cfg.signal_interval_ns,
                scored=scored,
                progress=log,
            )
            log.op(f"deceptive buckets={deceptive_bucket.height:,}")
        if cfg.include_reliability_evidence:
            log.op("attach_reliability_evidence")
            reliability = attach_reliability_evidence(
                score_src,
                interval_ns=cfg.signal_interval_ns,
                scored=scored,
                progress=log,
            )
            log.op(f"reliability bars={reliability.height:,}")
    else:
        log.op("deceptive/reliability skipped")
    log.step("level_flow", f"enabled={cfg.include_level_flow}")
    if cfg.include_level_flow:
        level_flow = attach_level_flow_features(
            mbo if score_mbo is None else score_mbo,
            states,
            config=LevelFlowConfig(interval_ns=cfg.signal_interval_ns),
            progress=log,
        )
        log.op(f"level_flow bars={level_flow.height:,}")
    else:
        log.op("level_flow skipped")

    # Layers 5–6: VP/FSM signals + behavior events
    log.step("auction_signals_from_states", f"bars={states.height:,}")
    signals = auction_signals_from_states(
        states,
        fixed_range_decisions=cfg.fixed_range,
        progress=log,
    )
    log.op(f"signal bars={signals.height:,}")
    blended = signals
    log.step("join_flow_and_projection")
    log.op("join deceptive scores")
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

    log.op("join reliability evidence")
    if reliability.height > 0 and AVAILABILITY_TS in reliability.columns:
        rel_cols = [c for c in (AVAILABILITY_TS, *RELIABILITY_COLUMNS) if c in reliability.columns]
        blended = blended.join(reliability.select(rel_cols), on=AVAILABILITY_TS, how="left")
        blended = blended.with_columns(
            pl.col(c).fill_null(0.0) for c in RELIABILITY_COLUMNS if c in blended.columns
        )
    else:
        blended = blended.with_columns(pl.lit(0.0).alias(c) for c in RELIABILITY_COLUMNS)

    log.op("join level_flow")
    if level_flow.height > 0 and AVAILABILITY_TS in level_flow.columns:
        lf_cols = [c for c in (AVAILABILITY_TS, *LEVEL_FLOW_COLUMNS) if c in level_flow.columns]
        blended = blended.join(level_flow.select(lf_cols), on=AVAILABILITY_TS, how="left")
        blended = blended.with_columns(
            pl.col(c).fill_null(0.0) for c in LEVEL_FLOW_COLUMNS if c in blended.columns
        )
    else:
        blended = blended.with_columns(pl.lit(0.0).alias(c) for c in LEVEL_FLOW_COLUMNS)

    # Attach decision_* + OHLC for structure FE (from states; decision already lagged)
    candidate_cols = (
        "decision_poc",
        "decision_vah",
        "decision_val",
        BUCKET_START,
        "close",
        "high",
        "low",
        "open",
        VP_LIQUIDITY_SESSION,
    )
    log.op("join decision_* from states")
    join_cols = [c for c in candidate_cols if c in states.columns and c not in blended.columns]
    state_decision = states.select(AVAILABILITY_TS, *join_cols)
    blended = blended.join(state_decision, on=AVAILABILITY_TS, how="left")
    log.op("session runs + projection asof + story runs")
    blended = _with_session_runs(blended)
    blended = _attach_projection(blended, projection, progress=log)
    log.op("attach_path_depth_confirmation")
    blended = attach_path_depth_confirmation(blended, progress=log)
    blended = _with_behavior_story_runs(blended, progress=log)
    log.op(f"blended bars={blended.height:,}")

    log.step("behavior_events", f"window={cfg.outcome_window}")
    events = build_behavior_events(
        blended,
        outcome_window=cfg.outcome_window,
        group_col="_behavior_story_run",
        progress=log,
    )
    log.op(f"event rows={events.height:,}")

    log.step("quality_structure_memory")
    log.op("attach_signal_quality")
    blended = attach_signal_quality(blended, progress=log)
    log.op("attach_structure_features")
    blended = attach_structure_features(blended, progress=log)
    mem_cols = [c for c in (*_MEMORY_BASE_COLS, *STRUCTURE_FEATURE_COLUMNS) if c in blended.columns]
    event_mem = [
        c
        for c in ("vp_fsm_break", "vp_fsm_retest", "vp_look_fail", "vp_absorb")
        if c in blended.columns
    ]
    log.op(f"attach_market_memory lags={cfg.memory_lags} cols={len(mem_cols)}")
    blended = attach_market_memory(
        blended,
        columns=mem_cols,
        lags=cfg.memory_lags,
        group_col="_behavior_story_run",
        event_columns=event_mem,
        progress=log,
    )
    log.op("attach_sequence_memory")
    blended = attach_sequence_memory(blended, group_col="_behavior_story_run", progress=log)
    log.op("attach_state_vector")
    blended = attach_state_vector(blended, progress=log)
    _assert_no_trade_columns(blended)
    _assert_no_trade_columns(events)
    log.op(f"feature cols={len(blended.columns)}")

    log.step("base_rate_probabilities", f"n_splits={cfg.n_splits}")
    probs, base_rate_fold_metrics = estimate_behavior_probabilities(
        blended,
        events,
        n_splits=cfg.n_splits,
        embargo=cfg.embargo,
        purge_samples=cfg.purge_samples,
        min_train_size=cfg.min_train_size,
        progress=log,
    )
    log.op(f"base_rate n_samples={probs.n_samples} folds={base_rate_fold_metrics.height}")
    science: BehaviorScienceReport | None = None
    predictions = pl.DataFrame()
    fold_metrics = base_rate_fold_metrics
    log.step("behavior_science", f"enabled={cfg.include_science}")
    if cfg.include_science:
        science = run_behavior_science(
            blended,
            config=ScienceConfig(
                outcome_window=cfg.outcome_window,
                n_splits=cfg.n_splits,
                embargo=cfg.embargo,
                purge_samples=cfg.purge_samples,
                min_train_size=cfg.min_train_size,
                holdout_frac=cfg.holdout_frac,
                evaluate_holdout=cfg.evaluate_holdout,
                group_col="_behavior_story_run",
            ),
            progress=log,
        )
        # لا تخلط: fold_metrics الشرطي منفصل؛ base-rate يبقى في diagnostics
        if science.fold_frame.height:
            fold_metrics = science.fold_frame
        # للتنبؤ التاريخي الصالح للباك تست فضّل OOF؛ وإلا الحي موثّق كغير مؤهل
        if science.conditional_oof_predictions.height:
            predictions = science.conditional_oof_predictions
        else:
            predictions = science.live_model_predictions
        log.op(
            f"science labeled={science.diagnostics.get('n_labeled', 0)} "
            f"oof={science.conditional_oof_predictions.height} "
            f"live={science.live_model_predictions.height}"
        )
    else:
        log.op("science skipped")

    log.step("validate_behavior_frame")
    validation = validate_behavior_frame(blended, fold_df=fold_metrics, progress=log)
    log.op(f"validation.ok={validation.ok} n_rows={validation.n_rows}")

    diagnostics: dict[str, Any] = {
        "n_mbo_rows": int(mbo.height),
        "n_state_bars": int(states.height),
        "n_signal_bars": int(signals.height),
        "n_events": int(events.height),
        "n_behavior_event_cols": len(BEHAVIOR_EVENT_COLUMNS),
        "n_state_feature_cols": len(STATE_FEATURE_COLUMNS),
        "n_structure_feature_cols": len(STRUCTURE_FEATURE_COLUMNS),
        "n_level_flow_cols": len(LEVEL_FLOW_COLUMNS),
        "n_reliability_cols": len(RELIABILITY_COLUMNS),
        "n_projection_bars": int(projection.height),
        "n_prediction_rows": int(predictions.height),
        "deceptive_scored_rows": scored_rows,
        "deceptive_filtered": False,
        "signal_quality_is_calibrated_probability": False,
        "probabilities_source": "train_only_walk_forward_base_rates",
        "conditional_probability_semantics": (
            "independent_binary_outcomes_not_a_joint_competing-risk_distribution"
        ),
        "fold_metrics_alias": "conditional_when_available_else_base_rate",
        "base_rate_fold_metrics_rows": int(base_rate_fold_metrics.height),
        "conditional_fold_metrics_rows": int(0 if science is None else science.fold_frame.height),
        "science": None if science is None else science.diagnostics,
        "causality": {
            "decision_columns_required": True,
            "profile_to_signal": "backward_asof_inside_auction_action_states",
            "deceptive_deletion": False,
            "reliability_deletion": False,
            "validation": "purged_walk_forward",
            "outcomes": "outcome_available_ts_gated_censored_excluded",
            "holdout": "frozen_final_tail",
            "holdout_evaluation": "explicit_opt_in_only",
            "prediction_vs_state": "separated",
            "oof_vs_live": "separated",
            "trade_outputs": False,
        },
        "config": {
            "profile_interval_ns": cfg.profile_interval_ns,
            "signal_interval_ns": cfg.signal_interval_ns,
            "fixed_range": cfg.fixed_range,
            "include_deceptive_scores": cfg.include_deceptive_scores,
            "include_level_flow": cfg.include_level_flow,
            "include_reliability_evidence": cfg.include_reliability_evidence,
            "n_splits": cfg.n_splits,
            "embargo": cfg.embargo,
            "purge_samples": cfg.purge_samples,
            "outcome_window": cfg.outcome_window,
            "include_asia_london_projection": cfg.include_asia_london_projection,
            "include_science": cfg.include_science,
            "holdout_frac": cfg.holdout_frac,
            "evaluate_holdout": cfg.evaluate_holdout,
            "projection_interval_ns": cfg.projection_config.interval_ns,
            "quiet": cfg.quiet,
            "progress_log_path": cfg.progress_log_path,
        },
    }

    log.done(f"bars={blended.height:,} events={events.height:,} ok={validation.ok}")
    return AuctionBehaviorResult(
        probabilities=probs,
        validation=validation,
        blended=blended,
        events=events,
        fold_metrics=fold_metrics,
        session_profiles=session_profiles,
        london_scenarios=london_scenarios,
        projection=projection,
        science=science,
        predictions=predictions,
        base_rate_fold_metrics=base_rate_fold_metrics,
        conditional_fold_metrics=(pl.DataFrame() if science is None else science.fold_frame),
        oof_predictions=(
            pl.DataFrame() if science is None else science.conditional_oof_predictions
        ),
        live_predictions=(pl.DataFrame() if science is None else science.live_model_predictions),
        diagnostics=diagnostics,
    )


def behavior_state_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """ما الذي أعرفه عن السوق الآن؟ — حالة فقط، بلا أعمدة ``p_*`` تنبؤية."""
    cols = [
        AVAILABILITY_TS,
        "signal_quality",
        "signal_evidence",
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
        *LEVEL_FLOW_COLUMNS,
        *RELIABILITY_COLUMNS,
        "mem_time_since_break",
        "mem_time_since_retest",
        "mem_dwell_inside_value",
        "mem_poc_migration_abs",
        "mem_value_transfer_gradual",
    ]
    frame = result.blended
    keep = [c for c in cols if c in frame.columns]
    out = frame.select(keep) if keep else pl.DataFrame()
    # ضمان فصل صارم: لا أعمدة احتمال شرطي داخل إطار الحالة
    leak = [c for c in out.columns if c.startswith("p_y_") or c.startswith("p_hat")]
    if leak:
        raise AssertionError(f"behavior_state_frame must not contain predictions: {leak}")
    return out


def behavior_prediction_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """احتمالات شرطية — يفضّل OOF المؤهل للباك تست إن وُجد.

    راجع ``prediction_is_oof`` / ``eligible_for_backtest`` / ``model_train_end_ts``.
    التنبؤ الحي (النموذج النهائي) ليس سلسلة تاريخية OOS.
    """
    if result.predictions.height > 0:
        return result.predictions
    if result.science is not None:
        if result.science.conditional_oof_predictions.height > 0:
            return result.science.conditional_oof_predictions
        if result.science.live_model_predictions.height > 0:
            return result.science.live_model_predictions
    return pl.DataFrame(
        schema={
            AVAILABILITY_TS: pl.Int64(),
            "prediction_source": pl.Utf8(),
            "prediction_is_oof": pl.Boolean(),
            "eligible_for_backtest": pl.Boolean(),
            "model_train_end_ts": pl.Int64(),
        }
    )


def behavior_oof_prediction_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """السلسلة التاريخية OOF فقط؛ آمنة للباك تست من ناحية أوزان النموذج."""
    if result.oof_predictions.height:
        return result.oof_predictions
    if result.science is not None:
        return result.science.conditional_oof_predictions
    return pl.DataFrame()


def behavior_live_prediction_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """تنبؤ النموذج النهائي للحالة الحية؛ لا يُستخدم كسلسلة باك تست تاريخية."""
    if result.live_predictions.height:
        return result.live_predictions
    if result.science is not None:
        return result.science.live_model_predictions
    return pl.DataFrame()


def behavior_competing_prediction_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """توزيع أول انتقال (مجموعه 1) — يفضّل OOF المؤهل للباك تست إن وُجد."""
    if result.science is not None:
        if result.science.competing_oof_predictions.height > 0:
            return result.science.competing_oof_predictions
        if result.science.competing_live_predictions.height > 0:
            return result.science.competing_live_predictions
    return pl.DataFrame(
        schema={
            AVAILABILITY_TS: pl.Int64(),
            "prediction_source": pl.Utf8(),
            "prediction_is_oof": pl.Boolean(),
            "eligible_for_backtest": pl.Boolean(),
            "model_train_end_ts": pl.Int64(),
        }
    )


def behavior_probabilities_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """توافق قديم → يُفضَّل :func:`behavior_state_frame` للحالة أو
    :func:`behavior_prediction_frame` للتنبؤ. يُرجع الحالة فقط (بلا ``p_*``).
    """
    return behavior_state_frame(result)


def behavior_probability_summary(result: AuctionBehaviorResult) -> pl.DataFrame:
    """صف واحد للتوقعات المجمعة مع وقت اكتمال التحقق الذي أتاحها.

    ``confidence`` هنا متوسط evidence (``signal_quality``) — ليس احتمالًا معايرًا.
    """
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
            "confidence_is_calibrated_probability": [False],
            "probability_source": ["train_only_walk_forward_base_rates"],
            "probabilities_are_joint_distribution": [False],
            "n_samples": [probs.n_samples],
            "detail": [probs.detail],
        }
    )


__all__ = [
    "AuctionBehaviorResult",
    "BehaviorConfig",
    "behavior_competing_prediction_frame",
    "behavior_live_prediction_frame",
    "behavior_oof_prediction_frame",
    "behavior_prediction_frame",
    "behavior_probabilities_frame",
    "behavior_probability_summary",
    "behavior_state_frame",
    "run_auction_behavior_analysis",
]
