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

import polars as pl

from nq.auction_behavior.events import BEHAVIOR_EVENT_COLUMNS, build_behavior_events
from nq.auction_behavior.memory import attach_causal_memory
from nq.auction_behavior.model import estimate_behavior_probabilities
from nq.auction_behavior.quality import attach_signal_quality
from nq.auction_behavior.state import STATE_FEATURE_COLUMNS, attach_state_vector
from nq.auction_behavior.types import BehaviorProbabilities
from nq.auction_behavior.validate import BehaviorValidationReport, validate_behavior_frame
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession, vp_liquidity_session_label
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
)

_TRADE_FORBIDDEN = (
    "edge_pnl",
    "entry_gate",
    "edge_entry",
    "edge_stop",
    "edge_target",
    "position_size",
    "responsive_long",
    "responsive_short",
    "initiative_long",
    "initiative_short",
    "stop_price",
    "target_price",
    "mfe",
    "mae",
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
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _require_decision_columns(states: pl.DataFrame) -> None:
    need = ("decision_poc", "decision_vah", "decision_val")
    missing = [c for c in need if c not in states.columns]
    if missing:
        raise ValueError(
            "auction_behavior requires lagged decision_* columns "
            f"(missing={missing}). Never use current-bar vah/val/poc for decisions."
        )


def _assert_no_trade_columns(frame: pl.DataFrame) -> None:
    hit = [c for c in _TRADE_FORBIDDEN if c in frame.columns]
    if hit:
        raise ValueError(f"auction_behavior must not emit trade columns; found {sorted(hit)}")


def _session_vp_summary(states: pl.DataFrame) -> pl.DataFrame:
    """ملخص بروفايل لكل جلسة سيولة (وصفي — حدود مكتملة decision_* عند آخر برميل)."""
    if states.height == 0 or VP_LIQUIDITY_SESSION not in states.columns:
        return pl.DataFrame(
            schema={
                "liquidity_session": pl.Int64(),
                "session_name": pl.Utf8(),
                "n_bars": pl.Int64(),
                "decision_poc": pl.Float64(),
                "decision_vah": pl.Float64(),
                "decision_val": pl.Float64(),
                "va_width": pl.Float64(),
            }
        )

    work = states.sort(AVAILABILITY_TS)
    rows: list[dict[str, Any]] = []
    for sess_id, g in work.group_by(VP_LIQUIDITY_SESSION, maintain_order=True):
        sid = int(sess_id[0]) if isinstance(sess_id, tuple) else int(sess_id)
        last = g.tail(1)
        vah = last["decision_vah"][0] if "decision_vah" in last.columns else None
        val = last["decision_val"][0] if "decision_val" in last.columns else None
        poc = last["decision_poc"][0] if "decision_poc" in last.columns else None
        va_w = float(vah) - float(val) if vah is not None and val is not None else None
        rows.append(
            {
                "liquidity_session": sid,
                "session_name": vp_liquidity_session_label(sid),
                "n_bars": int(g.height),
                "decision_poc": float(poc) if poc is not None else None,
                "decision_vah": float(vah) if vah is not None else None,
                "decision_val": float(val) if val is not None else None,
                "va_width": va_w,
            }
        )
    return pl.DataFrame(rows)


def _london_scenario_summary(states: pl.DataFrame) -> pl.DataFrame:
    """سيناريو لندن مقابل قيمة آسيا المكتملة (وصفي؛ بلا توصية صفقة).

    يستخدم آخر decision_* داخل آسيا كحدود معروفة قبل/عند أول برميل لندن في
    نفس تسلسل الجلسات — لا يعيد حساب VA من مستقبل لندن.
    """
    schema = {
        "asia_end_ts": pl.Int64(),
        "london_open_ts": pl.Int64(),
        "scenario": pl.Utf8(),
        "london_open": pl.Float64(),
        "asia_decision_poc": pl.Float64(),
        "asia_decision_vah": pl.Float64(),
        "asia_decision_val": pl.Float64(),
        "open_inside_asia_va": pl.Boolean(),
        "london_broke_asia_vah": pl.Boolean(),
        "london_broke_asia_val": pl.Boolean(),
    }
    need = {VP_LIQUIDITY_SESSION, "decision_vah", "decision_val", "close", AVAILABILITY_TS}
    if states.height == 0 or not need.issubset(states.columns):
        return pl.DataFrame(schema=schema)

    ordered = states.sort(AVAILABILITY_TS)
    asia_code = int(VpLiquiditySession.ASIA)
    london_code = int(VpLiquiditySession.LONDON)

    # تقسيم إلى شرائح جلسة متصلة (آسيا→لندن→نيويورك→آسيا…)
    runs = ordered.with_columns(
        (pl.col(VP_LIQUIDITY_SESSION) != pl.col(VP_LIQUIDITY_SESSION).shift(1).fill_null(-1))
        .cast(pl.Int64)
        .cum_sum()
        .alias("_sess_run")
    )

    rows: list[dict[str, Any]] = []
    # اجمع أزواج: آخر آسيا قبل أول لندن التالية
    asia_last: pl.DataFrame | None = None
    for _run_id, g in runs.group_by("_sess_run", maintain_order=True):
        sess = int(g[VP_LIQUIDITY_SESSION][0])
        if sess == asia_code:
            asia_last = g.tail(1)
            continue
        if sess != london_code or asia_last is None:
            continue

        a_vah = asia_last["decision_vah"][0]
        a_val = asia_last["decision_val"][0]
        a_poc = asia_last["decision_poc"][0] if "decision_poc" in asia_last.columns else None
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

        open_f = float(open_px)
        vah_f = float(a_vah)
        val_f = float(a_val)
        inside = bool(val_f <= open_f <= vah_f)
        if open_f > vah_f:
            scenario = "open_above_asia_vah"
        elif open_f < val_f:
            scenario = "open_below_asia_val"
        else:
            scenario = "inside_asia_va"

        hi = g["high"].max() if "high" in g.columns else None
        lo = g["low"].min() if "low" in g.columns else None
        broke_vah = bool(hi is not None and float(hi) > vah_f)
        broke_val = bool(lo is not None and float(lo) < val_f)

        rows.append(
            {
                "asia_end_ts": int(asia_last[AVAILABILITY_TS][0]),
                "london_open_ts": int(lon0[AVAILABILITY_TS][0]),
                "scenario": scenario,
                "london_open": open_f,
                "asia_decision_poc": float(a_poc) if a_poc is not None else None,
                "asia_decision_vah": vah_f,
                "asia_decision_val": val_f,
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
            diagnostics={"empty": True, "deceptive_filtered": False},
        )

    # Layers 1–3: session-aware auction states (decision_* lagged inside builder)
    states = auction_action_states(
        mbo,
        profile_interval_ns=cfg.profile_interval_ns,
        signal_interval_ns=cfg.signal_interval_ns,
        fixed_range=cfg.fixed_range,
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

    events = build_behavior_events(blended)

    # Layers 7–9: quality → memory → state vector
    blended = attach_signal_quality(blended)
    mem_cols = [c for c in _MEMORY_BASE_COLS if c in blended.columns]
    blended = attach_causal_memory(blended, columns=mem_cols, lags=cfg.memory_lags)
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
        diagnostics=diagnostics,
    )


def behavior_probabilities_frame(result: AuctionBehaviorResult) -> pl.DataFrame:
    """إطار خفيف: availability_ts + أعمدة حالة/جودة مفيدة للفحص (بلا تداول)."""
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
    ]
    frame = result.blended
    keep = [c for c in cols if c in frame.columns]
    return frame.select(keep) if keep else pl.DataFrame()


__all__ = [
    "AuctionBehaviorResult",
    "BehaviorConfig",
    "behavior_probabilities_frame",
    "run_auction_behavior_analysis",
]
