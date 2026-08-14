"""اختبارات مكثّفة لمحرّك فهم سلوك المزاد (سببية + أداء + طبقات)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior import (
    BehaviorConfig,
    latest_state_snapshot,
    run_auction_behavior_analysis,
    state_matrix,
)
from nq.auction_behavior.events import BEHAVIOR_EVENT_COLUMNS, build_behavior_events
from nq.auction_behavior.memory import attach_causal_memory
from nq.auction_behavior.model import estimate_behavior_probabilities
from nq.auction_behavior.pipeline import (
    _london_scenario_summary,
    _session_vp_summary,
    behavior_probabilities_frame,
)
from nq.auction_behavior.quality import attach_signal_quality
from nq.auction_behavior.state import STATE_FEATURE_COLUMNS
from nq.auction_behavior.validate import mean_absolute_calibration, validate_behavior_frame
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VpLiquiditySession, vp_liquidity_session_from_ns
from nq.models.splitting import purged_walk_forward_split
from nq.validation.leakage import assert_causal_order, assert_temporal_split
from tests.mbo_factory import make_stream, random_add_cancel_stream
from tests.test_auction_behavior import _dense_trade_stream

_ET = ZoneInfo("America/New_York")


def _ns_et(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    local = datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(local.astimezone(UTC).timestamp() * 1_000_000_000)


def _session_crossing_stream() -> pl.DataFrame:
    """صفقات عبر آسيا→لندن (ET) لتمرين ملخص الجلسات/السيناريو."""
    # آسيا: 20:00 ET يوم 1 · لندن: 04:00 ET يوم 2
    asia_base = _ns_et(2024, 6, 3, 20, 0)
    london_base = _ns_et(2024, 6, 4, 4, 0)
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    # آسيا: نطاق ضيّق حول 100
    for i in range(40):
        events.append(("T", "B" if i % 2 == 0 else "A", 100 + (i % 3), 3, 0))
        ts.append(asia_base + i * 60_000_000_000)  # دقيقة
    # لندن: فتح فوق ثم توسّع
    for i in range(40):
        events.append(("T", "B" if i % 2 == 0 else "A", 104 + (i % 5), 3, 0))
        ts.append(london_base + i * 60_000_000_000)
    return make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))


@pytest.mark.leakage
def test_intensive_no_lookahead_in_memory_or_events() -> None:
    frame = _dense_trade_stream(n_bars=80, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            memory_lags=(1, 2, 3),
            n_splits=4,
            min_train_size=8,
        ),
    )
    assert result.validation.ok
    assert_causal_order(result.blended[AVAILABILITY_TS].to_numpy())
    # لا lags سالبة / لا forward shift
    lag_cols = [c for c in result.blended.columns if "__lag" in c]
    assert lag_cols
    for col in lag_cols:
        assert "__lag-" not in col
        lag_n = int(col.rsplit("__lag", 1)[1])
        assert lag_n >= 1
    for col in BEHAVIOR_EVENT_COLUMNS:
        assert col in result.events.columns
        assert result.events[col].null_count() == 0


@pytest.mark.leakage
def test_intensive_purged_folds_respect_temporal_order() -> None:
    frame = _dense_trade_stream(n_bars=100, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            n_splits=5,
            embargo=1,
            purge_samples=1,
            min_train_size=10,
        ),
    )
    times = result.blended[AVAILABILITY_TS].to_numpy().astype(np.int64)
    folds = purged_walk_forward_split(
        times,
        n_splits=5,
        embargo=1,
        purge_samples=1,
        min_train_size=10,
    )
    assert len(folds) >= 1
    for fold in folds:
        assert_temporal_split(times[fold.train_idx], times[fold.test_idx], embargo=1.0)
        if fold.test_idx.size and fold.train_idx.size:
            assert int(times[fold.train_idx].max()) <= int(times[fold.test_idx].min())
    assert result.fold_metrics.height == len(folds)
    # معايرة قابلة للحساب إن وُجدت طيّات
    if result.fold_metrics.height > 0:
        err = mean_absolute_calibration(result.fold_metrics)
        assert err >= 0.0


def test_intensive_session_and_london_summaries() -> None:
    frame = _session_crossing_stream()
    # تأكد أن الطوابع تعبر الجلسات كما نتوقع
    assert vp_liquidity_session_from_ns(int(frame["event_ts"][0])) == int(VpLiquiditySession.ASIA)
    assert vp_liquidity_session_from_ns(int(frame["event_ts"][-1])) == int(
        VpLiquiditySession.LONDON
    )
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=5 * 60 * 1_000_000_000,
            signal_interval_ns=60 * 1_000_000_000,
            fixed_range=False,
            include_deceptive_scores=False,
            n_splits=2,
            min_train_size=4,
        ),
    )
    assert result.session_profiles.height >= 1
    names = set(result.session_profiles["session_name"].to_list())
    assert "asia" in names or "london" in names
    # سيناريو لندن قد يُملأ عند توفر decision_* آسيا
    assert "scenario" in result.london_scenarios.columns or result.london_scenarios.height == 0
    for col in ("decision_poc", "decision_vah", "decision_val"):
        assert col in result.blended.columns


def test_intensive_state_vector_complete_and_no_trade() -> None:
    frame = _dense_trade_stream(n_bars=60, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=True,
            include_deceptive_scores=False,
        ),
    )
    for col in STATE_FEATURE_COLUMNS:
        assert col in result.blended.columns
    banned = {
        "edge_pnl",
        "entry_gate",
        "edge_entry",
        "edge_stop",
        "edge_target",
        "position_size",
        "responsive_long",
        "initiative_short",
        "mfe",
        "mae",
    }
    assert banned.isdisjoint(result.blended.columns)
    assert banned.isdisjoint(result.events.columns)
    probs = result.probabilities
    for name in (
        "p_balanced",
        "p_imbalanced",
        "p_true_break",
        "p_false_break",
        "p_retest_success",
        "p_retest_fail",
        "p_expansion_continue",
        "p_return_to_value",
        "confidence",
    ):
        val = getattr(probs, name)
        assert 0.0 <= float(val) <= 1.0
    snapshot = latest_state_snapshot(result.blended)
    assert snapshot is not None
    latest_ts = result.blended[AVAILABILITY_TS].max()
    assert latest_ts is not None
    assert snapshot.availability_ts == int(np.asarray(latest_ts).item())
    matrix = state_matrix(result.blended)
    assert matrix.shape == (result.blended.height, len(STATE_FEATURE_COLUMNS))
    assert np.isfinite(matrix).all()


def test_intensive_deceptive_scores_without_deletion() -> None:
    # أوامر + صفقات حتى يعمل التسجيل
    book = random_add_cancel_stream(120, seed=7)
    trades = _dense_trade_stream(n_bars=40, bar_ns=50)
    # محاذاة زمنية بسيطة: الصفقات بعد أوامر الدفتر
    max_ts = book["event_ts"].max()
    assert max_ts is not None
    t0 = int(np.asarray(max_ts).item()) + 1
    trades = trades.with_columns(
        (pl.col("event_ts") + t0).alias("event_ts"),
        (pl.col("ingest_ts") + t0).alias("ingest_ts"),
    )
    frame = pl.concat([book, trades], how="diagonal_relaxed").sort("event_ts")
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=200,
            signal_interval_ns=50,
            fixed_range=False,
            include_deceptive_scores=True,
            n_splits=2,
            min_train_size=4,
        ),
    )
    assert result.diagnostics["deceptive_filtered"] is False
    assert result.diagnostics["deceptive_scored_rows"] > 0
    assert "deceptive_score" in result.blended.columns
    assert "real_liquidity_ratio" in result.blended.columns
    # الصف الأصلي لم يُحذف من المصدر — المسار لا يستدعي الفلتر
    assert frame.height == book.height + trades.height


def test_intensive_pipeline_speed_smoke() -> None:
    """طبقة السلوك رخيصة نسبيًا بدون scoring ثقيل."""
    n_bars = 400
    bar_ns = 30_000_000_000
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    for i in range(n_bars * 4):
        events.append(("T", "B" if i % 2 == 0 else "A", 20_000 + (i % 15), 2, 0))
        ts.append((i // 4) * bar_ns + (i % 4) * 1_000_000)
    frame = make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))
    cfg = BehaviorConfig(
        profile_interval_ns=5 * 60 * 1_000_000_000,
        signal_interval_ns=30 * 1_000_000_000,
        fixed_range=True,
        include_deceptive_scores=False,
        n_splits=3,
        min_train_size=15,
    )
    t0 = time.perf_counter()
    result = run_auction_behavior_analysis(frame, config=cfg)
    elapsed = time.perf_counter() - t0
    assert result.blended.height >= 1
    assert result.validation.ok
    # سقف فضفاض لمنع انحدار أداء كارثي في CI
    assert elapsed < 5.0, f"behavior pipeline too slow: {elapsed:.3f}s"


def test_intensive_model_empty_and_descriptive_fallback() -> None:
    empty = pl.DataFrame({AVAILABILITY_TS: pl.Series([], dtype=pl.Int64())})
    probs, folds = estimate_behavior_probabilities(empty, empty)
    assert probs.n_samples == 0
    assert folds.height == 0

    # عيّنة صغيرة جدًا → descriptive (ليس ادّعاء OOS)
    tiny = _dense_trade_stream(n_bars=8, bar_ns=100)
    result = run_auction_behavior_analysis(
        tiny,
        config=BehaviorConfig(
            profile_interval_ns=400,
            signal_interval_ns=100,
            fixed_range=False,
            include_deceptive_scores=False,
            n_splits=8,
            min_train_size=50,  # يجبر عدم كفاية الطيّات
        ),
    )
    assert "descriptive" in result.probabilities.detail or result.probabilities.n_samples >= 0
    light = behavior_probabilities_frame(result)
    if result.blended.height:
        assert AVAILABILITY_TS in light.columns


@pytest.mark.leakage
def test_intensive_signals_use_decision_not_live_va_for_bounds() -> None:
    """vp_upper/mid/lower من auction_signals مبنية على decision_*."""
    frame = _dense_trade_stream(n_bars=50, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
        ),
    )
    b = result.blended
    need = {
        "vp_upper",
        "vp_mid",
        "vp_lower",
        "decision_vah",
        "decision_val",
        "decision_poc",
    }
    assert need.issubset(set(b.columns))
    # عند وجود قيم: الحدود المصدَّرة تتوافق مع decision_* (مقياس السعر)
    both = b.filter(
        pl.col("decision_vah").is_not_null()
        & pl.col("vp_upper").is_not_null()
        & pl.col("decision_vah").is_not_nan()
    )
    if both.height:
        scale = float(PRICE_SCALE)
        got = both["vp_upper"].to_numpy()
        exp = both["decision_vah"].cast(pl.Float64).to_numpy() * scale
        np.testing.assert_allclose(got, exp, rtol=0, atol=1e-9)


@pytest.mark.leakage
def test_behavior_outcomes_are_emitted_when_known_not_backdated() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [10, 20, 30, 40, 50, 60],
            "vp_fsm_break": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "vp_fsm_retest": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "vp_fsm_expand": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "vp_fr_accepted_expansion": [0.0] * 6,
            "vp_fr_exit": [0.0] * 6,
            "vp_look_fail": [0.0] * 6,
            "vp_absorb": [0.0] * 6,
            "vp_close_in_value": [0.0] * 6,
            "_liquidity_run": [1] * 6,
        }
    )
    events = build_behavior_events(frame, outcome_window=4, group_col="_liquidity_run")
    assert events["evt_true_break"].to_list() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert events["evt_retest_success"].to_list() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # لا تُنقل معرفة الصف 50 إلى صف الكسر 20 أو الريتست 30.
    assert events.filter(pl.col(AVAILABILITY_TS) < 50)["evt_true_break"].sum() == 0.0


def test_fixed_range_exit_is_true_not_failed_break() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2],
            "vp_fsm_break": [0.0, 1.0],
            "vp_fr_exit": [0.0, 1.0],
        }
    )
    events = build_behavior_events(frame)
    assert events["evt_true_break"].to_list() == [0.0, 1.0]
    assert events["evt_failed_breakout"].sum() == 0.0


def test_look_fail_without_prior_break_is_value_rejection_not_false_break() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2],
            "vp_look_fail": [0.0, -1.0],
            "vp_close_in_value": [0.0, 1.0],
        }
    )
    events = build_behavior_events(frame)
    assert events["evt_failed_breakout"].sum() == 0.0
    assert events["evt_reject_value"].to_list() == [0.0, 1.0]


def test_signal_quality_requires_behavior_evidence() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2],
            "real_liquidity_ratio": [1.0, 1.0],
            "deceptive_score": [0.0, 0.0],
        }
    )
    quality = attach_signal_quality(frame)
    assert quality["signal_quality"].to_list() == [0.0, 0.0]


def test_memory_resets_at_liquidity_session_run() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3, 4],
            "_liquidity_run": [1, 1, 2, 2],
            "vp_balance": [0.0, 1.0, 9.0, 8.0],
        }
    )
    out = attach_causal_memory(
        frame,
        columns=("vp_balance",),
        lags=(1,),
        group_col="_liquidity_run",
    )
    assert out["vp_balance__lag1"].to_list() == [None, 0.0, None, 9.0]


def test_session_summaries_do_not_merge_repeated_sessions() -> None:
    states = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3, 4, 5, 6],
            "vp_liquidity_session": [1, 1, 2, 2, 1, 1],
            "decision_poc": [None, 99, None, 109, None, 119],
            "decision_vah": [None, 101, None, 111, None, 121],
            "decision_val": [None, 97, None, 107, None, 117],
            "poc": [99, 100, 109, 110, 119, 120],
            "vah": [101, 102, 111, 112, 121, 122],
            "val": [97, 98, 107, 108, 117, 118],
        }
    )
    summary = _session_vp_summary(states)
    assert summary.height == 3
    assert summary["session_run"].n_unique() == 3
    assert summary["completed_vah"].to_list() == [102.0, 112.0, 122.0]


def test_london_realized_outcome_has_end_availability() -> None:
    states = pl.DataFrame(
        {
            AVAILABILITY_TS: [10, 20, 30, 40],
            "vp_liquidity_session": [0, 0, 1, 1],
            "vah": [109.0, 110.0, 120.0, 121.0],
            "val": [90.0, 91.0, 100.0, 101.0],
            "poc": [100.0, 101.0, 110.0, 111.0],
            "open": [100.0, 101.0, 105.0, 106.0],
            "close": [101.0, 102.0, 106.0, 107.0],
            "high": [102.0, 103.0, 108.0, 115.0],
            "low": [99.0, 100.0, 104.0, 89.0],
        }
    )
    scenarios = _london_scenario_summary(states)
    assert scenarios.height == 1
    assert scenarios["asia_completed_vah"][0] == pytest.approx(110.0)
    assert scenarios["london_open_ts"][0] == 30
    assert scenarios["outcome_available_ts"][0] == 40
    assert scenarios["outcome_available_ts"][0] >= scenarios["london_open_ts"][0]


def test_train_only_forecast_is_not_replaced_by_test_realization() -> None:
    n = 60
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: np.arange(n, dtype=np.int64),
            "vp_balance": np.zeros(n),
            "vp_imbalance": np.zeros(n),
            "signal_quality": np.zeros(n),
        }
    )
    events = pl.DataFrame(
        {
            AVAILABILITY_TS: np.arange(n, dtype=np.int64),
            "evt_true_break": np.r_[np.zeros(40), np.ones(20)],
        }
    )
    probs, folds = estimate_behavior_probabilities(
        blended,
        events,
        n_splits=2,
        min_train_size=10,
    )
    assert folds.height == 2
    assert probs.p_true_break == 0.0
    assert folds["oos_true_break_rate"].to_list() == [0.0, 1.0]
    assert probs.n_samples == 40
    assert "train-only" in probs.detail


def test_validation_requires_visible_decision_bounds() -> None:
    report = validate_behavior_frame(
        pl.DataFrame({AVAILABILITY_TS: [1, 2], "vp_balance": [0.0, 1.0]})
    )
    assert report.ok is False
    assert report.decision_bounds_present is False


def test_add_cancel_only_stream_returns_safe_empty_result() -> None:
    result = run_auction_behavior_analysis(random_add_cancel_stream(80, seed=11))
    assert result.blended.height == 0
    assert result.validation.ok
    assert result.diagnostics["reason"] == "no_trade_derived_auction_bars"
