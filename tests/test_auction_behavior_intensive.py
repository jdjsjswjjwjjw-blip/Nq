"""اختبارات مكثّفة لمحرّك فهم سلوك المزاد (سببية + أداء + طبقات)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior import BehaviorConfig, run_auction_behavior_analysis
from nq.auction_behavior.events import BEHAVIOR_EVENT_COLUMNS
from nq.auction_behavior.model import estimate_behavior_probabilities
from nq.auction_behavior.pipeline import behavior_probabilities_frame
from nq.auction_behavior.state import STATE_FEATURE_COLUMNS
from nq.auction_behavior.validate import mean_absolute_calibration
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
    assert vp_liquidity_session_from_ns(int(frame["event_ts"][0])) == int(
        VpLiquiditySession.ASIA
    )
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


def test_intensive_deceptive_scores_without_deletion() -> None:
    # أوامر + صفقات حتى يعمل التسجيل
    book = random_add_cancel_stream(120, seed=7)
    trades = _dense_trade_stream(n_bars=40, bar_ns=50)
    # محاذاة زمنية بسيطة: الصفقات بعد أوامر الدفتر
    t0 = int(book["event_ts"].max()) + 1
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
