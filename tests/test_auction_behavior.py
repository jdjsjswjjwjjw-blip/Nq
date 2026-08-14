"""اختبارات محرّك فهم سلوك المزاد — سببية + بلا مخرجات تداول."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior import (
    BehaviorConfig,
    run_auction_behavior_analysis,
)
from nq.auction_behavior.events import BEHAVIOR_EVENT_COLUMNS, build_behavior_events
from nq.auction_behavior.memory import attach_causal_memory
from nq.auction_behavior.pipeline import behavior_probabilities_frame
from nq.auction_behavior.quality import attach_signal_quality
from nq.auction_behavior.validate import validate_behavior_frame
from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.auction import auction_action_states
from nq.simulation.deceptive_liquidity import filter_deceptive_liquidity
from nq.validation.leakage import assert_causal_order
from tests.mbo_factory import make_stream


def _dense_trade_stream(*, n_bars: int = 40, bar_ns: int = 200) -> pl.DataFrame:
    """سلسلة صفقات كافية لبناء رينج+فعل (مصغّرة للاختبار)."""
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    seq = 1
    for i in range(n_bars * 4):
        px = 100 + (i % 5)
        events.append(("T", "B" if i % 2 == 0 else "A", px, 2, 0))
        ts.append((i // 4) * bar_ns + (i % 4))
        seq += 1
    return make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))


def test_behavior_pipeline_runs_without_trade_outputs() -> None:
    frame = _dense_trade_stream(n_bars=48, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            n_splits=3,
            min_train_size=6,
        ),
    )
    assert result.blended.height >= 1
    assert result.validation.ok
    assert result.validation.no_trade_outputs
    assert result.validation.causal_ok
    assert result.diagnostics["deceptive_filtered"] is False
    assert result.diagnostics["causality"]["trade_outputs"] is False
    for col in (
        "edge_pnl",
        "entry_gate",
        "stop_price",
        "target_price",
        "responsive_long",
    ):
        assert col not in result.blended.columns
    light = behavior_probabilities_frame(result)
    assert AVAILABILITY_TS in light.columns
    assert "signal_quality" in light.columns


def test_behavior_requires_decision_columns_on_states_path() -> None:
    frame = _dense_trade_stream(n_bars=24, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            n_splits=2,
            min_train_size=4,
        ),
    )
    assert result.validation.decision_bounds_present
    for col in ("decision_poc", "decision_vah", "decision_val"):
        assert col in result.blended.columns


def test_behavior_events_and_memory_are_causal() -> None:
    frame = _dense_trade_stream(n_bars=36, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            memory_lags=(1, 2),
        ),
    )
    assert_causal_order(result.blended[AVAILABILITY_TS].to_numpy())
    for col in BEHAVIOR_EVENT_COLUMNS:
        assert col in result.events.columns
    # ذاكرة سببية: lag1 في الصف i == قيمة الصف i-1
    if "vp_balance__lag1" in result.blended.columns and result.blended.height >= 2:
        bal = result.blended["vp_balance"].to_numpy()
        lag1 = result.blended["vp_balance__lag1"].to_numpy()
        # أول صف NaN/null؛ الباقي يطابق السابق
        for i in range(1, len(bal)):
            if np.isfinite(lag1[i]):
                assert lag1[i] == pytest.approx(bal[i - 1])


def test_behavior_path_never_calls_filter_deceptive(monkeypatch: pytest.MonkeyPatch) -> None:
    """مسار السلوك يسجّل درجات فقط — لا يحذف أحداثًا."""
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("filter_deceptive_liquidity must not run on behavior path")

    monkeypatch.setattr(
        "nq.simulation.deceptive_liquidity.filter_deceptive_liquidity",
        _boom,
    )
    # Also patch the name if pipeline imported it (it must not import filter at all).
    monkeypatch.setattr(
        "nq.auction_behavior.pipeline.filter_deceptive_liquidity",
        _boom,
        raising=False,
    )

    # include add/cancel so scoring has work; trades alone are fine too
    events = [("A", "B", 100, 5, 1), ("C", "B", 100, 5, 1)]
    for i in range(80):
        events.append(("T", "B" if i % 2 == 0 else "A", 100 + (i % 4), 2, 0))
    ts = list(range(len(events)))
    frame = make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))

    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=20,
            signal_interval_ns=10,
            fixed_range=False,
            include_deceptive_scores=True,
            n_splits=2,
            min_train_size=4,
        ),
    )
    assert called["n"] == 0
    assert result.diagnostics["deceptive_filtered"] is False
    assert "deceptive_score" in result.blended.columns
    # Sanity: filter still exists for other paths
    assert callable(filter_deceptive_liquidity)


def test_validate_rejects_trade_columns() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3],
            "decision_vah": [1.0, 1.0, 1.0],
            "decision_val": [0.0, 0.0, 0.0],
            "decision_poc": [0.5, 0.5, 0.5],
            "edge_pnl": [0.1, -0.2, 0.0],
        }
    )
    report = validate_behavior_frame(frame)
    assert report.ok is False
    assert report.no_trade_outputs is False


def test_empty_mbo_returns_empty_result() -> None:
    result = run_auction_behavior_analysis(make_stream([]))
    assert result.probabilities.n_samples == 0
    assert result.blended.height == 0
    assert result.validation.ok


def test_decision_bounds_lag_current_profile() -> None:
    """decision_* = shift(1) داخل جلسة السيولة (لا look-ahead داخل البرميل)."""
    frame = _dense_trade_stream(n_bars=30, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
        ),
    )
    states = auction_action_states(
        frame,
        profile_interval_ns=1000,
        signal_interval_ns=200,
        fixed_range=False,
    ).sort(AVAILABILITY_TS)
    assert states.height >= 2
    # decision_vah[i] == vah[i-1] عندما كلاهما غير null (على سلسلة جلسة واحدة في الاختبار)
    lagged = states.select(
        "vah",
        pl.col("vah").shift(1).alias("prev_vah"),
        "decision_vah",
    ).filter(pl.col("decision_vah").is_not_null() & pl.col("prev_vah").is_not_null())
    assert lagged.height >= 1
    assert (lagged["decision_vah"] == lagged["prev_vah"]).all()
    assert result.validation.decision_bounds_present


@pytest.mark.leakage
def test_behavior_availability_strictly_sorted() -> None:
    frame = _dense_trade_stream(n_bars=40, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
        ),
    )
    ts = result.blended[AVAILABILITY_TS].to_numpy()
    assert_causal_order(ts)
    assert np.all(np.diff(ts) >= 0)


def test_attach_helpers_compose() -> None:
    base = pl.DataFrame(
        {
            AVAILABILITY_TS: [10, 20, 30],
            "vp_fsm_break": [0.0, 1.0, 0.0],
            "vp_look_fail": [0.0, 0.0, 1.0],
            "vp_absorb": [0.0, 1.0, -1.0],
            "vp_fsm_retest": [0.0, 0.0, 1.0],
            "vp_fsm_expand": [0.0, 1.0, 0.0],
            "vp_fr_accepted_expansion": [0.0, 1.0, 0.0],
            "vp_fr_exit": [0.0, 0.0, 0.0],
            "vp_close_in_value": [1.0, 0.0, 1.0],
            "vp_imbalance": [0.0, 1.0, 0.0],
            "vp_expansion": [0.0, 1.0, 0.0],
            "vp_pullback_defense": [0.0, 0.0, 1.0],
            "vp_early_imbalance": [0.0, 1.0, 0.0],
        }
    )
    ev = build_behavior_events(base)
    assert ev["evt_breakout"].to_list()[1] == 1.0
    q = attach_signal_quality(base)
    assert "signal_quality" in q.columns
    mem = attach_causal_memory(q, columns=("signal_quality",), lags=(1,))
    assert "signal_quality__lag1" in mem.columns
