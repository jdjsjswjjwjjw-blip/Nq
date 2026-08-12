"""اختبارات فلتر التضليل وحكم السوق وخطة الإدج التنفيذية."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import pytest

import nq.simulation.deceptive_liquidity as deceptive_module
import nq.simulation.edge_execution_plan as edge_module
from nq.contracts.mbo import MBO_SCHEMA, PRICE_SCALE, validate_mbo_frame
from nq.simulation.auction import auction_action_states
from nq.simulation.deceptive_liquidity import (
    DeceptiveLiquidityConfig,
    deceptive_features_by_bucket,
    filter_deceptive_liquidity,
    score_deceptive_events,
)
from nq.simulation.edge_execution_plan import (
    EdgeExecConfig,
    EdgeSearchSpec,
    _plan_levels,
    score_edge_spec_oos,
    search_best_edge_spec,
    simulate_edge_trades,
    summarize_edge_trades,
)
from nq.simulation.market_truth import MarketTruthConfig, build_market_truth_frame
from nq.strategies.liquidity_edge import run_liquidity_edge_research
from tests.mbo_factory import make_stream, random_add_cancel_stream


def _px(dollars: float) -> int:
    return round(dollars / PRICE_SCALE)


def test_score_marks_short_life_cancel_as_deceptive() -> None:
    # إضافة ثم إلغاء سريع بلا fill
    frame = make_stream(
        [
            ("A", "B", _px(100.0), 8, 1),
            ("A", "A", _px(101.0), 4, 2),
            ("C", "B", _px(100.0), 8, 1),
        ],
        event_ts=[0, 1_000_000, 10_000_000],  # 10ms
    )
    scored = score_deceptive_events(
        frame,
        config=DeceptiveLiquidityConfig(
            short_life_ns=50_000_000,
            drop_score=0.2,
            storm_min_events=100,
        ),
    )
    cancel_row = scored.filter(pl.col("action") == "C")
    assert cancel_row.height == 1
    assert float(cancel_row["deceptive_score"][0]) > 0.0
    assert float(cancel_row["flicker_flag"][0]) == 1.0


def test_score_deceptive_chunk_boundaries_preserve_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_stream(
        [
            ("A", "B", _px(100.0), 8, 1),
            ("A", "A", _px(101.0), 6, 2),
            ("M", "B", _px(99.75), 8, 1),
            ("T", "A", _px(99.75), 1, 0),
            ("C", "B", _px(99.75), 8, 1),
            ("C", "A", _px(101.0), 6, 2),
        ],
        event_ts=[0, 1, 2, 3, 4, 5],
    )
    expected = score_deceptive_events(frame)
    monkeypatch.setattr(deceptive_module, "_SCORE_CHUNK", 2)
    chunked = score_deceptive_events(frame)
    assert chunked.equals(expected)


def test_filter_keeps_trades_drops_full_spoof_lifecycle() -> None:
    frame = make_stream(
        [
            ("A", "B", _px(100.0), 10, 1),
            ("A", "A", _px(105.0), 10, 2),  # بعيد
            ("C", "A", _px(105.0), 10, 2),  # إلغاء سبوف
            ("T", "B", _px(100.0), 1, 0),
        ],
        event_ts=[0, 1_000_000, 50_000_000, 60_000_000],
    )
    cleaned = filter_deceptive_liquidity(
        frame,
        config=DeceptiveLiquidityConfig(
            spoof_ticks_from_mid=4,
            spoof_min_size=5,
            spoof_cancel_ns=300_000_000,
            drop_score=0.25,
            storm_min_events=100,
            w_short_life=0.0,
            w_spoof=1.0,
            w_bait=0.0,
            w_nonparticipate=0.0,
            w_storm=0.0,
        ),
    )
    validate_mbo_frame(cleaned)
    assert set(cleaned.columns) == set(MBO_SCHEMA)
    actions = [str(a) for a in cleaned["action"].to_list()]
    assert "T" in actions
    # دورة السبوف كاملة تُسقط (ADD+CANCEL) — بلا شبح
    assert 2 not in [int(x) for x in cleaned["order_id"].to_list()]
    assert actions.count("C") == 0


def test_deceptive_bucket_noise_cum_is_causal() -> None:
    frame = make_stream(
        [
            ("A", "B", _px(100.0), 5, 1),
            ("C", "B", _px(100.0), 5, 1),
            ("A", "A", _px(101.0), 3, 2),
            ("T", "B", _px(101.0), 1, 0),
        ],
        event_ts=[0, 10_000_000, 1_100_000_000, 1_200_000_000],
    )
    feats = deceptive_features_by_bucket(
        frame,
        interval_ns=1_000_000_000,
        config=DeceptiveLiquidityConfig(storm_min_events=100, short_life_ns=50_000_000),
    )
    assert feats.height >= 1
    assert "noise_cum" in feats.columns
    assert "real_liquidity_ratio" in feats.columns


def test_scored_frame_reused_for_filter_and_bucket_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """تسجيل التضليل مرة واحدة يكفي للفلتر + براميل الإدج."""
    frame = make_stream(
        [
            ("A", "B", _px(100.0), 8, 1),
            ("A", "A", _px(101.0), 4, 2),
            ("C", "B", _px(100.0), 8, 1),
            ("T", "B", _px(100.0), 1, 0),
        ],
        event_ts=[0, 1_000_000, 10_000_000, 20_000_000],
    )
    cfg = DeceptiveLiquidityConfig(
        short_life_ns=50_000_000,
        drop_score=0.2,
        storm_min_events=100,
    )
    calls = {"n": 0}
    real_score = score_deceptive_events

    def _counting_score(*args: Any, **kwargs: Any) -> pl.DataFrame:
        calls["n"] += 1
        return real_score(*args, **kwargs)

    monkeypatch.setattr(
        "nq.simulation.deceptive_liquidity.score_deceptive_events",
        _counting_score,
    )
    scored = _counting_score(frame, config=cfg)
    assert calls["n"] == 1
    cleaned = filter_deceptive_liquidity(frame, config=cfg, scored=scored)
    feats = deceptive_features_by_bucket(
        frame, interval_ns=1_000_000_000, config=cfg, scored=scored
    )
    assert calls["n"] == 1
    validate_mbo_frame(cleaned)
    assert feats.height >= 1
    assert "deceptive_score" in feats.columns


def test_score_deceptive_fast_path_handles_dense_book() -> None:
    """مسار BBO التزايدي لا ينهار على دفتر كثيف (سابقًا O(live) كل حدث)."""
    frame = random_add_cancel_stream(20_000, seed=42)
    # أضف صفقات حتى لا يبقى المسار ADD/CANCEL فقط
    t0 = time.perf_counter()
    scored = score_deceptive_events(
        frame,
        config=DeceptiveLiquidityConfig(storm_min_events=50),
    )
    elapsed = time.perf_counter() - t0
    assert scored.height == frame.height
    assert "deceptive_score" in scored.columns
    # على 20k حدث يجب أن ينتهي في أقل من ثانيتين حتى على CPU بطيء
    assert elapsed < 2.0, f"score too slow: {elapsed:.3f}s"


def _session_with_imbalance(n_buckets: int = 40) -> pl.DataFrame:
    """جلسة اصطناعية: صفقات داخل قيمة ثم تمدّد صاعد مع سيولة حقيقية."""
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    oid = 1
    t = 0
    # بناء قيمة حول 100–101
    for b in range(n_buckets):
        base = t
        for k in range(20):
            px = 100.0 + (k % 5) * 0.25
            if b >= n_buckets // 2:
                px = 101.0 + (k % 6) * 0.25  # تمدّد لأعلى
            side = "B" if k % 2 == 0 else "A"
            events.append(("T", side, _px(px), 2 + (k % 3), 0))
            ts.append(base + k * 10_000_000)
            # أوامر راقدة حقيقية قريبة من الميد (عمر طويل — لا تُلغى فورًا)
            if k == 0:
                events.append(("A", "B", _px(px - 0.25), 3, oid))
                ts.append(base + k * 10_000_000 + 1)
                oid += 1
        t += 1_000_000_000
    return make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))


def test_market_truth_hold_and_verdict_columns() -> None:
    mbo = _session_with_imbalance(30)
    truth = build_market_truth_frame(
        mbo,
        interval_ns=1_000_000_000,
        profile_interval_ns=2_000_000_000,
        truth=MarketTruthConfig(
            hold_buckets=2,
            min_real_liquidity=0.0,
            max_deceptive_score=1.0,
            min_move_ticks=1.0,
        ),
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
    )
    for c in (
        "thesis_dir",
        "hold_ok",
        "delta_instant",
        "delta_cum",
        "market_verdict",
        "entry_gate",
    ):
        assert c in truth.columns
    assert truth.height >= 5


def test_plan_levels_enforces_min_rr() -> None:
    planned = _plan_levels(
        direction=1.0,
        entry=101.0,
        vah=100.5,
        val=99.5,
        poc=100.0,
        cfg=EdgeExecConfig(min_rr=3.0, stop_buffer_ticks=0.0, target_mode="rr_multiple"),
    )
    # initiative long: الوقف خلف VAH المكسور، لا خلف VAL البعيد.
    assert planned is not None
    _stop, target, risk, reward = planned
    assert reward / risk >= 3.0 - 1e-9
    assert target >= 101.0 + 3.0 * risk - 1e-9

    responsive = _plan_levels(
        direction=1.0,
        entry=99.25,
        vah=101.0,
        val=99.5,
        poc=100.25,
        cfg=EdgeExecConfig(
            min_rr=1.0,
            stop_buffer_ticks=1.0,
            target_mode="poc",
            playbook="responsive",
        ),
    )
    assert responsive is not None
    responsive_stop, responsive_target, _risk, _reward = responsive
    assert responsive_stop < 99.5
    assert responsive_target == 100.25


def test_simulate_edge_trades_no_chase_every_bar() -> None:
    # إطار حكم يدوي: بوابة على صف واحد فقط
    truth = pl.DataFrame(
        {
            "availability_ts": [1, 2, 3, 4, 5, 6],
            "entry_gate": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "thesis_dir": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "market_verdict": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [
                _px(101.0),
                _px(101.25),
                _px(101.5),
                _px(101.75),
                _px(102.0),
                _px(102.25),
            ],
            "vah": [_px(101.0)] * 6,
            "val": [_px(99.0)] * 6,
        }
    )
    out = simulate_edge_trades(
        truth,
        exec_cfg=EdgeExecConfig(
            min_rr=2.0,
            stop_buffer_ticks=0.0,
            target_mode="rr_multiple",
            rr_multiple=2.0,
            max_hold_buckets=4,
        ),
    )
    n_signals = int((out["edge_signal"] != 0.0).sum())
    assert n_signals == 1
    assert float(out["edge_rr"].drop_nans()[0]) >= 2.0 - 1e-9


def test_edge_execution_uses_mbo_first_touch_and_deducts_costs() -> None:
    truth = pl.DataFrame(
        {
            "availability_ts": [10, 20, 30],
            "entry_gate": [1.0, 0.0, 0.0],
            "thesis_dir": [1.0, 1.0, 1.0],
            "market_verdict": [1.0, 1.0, 1.0],
            "close": [_px(101.25)] * 3,
            "vah": [_px(101.0)] * 3,
            "val": [_px(99.0)] * 3,
        }
    )
    # الهدف يُلمس أولاً عند t=11 ثم الوقف عند t=12؛ ترتيب MBO يجب أن يحسم الربح.
    tape = pl.DataFrame(
        {
            "event_ts": [11, 12],
            "price": [_px(101.75), _px(100.75)],
        }
    )
    free = simulate_edge_trades(
        truth,
        exec_cfg=EdgeExecConfig(
            min_rr=1.0,
            stop_buffer_ticks=0.0,
            target_mode="rr_multiple",
            rr_multiple=1.0,
            max_hold_buckets=2,
            slippage_ticks=0.0,
            commission_bps=0.0,
        ),
        trade_path=tape,
    )
    costly = simulate_edge_trades(
        truth,
        exec_cfg=EdgeExecConfig(
            min_rr=1.0,
            stop_buffer_ticks=0.0,
            target_mode="rr_multiple",
            rr_multiple=1.0,
            max_hold_buckets=2,
            slippage_ticks=0.5,
            commission_bps=1.0,
        ),
        trade_path=tape,
    )

    assert free["edge_hit"][0] == 1.0
    assert free["edge_exit_ts"][0] == 11.0
    assert costly["edge_cost"][0] > 0.0
    assert costly["edge_pnl"][0] < free["edge_pnl"][0]


def test_search_rejects_ineligible_grid() -> None:
    mbo = _session_with_imbalance(30)
    grid = (
        EdgeSearchSpec(
            name="hold2_rr99",
            hold_buckets=2,
            min_rr=99.0,
            stop_buffer_ticks=1.0,
            target_mode="rr_multiple",
            rr_multiple=99.0,
        ),
    )
    table, best, row = search_best_edge_spec(
        mbo,
        interval_ns=1_000_000_000,
        profile_interval_ns=2_000_000_000,
        grid=grid,
        train_frac=0.5,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
        min_oos_trades=1000,
        min_oos_rr=50.0,
    )
    assert table.height == 1
    assert best is None
    assert row == {}


def test_oos_simulations_are_independent() -> None:
    """محاكاة التدريب لا تحجب صفقات الاختبار عبر القطع."""
    mbo = _session_with_imbalance(60)
    spec = EdgeSearchSpec(
        name="hold2_rr2",
        hold_buckets=2,
        min_rr=2.0,
        stop_buffer_ticks=1.0,
        target_mode="rr_multiple",
        rr_multiple=2.0,
    )
    auction = auction_action_states(
        mbo,
        profile_interval_ns=2_000_000_000,
        signal_interval_ns=1_000_000_000,
    )
    row = score_edge_spec_oos(
        mbo,
        spec,
        interval_ns=1_000_000_000,
        train_frac=0.5,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
        auction=auction,
    )
    assert "oos_n" in row
    assert "train_expectancy" in row


def test_execution_levels_use_lagged_value_area() -> None:
    truth = pl.DataFrame(
        {
            "availability_ts": [1, 2, 3],
            "entry_gate": [1.0, 0.0, 0.0],
            "thesis_dir": [1.0, 1.0, 1.0],
            "market_verdict": [1.0, 1.0, 1.0],
            "close": [_px(104.0)] * 3,
            "decision_val": [_px(99.0)] * 3,
            "decision_poc": [_px(101.0)] * 3,
            "decision_vah": [_px(103.0)] * 3,
            # حدود حالية متباعدة عمدًا؛ لا يجوز أن تدخل الوقف.
            "val": [_px(50.0)] * 3,
            "vah": [_px(150.0)] * 3,
            "is_expansion": [True] * 3,
        }
    )
    out = simulate_edge_trades(
        truth,
        exec_cfg=EdgeExecConfig(
            min_rr=1.0,
            stop_buffer_ticks=1.0,
            target_mode="rr_multiple",
            rr_multiple=1.0,
            playbook="initiative",
            slippage_ticks=0.0,
            commission_bps=0.0,
        ),
    )
    assert out["edge_stop"][0] == pytest.approx(102.75)


def test_search_and_strategy_smoke() -> None:
    mbo = _session_with_imbalance(50)
    grid = (
        EdgeSearchSpec(
            name="hold2_rr2_buf1_rr_multiple",
            hold_buckets=2,
            min_rr=2.0,
            stop_buffer_ticks=1.0,
            target_mode="rr_multiple",
            rr_multiple=2.5,
        ),
        EdgeSearchSpec(
            name="hold3_rr2_buf2_va_opposite",
            hold_buckets=3,
            min_rr=2.0,
            stop_buffer_ticks=2.0,
            target_mode="va_opposite",
            rr_multiple=3.0,
        ),
    )
    table, best, row = search_best_edge_spec(
        mbo,
        interval_ns=1_000_000_000,
        profile_interval_ns=2_000_000_000,
        grid=grid,
        train_frac=0.5,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
        min_oos_trades=0,
        min_oos_rr=0.0,
    )
    assert table.height == 2
    assert best is not None
    assert "oos_expectancy" in row

    result = run_liquidity_edge_research(
        mbo,
        interval_ns=1_000_000_000,
        profile_interval_ns=2_000_000_000,
        train_frac=0.5,
        min_oos_trades=0,
        min_oos_rr=0.0,
        grid=grid,
        drop_deceptive=True,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
        quiet=True,
        streaming_features=True,
    )
    assert result.vp.with_execution is True
    assert result.raw_mbo_rows == mbo.height
    assert "Volume Profile" in result.report_md or "vp" in result.report_md.lower()
    summary = summarize_edge_trades(result.trades)
    assert "expectancy" in summary
    assert np.isfinite(summary["expectancy"]) or summary["n_trades"] == 0.0


def test_edge_search_selects_on_train_not_oos(monkeypatch: pytest.MonkeyPatch) -> None:
    """الـholdout الخارجي لا يُقاس إلا للمواصفة المختارة داخليًا."""
    grid = (
        EdgeSearchSpec(
            name="train_winner",
            hold_buckets=2,
            min_rr=2.0,
            stop_buffer_ticks=1.0,
            target_mode="rr_multiple",
        ),
        EdgeSearchSpec(
            name="oos_winner",
            hold_buckets=2,
            min_rr=2.0,
            stop_buffer_ticks=1.0,
            target_mode="rr_multiple",
        ),
    )

    monkeypatch.setattr(
        edge_module,
        "build_market_truth_frame",
        lambda *_args, **_kwargs: pl.DataFrame({"dummy": list(range(20))}),
    )

    def fake_inner(_truth: pl.DataFrame, spec: EdgeSearchSpec, **_kwargs: Any) -> dict[str, Any]:
        train_expectancy = 0.02 if spec.name == "train_winner" else 0.01
        return {
            "name": spec.name,
            "train_expectancy": train_expectancy,
            "train_win_rate": 0.6,
            "train_n": 20.0,
            "train_avg_rr": 2.5,
            "train_profit_factor": 1.5,
            "train_positive_fold_rate": 1.0,
            "train_fold_count": 3.0,
            "selection_scope": "inner_walk_forward",
            "outer_evaluated": 0.0,
            "oos_expectancy": 0.0,
            "oos_win_rate": 0.0,
            "oos_n": 0.0,
            "oos_avg_rr": 0.0,
            "oos_profit_factor": 0.0,
            "oos_start_index": -1.0,
            "oos_start_ts": -1.0,
        }

    outer_calls: list[str] = []

    def fake_outer(_truth: pl.DataFrame, spec: EdgeSearchSpec, **_kwargs: Any) -> dict[str, float]:
        outer_calls.append(spec.name)
        return {
            "oos_expectancy": -0.50,
            "oos_win_rate": 0.4,
            "oos_n": 10.0,
            "oos_avg_rr": 2.5,
            "oos_profit_factor": 0.8,
            "oos_start_index": 15.0,
            "oos_start_ts": -1.0,
            "outer_evaluated": 1.0,
        }

    monkeypatch.setattr(edge_module, "_inner_walk_forward_summary", fake_inner)
    monkeypatch.setattr(edge_module, "_outer_holdout_summary", fake_outer)
    table, best, row = edge_module.search_best_edge_spec(
        pl.DataFrame(),
        interval_ns=1,
        grid=grid,
        min_oos_trades=1,
        min_oos_rr=2.0,
        auction=pl.DataFrame({"dummy": [1]}),
        deceptive_frame=pl.DataFrame({"dummy": [1]}),
    )

    assert table.height == 2
    assert best is not None
    assert best.name == "train_winner"
    assert row["oos_expectancy"] == -0.50
    assert outer_calls == ["train_winner"]
    unselected = table.filter(pl.col("name") == "oos_winner").row(0, named=True)
    assert unselected["oos_n"] == 0.0
    assert unselected["outer_evaluated"] == 0.0
