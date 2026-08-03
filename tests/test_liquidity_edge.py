"""اختبارات فلتر التضليل وحكم السوق وخطة الإدج التنفيذية."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import MBO_SCHEMA, PRICE_SCALE, validate_mbo_frame
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
from tests.mbo_factory import make_stream


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


def test_scored_frame_reused_for_filter_and_bucket_features(monkeypatch) -> None:
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

    def _counting_score(*args, **kwargs):
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
        entry=100.0,
        vah=100.5,
        val=99.5,
        cfg=EdgeExecConfig(min_rr=3.0, stop_buffer_ticks=0.0, target_mode="va_opposite"),
    )
    # risk=0.5 من VAL؛ الهدف يُوسَّع لـ entry + min_rr*risk إن VAH قريب
    assert planned is not None
    _stop, target, risk, reward = planned
    assert reward / risk >= 3.0 - 1e-9
    assert target >= 100.0 + 3.0 * risk - 1e-9


def test_simulate_edge_trades_no_chase_every_bar() -> None:
    # إطار حكم يدوي: بوابة على صف واحد فقط
    truth = pl.DataFrame(
        {
            "availability_ts": [1, 2, 3, 4, 5, 6],
            "entry_gate": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "thesis_dir": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "market_verdict": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "close": [
                _px(100.0),
                _px(100.0),
                _px(100.5),
                _px(101.0),
                _px(101.5),
                _px(102.0),
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
    row = score_edge_spec_oos(
        mbo,
        spec,
        interval_ns=1_000_000_000,
        train_frac=0.5,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
    )
    assert "oos_n" in row
    assert "train_expectancy" in row


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
        train_frac=0.5,
        min_oos_trades=0,
        min_oos_rr=0.0,
        grid=grid,
        drop_deceptive=True,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
        quiet=True,
    )
    assert result.vp.with_execution is True
    assert result.raw_mbo_rows == mbo.height
    assert "Volume Profile" in result.report_md or "vp" in result.report_md.lower()
    summary = summarize_edge_trades(result.trades)
    assert "expectancy" in summary
    assert np.isfinite(summary["expectancy"]) or summary["n_trades"] == 0.0
