"""بطارية علمية مكثّفة لاستراتيجية Volume Profile / Auction (مسار متصل).

الهدف: إثبات أن آلة الاستخراج اليومية (كسر → بناء → انطلاق → إدج) تعمل
بصرامة المبادئ الأربعة قبل الاعتماد على setup يومي.

لا تدّعي ربحية على بيانات حية — تثبت صحّة الآلة السببية والإشارات والطبقات.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import pytest

from nq.contracts.mbo import MBO_SCHEMA, PRICE_SCALE, validate_mbo_frame
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.determinism import make_generator
from nq.core.time import assert_sorted_causal
from nq.orderbook import reconstruct, scan_book_tob_and_depth
from nq.simulation.auction import (
    VP_PROFILE_INTERVAL_NS,
    VP_SIGNAL_INTERVAL_NS,
    auction_action_states,
    auction_fsm_columns,
    auction_signal_frame,
    auction_signals_from_states,
)
from nq.simulation.deceptive_liquidity import (
    DeceptiveLiquidityConfig,
    deceptive_features_by_bucket,
    filter_deceptive_liquidity,
    score_deceptive_events,
)
from nq.simulation.depth_lifecycle import depth_at_bar_close_multi
from nq.simulation.edge_execution_plan import (
    EdgeSearchSpec,
    default_edge_search_grid,
    search_best_edge_spec,
    summarize_edge_trades,
)
from nq.strategies.vp_auction import (
    _VP_AUCTION_FOCUS,
    _VP_LEVEL_DISTANCE_FEATURES,
    _VP_REGIME_STATE_FEATURES,
    run_vp_auction_research,
)
from nq.validation import assert_availability_not_before_event, assert_causal_order
from tests.test_coverage import _paired_streams
from tests.test_liquidity_edge import _session_with_imbalance

_NS = 1_000_000_000
_PX = int(PRICE_SCALE)


def _synth_session(
    n: int,
    *,
    seed: int,
    hours: float = 6.5,
    trade_every: int = 4,
) -> pl.DataFrame:
    """جلسة اصطناعية كثيفة: أوامر + صفقات على مدى ساعات (سببية صارمة)."""
    rng = np.random.default_rng(seed)
    n_add = max(n // 2, 1)
    actions: list[str] = []
    sides: list[str] = []
    prices: list[int] = []
    sizes: list[int] = []
    oids: list[int] = []
    for i in range(n_add):
        actions.append("A")
        sides.append("B" if rng.random() < 0.5 else "A")
        # مسار سعري بطيء ثم توسّع — يحاكي بناء ثم انطلاق
        drift = int((i / max(n_add - 1, 1)) * 40)
        mid = 20_000 + drift
        prices.append(int((mid + int(rng.integers(-8, 9))) * _PX))
        sizes.append(int(rng.integers(1, 25)))
        oids.append(i + 1)
    for i in range(n - n_add):
        r = int(rng.integers(0, 5))
        if r == 0:
            actions.append("C")
            sides.append("N")
            prices.append(0)
            sizes.append(0)
            oids.append(int(rng.integers(1, n_add + 1)))
        elif r == 1 or (i % trade_every == 0):
            actions.append("T")
            sides.append("B" if rng.random() < 0.55 else "A")
            prices.append(int((20_000 + int(i / 50) + int(rng.integers(-3, 4))) * _PX))
            sizes.append(int(rng.integers(1, 8)))
            oids.append(0)
        elif r == 2:
            actions.append("F")
            sides.append("N")
            prices.append(0)
            sizes.append(int(rng.integers(1, 3)))
            oids.append(int(rng.integers(1, n_add + 1)))
        else:
            actions.append("A")
            sides.append("B" if rng.random() < 0.5 else "A")
            prices.append(int((20_010 + int(rng.integers(-5, 6))) * _PX))
            sizes.append(int(rng.integers(1, 20)))
            oids.append(n_add + i + 1)
    span = max(n, int(hours * 3600 * _NS))
    event_ts = (np.arange(n, dtype=np.int64) * (span // n)).astype(np.int64)
    frame = pl.DataFrame(
        {
            "event_ts": event_ts,
            "ingest_ts": event_ts,
            "sequence": np.arange(1, n + 1, dtype=np.uint64),
            "instrument_id": np.ones(n, dtype=np.uint32),
            "symbol": ["MES"] * n,
            "action": actions,
            "side": sides,
            "price": np.asarray(prices, dtype=np.int64),
            "size": np.asarray(sizes, dtype=np.uint32),
            "order_id": np.asarray(oids, dtype=np.uint64),
            "flags": np.zeros(n, dtype=np.uint8),
        },
        schema=MBO_SCHEMA,
    )
    validate_mbo_frame(frame)
    assert_sorted_causal(frame)
    return frame


# ---------------------------------------------------------------------------
# 1) مبادئ / عقودات زمنية
# ---------------------------------------------------------------------------


def test_dual_tf_defaults_match_research_doc() -> None:
    assert VP_PROFILE_INTERVAL_NS == 5 * 60 * _NS
    assert VP_SIGNAL_INTERVAL_NS == 30 * _NS
    assert VP_PROFILE_INTERVAL_NS >= VP_SIGNAL_INTERVAL_NS


def test_auction_signals_availability_never_before_bucket_end() -> None:
    frame = _synth_session(8_000, seed=11, hours=3.0)
    sig = auction_signal_frame(
        frame,
        profile_interval_ns=VP_PROFILE_INTERVAL_NS,
        signal_interval_ns=VP_SIGNAL_INTERVAL_NS,
    )
    if sig.height == 0:
        pytest.skip("no trade buckets in synthetic draw")
    assert_causal_order(sig[AVAILABILITY_TS].to_list())
    # كل إشارة متاحة عند نهاية برميلها (availability == bucket_end في المسار)
    states = auction_action_states(
        frame,
        profile_interval_ns=VP_PROFILE_INTERVAL_NS,
        signal_interval_ns=VP_SIGNAL_INTERVAL_NS,
    )
    assert states.height == sig.height
    assert_availability_not_before_event(
        states["bucket_start"].to_list(),
        states[AVAILABILITY_TS].to_list(),
    )


def test_auction_signal_frame_suffix_perturbation_is_causal() -> None:
    """تشويش آخر 20% من MBO لا يغيّر إشارات النصف الأول (منع تسريب)."""
    frame = _synth_session(12_000, seed=21, hours=4.0)
    sig = auction_signal_frame(
        frame,
        profile_interval_ns=2 * 60 * _NS,
        signal_interval_ns=30 * _NS,
    )
    if sig.height < 20:
        pytest.skip("need enough signal bars")
    cut = sig.height // 2
    mid_ts = int(sig[AVAILABILITY_TS][cut])
    # شوّه أحداث ما بعد منتصف الإشارات فقط
    noisy = frame.with_columns(
        pl.when(pl.col(EVENT_TS) > mid_ts)
        .then(pl.col("price") + 50 * _PX)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    sig2 = auction_signal_frame(
        noisy,
        profile_interval_ns=2 * 60 * _NS,
        signal_interval_ns=30 * _NS,
    )
    early = sig.filter(pl.col(AVAILABILITY_TS) <= mid_ts)
    early2 = sig2.filter(pl.col(AVAILABILITY_TS) <= mid_ts)
    # نفس الطوابع المبكرة ونفس قيم الحدود/التوازن على التقاطع
    joined = early.join(early2, on=AVAILABILITY_TS, how="inner", suffix="_b")
    assert joined.height >= max(5, cut // 3)
    for col in ("vp_upper", "vp_mid", "vp_lower", "vp_balance", "vp_imbalance"):
        a = joined[col].to_numpy()
        b = joined[f"{col}_b"].to_numpy()
        assert np.allclose(a, b, equal_nan=True), col


# ---------------------------------------------------------------------------
# 2) FSM علمي: كسر → بناء → انطلاق (بدون قتل مبكر)
# ---------------------------------------------------------------------------


def test_fsm_phase_order_break_before_build_before_expand() -> None:
    n = 16
    states = pl.DataFrame(
        {
            "bucket_start": list(range(n)),
            "is_balanced": [True, True] + [False] * (n - 2),
            "close": [
                100.0,
                100.0,
                103.0,
                100.4,
                100.1,
                99.8,
                100.2,
                100.5,
                100.0,
                102.0,
                100.3,
                106.0,
                108.0,
                110.0,
                112.0,
                114.0,
            ],
            "vah": [101.0] * n,
            "poc": [100.0] * n,
            "val": [99.0] * n,
            "bucket_volume": [10.0] * n,
            "is_expansion": [False] * 11 + [True] * 5,
            "pullback_defended": [False] * 3 + [True] * 6 + [False] * 7,
            "close_in_value": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            "delta": [0.0, 0.0] + [4.0] * (n - 2),
            "absorb": [0.0] * n,
            "look_fail": [0.0] * n,
        }
    )
    fsm = auction_fsm_columns(states, rebalance_confirm=5, build_max_age=40)
    brk_i = int(np.flatnonzero(fsm["vp_fsm_break"].to_numpy())[0])
    build_idx = np.flatnonzero(fsm["vp_fsm_build"].to_numpy())
    exp_idx = np.flatnonzero(fsm["vp_fsm_expand"].to_numpy())
    assert brk_i == 2
    assert float(fsm["vp_fsm_expand"][brk_i]) == 0.0
    assert build_idx.size > 0
    assert int(build_idx[0]) > brk_i
    assert exp_idx.size > 0
    assert int(exp_idx[0]) > int(build_idx[0])
    assert int(exp_idx[0]) >= 11


def test_fsm_never_expands_without_is_expansion_flag() -> None:
    n = 12
    states = pl.DataFrame(
        {
            "bucket_start": list(range(n)),
            "is_balanced": [True, True] + [False] * (n - 2),
            "close": [100.0, 100.0] + [103.0 + i * 0.5 for i in range(n - 2)],
            "vah": [101.0] * n,
            "poc": [100.0] * n,
            "val": [99.0] * n,
            "bucket_volume": [10.0] * n,
            "is_expansion": [False] * n,
            "pullback_defended": [False] * n,
            "close_in_value": [True, True] + [False] * (n - 2),
            "delta": [0.0, 0.0] + [5.0] * (n - 2),
            "absorb": [0.0] * n,
            "look_fail": [0.0] * n,
        }
    )
    fsm = auction_fsm_columns(states, build_max_age=30)
    assert float(fsm["vp_fsm_break"][2]) == 1.0
    assert float(fsm["vp_fsm_expand"].sum()) == 0.0
    assert float(fsm["vp_auction_setup"].sum()) == 0.0


def test_focus_columns_include_build_and_setup_for_daily_ontology() -> None:
    """بركة IC الاتجاهية تشمل FSM/setup؛ مسافات VP تبقى وصفيّة خارج الفرز."""
    directional = {
        "vp_fsm_break",
        "vp_fsm_build",
        "vp_fsm_retest",
        "vp_auction_setup",
        "vp_flip_to_imbalance",
    }
    assert directional <= set(_VP_AUCTION_FOCUS)
    level = {"vp_rel_upper", "vp_rel_mid", "vp_rel_lower"}
    assert level <= set(_VP_LEVEL_DISTANCE_FEATURES)
    assert level.isdisjoint(_VP_AUCTION_FOCUS)
    assert "vp_expansion" in _VP_REGIME_STATE_FEATURES
    assert "vp_expansion" not in _VP_AUCTION_FOCUS


# ---------------------------------------------------------------------------
# 3) دفتر موحّد + تضليل + إعادة استخدام (مسار يومي)
# ---------------------------------------------------------------------------


def test_unified_book_scan_matches_separate_passes_on_dense_session() -> None:
    frame = _synth_session(40_000, seed=31, hours=2.0)
    intervals = (VP_SIGNAL_INTERVAL_NS, VP_PROFILE_INTERVAL_NS)
    unified, u_depth = scan_book_tob_and_depth(frame, interval_ns_list=intervals, n_levels=5)
    separate = reconstruct(frame)
    sep_depth = depth_at_bar_close_multi(frame, interval_ns_list=intervals, n_levels=5)
    assert unified.top_of_book.equals(separate.top_of_book)
    assert unified.integrity == separate.integrity
    for iv in intervals:
        assert u_depth[iv].equals(sep_depth[iv])


def test_deceptive_scored_once_and_reused_on_dense_session(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _synth_session(6_000, seed=41, hours=1.5)
    calls = {"n": 0}
    real = score_deceptive_events

    def _counting(*args: Any, **kwargs: Any) -> pl.DataFrame:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "nq.simulation.deceptive_liquidity.score_deceptive_events",
        _counting,
    )
    cfg = DeceptiveLiquidityConfig(storm_min_events=50_000, drop_score=0.35)
    scored = _counting(frame, config=cfg)
    assert calls["n"] == 1
    cleaned = filter_deceptive_liquidity(frame, config=cfg, scored=scored)
    buckets = deceptive_features_by_bucket(
        frame, interval_ns=VP_SIGNAL_INTERVAL_NS, config=cfg, scored=scored
    )
    assert calls["n"] == 1
    assert cleaned.height <= frame.height
    assert buckets.height >= 1
    assert "deceptive_score" in buckets.columns


# ---------------------------------------------------------------------------
# 4) مسار الاستراتيجية الكامل: WF قبل التنفيذ + حتمية + إدج
# ---------------------------------------------------------------------------


def test_vp_research_determinism_same_seed() -> None:
    nq, _ = _paired_streams(3_000, seed=91)
    a = run_vp_auction_research(
        nq,
        n_permutations=16,
        n_splits=2,
        rng=make_generator(101),
        quiet=True,
        with_execution=False,
        interval_ns=10_000,
        profile_interval_ns=20_000,
    )
    b = run_vp_auction_research(
        nq,
        n_permutations=16,
        n_splits=2,
        rng=make_generator(101),
        quiet=True,
        with_execution=False,
        interval_ns=10_000,
        profile_interval_ns=20_000,
    )
    assert a.oos_ic == b.oos_ic
    assert a.oos_pvalue == b.oos_pvalue
    assert a.oos_n == b.oos_n
    assert a.best_signal == b.best_signal
    assert a.features.select(sorted(a.features.columns)).equals(
        b.features.select(sorted(b.features.columns))
    )


def test_vp_research_wf_before_execution_no_edge_pnl_in_signal_focus() -> None:
    mbo = _session_with_imbalance(48)
    grid = (
        EdgeSearchSpec(
            name="hold2_rr2_buf1_rr_multiple",
            hold_buckets=2,
            min_rr=2.0,
            stop_buffer_ticks=1.0,
            target_mode="rr_multiple",
            rr_multiple=2.5,
        ),
    )
    result = run_vp_auction_research(
        mbo,
        n_permutations=16,
        n_splits=2,
        rng=make_generator(77),
        quiet=True,
        with_execution=True,
        drop_deceptive=True,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
        edge_grid=grid,
        edge_train_frac=0.5,
        min_oos_trades=0,
        min_oos_rr=0.0,
        streaming_features=True,
        interval_ns=1_000_000_000,
        profile_interval_ns=2_000_000_000,
    )
    assert result.with_execution is True
    assert result.fold_df is not None
    assert result.fold_df.height >= 1
    for col in result.signal_columns:
        assert not str(col).startswith("edge_pnl")
        assert col != "entry_gate"
    assert "edge_pnl" not in result.features.columns
    assert "entry_gate" not in result.signal_columns
    # WF اكتمل قبل طبقة التنفيذ: جدول الطيّ موجود + أعمدة التنفيذ ملحقة وصفيًا
    assert result.edge_search_table.height >= 1
    for col in ("entry_gate", "deceptive_score", "market_verdict"):
        assert col in result.features.columns
    assert isinstance(result.oos_ic, float)
    assert result.raw_mbo_rows == mbo.height
    assert result.cleaned_mbo_rows <= result.raw_mbo_rows
    assert "قناة" in result.unified.to_markdown()


def test_edge_search_grid_structural_rr_specs_nonempty() -> None:
    grid = default_edge_search_grid()
    assert len(grid) >= 8
    for spec in grid:
        assert spec.min_rr >= 2.0
        assert spec.hold_buckets >= 1
        assert spec.target_mode in ("poc", "va_opposite", "rr_multiple")
        assert spec.playbook in ("responsive", "initiative")
        if spec.playbook == "responsive":
            assert spec.target_mode == "poc"
        else:
            assert spec.target_mode == "rr_multiple"


def test_edge_search_reuses_auction_and_deceptive_frames() -> None:
    mbo = _session_with_imbalance(36)
    iv = 1_000_000_000
    auction = auction_action_states(mbo, profile_interval_ns=2 * iv, signal_interval_ns=iv)
    deco = deceptive_features_by_bucket(
        mbo,
        interval_ns=iv,
        config=DeceptiveLiquidityConfig(storm_min_events=10_000),
    )
    grid = (
        EdgeSearchSpec(
            name="hold2_rr2_buf1_rr_multiple",
            hold_buckets=2,
            min_rr=2.0,
            stop_buffer_ticks=1.0,
            target_mode="rr_multiple",
            rr_multiple=2.5,
        ),
    )
    table, _best, _row = search_best_edge_spec(
        mbo,
        interval_ns=iv,
        grid=grid,
        train_frac=0.5,
        min_oos_trades=0,
        min_oos_rr=0.0,
        auction=auction,
        deceptive_frame=deco,
    )
    assert table.height >= 1
    assert "oos_expectancy" in table.columns


# ---------------------------------------------------------------------------
# 5) ضغط ثقيل سريع: آلة يوم كامل اصطناعي
# ---------------------------------------------------------------------------


def test_heavy_daily_machine_pipeline_wallclock_and_invariants() -> None:
    """يوم اصطناعي كثيف: إشارات + تضليل + بحث إدج مصغّر خلال زمن محدود."""
    t0 = time.perf_counter()
    frame = _synth_session(25_000, seed=55, hours=6.5)
    assert frame.height == 25_000

    scored = score_deceptive_events(frame, config=DeceptiveLiquidityConfig(storm_min_events=50_000))
    cleaned = filter_deceptive_liquidity(
        frame,
        config=DeceptiveLiquidityConfig(storm_min_events=50_000),
        scored=scored,
    )
    states = auction_action_states(
        cleaned if cleaned.height else frame,
        profile_interval_ns=VP_PROFILE_INTERVAL_NS,
        signal_interval_ns=VP_SIGNAL_INTERVAL_NS,
    )
    sigs = auction_signals_from_states(states)
    assert sigs.height == states.height
    if sigs.height:
        assert_causal_order(sigs[AVAILABILITY_TS].to_list())
        for col in (
            "vp_upper",
            "vp_mid",
            "vp_lower",
            "vp_fsm_break",
            "vp_fsm_build",
            "vp_fsm_expand",
            "vp_auction_setup",
        ):
            assert col in sigs.columns

    # مسار بحث مصغّر على عيّنة paired (حتمي)
    nq, _ = _paired_streams(4_000, seed=56)
    result = run_vp_auction_research(
        nq,
        n_permutations=12,
        n_splits=2,
        rng=make_generator(56),
        quiet=True,
        with_execution=False,
        interval_ns=8_000,
        profile_interval_ns=24_000,
    )
    assert result.features.height > 0
    assert isinstance(result.oos_ic, float)
    elapsed = time.perf_counter() - t0
    # ثقيل لكن سريع بما يكفي لبوابة CI محلية
    assert elapsed < 90.0, f"heavy pipeline too slow: {elapsed:.1f}s"


def test_summarize_edge_trades_handles_empty() -> None:
    empty = summarize_edge_trades(pl.DataFrame())
    assert empty["n_trades"] == 0.0
    assert empty["expectancy"] == 0.0
