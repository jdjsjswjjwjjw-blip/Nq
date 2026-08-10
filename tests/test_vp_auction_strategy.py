"""اختبارات تركيز فرضيات Volume Profile / Auction (مسار متصل)."""

from __future__ import annotations

from nq.core.determinism import make_generator
from nq.simulation.deceptive_liquidity import DeceptiveLiquidityConfig
from nq.simulation.edge_execution_plan import EdgeSearchSpec
from nq.strategies.vp_auction import VP_DEFAULT_SELECTION_HORIZON, run_vp_auction_research
from tests.test_coverage import _paired_streams
from tests.test_liquidity_edge import _session_with_imbalance


def test_default_vp_ic_horizon_matches_execution_time_scale() -> None:
    assert VP_DEFAULT_SELECTION_HORIZON == 10  # 5m on 30s action bars; max hold is 15m


def test_run_vp_auction_research_uses_unified_features() -> None:
    nq, _mnq = _paired_streams(2500, seed=82)
    result = run_vp_auction_research(
        nq,
        n_permutations=20,
        n_splits=2,
        rng=make_generator(5),
        quiet=True,
        with_execution=False,
        # مصغّر: فعل 10μs · رينج 20μs (بدل 30ث / 5د على بيانات اصطناعية قصيرة)
        interval_ns=10_000,
        profile_interval_ns=20_000,
    )
    assert "vp_balance" in result.features.columns
    assert "vp_imbalance" in result.features.columns
    assert "vp_of_delta" in result.signal_columns
    assert "vp_look_fail" in result.signal_columns
    assert "vp_rel_upper" not in result.signal_columns
    assert "vp_balance" not in result.signal_columns
    assert "fail_fvg" not in result.signal_columns
    assert result.unified is not None
    assert result.fold_df is not None
    assert result.exploratory_only is False
    assert result.with_execution is False


def test_run_vp_auction_research_produces_report() -> None:
    nq, _mnq = _paired_streams(2000, seed=83)
    result = run_vp_auction_research(
        nq,
        n_permutations=20,
        n_splits=2,
        rng=make_generator(6),
        quiet=True,
        with_execution=False,
        interval_ns=10_000,
        profile_interval_ns=20_000,
    )
    md = result.unified.to_markdown()
    assert "قناة 1 — SSL" in md
    assert result.features.height > 0
    assert isinstance(result.oos_ic, float)
    assert result.oos_n >= 0


def test_run_vp_auction_connected_execution_layer() -> None:
    """التضليل + الهولد + R:R داخل نفس استراتيجية VP — بلا مسار منفصل."""
    mbo = _session_with_imbalance(40)
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
        n_permutations=20,
        n_splits=2,
        rng=make_generator(7),
        quiet=True,
        with_execution=True,
        drop_deceptive=True,
        deceptive=DeceptiveLiquidityConfig(storm_min_events=10_000),
        edge_grid=grid,
        edge_train_frac=0.5,
        min_oos_trades=0,
        min_oos_rr=0.0,
        # عيّنة اصطناعية صغيرة؛ batch على هذا المصنع قد يُفرّغ البراميل
        streaming_features=True,
        interval_ns=1_000_000_000,
        profile_interval_ns=2_000_000_000,
    )
    assert result.with_execution is True
    assert result.raw_mbo_rows == mbo.height
    assert result.cleaned_mbo_rows <= result.raw_mbo_rows
    # أعمدة التنفيذ مدمجة وصفيًا — لكن اختيار الإشارة = vp_* فقط
    for col in ("entry_gate", "deceptive_score", "market_verdict", "vp_flip_gated"):
        assert col in result.features.columns, f"missing connected col {col}"
    assert "edge_pnl" not in result.features.columns
    assert "entry_gate" not in result.signal_columns
    assert "vp_look_fail" in result.signal_columns
    assert "vp_rel_upper" not in result.signal_columns
    assert result.edge_search_table.height >= 1
