"""اختبارات تنفيذ intraday المبسّط (spread + slippage)."""

from __future__ import annotations

import numpy as np

from nq.alpha.signals import evaluate_signal_intraday
from nq.core.determinism import make_generator
from nq.simulation.execution import (
    buy_fill_price,
    directional_execution_returns,
    execution_forward_returns,
    execution_forward_returns_depth,
    realistic_depth_execution_simulation,
    realistic_execution_forward_returns,
    realistic_execution_simulation,
    sell_fill_price,
)


def test_spread_crossing_prices() -> None:
    assert buy_fill_price(100.0, slippage_ticks=0.0, tick_size=0.25) == 100.0
    assert sell_fill_price(99.75, slippage_ticks=0.0, tick_size=0.25) == 99.75
    assert buy_fill_price(100.0, slippage_ticks=1.0, tick_size=0.25) == 100.25


def test_long_execution_return_less_than_mid() -> None:
    bid = np.array([100.0, 100.5, 101.0])
    ask = np.array([100.25, 100.75, 101.25])
    long_fwd, _ = execution_forward_returns(bid, ask, horizon=1, slippage_ticks=0.0)
    mid0 = (bid[0] + ask[0]) / 2.0
    mid1 = (bid[1] + ask[1]) / 2.0
    mid_fwd = (mid1 - mid0) / mid0
    assert long_fwd[0] < mid_fwd


def test_realistic_execution_applies_latency_before_entry() -> None:
    bid = np.array([100.0, 90.0, 95.0])
    ask = np.array([101.0, 91.0, 96.0])
    long_fwd, _ = realistic_execution_forward_returns(
        bid,
        ask,
        horizon=1,
        latency_steps=1,
        slippage_ticks=0.0,
    )
    assert np.isclose(long_fwd[0], (95.0 - 91.0) / 91.0)
    assert np.isnan(long_fwd[1])


def test_realistic_execution_reports_fill_timestamps_and_order_size() -> None:
    bid = np.array([100.0, 90.0, 95.0])
    ask = np.array([101.0, 91.0, 96.0])
    timestamps = np.array([10, 20, 30], dtype=np.int64)

    report = realistic_execution_simulation(
        bid,
        ask,
        timestamps=timestamps,
        horizon=1,
        latency_steps=1,
        order_qty=3,
        slippage_ticks=0.0,
        commission_bps=0.0,
    )

    assert report.long_entry_ts[0] == 20
    assert report.long_exit_ts[0] == 30
    assert report.long_filled_qty[0] == 3
    assert np.isclose(report.long_returns[0], (95.0 - 91.0) / 91.0)
    assert np.isnan(report.long_returns[1])


def test_depth_execution_rejects_insufficient_liquidity_even_with_l1_fallback_by_default() -> None:
    bid_px = np.array([[100.0], [100.5], [101.0]])
    bid_sz = np.array([[1.0], [1.0], [1.0]])
    ask_px = np.array([[100.25], [100.75], [101.25]])
    ask_sz = np.array([[1.0], [1.0], [1.0]])

    long_fwd, short_fwd = execution_forward_returns_depth(
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
        horizon=1,
        order_qty=2,
        n_levels=1,
        fallback_bid=np.array([100.0, 100.5, 101.0]),
        fallback_ask=np.array([100.25, 100.75, 101.25]),
        slippage_ticks=0.0,
    )

    assert np.isnan(long_fwd[0])
    assert np.isnan(short_fwd[0])


def test_depth_execution_allows_partial_fills_with_reported_qty_and_timestamps() -> None:
    bid_px = np.array([[100.0, 99.75], [100.5, 100.25], [101.0, 100.75]])
    bid_sz = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    ask_px = np.array([[100.25, 100.5], [100.75, 101.0], [101.25, 101.5]])
    ask_sz = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    timestamps = np.array([10, 20, 30], dtype=np.int64)

    report = realistic_depth_execution_simulation(
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
        timestamps=timestamps,
        horizon=1,
        order_qty=3,
        n_levels=2,
        latency_steps=1,
        allow_partial_fills=True,
        commission_bps=0.0,
    )

    assert report.long_entry_ts[0] == 20
    assert report.long_exit_ts[0] == 30
    assert report.long_filled_qty[0] == 2
    assert report.long_partial[0]
    assert np.isfinite(report.long_returns[0])


def test_directional_returns_follow_signal_sign() -> None:
    long_fwd = np.array([0.01, 0.02, np.nan])
    short_fwd = np.array([0.03, 0.04, np.nan])
    signal = np.array([1.0, -1.0, 0.0])
    directional = directional_execution_returns(signal, long_fwd, short_fwd)
    assert directional[0] == 0.01
    assert directional[1] == 0.04
    assert np.isnan(directional[2])


def test_evaluate_signal_intraday_runs() -> None:
    rng = make_generator(0)
    n = 120
    bid = np.linspace(100.0, 101.0, n)
    ask = bid + 0.25
    signal = rng.normal(0, 1, n)
    result = evaluate_signal_intraday(
        "test",
        signal,
        bid,
        ask,
        horizon=1,
        n_permutations=200,
        rng=rng,
    )
    assert result.n > 0
