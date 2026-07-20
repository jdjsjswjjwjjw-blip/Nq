"""اختبارات تنفيذ intraday المبسّط (spread + slippage)."""

from __future__ import annotations

import numpy as np

from nq.alpha.execution import evaluate_signal_intraday
from nq.core.determinism import make_generator
from nq.simulation.execution import (
    buy_fill_price,
    directional_execution_returns,
    execution_forward_returns,
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
