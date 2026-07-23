"""عبور الـ spread — أسعار تنفيذ intraday من top-of-book."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nq.contracts.instruments import NQ_METADATA

FloatArray = npt.NDArray[np.float64]


def buy_fill_price(
    ask: npt.NDArray[np.floating] | float,
    *,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
) -> FloatArray | float:
    """سعر شراء عدواني: ask + انزلاق."""
    slip = slippage_ticks * tick_size
    if isinstance(ask, float):
        return ask + slip
    return np.asarray(ask, dtype=np.float64) + slip


def sell_fill_price(
    bid: npt.NDArray[np.floating] | float,
    *,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
) -> FloatArray | float:
    """سعر بيع عدواني: bid − انزلاق."""
    slip = slippage_ticks * tick_size
    if isinstance(bid, float):
        return bid - slip
    return np.asarray(bid, dtype=np.float64) - slip


__all__ = ["buy_fill_price", "sell_fill_price"]
