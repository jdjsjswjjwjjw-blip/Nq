"""تنفيذ intraday مبسّط: bid/ask + انزلاق (بدون تدفق طابور)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from nq.simulation.execution.costs import commission_rate

FloatArray = npt.NDArray[np.float64]


def execution_forward_returns(
    bid: npt.NDArray[np.floating] | Sequence[float],
    ask: npt.NDArray[np.floating] | Sequence[float],
    *,
    horizon: int = 1,
    slippage_ticks: float = 0.5,
    tick_size: float = 0.25,
    commission_bps: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """عوائد أمامية لصفقة شراء/بيع وحدة عند ``t`` مع إغلاق عند ``t+horizon``.

    * شراء عند ``ask + slippage``، إغلاق بيع عند ``bid - slippage``.
    * بيع عند ``bid - slippage``، إغلاق شراء عند ``ask + slippage``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    b = np.asarray(bid, dtype=np.float64)
    a = np.asarray(ask, dtype=np.float64)
    if b.shape != a.shape:
        raise ValueError(f"bid and ask must align, got {b.shape} vs {a.shape}")

    n = b.shape[0]
    long_fwd = np.full(n, np.nan, dtype=np.float64)
    short_fwd = np.full(n, np.nan, dtype=np.float64)
    comm = commission_rate(commission_bps=commission_bps)
    slip = slippage_ticks * tick_size

    entry_long = a + slip
    entry_short = b - slip
    exit_long = b - slip
    exit_short = a + slip

    valid = np.arange(n - horizon)
    entry_long_v = entry_long[valid]
    exit_long_v = exit_long[valid + horizon]
    entry_short_v = entry_short[valid]
    exit_short_v = exit_short[valid + horizon]

    with np.errstate(divide="ignore", invalid="ignore"):
        long_fwd[valid] = np.where(
            entry_long_v > 0,
            (exit_long_v - entry_long_v) / entry_long_v - comm,
            np.nan,
        )
        short_fwd[valid] = np.where(
            entry_short_v > 0,
            (entry_short_v - exit_short_v) / entry_short_v - comm,
            np.nan,
        )
    return long_fwd, short_fwd


def directional_execution_returns(
    signal: npt.NDArray[np.floating] | Sequence[float],
    long_fwd: npt.NDArray[np.floating],
    short_fwd: npt.NDArray[np.floating],
) -> FloatArray:
    """يعيد العائد الأمامي المناسب لاتجاه الإشارة (موجب=شراء، سالب=بيع)."""
    s = np.asarray(signal, dtype=np.float64)
    out = np.full(s.shape[0], np.nan, dtype=np.float64)
    long_mask = s > 0
    short_mask = s < 0
    out[long_mask] = long_fwd[long_mask]
    out[short_mask] = short_fwd[short_mask]
    return out


__all__ = ["directional_execution_returns", "execution_forward_returns"]
