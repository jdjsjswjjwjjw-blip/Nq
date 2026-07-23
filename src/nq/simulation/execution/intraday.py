"""تنفيذ intraday: research forward labels plus causal latency simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nq.contracts.instruments import NQ_METADATA
from nq.simulation.execution.costs import commission_rate

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ExecutionSimulationReport:
    """Detailed L1 causal execution report; no queue-position realism is implied."""

    long_returns: FloatArray
    short_returns: FloatArray
    long_entry_ts: IntArray
    long_exit_ts: IntArray
    short_entry_ts: IntArray
    short_exit_ts: IntArray
    long_filled_qty: FloatArray
    short_filled_qty: FloatArray
    long_rejected: BoolArray
    short_rejected: BoolArray


def _timestamps_or_index(
    timestamps: npt.NDArray[np.integer] | Sequence[int] | None,
    *,
    n: int,
) -> IntArray:
    if timestamps is None:
        return np.arange(n, dtype=np.int64)
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.shape != (n,):
        raise ValueError(f"timestamps must have shape {(n,)}, got {ts.shape}")
    return ts


def _valid_l1_quote(bid: float, ask: float) -> bool:
    return bool(np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and bid < ask)


def execution_forward_returns(
    bid: npt.NDArray[np.floating] | Sequence[float],
    ask: npt.NDArray[np.floating] | Sequence[float],
    *,
    horizon: int = 1,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
    commission_bps: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Research forward-return labels crossing the spread at decision row ``t``.

    * شراء عند ``ask + slippage``، إغلاق بيع عند ``bid - slippage``.
    * بيع عند ``bid - slippage``، إغلاق شراء عند ``ask + slippage``.

    هذا مسار labeling بحثي، وليس محاكاة تنفيذ واقعية. للاختبار السببي استخدم
    ``realistic_execution_forward_returns`` الذي يفرض latency قبل الدخول.
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


def realistic_execution_forward_returns(
    bid: npt.NDArray[np.floating] | Sequence[float],
    ask: npt.NDArray[np.floating] | Sequence[float],
    *,
    horizon: int = 1,
    latency_steps: int = 1,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
    commission_bps: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Causal decision → latency → order → eligible fill simulation on L1 quotes."""
    report = realistic_execution_simulation(
        bid,
        ask,
        horizon=horizon,
        latency_steps=latency_steps,
        slippage_ticks=slippage_ticks,
        tick_size=tick_size,
        commission_bps=commission_bps,
    )
    return report.long_returns, report.short_returns


def realistic_execution_simulation(
    bid: npt.NDArray[np.floating] | Sequence[float],
    ask: npt.NDArray[np.floating] | Sequence[float],
    *,
    timestamps: npt.NDArray[np.integer] | Sequence[int] | None = None,
    horizon: int = 1,
    latency_steps: int = 1,
    order_qty: int = 1,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
    commission_bps: float = 0.0,
) -> ExecutionSimulationReport:
    """Detailed causal L1 market-order simulation.

    A decision at row ``t`` can only submit after ``latency_steps`` and fill at
    the next eligible quote row. Buys cross the ask; sells cross the bid.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if latency_steps < 1:
        raise ValueError(f"latency_steps must be >= 1, got {latency_steps}")
    if order_qty < 1:
        raise ValueError(f"order_qty must be >= 1, got {order_qty}")
    b = np.asarray(bid, dtype=np.float64)
    a = np.asarray(ask, dtype=np.float64)
    if b.shape != a.shape:
        raise ValueError(f"bid and ask must align, got {b.shape} vs {a.shape}")

    n = b.shape[0]
    ts = _timestamps_or_index(timestamps, n=n)
    long_fwd = np.full(n, np.nan, dtype=np.float64)
    short_fwd = np.full(n, np.nan, dtype=np.float64)
    long_entry_ts = np.full(n, -1, dtype=np.int64)
    long_exit_ts = np.full(n, -1, dtype=np.int64)
    short_entry_ts = np.full(n, -1, dtype=np.int64)
    short_exit_ts = np.full(n, -1, dtype=np.int64)
    long_filled_qty = np.zeros(n, dtype=np.float64)
    short_filled_qty = np.zeros(n, dtype=np.float64)
    long_rejected = np.zeros(n, dtype=np.bool_)
    short_rejected = np.zeros(n, dtype=np.bool_)
    comm = commission_rate(commission_bps=commission_bps)
    slip = slippage_ticks * tick_size
    n_decisions = n - latency_steps - horizon
    if n_decisions <= 0:
        return ExecutionSimulationReport(
            long_fwd,
            short_fwd,
            long_entry_ts,
            long_exit_ts,
            short_entry_ts,
            short_exit_ts,
            long_filled_qty,
            short_filled_qty,
            long_rejected,
            short_rejected,
        )

    decisions = np.arange(n_decisions)
    for decision in decisions:
        entry_idx = int(decision + latency_steps)
        exit_idx = int(entry_idx + horizon)
        entry_quote_ok = _valid_l1_quote(float(b[entry_idx]), float(a[entry_idx]))
        exit_quote_ok = _valid_l1_quote(float(b[exit_idx]), float(a[exit_idx]))
        if not entry_quote_ok or not exit_quote_ok:
            long_rejected[decision] = True
            short_rejected[decision] = True
            continue

        entry_long = float(a[entry_idx]) + slip
        exit_long = float(b[exit_idx]) - slip
        entry_short = float(b[entry_idx]) - slip
        exit_short = float(a[exit_idx]) + slip
        if entry_long <= 0 or entry_short <= 0:
            long_rejected[decision] = entry_long <= 0
            short_rejected[decision] = entry_short <= 0
            continue

        long_entry_ts[decision] = ts[entry_idx]
        long_exit_ts[decision] = ts[exit_idx]
        short_entry_ts[decision] = ts[entry_idx]
        short_exit_ts[decision] = ts[exit_idx]
        long_filled_qty[decision] = float(order_qty)
        short_filled_qty[decision] = float(order_qty)
        long_fwd[decision] = (exit_long - entry_long) / entry_long - comm
        short_fwd[decision] = (entry_short - exit_short) / entry_short - comm

    return ExecutionSimulationReport(
        long_fwd,
        short_fwd,
        long_entry_ts,
        long_exit_ts,
        short_entry_ts,
        short_exit_ts,
        long_filled_qty,
        short_filled_qty,
        long_rejected,
        short_rejected,
        )


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


__all__ = [
    "ExecutionSimulationReport",
    "directional_execution_returns",
    "execution_forward_returns",
    "realistic_execution_forward_returns",
    "realistic_execution_simulation",
]
