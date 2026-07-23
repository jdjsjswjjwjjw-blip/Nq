"""تنفيذ بمسح عمق الدفتر — دخول وخروج من السيولة الظاهرة فقط."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.instruments import NQ_METADATA
from nq.simulation.execution.costs import commission_rate

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
_MATRIX_NDIM = 2


@dataclass(frozen=True, slots=True)
class DepthExecutionSimulationReport:
    """Detailed visible-depth execution report; no queue priority is simulated."""

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
    long_partial: BoolArray
    short_partial: BoolArray


@dataclass(frozen=True, slots=True)
class _DepthFill:
    price: float
    filled_qty: int
    partial: bool


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


def _levels_at(
    bid_px: npt.NDArray[np.floating],
    bid_sz: npt.NDArray[np.floating],
    ask_px: npt.NDArray[np.floating],
    ask_sz: npt.NDArray[np.floating],
    index: int,
    *,
    n_levels: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bids: list[tuple[int, int]] = []
    asks: list[tuple[int, int]] = []
    for k in range(n_levels):
        bp = float(bid_px[index, k]) if bid_px.ndim == _MATRIX_NDIM else float(bid_px[index])
        bs = float(bid_sz[index, k]) if bid_sz.ndim == _MATRIX_NDIM else float(bid_sz[index])
        ap = float(ask_px[index, k]) if ask_px.ndim == _MATRIX_NDIM else float(ask_px[index])
        asz = float(ask_sz[index, k]) if ask_sz.ndim == _MATRIX_NDIM else float(ask_sz[index])
        if np.isfinite(bp) and np.isfinite(bs) and bs > 0:
            bids.append((round(bp / PRICE_SCALE), int(bs)))
        if np.isfinite(ap) and np.isfinite(asz) and asz > 0:
            asks.append((round(ap / PRICE_SCALE), int(asz)))
    return bids, asks


def _walk_vwap_fill(
    levels: list[tuple[int, int]],
    qty: int,
    *,
    allow_partial: bool,
) -> _DepthFill | None:
    remaining = qty
    notional = 0
    filled = 0
    for px, sz in levels:
        if sz <= 0:
            continue
        take = min(remaining, int(sz))
        notional += take * int(px)
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0:
        return None
    if filled < qty and not allow_partial:
        return None
    return _DepthFill(
        price=(notional / filled) * PRICE_SCALE,
        filled_qty=filled,
        partial=filled < qty,
    )


def _fallback_fill(
    price: float,
    *,
    qty: int,
    side: str,
    slippage_ticks: float,
    tick_size: float,
) -> _DepthFill | None:
    if not np.isfinite(price) or price <= 0:
        return None
    slip = slippage_ticks * tick_size
    fill_price = price + slip if side == "buy" else price - slip
    if fill_price <= 0:
        return None
    return _DepthFill(price=float(fill_price), filled_qty=qty, partial=False)


def _empty_report(n: int) -> DepthExecutionSimulationReport:
    return DepthExecutionSimulationReport(
        long_returns=np.full(n, np.nan, dtype=np.float64),
        short_returns=np.full(n, np.nan, dtype=np.float64),
        long_entry_ts=np.full(n, -1, dtype=np.int64),
        long_exit_ts=np.full(n, -1, dtype=np.int64),
        short_entry_ts=np.full(n, -1, dtype=np.int64),
        short_exit_ts=np.full(n, -1, dtype=np.int64),
        long_filled_qty=np.zeros(n, dtype=np.float64),
        short_filled_qty=np.zeros(n, dtype=np.float64),
        long_rejected=np.zeros(n, dtype=np.bool_),
        short_rejected=np.zeros(n, dtype=np.bool_),
        long_partial=np.zeros(n, dtype=np.bool_),
        short_partial=np.zeros(n, dtype=np.bool_),
    )


def realistic_depth_execution_simulation(
    bid_px: npt.NDArray[np.floating],
    bid_sz: npt.NDArray[np.floating],
    ask_px: npt.NDArray[np.floating],
    ask_sz: npt.NDArray[np.floating],
    *,
    timestamps: npt.NDArray[np.integer] | Sequence[int] | None = None,
    horizon: int = 1,
    order_qty: int = 1,
    n_levels: int = 5,
    latency_steps: int = 1,
    allow_partial_fills: bool = False,
    commission_bps: float = 0.0,
    fallback_bid: npt.NDArray[np.floating] | None = None,
    fallback_ask: npt.NDArray[np.floating] | None = None,
    allow_l1_fallback: bool = False,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
) -> DepthExecutionSimulationReport:
    """Causal visible-depth execution with latency, rejection, and partial fills."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if order_qty < 1:
        raise ValueError(f"order_qty must be >= 1, got {order_qty}")
    if latency_steps < 1:
        raise ValueError(f"latency_steps must be >= 1, got {latency_steps}")

    bp = np.asarray(bid_px, dtype=np.float64)
    bs = np.asarray(bid_sz, dtype=np.float64)
    ap = np.asarray(ask_px, dtype=np.float64)
    az = np.asarray(ask_sz, dtype=np.float64)
    if bp.ndim == 1:
        bp = bp.reshape(-1, 1)
        bs = bs.reshape(-1, 1)
        ap = ap.reshape(-1, 1)
        az = az.reshape(-1, 1)
    if not (bp.shape == bs.shape == ap.shape == az.shape):
        raise ValueError("bid/ask price and size matrices must align")

    n = bp.shape[0]
    report = _empty_report(n)
    ts = _timestamps_or_index(timestamps, n=n)
    comm = commission_rate(commission_bps=commission_bps)
    fb = None if fallback_bid is None else np.asarray(fallback_bid, dtype=np.float64)
    fa = None if fallback_ask is None else np.asarray(fallback_ask, dtype=np.float64)
    if fb is not None and fb.shape != (n,):
        raise ValueError(f"fallback_bid must have shape {(n,)}, got {fb.shape}")
    if fa is not None and fa.shape != (n,):
        raise ValueError(f"fallback_ask must have shape {(n,)}, got {fa.shape}")

    for t in range(max(0, n - latency_steps - horizon)):
        entry_t = t + latency_steps
        exit_t = entry_t + horizon
        e_bids, e_asks = _levels_at(bp, bs, ap, az, entry_t, n_levels=n_levels)
        x_bids, x_asks = _levels_at(bp, bs, ap, az, exit_t, n_levels=n_levels)

        entry_long = _walk_vwap_fill(
            e_asks,
            order_qty,
            allow_partial=allow_partial_fills,
        )
        if entry_long is None and allow_l1_fallback and fa is not None:
            entry_long = _fallback_fill(
                float(fa[entry_t]),
                qty=order_qty,
                side="buy",
                slippage_ticks=slippage_ticks,
                tick_size=tick_size,
            )
        if entry_long is None:
            report.long_rejected[t] = True
        else:
            exit_long = _walk_vwap_fill(
                x_bids,
                entry_long.filled_qty,
                allow_partial=False,
            )
            if exit_long is None and allow_l1_fallback and fb is not None:
                exit_long = _fallback_fill(
                    float(fb[exit_t]),
                    qty=entry_long.filled_qty,
                    side="sell",
                    slippage_ticks=slippage_ticks,
                    tick_size=tick_size,
                )
            if exit_long is None:
                report.long_rejected[t] = True
            elif entry_long.price > 0:
                report.long_returns[t] = (exit_long.price - entry_long.price) / entry_long.price - comm
                report.long_entry_ts[t] = ts[entry_t]
                report.long_exit_ts[t] = ts[exit_t]
                report.long_filled_qty[t] = float(entry_long.filled_qty)
                report.long_partial[t] = entry_long.partial

        entry_short = _walk_vwap_fill(
            e_bids,
            order_qty,
            allow_partial=allow_partial_fills,
        )
        if entry_short is None and allow_l1_fallback and fb is not None:
            entry_short = _fallback_fill(
                float(fb[entry_t]),
                qty=order_qty,
                side="sell",
                slippage_ticks=slippage_ticks,
                tick_size=tick_size,
            )
        if entry_short is None:
            report.short_rejected[t] = True
        else:
            exit_short = _walk_vwap_fill(
                x_asks,
                entry_short.filled_qty,
                allow_partial=False,
            )
            if exit_short is None and allow_l1_fallback and fa is not None:
                exit_short = _fallback_fill(
                    float(fa[exit_t]),
                    qty=entry_short.filled_qty,
                    side="buy",
                    slippage_ticks=slippage_ticks,
                    tick_size=tick_size,
                )
            if exit_short is None:
                report.short_rejected[t] = True
            elif entry_short.price > 0:
                report.short_returns[t] = (
                    entry_short.price - exit_short.price
                ) / entry_short.price - comm
                report.short_entry_ts[t] = ts[entry_t]
                report.short_exit_ts[t] = ts[exit_t]
                report.short_filled_qty[t] = float(entry_short.filled_qty)
                report.short_partial[t] = entry_short.partial

    return report


def execution_forward_returns_depth(
    bid_px: npt.NDArray[np.floating],
    bid_sz: npt.NDArray[np.floating],
    ask_px: npt.NDArray[np.floating],
    ask_sz: npt.NDArray[np.floating],
    *,
    horizon: int = 1,
    order_qty: int = 1,
    n_levels: int = 5,
    latency_steps: int = 1,
    commission_bps: float = 0.0,
    fallback_bid: npt.NDArray[np.floating] | None = None,
    fallback_ask: npt.NDArray[np.floating] | None = None,
    allow_l1_fallback: bool = False,
    allow_partial_fills: bool = False,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
) -> tuple[FloatArray, FloatArray]:
    """عوائد أمامية بمسح عمق ظاهر عند الدخول (t) والخروج (t+h).

    إن نقصت السيولة الظاهرة يُستخدم مسار L1+slippage كاحتياطي إن وُجد،
    وإلا NaN (لا اختلاق عمق).
    """
    report = realistic_depth_execution_simulation(
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
        horizon=horizon,
        order_qty=order_qty,
        n_levels=n_levels,
        latency_steps=latency_steps,
        allow_partial_fills=allow_partial_fills,
        commission_bps=commission_bps,
        fallback_bid=fallback_bid,
        fallback_ask=fallback_ask,
        allow_l1_fallback=allow_l1_fallback,
        slippage_ticks=slippage_ticks,
        tick_size=tick_size,
    )
    return report.long_returns, report.short_returns


def stack_depth_levels(
    frame: pl.DataFrame,
    *,
    n_levels: int = 5,
    side: str,
) -> npt.NDArray[np.float64]:
    """يجمع أعمدة ``depth_{side}_px/sz_k`` إلى مصفوفة ``(n, n_levels)``."""
    prefix = "bid" if side == "B" else "ask"
    px_cols = [f"depth_{prefix}_px_{k}" for k in range(1, n_levels + 1)]
    sz_cols = [f"depth_{prefix}_sz_{k}" for k in range(1, n_levels + 1)]
    for c in px_cols + sz_cols:
        if c not in frame.columns:
            n = frame.height
            return np.zeros((n, n_levels), dtype=np.float64)
    px = np.column_stack([frame[c].to_numpy().astype(np.float64) for c in px_cols])
    return px  # caller stacks sz separately


def depth_matrices_from_frame(
    frame: pl.DataFrame,
    *,
    n_levels: int = 5,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """يعيد ``(bid_px, bid_sz, ask_px, ask_sz)`` شكل ``(n, n_levels)``."""
    n = frame.height
    bid_px = np.zeros((n, n_levels), dtype=np.float64)
    bid_sz = np.zeros((n, n_levels), dtype=np.float64)
    ask_px = np.zeros((n, n_levels), dtype=np.float64)
    ask_sz = np.zeros((n, n_levels), dtype=np.float64)
    for k in range(1, n_levels + 1):
        i = k - 1
        bp, bs = f"depth_bid_px_{k}", f"depth_bid_sz_{k}"
        ap, asz = f"depth_ask_px_{k}", f"depth_ask_sz_{k}"
        if bp in frame.columns:
            bid_px[:, i] = frame[bp].to_numpy().astype(np.float64)
        if bs in frame.columns:
            bid_sz[:, i] = frame[bs].to_numpy().astype(np.float64)
        if ap in frame.columns:
            ask_px[:, i] = frame[ap].to_numpy().astype(np.float64)
        if asz in frame.columns:
            ask_sz[:, i] = frame[asz].to_numpy().astype(np.float64)
    return bid_px, bid_sz, ask_px, ask_sz


__all__ = [
    "DepthExecutionSimulationReport",
    "depth_matrices_from_frame",
    "execution_forward_returns_depth",
    "realistic_depth_execution_simulation",
    "stack_depth_levels",
]
