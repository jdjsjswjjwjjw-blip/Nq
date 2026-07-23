"""تنفيذ بمسح عمق الدفتر — دخول وخروج من السيولة الظاهرة فقط."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from nq.contracts.mbo import PRICE_SCALE
from nq.orderbook.depth import walk_buy_vwap, walk_sell_vwap
from nq.simulation.execution.costs import commission_rate

FloatArray = npt.NDArray[np.float64]


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
        bp = float(bid_px[index, k]) if bid_px.ndim == 2 else float(bid_px[index])
        bs = float(bid_sz[index, k]) if bid_sz.ndim == 2 else float(bid_sz[index])
        ap = float(ask_px[index, k]) if ask_px.ndim == 2 else float(ask_px[index])
        asz = float(ask_sz[index, k]) if ask_sz.ndim == 2 else float(ask_sz[index])
        if np.isfinite(bp) and np.isfinite(bs) and bs > 0:
            bids.append((int(round(bp / PRICE_SCALE)), int(bs)))
        if np.isfinite(ap) and np.isfinite(asz) and asz > 0:
            asks.append((int(round(ap / PRICE_SCALE)), int(asz)))
    return bids, asks


def execution_forward_returns_depth(
    bid_px: npt.NDArray[np.floating],
    bid_sz: npt.NDArray[np.floating],
    ask_px: npt.NDArray[np.floating],
    ask_sz: npt.NDArray[np.floating],
    *,
    horizon: int = 1,
    order_qty: int = 1,
    n_levels: int = 5,
    commission_bps: float = 0.0,
    fallback_bid: npt.NDArray[np.floating] | None = None,
    fallback_ask: npt.NDArray[np.floating] | None = None,
    slippage_ticks: float = 0.5,
    tick_size: float = 0.25,
) -> tuple[FloatArray, FloatArray]:
    """عوائد أمامية بمسح عمق ظاهر عند الدخول (t) والخروج (t+h).

    إن نقصت السيولة الظاهرة يُستخدم مسار L1+slippage كاحتياطي إن وُجد،
    وإلا NaN (لا اختلاق عمق).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if order_qty < 1:
        raise ValueError(f"order_qty must be >= 1, got {order_qty}")

    bp = np.asarray(bid_px, dtype=np.float64)
    bs = np.asarray(bid_sz, dtype=np.float64)
    ap = np.asarray(ask_px, dtype=np.float64)
    az = np.asarray(ask_sz, dtype=np.float64)
    if bp.ndim == 1:
        bp = bp.reshape(-1, 1)
        bs = bs.reshape(-1, 1)
        ap = ap.reshape(-1, 1)
        az = az.reshape(-1, 1)
    n = bp.shape[0]
    long_fwd = np.full(n, np.nan, dtype=np.float64)
    short_fwd = np.full(n, np.nan, dtype=np.float64)
    comm = commission_rate(commission_bps=commission_bps)
    slip = slippage_ticks * tick_size
    fb = None if fallback_bid is None else np.asarray(fallback_bid, dtype=np.float64)
    fa = None if fallback_ask is None else np.asarray(fallback_ask, dtype=np.float64)

    for t in range(n - horizon):
        e_bids, e_asks = _levels_at(bp, bs, ap, az, t, n_levels=n_levels)
        x_bids, x_asks = _levels_at(bp, bs, ap, az, t + horizon, n_levels=n_levels)

        entry_long = walk_buy_vwap(e_asks, order_qty)
        exit_long = walk_sell_vwap(x_bids, order_qty)
        entry_short = walk_sell_vwap(e_bids, order_qty)
        exit_short = walk_buy_vwap(x_asks, order_qty)

        if entry_long is None and fa is not None and np.isfinite(fa[t]):
            entry_long = float(fa[t]) + slip
        if exit_long is None and fb is not None and np.isfinite(fb[t + horizon]):
            exit_long = float(fb[t + horizon]) - slip
        if entry_short is None and fb is not None and np.isfinite(fb[t]):
            entry_short = float(fb[t]) - slip
        if exit_short is None and fa is not None and np.isfinite(fa[t + horizon]):
            exit_short = float(fa[t + horizon]) + slip

        if entry_long is not None and exit_long is not None and entry_long > 0:
            long_fwd[t] = (exit_long - entry_long) / entry_long - comm
        if entry_short is not None and exit_short is not None and entry_short > 0:
            short_fwd[t] = (entry_short - exit_short) / entry_short - comm

    return long_fwd, short_fwd


def stack_depth_levels(
    frame,
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
    frame,
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
    "depth_matrices_from_frame",
    "execution_forward_returns_depth",
    "stack_depth_levels",
]
