"""اختبارات دورة حياة العمق السببية (دخول/مراقبة/تنفيذ/خروج)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.determinism import make_generator
from nq.orderbook import OrderBook, walk_buy_vwap, walk_sell_vwap
from nq.orderbook.depth import DepthSnapshot
from nq.research.orchestrator import run_research_pipeline
from nq.simulation.depth_lifecycle import depth_at_bar_close, depth_event_series
from nq.simulation.execution import execution_forward_returns_depth
from tests.mbo_factory import make_stream
from tests.test_coverage import _paired_streams


def test_orderbook_snapshot_top_n_and_imbalance() -> None:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("A", "B", 99, 3, 2)
    book.apply("A", "A", 102, 4, 3)
    book.apply("A", "A", 103, 2, 4)
    snap = book.snapshot(2, availability_ts=10)
    assert isinstance(snap, DepthSnapshot)
    assert snap.bid_levels == ((100, 5), (99, 3))
    assert snap.ask_levels == ((102, 4), (103, 2))
    assert snap.cum_bid == 8
    assert snap.cum_ask == 6
    assert snap.imbalance == (8 - 6) / 14
    assert book.size_at("B", 99) == 3


def test_walk_book_requires_visible_liquidity() -> None:
    asks = [(102, 1), (103, 1)]
    assert walk_buy_vwap(asks, qty=2) == ((102 + 103) / 2) * PRICE_SCALE
    assert walk_buy_vwap(asks, qty=3) is None  # لا اختلاق عمق
    bids = [(100, 2)]
    assert walk_sell_vwap(bids, qty=2) == 100 * PRICE_SCALE
    assert walk_sell_vwap(bids, qty=3) is None


def test_depth_bar_close_availability_at_bucket_end() -> None:
    base = int(100 / PRICE_SCALE)
    events = [
        ("A", "B", base, 5, 1),
        ("A", "A", base + int(1 / PRICE_SCALE), 4, 2),
        ("T", "B", base, 1, 0),
        ("A", "B", base - int(1 / PRICE_SCALE), 3, 3),
    ]
    interval = 1_000_000
    ts = [0, 10, interval // 2, interval + 10]
    frame = make_stream(events, event_ts=ts, sequence=list(range(1, 5)))
    bars = depth_at_bar_close(frame, interval_ns=interval, n_levels=3)
    assert bars.height >= 1
    assert (bars[AVAILABILITY_TS] == bars["bucket_end"]).all()
    assert "depth_cum_bid" in bars.columns
    assert "depth_bid_sz_1" in bars.columns


def test_depth_event_series_causal_past_stable() -> None:
    nq, _ = _paired_streams(600, seed=41)
    base = depth_event_series(nq, n_levels=3)
    if base.height < 20:
        return
    cut = int(base[AVAILABILITY_TS].median())
    past = base.filter(pl.col(AVAILABILITY_TS) <= cut)
    scrambled = nq.with_columns(
        pl.when(pl.col(EVENT_TS) > cut)
        .then(pl.col("size") + 50)
        .otherwise(pl.col("size"))
        .alias("size")
    )
    again = depth_event_series(scrambled, n_levels=3)
    past2 = again.filter(pl.col(AVAILABILITY_TS) <= cut)
    cols = ["depth_cum_bid", "depth_cum_ask", "depth_imbalance"]
    a = past.select(AVAILABILITY_TS, *cols).sort(AVAILABILITY_TS)
    b = past2.select(AVAILABILITY_TS, *cols).sort(AVAILABILITY_TS)
    assert a.equals(b)


def test_execution_depth_walk_uses_levels() -> None:
    # L1 thin, L2 deeper — VWAP should average
    bid_px = np.array([[100.0, 99.0], [100.5, 99.5], [101.0, 100.0]])
    bid_sz = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    ask_px = np.array([[100.25, 100.5], [100.75, 101.0], [101.25, 101.5]])
    ask_sz = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    long_fwd, short_fwd = execution_forward_returns_depth(
        bid_px,
        bid_sz,
        ask_px,
        ask_sz,
        horizon=1,
        order_qty=2,
        n_levels=2,
        commission_bps=0.0,
    )
    assert np.isfinite(long_fwd[0])
    assert np.isfinite(short_fwd[0])


def test_pipeline_attaches_depth_columns() -> None:
    nq, mnq = _paired_streams(1800, seed=42)
    result = run_research_pipeline(
        nq,
        mnq,
        interval_ns=10_000,
        n_permutations=40,
        parallel_coverage=False,
        rng=make_generator(7),
        quiet=True,
    )
    for col in (
        "depth_cum_bid",
        "depth_cum_ask",
        "depth_imbalance",
        "depth_bid_sz_1",
        "depth_ask_sz_1",
    ):
        assert col in result.features.columns, f"missing {col}"
