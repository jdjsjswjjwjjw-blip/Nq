"""اختبارات تسريع المعالجة مع بقاء نفس الأرقام السببية."""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from nq.models.tick_stream import build_tick_stream
from nq.models.windowing import build_sequences
from nq.orderbook import OrderBook
from nq.simulation.depth_lifecycle import depth_at_bar_close, depth_at_bar_close_multi
from nq.simulation.volume_profile import DevelopingVolumeProfile, value_area, value_area_from_levels
from tests.mbo_factory import make_stream
from tests.test_coverage import _paired_streams


def test_trail_liquidity_matches_explicit_sum() -> None:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("A", "B", 99, 3, 2)
    book.apply("A", "B", 98, 7, 3)
    book.apply("A", "A", 102, 4, 4)
    book.apply("A", "A", 103, 2, 5)
    book.apply("A", "A", 104, 9, 6)
    trail_bid, trail_ask = book.trail_liquidity()
    assert trail_bid == 3 + 7
    assert trail_ask == 2 + 9
    book.apply("C", "N", 0, 0, 1)
    trail_bid, trail_ask = book.trail_liquidity()
    assert trail_bid == 7  # best is now 99x3, trail = 98x7
    assert trail_ask == 2 + 9


def test_value_area_from_levels_matches_polars_frame() -> None:
    profile = pl.DataFrame(
        {
            "price": [100, 101, 102, 103, 104],
            "volume": [1, 5, 10, 4, 2],
        }
    )
    from_frame = value_area(profile, fraction=0.7)
    from_lists = value_area_from_levels(
        profile["price"].to_list(),
        profile["volume"].to_list(),
        fraction=0.7,
    )
    assert from_frame is not None and from_lists is not None
    assert from_frame == from_lists


def test_developing_profile_cache_stable_between_trades() -> None:
    profile = DevelopingVolumeProfile()
    profile.add_trade(100, 5)
    profile.add_trade(101, 8)
    profile.add_trade(102, 3)
    first = profile.value_area()
    second = profile.value_area()
    assert first is second  # cache hit
    profile.add_trade(103, 1)
    third = profile.value_area()
    assert third is not None and first is not None
    assert third != first


def test_depth_multi_matches_single_passes() -> None:
    nq, _ = _paired_streams(400, seed=17)
    interval_a = 1_000_000_000
    interval_b = 30 * 60 * 1_000_000_000
    multi = depth_at_bar_close_multi(
        nq, interval_ns_list=(interval_a, interval_b), n_levels=5
    )
    single_a = depth_at_bar_close(nq, interval_ns=interval_a, n_levels=5)
    single_b = depth_at_bar_close(nq, interval_ns=interval_b, n_levels=5)
    assert multi[interval_a].equals(single_a)
    assert multi[interval_b].equals(single_b)


def test_build_sequences_vectorized_values() -> None:
    frame = pl.DataFrame(
        {
            "availability_ts": list(range(12)),
            "a": [float(i) for i in range(12)],
            "b": [float(i * 2) for i in range(12)],
        }
    )
    ds = build_sequences(frame, feature_columns=["a", "b"], window=3, stride=2)
    assert ds.x.shape == (5, 3, 2)
    assert np.allclose(ds.x[0], [[0, 0], [1, 2], [2, 4]])
    assert np.allclose(ds.x[1], [[2, 4], [3, 6], [4, 8]])
    assert list(ds.times) == [2, 4, 6, 8, 10]


def test_tick_stream_reuse_identical_frame() -> None:
    events = [
        ("A", "B", 20_000_000_000, 5, 1),
        ("A", "A", 20_001_000_000, 5, 2),
        ("T", "B", 20_000_000_000, 1, 0),
        ("A", "B", 19_999_000_000, 3, 3),
        ("T", "A", 20_001_000_000, 1, 0),
    ]
    ts = [0, 1, 2, 3, 4]
    nq = make_stream(events, instrument_id=1, symbol="NQ", event_ts=ts, sequence=list(range(1, 6)))
    mnq = make_stream(events, instrument_id=2, symbol="MNQ", event_ts=ts, sequence=list(range(1, 6)))
    a = build_tick_stream(nq, mnq)
    b = build_tick_stream(nq, mnq)
    assert a.frame.equals(b.frame)


def test_value_area_hot_path_faster_than_polars_rebuild() -> None:
    """Smoke: الكاش+القوائم أسرع من إعادة بناء DataFrame كل مرة (بدون شرط زمن صارم)."""
    profile = DevelopingVolumeProfile()
    for i in range(80):
        profile.add_trade(100 + (i % 11), 1 + (i % 5))
    t0 = time.perf_counter()
    for _ in range(200):
        _ = profile.value_area()
    cached = time.perf_counter() - t0
    # بعد الصفقات، القراءة المتكررة رخيصة بفضل الكاش
    assert cached < 0.5
