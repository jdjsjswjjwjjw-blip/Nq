"""اختبارات مُحاكي ملف الحجم."""

from __future__ import annotations

import polars as pl

from nq.contracts.instruments import instrument_metadata
from nq.contracts.mbo import PRICE_SCALE
from nq.simulation.volume_profile import (
    DevelopingVolumeProfile,
    build_volume_profile,
    classify_nodes,
    developing_value_area,
    value_area,
)
from tests.mbo_factory import make_stream


def _profile_stream() -> pl.DataFrame:
    # حجم مركّز عند 100 (POC)، وأقل حوله.
    return make_stream(
        [
            ("T", "B", 98, 1, 0),
            ("T", "B", 99, 3, 0),
            ("T", "A", 100, 10, 0),
            ("T", "B", 101, 3, 0),
            ("T", "A", 102, 1, 0),
        ],
        event_ts=[0, 1, 2, 3, 4],
        sequence=[1, 2, 3, 4, 5],
    )


def test_build_volume_profile() -> None:
    profile = build_volume_profile(_profile_stream())
    assert profile["price"].to_list() == [98, 99, 100, 101, 102]
    assert profile["volume"].to_list() == [1, 3, 10, 3, 1]


def test_value_area_poc_and_bounds() -> None:
    profile = build_volume_profile(_profile_stream())
    va = value_area(profile, fraction=0.7)
    assert va is not None
    assert va.poc == 100
    assert va.total_volume == 18
    # target = 0.7*18 = 12.6; POC=10, add 99(3)->13 covers -> then 101(3)->16
    assert va.val <= 100 <= va.vah
    assert va.value_volume >= 12.6


def test_value_area_empty_returns_none() -> None:
    empty = make_stream([])
    assert value_area(build_volume_profile(empty)) is None


def test_classify_nodes_hvn_lvn() -> None:
    profile = build_volume_profile(_profile_stream())
    nodes = classify_nodes(profile)
    # 100 is a local max (HVN); interior local minima none here besides edges
    hvn_prices = nodes.filter(nodes["is_hvn"])["price"].to_list()
    assert 100 in hvn_prices


def test_developing_volume_profile_incremental() -> None:
    profile = DevelopingVolumeProfile()
    for price, size in [(100, 5), (100, 5), (110, 3)]:
        profile.add_trade(price, size)
    va = profile.value_area()
    assert va is not None
    assert va.poc == 100
    feats = profile.features_at_mid(100.0, ref_price=1.0, near_ticks=2)
    assert feats[5] == 1.0  # in_value_area


def test_developing_volume_profile_near_levels_use_nq_tick_size() -> None:
    meta = instrument_metadata("NQU4")
    level = int(round(20_000.0 / PRICE_SCALE))
    profile = DevelopingVolumeProfile()
    profile.add_trade(level, 10)

    at_two_ticks = profile.features_at_mid(
        level + 2 * meta.tick_size_fixed,
        ref_price=float(level),
        near_ticks=2,
        tick_size_fixed=meta.tick_size_fixed,
    )
    at_three_ticks = profile.features_at_mid(
        level + 3 * meta.tick_size_fixed,
        ref_price=float(level),
        near_ticks=2,
        tick_size_fixed=meta.tick_size_fixed,
    )

    assert at_two_ticks[3] == 1.0
    assert at_two_ticks[4] == 1.0
    assert at_three_ticks[3] == 0.0
    assert at_three_ticks[4] == 0.0


def test_instrument_metadata_distinguishes_nq_and_mnq_contract_specs() -> None:
    nq = instrument_metadata("NQU4")
    mnq = instrument_metadata("MNQU4")

    assert nq.root_symbol == "NQ"
    assert mnq.root_symbol == "MNQ"
    assert nq.tick_size == 0.25
    assert mnq.tick_size == 0.25
    assert nq.tick_size_fixed == mnq.tick_size_fixed == int(round(0.25 / PRICE_SCALE))
    assert nq.point_value == 20.0
    assert mnq.point_value == 2.0


def test_developing_value_area_accumulates_within_session() -> None:
    stream = make_stream(
        [
            ("T", "B", 100, 5, 0),  # bucket 0 POC 100
            ("T", "B", 100, 5, 0),
            ("T", "A", 110, 1, 0),  # bucket-local POC would be 110
        ],
        event_ts=[0, 1, 11],
        sequence=[1, 2, 3],
    )
    dev = developing_value_area(stream, interval_ns=10).sort("bucket_start")
    assert dev["poc"].to_list() == [100, 100]
    assert dev["poc_migration"].to_list() == [0, 0]
    assert dev["availability_ts"].to_list() == [10, 20]
