"""عمق عند منطقة يحددها العين: يوم واحد، بلا مستقبل، بلا تصدير."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.contracts.mbo import PRICE_SCALE
from nq.research.manual_zone_depth import (
    TICK_POINTS,
    book_at,
    et_clock_to_ns,
    fixed_to_points,
    parse_zone,
    points_to_fixed,
    write_zone_depth_report,
    zone_depth,
)
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_PX_ASK = 20_000_250_000


def _setup_ts() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "zone_depth" not in nq.research.__all__
    assert not hasattr(nq.research, "zone_depth")
    assert not hasattr(nq.research, "book_at")


def test_points_round_trip_on_tick() -> None:
    assert TICK_POINTS == 0.25
    assert abs(TICK_POINTS / PRICE_SCALE - 250_000_000) < 1e-6
    for pts in (21000.00, 21000.25, 21450.75):
        assert abs(fixed_to_points(points_to_fixed(pts)) - pts) < 1e-9


def test_et_clock_matches_explicit_datetime() -> None:
    assert et_clock_to_ns("2025-06-03", "10:35:00") == _setup_ts()


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mbo = make_stream(
        [("A", "B", _PX, 1, 1), ("A", "A", _PX_ASK, 1, 2)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    zone = parse_zone(day="2025-06-03", clock="10:35:00", levels_points=(20.0,))
    with pytest.raises(ValueError, match="multi-day"):
        zone_depth(mbo, zone)


def test_future_add_absent_from_depth_at_t() -> None:
    t = _setup_ts()
    mbo = make_stream(
        [
            ("A", "B", _PX, 4, 1),
            ("A", "A", _PX_ASK, 2, 2),
            ("A", "B", _PX, 9, 3),
        ],
        event_ts=[t - 1_000, t - 500, t + 1_000],
    )
    zone = parse_zone(day="2025-06-03", clock="10:35:00", levels_points=(20.0,), band_ticks=0)
    depth = zone_depth(mbo, zone)
    eye = depth.filter(pl.col("is_eye_level"))
    assert eye.height == 1
    assert int(eye["bid_size"][0]) == 4
    book = book_at(mbo, t)
    assert book.size_at("B", _PX) == 4


def test_two_adds_at_eye_level_sum_size() -> None:
    t = _setup_ts()
    mbo = make_stream(
        [
            ("A", "B", _PX, 3, 1),
            ("A", "B", _PX, 5, 2),
            ("A", "A", _PX_ASK, 1, 3),
        ],
        event_ts=[t - 3, t - 2, t - 1],
    )
    zone = parse_zone(day="2025-06-03", clock="10:35:00", levels_points=(20.0,), band_ticks=0)
    depth = zone_depth(mbo, zone)
    eye = depth.filter(pl.col("is_eye_level"))
    assert int(eye["bid_size"][0]) == 8
    assert int(eye["zone_bid_stack"][0]) == 8


def test_report_writes(tmp_path: Path) -> None:
    t = _setup_ts()
    mbo = make_stream(
        [("A", "B", _PX, 2, 1), ("A", "A", _PX_ASK, 1, 2)],
        event_ts=[t - 2, t - 1],
    )
    zone = parse_zone(
        day="2025-06-03",
        clock="10:35:00",
        levels_points=(20.0, 20.25),
        label="eye",
        band_ticks=0,
    )
    depth = zone_depth(mbo, zone)
    written = write_zone_depth_report(
        depth,
        {
            "reconstructed_order_book": True,
            "concatenated_raw_mbo": False,
            "live_overlay": False,
        },
        tmp_path,
    )
    text = (written / "MANUAL_ZONE_DEPTH.md").read_text(encoding="utf-8")
    assert "Not a live overlay" in text
    assert (written / "manual_zone_depth.parquet").is_file()
