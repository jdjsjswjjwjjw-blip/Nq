"""اختبارات مراحل الجلسة intraday."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from nq.core.session import (
    SessionPhase,
    VpLiquiditySession,
    clip_zone_end_ns,
    clip_zone_start_ns,
    session_date_from_ns,
    session_phase_from_ns,
    vp_liquidity_session_bounds_ns,
    vp_liquidity_session_from_ns,
)


def test_rth_open_phase() -> None:
    et = ZoneInfo("America/New_York")
    # 2024-07-15 09:45 ET — داخل فترة الافتتاح
    ts = int(dt.datetime(2024, 7, 15, 9, 45, tzinfo=et).timestamp() * 1e9)
    assert session_phase_from_ns(ts) == int(SessionPhase.OPEN)


def test_eth_outside_rth() -> None:
    et = ZoneInfo("America/New_York")
    ts = int(dt.datetime(2024, 7, 15, 3, 0, tzinfo=et).timestamp() * 1e9)
    assert session_phase_from_ns(ts) == int(SessionPhase.ETH)


def test_cme_trade_date_rolls_at_1800_et() -> None:
    et = ZoneInfo("America/New_York")
    before = int(dt.datetime(2024, 6, 3, 17, 59, tzinfo=et).timestamp() * 1e9)
    after = int(dt.datetime(2024, 6, 3, 18, 0, tzinfo=et).timestamp() * 1e9)
    assert session_date_from_ns(before) == "2024-06-03"
    assert session_date_from_ns(after) == "2024-06-04"


def test_zone_ends_at_data_when_london_is_truncated() -> None:
    et = ZoneInfo("America/New_York")
    london_open = int(dt.datetime(2024, 6, 4, 3, 0, tzinfo=et).timestamp() * 1e9)
    data_end = int(dt.datetime(2024, 6, 4, 5, 0, tzinfo=et).timestamp() * 1e9)
    calendar_ny = int(dt.datetime(2024, 6, 4, 9, 30, tzinfo=et).timestamp() * 1e9)
    start, end = vp_liquidity_session_bounds_ns(london_open + 1)
    assert start == london_open
    assert end == calendar_ny
    assert clip_zone_end_ns(london_open + 1, data_end) == data_end
    assert vp_liquidity_session_from_ns(data_end) == int(VpLiquiditySession.LONDON)


def test_zone_starts_at_data_when_asia_is_truncated() -> None:
    et = ZoneInfo("America/New_York")
    asia_open = int(dt.datetime(2024, 6, 3, 18, 0, tzinfo=et).timestamp() * 1e9)
    data_start = int(dt.datetime(2024, 6, 3, 20, 0, tzinfo=et).timestamp() * 1e9)
    london_open = int(dt.datetime(2024, 6, 4, 3, 0, tzinfo=et).timestamp() * 1e9)
    assert clip_zone_start_ns(data_start, data_start) == data_start
    assert clip_zone_start_ns(asia_open, data_start) == data_start
    assert clip_zone_end_ns(data_start, london_open) == london_open
