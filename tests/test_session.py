"""اختبارات مراحل الجلسة intraday."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from nq.core.session import (
    SessionPhase,
    calendar_date_from_ns,
    is_globex_from_ns,
    is_rth_from_ns,
    session_phase_from_ns,
    trading_session_id_from_ns,
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


def test_cme_trading_session_id_boundaries() -> None:
    et = ZoneInfo("America/New_York")

    before_globex_close = int(dt.datetime(2024, 7, 15, 16, 59, tzinfo=et).timestamp() * 1e9)
    maintenance = int(dt.datetime(2024, 7, 15, 17, 30, tzinfo=et).timestamp() * 1e9)
    globex_reopen = int(dt.datetime(2024, 7, 15, 18, 30, tzinfo=et).timestamp() * 1e9)
    midnight = int(dt.datetime(2024, 7, 16, 0, 5, tzinfo=et).timestamp() * 1e9)
    rth_open = int(dt.datetime(2024, 7, 16, 9, 45, tzinfo=et).timestamp() * 1e9)
    rth_close = int(dt.datetime(2024, 7, 16, 16, 0, tzinfo=et).timestamp() * 1e9)

    assert calendar_date_from_ns(globex_reopen) == "2024-07-15"
    assert trading_session_id_from_ns(before_globex_close) == "2024-07-15"
    assert trading_session_id_from_ns(maintenance) == "2024-07-15"
    assert trading_session_id_from_ns(globex_reopen) == "2024-07-16"
    assert trading_session_id_from_ns(midnight) == "2024-07-16"
    assert trading_session_id_from_ns(rth_open) == "2024-07-16"
    assert trading_session_id_from_ns(rth_close) == "2024-07-16"
    assert is_globex_from_ns(globex_reopen)
    assert is_rth_from_ns(rth_open)
    assert not is_rth_from_ns(rth_close)


def test_cme_trading_session_id_survives_dst_transition() -> None:
    et = ZoneInfo("America/New_York")
    before_dst_jump = int(dt.datetime(2024, 3, 10, 1, 30, tzinfo=et).timestamp() * 1e9)
    after_evening_reopen = int(dt.datetime(2024, 3, 10, 18, 30, tzinfo=et).timestamp() * 1e9)

    assert trading_session_id_from_ns(before_dst_jump) == "2024-03-10"
    assert trading_session_id_from_ns(after_evening_reopen) == "2024-03-11"
