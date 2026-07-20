"""اختبارات مراحل الجلسة intraday."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from nq.core.session import SessionPhase, session_phase_from_ns


def test_rth_open_phase() -> None:
    et = ZoneInfo("America/New_York")
    # 2024-07-15 09:45 ET — داخل فترة الافتتاح
    ts = int(dt.datetime(2024, 7, 15, 9, 45, tzinfo=et).timestamp() * 1e9)
    assert session_phase_from_ns(ts) == int(SessionPhase.OPEN)


def test_eth_outside_rth() -> None:
    et = ZoneInfo("America/New_York")
    ts = int(dt.datetime(2024, 7, 15, 3, 0, tzinfo=et).timestamp() * 1e9)
    assert session_phase_from_ns(ts) == int(SessionPhase.ETH)
