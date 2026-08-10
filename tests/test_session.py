"""اختبارات مراحل الجلسة intraday."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from nq.core.session import SessionPhase, session_date_from_ns, session_phase_from_ns


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
