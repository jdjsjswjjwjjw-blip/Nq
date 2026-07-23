"""مراحل جلسة التداول intraday (Session Phases).

يُصنّف كل ``bucket_end`` / ``availability_ts`` إلى مرحلة جلسة سببية
(متاحة عند إغلاق النافذة فقط). الأوقات الافتراضية: RTH بتوقيت America/New_York
لعقود NQ (09:30–16:00 ET).
"""

from __future__ import annotations

import datetime as dt
from enum import IntEnum
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

_ET: Final = ZoneInfo("America/New_York")
_RTH_OPEN: Final = dt.time(9, 30)
_RTH_CLOSE: Final = dt.time(16, 0)
_MAINTENANCE_START: Final = dt.time(17, 0)
_GLOBEX_OPEN: Final = dt.time(18, 0)
_OPEN_END: Final = dt.time(10, 0)
_MORNING_END: Final = dt.time(12, 0)
_LUNCH_END: Final = dt.time(13, 30)
_AFTERNOON_END: Final = dt.time(15, 30)

SESSION_PHASE: Final = "session_phase"
MINUTES_SINCE_RTH_OPEN: Final = "minutes_since_rth_open"
CALENDAR_DATE: Final = "calendar_date"
TRADING_SESSION_ID: Final = "trading_session_id"
SESSION_DATE: Final = TRADING_SESSION_ID
IS_RTH: Final = "is_rth"
IS_ETH: Final = "is_eth"
IS_GLOBEX: Final = "is_globex"


class SessionPhase(IntEnum):
    """مراحل الجلسة intraday (قيم صحيحة للتعلّم الآلي)."""

    ETH = 0
    OPEN = 1
    MORNING = 2
    LUNCH = 3
    AFTERNOON = 4
    CLOSE = 5
    MAINTENANCE = 6


def _phase_for_time(local_time: dt.time) -> SessionPhase:
    if _MAINTENANCE_START <= local_time < _GLOBEX_OPEN:
        phase = SessionPhase.MAINTENANCE
    elif local_time < _RTH_OPEN or local_time >= _RTH_CLOSE:
        phase = SessionPhase.ETH
    elif local_time < _OPEN_END:
        phase = SessionPhase.OPEN
    elif local_time < _MORNING_END:
        phase = SessionPhase.MORNING
    elif local_time < _LUNCH_END:
        phase = SessionPhase.LUNCH
    elif local_time < _AFTERNOON_END:
        phase = SessionPhase.AFTERNOON
    else:
        phase = SessionPhase.CLOSE
    return phase


def _minutes_since_rth_open(local_time: dt.time) -> int | None:
    if local_time < _RTH_OPEN or local_time >= _RTH_CLOSE:
        return None
    open_dt = dt.datetime.combine(dt.date(2000, 1, 1), _RTH_OPEN)
    now_dt = dt.datetime.combine(dt.date(2000, 1, 1), local_time)
    return int((now_dt - open_dt).total_seconds() // 60)


def _local_from_ns(ts_ns: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET)


def _is_rth_time(local_time: dt.time) -> bool:
    return _RTH_OPEN <= local_time < _RTH_CLOSE


def _is_globex_time(local_time: dt.time) -> bool:
    return local_time >= _GLOBEX_OPEN or local_time < _RTH_OPEN


def session_phase_from_ns(ts_ns: int) -> int:
    """يُرجع ``session_phase`` كعدد صحيح من طابع نانوثانية."""
    local = _local_from_ns(ts_ns).time()
    return int(_phase_for_time(local))


def minutes_since_rth_open_from_ns(ts_ns: int) -> int | None:
    local = _local_from_ns(ts_ns).time()
    return _minutes_since_rth_open(local)


def calendar_date_from_ns(ts_ns: int) -> str:
    """تاريخ التقويم المحلي ET فقط، وليس معرّف جلسة futures."""
    local = _local_from_ns(ts_ns)
    return local.date().isoformat()


def trading_session_id_from_ns(ts_ns: int) -> str:
    """معرّف جلسة CME futures: جلسة المساء 18:00 ET تُنسب ليوم التداول التالي."""
    local = _local_from_ns(ts_ns)
    trading_date = (
        local.date() + dt.timedelta(days=1) if local.time() >= _GLOBEX_OPEN else local.date()
    )
    return trading_date.isoformat()


def session_date_from_ns(ts_ns: int) -> str:
    """توافق قديم: يُعيد الآن معرّف جلسة CME، لا تاريخ التقويم ET."""
    return trading_session_id_from_ns(ts_ns)


def is_rth_from_ns(ts_ns: int) -> bool:
    local = _local_from_ns(ts_ns).time()
    return _is_rth_time(local)


def is_globex_from_ns(ts_ns: int) -> bool:
    local = _local_from_ns(ts_ns).time()
    return _is_globex_time(local)


def add_session_columns(frame: pl.DataFrame, *, time_col: str) -> pl.DataFrame:
    """يضيف أعمدة الجلسة المركزية من طابع إتاحة واحد."""
    if time_col not in frame.columns:
        raise ValueError(f"time column {time_col!r} not found")

    times = frame[time_col].to_list()
    phases = [session_phase_from_ns(int(t)) for t in times]
    minutes = [minutes_since_rth_open_from_ns(int(t)) for t in times]
    calendar_dates = [calendar_date_from_ns(int(t)) for t in times]
    trading_sessions = [trading_session_id_from_ns(int(t)) for t in times]
    rth_flags = [is_rth_from_ns(int(t)) for t in times]
    globex_flags = [is_globex_from_ns(int(t)) for t in times]
    return frame.with_columns(
        pl.Series(SESSION_PHASE, phases, dtype=pl.Int8()),
        pl.Series(MINUTES_SINCE_RTH_OPEN, minutes, dtype=pl.Int64()),
        pl.Series(CALENDAR_DATE, calendar_dates, dtype=pl.Utf8()),
        pl.Series(TRADING_SESSION_ID, trading_sessions, dtype=pl.Utf8()),
        pl.Series(IS_RTH, rth_flags, dtype=pl.Boolean()),
        pl.Series(IS_ETH, [not flag for flag in rth_flags], dtype=pl.Boolean()),
        pl.Series(IS_GLOBEX, globex_flags, dtype=pl.Boolean()),
    )


__all__ = [
    "CALENDAR_DATE",
    "IS_ETH",
    "IS_GLOBEX",
    "IS_RTH",
    "MINUTES_SINCE_RTH_OPEN",
    "SESSION_DATE",
    "SESSION_PHASE",
    "TRADING_SESSION_ID",
    "SessionPhase",
    "add_session_columns",
    "calendar_date_from_ns",
    "is_globex_from_ns",
    "is_rth_from_ns",
    "minutes_since_rth_open_from_ns",
    "session_date_from_ns",
    "session_phase_from_ns",
    "trading_session_id_from_ns",
]
