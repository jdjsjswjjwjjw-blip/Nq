"""مراحل جلسة التداول + جلسات سيولة Volume Profile.

* ``SessionPhase``: مراحل RTH التفصيلية (فتح/صباح/…) بتوقيت America/New_York.
* ``VpLiquiditySession``: تقسيم غير متداخل لرسم/تصفير الفوليوم:
  آسيا / لندن / نيويورك.

كل التصنيفات سببية من الطابع الزمني فقط (point-in-time).
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
_OPEN_END: Final = dt.time(10, 0)
_MORNING_END: Final = dt.time(12, 0)
_LUNCH_END: Final = dt.time(13, 30)
_AFTERNOON_END: Final = dt.time(15, 30)

SESSION_PHASE: Final = "session_phase"
MINUTES_SINCE_RTH_OPEN: Final = "minutes_since_rth_open"
SESSION_DATE: Final = "session_date"  # تاريخ الجلسة بتوقيت ET (YYYY-MM-DD) — سببي من ts

#: جلسة سيولة لملف الحجم (آسيا / لندن / نيويورك).
VP_LIQUIDITY_SESSION: Final = "vp_liquidity_session"

# حدود تصفير الفوليوم (America/New_York) — فترات نصف-مفتوحة تغطي 24 ساعة:
# آسيا [18:00, 03:00) · لندن [03:00, 09:30) · نيويورك [09:30, 18:00)
_ASIA_START: Final = dt.time(18, 0)
_LONDON_START: Final = dt.time(3, 0)
_NY_START: Final = dt.time(9, 30)


class SessionPhase(IntEnum):
    """مراحل الجلسة intraday (قيم صحيحة للتعلّم الآلي)."""

    ETH = 0
    OPEN = 1
    MORNING = 2
    LUNCH = 3
    AFTERNOON = 4
    CLOSE = 5


class VpLiquiditySession(IntEnum):
    """جلسات سيولة لرسم/تصفير Volume Profile (غير متداخلة، ET)."""

    ASIA = 0
    LONDON = 1
    NEW_YORK = 2


def vp_liquidity_session_from_ns(ts_ns: int) -> int:
    """يصنّف الطابع إلى آسيا/لندن/نيويورك بتوقيت America/New_York (سببي)."""
    local = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET).time()
    if local >= _ASIA_START or local < _LONDON_START:
        return int(VpLiquiditySession.ASIA)
    if local < _NY_START:
        return int(VpLiquiditySession.LONDON)
    return int(VpLiquiditySession.NEW_YORK)


def vp_liquidity_session_label(session_id: int) -> str:
    """اسم نصّي ثابت لجلسة السيولة."""
    return {
        int(VpLiquiditySession.ASIA): "asia",
        int(VpLiquiditySession.LONDON): "london",
        int(VpLiquiditySession.NEW_YORK): "new_york",
    }.get(int(session_id), "unknown")


def _phase_for_time(local_time: dt.time) -> SessionPhase:
    if local_time < _RTH_OPEN or local_time >= _RTH_CLOSE:
        return SessionPhase.ETH
    if local_time < _OPEN_END:
        return SessionPhase.OPEN
    if local_time < _MORNING_END:
        return SessionPhase.MORNING
    if local_time < _LUNCH_END:
        return SessionPhase.LUNCH
    if local_time < _AFTERNOON_END:
        return SessionPhase.AFTERNOON
    return SessionPhase.CLOSE


def _minutes_since_rth_open(local_time: dt.time) -> int | None:
    if local_time < _RTH_OPEN or local_time >= _RTH_CLOSE:
        return None
    open_dt = dt.datetime.combine(dt.date(2000, 1, 1), _RTH_OPEN)
    now_dt = dt.datetime.combine(dt.date(2000, 1, 1), local_time)
    return int((now_dt - open_dt).total_seconds() // 60)


def session_phase_from_ns(ts_ns: int) -> int:
    """يُرجع ``session_phase`` كعدد صحيح من طابع نانوثانية."""
    local = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET).time()
    return int(_phase_for_time(local))


def minutes_since_rth_open_from_ns(ts_ns: int) -> int | None:
    local = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET).time()
    return _minutes_since_rth_open(local)


def session_date_from_ns(ts_ns: int) -> str:
    """تاريخ الجلسة (America/New_York) من طابع نانوثانية — سببي point-in-time."""
    local = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=_ET)
    return local.date().isoformat()


def add_session_columns(frame: pl.DataFrame, *, time_col: str) -> pl.DataFrame:
    """يضيف ``session_phase`` و ``minutes_since_rth_open`` و ``session_date``."""
    if time_col not in frame.columns:
        raise ValueError(f"time column {time_col!r} not found")

    times = frame[time_col].to_list()
    phases = [session_phase_from_ns(int(t)) for t in times]
    minutes = [minutes_since_rth_open_from_ns(int(t)) for t in times]
    dates = [session_date_from_ns(int(t)) for t in times]
    return frame.with_columns(
        pl.Series(SESSION_PHASE, phases, dtype=pl.Int8()),
        pl.Series(MINUTES_SINCE_RTH_OPEN, minutes, dtype=pl.Int64()),
        pl.Series(SESSION_DATE, dates, dtype=pl.Utf8()),
    )


__all__ = [
    "MINUTES_SINCE_RTH_OPEN",
    "SESSION_DATE",
    "SESSION_PHASE",
    "VP_LIQUIDITY_SESSION",
    "SessionPhase",
    "VpLiquiditySession",
    "add_session_columns",
    "minutes_since_rth_open_from_ns",
    "session_date_from_ns",
    "session_phase_from_ns",
    "vp_liquidity_session_from_ns",
    "vp_liquidity_session_label",
]
