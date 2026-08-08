"""اختبارات تصفير Volume Profile حسب جلسة السيولة (آسيا/لندن/نيويورك)."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl

from nq.core.session import (
    VP_LIQUIDITY_SESSION,
    VpLiquiditySession,
    vp_liquidity_session_from_ns,
    vp_liquidity_session_label,
)
from nq.simulation.auction import auction_action_states, auction_signals_from_states
from nq.simulation.volume_profile import developing_value_area
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_NS = 1_000_000_000


def _et_ns(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    local = dt.datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(local.timestamp() * _NS)


def test_vp_liquidity_session_boundaries_et() -> None:
    # 2024-06-03 Monday (EDT)
    assert vp_liquidity_session_from_ns(_et_ns(2024, 6, 3, 18, 0)) == int(VpLiquiditySession.ASIA)
    assert vp_liquidity_session_from_ns(_et_ns(2024, 6, 3, 2, 59)) == int(VpLiquiditySession.ASIA)
    assert vp_liquidity_session_from_ns(_et_ns(2024, 6, 3, 3, 0)) == int(VpLiquiditySession.LONDON)
    assert vp_liquidity_session_from_ns(_et_ns(2024, 6, 3, 9, 29)) == int(VpLiquiditySession.LONDON)
    assert vp_liquidity_session_from_ns(_et_ns(2024, 6, 3, 9, 30)) == int(
        VpLiquiditySession.NEW_YORK
    )
    assert vp_liquidity_session_from_ns(_et_ns(2024, 6, 3, 17, 59)) == int(
        VpLiquiditySession.NEW_YORK
    )
    assert vp_liquidity_session_label(0) == "asia"
    assert vp_liquidity_session_label(1) == "london"
    assert vp_liquidity_session_label(2) == "new_york"


def test_developing_va_resets_total_volume_across_sessions() -> None:
    """حجم آسيا لا يدخل تراكم لندن عند التصفير."""
    # شمعتان 5د في آسيا ثم شمعتان في لندن — أسعار مختلفة تمامًا
    events = [
        ("T", "B", 100, 50, 0),
        ("T", "B", 100, 50, 0),
        ("T", "B", 200, 10, 0),
        ("T", "B", 200, 10, 0),
    ]
    ts = [
        _et_ns(2024, 6, 3, 19, 0),
        _et_ns(2024, 6, 3, 19, 5),
        _et_ns(2024, 6, 3, 4, 0),  # next calendar morning London
        _et_ns(2024, 6, 3, 4, 5),
    ]
    # Fix date: London 4:00 on June 3 is after Asia June 2 19:00
    ts = [
        _et_ns(2024, 6, 2, 19, 0),
        _et_ns(2024, 6, 2, 19, 5),
        _et_ns(2024, 6, 3, 4, 0),
        _et_ns(2024, 6, 3, 4, 5),
    ]
    frame = make_stream(events, event_ts=ts, sequence=[1, 2, 3, 4])
    iv = 5 * 60 * _NS
    reset = developing_value_area(
        frame, interval_ns=iv, cumulative=True, reset_by_liquidity_session=True
    )
    no_reset = developing_value_area(
        frame, interval_ns=iv, cumulative=True, reset_by_liquidity_session=False
    )
    assert reset.height >= 2
    assert VP_LIQUIDITY_SESSION in reset.columns
    # أول برميل لندن بعد التصفير: الحجم ≈ حجم لندن فقط (10 أو 20) لا مجموع آسيا+لندن
    london = reset.filter(pl.col(VP_LIQUIDITY_SESSION) == int(VpLiquiditySession.LONDON))
    assert london.height >= 1
    first_london_vol = int(london["total_volume"][0])
    assert first_london_vol <= 20
    # بدون تصفير: الحجم التراكمي أكبر من حجم لندن وحدها
    london_nr = no_reset.filter(pl.col(VP_LIQUIDITY_SESSION) == int(VpLiquiditySession.LONDON))
    assert int(london_nr["total_volume"][0]) > first_london_vol
    # POC لندن بعد التصفير حول 200 وليس 100
    assert int(london["poc"][0]) == 200


def test_auction_action_states_default_session_reset_exposes_column() -> None:
    events = [("T", "B", 100 + (i % 3), 2, 0) for i in range(30)]
    # امتداد عبر حدود لندن→نيويورك
    base = _et_ns(2024, 6, 3, 8, 0)
    ts = [base + i * 5 * 60 * _NS for i in range(30)]
    frame = make_stream(events, event_ts=ts, sequence=list(range(1, 31)))
    states = auction_action_states(
        frame,
        profile_interval_ns=5 * 60 * _NS,
        signal_interval_ns=5 * 60 * _NS,
    )
    assert states.height >= 1
    assert VP_LIQUIDITY_SESSION in states.columns
    sigs = auction_signals_from_states(states)
    assert "vp_liquidity_session" in sigs.columns
    # يجب أن تظهر جلستان على الأقل حول 09:30
    assert states[VP_LIQUIDITY_SESSION].n_unique() >= 2


def test_poc_migration_zero_on_session_boundary() -> None:
    events = [
        ("T", "B", 100, 20, 0),
        ("T", "B", 101, 20, 0),
        ("T", "B", 300, 20, 0),
        ("T", "B", 301, 20, 0),
    ]
    ts = [
        _et_ns(2024, 6, 3, 8, 0),
        _et_ns(2024, 6, 3, 8, 5),
        _et_ns(2024, 6, 3, 9, 30),
        _et_ns(2024, 6, 3, 9, 35),
    ]
    frame = make_stream(events, event_ts=ts, sequence=[1, 2, 3, 4])
    dva = developing_value_area(
        frame,
        interval_ns=5 * 60 * _NS,
        cumulative=True,
        reset_by_liquidity_session=True,
    )
    # أول برميل نيويورك: poc_migration == 0 رغم قفزة POC
    ny = dva.filter(pl.col(VP_LIQUIDITY_SESSION) == int(VpLiquiditySession.NEW_YORK))
    assert ny.height >= 1
    assert int(ny["poc_migration"][0]) == 0
