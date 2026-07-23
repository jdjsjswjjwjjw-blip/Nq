"""اختبارات تدفّق tick/event (الأبعاد 1–4)."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl

from nq.models.tick_stream import MarketPhase, MaskPath, _trap_setup, build_tick_stream
from tests.mbo_factory import Event, make_stream


def _paired_mbo(n: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    events: list[Event] = []
    ts: list[int] = []
    seq: list[int] = []
    price = 20_000_000_000
    for i in range(n):
        t = i * 1000
        events.extend(
            [
                ("A", "B", price, 5, i * 2 + 1),
                ("A", "A", price + 1_000_000, 5, i * 2 + 2),
                ("T", "B", price, 1, 0),
            ]
        )
        ts.extend([t, t + 1, t + 2])
        seq.extend([i * 3 + 1, i * 3 + 2, i * 3 + 3])
        price += 250_000
    nq = make_stream(events, instrument_id=1, symbol="NQ", event_ts=ts, sequence=seq)
    mnq = make_stream(events, instrument_id=2, symbol="MNQ", event_ts=ts, sequence=seq)
    return nq, mnq


def test_build_tick_stream_has_book_and_vp_columns() -> None:
    nq, mnq = _paired_mbo(8)
    stream = build_tick_stream(nq, mnq)
    assert stream.height > 0
    assert "nq_best_bid_norm" in stream.frame.columns
    assert "nq_vah_bid_liq_log" in stream.frame.columns
    assert "poc_dist_norm" in stream.frame.columns
    assert "mask_path" in stream.frame.columns
    assert "market_phase" in stream.frame.columns


def test_tick_stream_causal_order() -> None:
    nq, mnq = _paired_mbo(5)
    stream = build_tick_stream(nq, mnq)
    times = stream.frame["event_ts"].to_list()
    assert times == sorted(times)


def test_mask_path_values() -> None:
    nq, mnq = _paired_mbo(4)
    stream = build_tick_stream(nq, mnq)
    paths = set(stream.frame["mask_path"].to_list())
    assert paths.issubset({int(MaskPath.STANDALONE), int(MaskPath.CROSS_TRAP)})


def test_bearish_trap_uses_lower_low_nonconfirmation() -> None:
    trap = _trap_setup(
        mnq_delta=-2,
        nq_mid=99.0,
        mnq_mid=98.0,
        nq_high=105.0,
        mnq_high=105.0,
        nq_low=99.0,
        mnq_low=100.0,
    )
    assert trap == -1.0


def test_market_phase_values() -> None:
    nq, mnq = _paired_mbo(4)
    stream = build_tick_stream(nq, mnq)
    phases = set(stream.frame["market_phase"].to_list())
    assert phases.issubset(
        {int(MarketPhase.BALANCE), int(MarketPhase.EXPANSION), int(MarketPhase.NEUTRAL)}
    )


def test_tick_stream_resets_session_scoped_state_at_cme_session_boundary() -> None:
    et = ZoneInfo("America/New_York")
    session_one = int(dt.datetime(2024, 7, 15, 16, 50, tzinfo=et).timestamp() * 1e9)
    session_two = int(dt.datetime(2024, 7, 15, 18, 30, tzinfo=et).timestamp() * 1e9)
    events: list[Event] = [
        ("A", "B", 20_000_000_000, 5, 1),
        ("A", "A", 20_001_000_000, 5, 2),
        ("T", "B", 20_001_000_000, 1, 0),
        ("A", "B", 20_010_000_000, 5, 3),
        ("A", "A", 20_011_000_000, 5, 4),
        ("T", "B", 20_011_000_000, 1, 0),
    ]
    times = [
        session_one,
        session_one + 1,
        session_one + 2,
        session_two,
        session_two + 1,
        session_two + 2,
    ]
    seq = [1, 2, 3, 4, 5, 6]
    nq = make_stream(events, instrument_id=1, symbol="NQ", event_ts=times, sequence=seq)
    mnq = make_stream(events, instrument_id=2, symbol="MNQ", event_ts=times, sequence=seq)

    stream = build_tick_stream(nq, mnq)
    mnq_trades = stream.frame.filter(
        (pl.col("instrument_id") == 2)
        & pl.col("event_ts").is_in([session_one + 2, session_two + 2])
    )

    assert mnq_trades["mnq_signed_vol"].to_list() == [1.0, 1.0]
