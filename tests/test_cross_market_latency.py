"""اختبارات latency في cross_market."""

from __future__ import annotations

import polars as pl

from nq.simulation.cross_market import cross_market_features
from tests.mbo_factory import Event, make_stream


def _simple_market(*, symbol: str, instrument_id: int, base_ts: int = 0) -> pl.DataFrame:
    events: list[Event] = []
    ts: list[int] = []
    seq: list[int] = []
    price = 20_000_000_000
    for i in range(6):
        t = base_ts + i * 50_000
        events.extend(
            [
                ("A", "B", price, 5, i * 2 + 1),
                ("A", "A", price + 1_000_000, 5, i * 2 + 2),
                ("T", "B", price, 1, 0),
            ]
        )
        ts.extend([t, t + 1, t + 2])
        seq.extend([i * 3 + 1, i * 3 + 2, i * 3 + 3])
    return make_stream(
        events, instrument_id=instrument_id, symbol=symbol, event_ts=ts, sequence=seq
    )


def test_latency_changes_alignment() -> None:
    nq = _simple_market(symbol="NQ", instrument_id=1)
    mnq = _simple_market(symbol="MNQ", instrument_id=2, base_ts=10_000)
    zero = cross_market_features(nq, mnq, interval_ns=100_000, lead_lag_window=2, latency_ns=0)
    shifted = cross_market_features(
        nq, mnq, interval_ns=100_000, lead_lag_window=2, latency_ns=50_000
    )
    assert zero.height > 0 and shifted.height > 0
    assert not zero.select("mnq_delta").equals(shifted.select("mnq_delta"))
