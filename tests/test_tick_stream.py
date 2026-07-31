"""اختبارات تدفّق tick/event (الأبعاد 1–4)."""

from __future__ import annotations

import polars as pl

from nq.models.tick_stream import MarketPhase, MaskPath, build_tick_stream
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


def test_market_phase_values() -> None:
    nq, mnq = _paired_mbo(4)
    stream = build_tick_stream(nq, mnq)
    phases = set(stream.frame["market_phase"].to_list())
    assert phases.issubset(
        {int(MarketPhase.BALANCE), int(MarketPhase.EXPANSION), int(MarketPhase.NEUTRAL)}
    )


def test_nq_only_tick_stream_does_not_double_events() -> None:
    """nq is mnq → مسار أحادي؛ لا مضاعفة الأحداث ولا trap من الذات."""
    nq, _ = _paired_mbo(6)
    dual = build_tick_stream(nq, nq.clone())  # كائنان مختلفان → مسار ثنائي
    single = build_tick_stream(nq, nq)  # نفس الكائن → nq_only
    assert single.height == nq.height
    assert dual.height == 2 * nq.height
    assert "nq_best_bid_norm" in single.frame.columns
    assert "mnq_best_bid_norm" in single.frame.columns
    # دفتر MNQ فارغ في المسار الأحادي
    assert float(single.frame["mnq_best_bid_norm"].abs().max()) == 0.0
    assert set(single.frame["mask_path"].to_list()).issubset({int(MaskPath.STANDALONE)})
