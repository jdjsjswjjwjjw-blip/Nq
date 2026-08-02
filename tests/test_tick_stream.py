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


def test_tick_stream_snapshot_emit_scales() -> None:
    """على timestamps حقيقية: snapshots لكل ثانية << عدد الأحداث."""
    import polars as pl

    from nq.models.tick_stream import build_tick_stream
    from tests.mbo_factory import random_add_cancel_stream

    n = 20_000
    nq = random_add_cancel_stream(n, seed=9)
    start = 1_700_000_000_000_000_000
    step = (3600 * 1_000_000_000) // n
    nq = nq.with_columns(
        (pl.int_range(0, n) * step + start).alias("event_ts"),
        (pl.int_range(0, n) * step + start).alias("ingest_ts"),
    )
    snap = build_tick_stream(nq, nq, emit_interval_ns=1_000_000_000)
    assert 2_000 < snap.height < 5_000
    assert "nq_best_bid_norm" in snap.frame.columns


def test_build_tick_stream_has_book_and_vp_columns() -> None:
    nq, mnq = _paired_mbo(8)
    stream = build_tick_stream(nq, mnq, emit_interval_ns=None)
    assert stream.height > 0
    assert "nq_best_bid_norm" in stream.frame.columns
    assert "nq_vah_bid_liq_log" in stream.frame.columns
    assert "poc_dist_norm" in stream.frame.columns
    assert "mask_path" in stream.frame.columns
    assert "market_phase" in stream.frame.columns


def test_tick_stream_causal_order() -> None:
    nq, mnq = _paired_mbo(5)
    stream = build_tick_stream(nq, mnq, emit_interval_ns=None)
    times = stream.frame["event_ts"].to_list()
    assert times == sorted(times)


def test_mask_path_values() -> None:
    nq, mnq = _paired_mbo(4)
    stream = build_tick_stream(nq, mnq, emit_interval_ns=None)
    paths = set(stream.frame["mask_path"].to_list())
    assert paths.issubset({int(MaskPath.STANDALONE), int(MaskPath.CROSS_TRAP)})


def test_market_phase_values() -> None:
    nq, mnq = _paired_mbo(4)
    stream = build_tick_stream(nq, mnq, emit_interval_ns=None)
    phases = set(stream.frame["market_phase"].to_list())
    assert phases.issubset(
        {int(MarketPhase.BALANCE), int(MarketPhase.EXPANSION), int(MarketPhase.NEUTRAL)}
    )


def test_nq_only_tick_stream_does_not_double_events() -> None:
    """nq is mnq → مسار أحادي؛ لا مضاعفة الأحداث ولا trap من الذات."""
    nq, _ = _paired_mbo(6)
    dual = build_tick_stream(nq, nq.clone(), emit_interval_ns=None)  # كائنان مختلفان → مسار ثنائي
    single = build_tick_stream(nq, nq, emit_interval_ns=None)  # نفس الكائن → nq_only
    assert single.height == nq.height
    assert dual.height == 2 * nq.height
    assert "nq_best_bid_norm" in single.frame.columns
    assert "mnq_best_bid_norm" in single.frame.columns
    # دفتر MNQ فارغ في المسار الأحادي
    assert single.frame["mnq_best_bid_norm"].abs().max() == 0.0
    assert set(single.frame["mask_path"].to_list()).issubset({int(MaskPath.STANDALONE)})
