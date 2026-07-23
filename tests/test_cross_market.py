"""اختبارات مُحاكي عبر السوقين (NQ ↔ MNQ)."""

from __future__ import annotations

import polars as pl

from nq.core.session import SESSION_DATE
from nq.simulation.cross_market import cross_market_features
from tests.mbo_factory import Event, make_stream


def _market(prices: list[int], *, symbol: str, instrument_id: int) -> pl.DataFrame:
    events: list[Event] = []
    ts: list[int] = []
    seq: list[int] = []
    oid = 1
    t = 0
    for i, price in enumerate(prices):
        # ثبّت السوق بأفضل طلب/عرض حول السعر ثم صفقة عدوانية عند السعر.
        events.append(("A", "B", price - 1_000_000, 5, oid))
        events.append(("A", "A", price + 1_000_000, 5, oid + 1))
        events.append(("T", "B", price, 3, 0))
        ts.extend([t, t + 1, t + 2])
        seq.extend([3 * i + 1, 3 * i + 2, 3 * i + 3])
        oid += 2
        t += 100  # نافذة جديدة كل خطوة
    return make_stream(
        events, instrument_id=instrument_id, symbol=symbol, event_ts=ts, sequence=seq
    )


def test_alignment_and_returns() -> None:
    nq = _market([100_000_000, 101_000_000, 102_000_000], symbol="NQ", instrument_id=1)
    mnq = _market([100_000_000, 101_000_000, 102_000_000], symbol="MNQ", instrument_id=2)
    feat = cross_market_features(nq, mnq, interval_ns=100, lead_lag_window=2)
    assert feat.height == 3
    assert feat["availability_ts"].to_list() == feat["bucket_end"].to_list()
    assert "lead_lag" in feat.columns
    assert "trap_setup" in feat.columns


def test_divergence_detected() -> None:
    # NQ يصعد، MNQ يهبط -> تباعد.
    nq = _market([100_000_000, 101_000_000, 102_000_000], symbol="NQ", instrument_id=1)
    mnq = _market([102_000_000, 101_000_000, 100_000_000], symbol="MNQ", instrument_id=2)
    feat = cross_market_features(nq, mnq, interval_ns=100, lead_lag_window=2)
    assert feat["divergence"].to_list()[1:] == [True, True]


def test_trader_trap_setup_when_mnq_new_high_unconfirmed() -> None:
    # MNQ يصنع قممًا جديدة بدلتا شراء عدوانية، بينما NQ ثابت (لا يؤكّد) -> مصيدة صعودية.
    nq = _market([100_000_000, 100_000_000, 100_000_000], symbol="NQ", instrument_id=1)
    mnq = _market([100_000_000, 101_000_000, 102_000_000], symbol="MNQ", instrument_id=2)
    feat = cross_market_features(nq, mnq, interval_ns=100, lead_lag_window=2, min_trap_delta=1)
    traps = feat["trap_setup"].to_list()
    assert 1 in traps  # على الأقل نافذة بها إعداد مصيدة صعودية
    conf = feat.filter(feat["trap_setup"] == 1)["mnq_new_high"].to_list()
    assert all(conf)


def test_session_partitioned_new_high_math() -> None:
    """إثبات رياضي: cum_max.over(session) يُصفَّر يوميًا؛ العالمي لا يفعل."""
    closes = [100.0, 101.0, 102.0, 100.0, 100.5, 101.0]
    dates = ["2024-06-03", "2024-06-03", "2024-06-03", "2024-06-04", "2024-06-04", "2024-06-04"]
    df = pl.DataFrame({SESSION_DATE: dates, "nq_close": closes})
    session_prev = pl.col("nq_close").cum_max().over(SESSION_DATE).shift(1).over(SESSION_DATE)
    global_prev = pl.col("nq_close").cum_max().shift(1)
    out = df.with_columns(
        (pl.col("nq_close") > session_prev).fill_null(value=False).alias("session_high"),
        (pl.col("nq_close") > global_prev).fill_null(value=False).alias("global_high"),
    )
    day1 = out.filter(pl.col(SESSION_DATE) == "2024-06-04")
    assert day1["session_high"].to_list() == [False, True, True]
    assert day1["global_high"].to_list() == [False, False, False]


def test_cross_market_features_include_session_date() -> None:
    nq = _market([100_000_000, 101_000_000, 102_000_000], symbol="NQ", instrument_id=1)
    mnq = _market([100_000_000, 101_000_000, 102_000_000], symbol="MNQ", instrument_id=2)
    feat = cross_market_features(nq, mnq, interval_ns=100, lead_lag_window=2)
    assert SESSION_DATE in feat.columns
    assert feat[SESSION_DATE].null_count() == 0
