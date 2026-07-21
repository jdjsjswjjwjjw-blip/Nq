"""اختبارات محاكي FVG / Failed FVG السببي."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.fvg import (
    NS_1H,
    NS_30M,
    build_ohlcv_bars,
    detect_h1_fvgs,
    failed_fvg_features,
)
from nq.validation.leakage import assert_availability_not_before_event
from tests.mbo_factory import make_stream


def _trades_at_prices(
    prices: list[int],
    *,
    start_ts: int = 0,
    step_ns: int = NS_30M // 4,
) -> pl.DataFrame:
    events = [("T", "B", p, 2, 0) for p in prices]
    ts = [start_ts + i * step_ns for i in range(len(prices))]
    return make_stream(events, event_ts=ts, sequence=list(range(1, len(prices) + 1)))


def test_build_ohlcv_bars_availability_at_bucket_end() -> None:
    base = int(100 / PRICE_SCALE)
    prices = [base, base + int(1 / PRICE_SCALE), base - int(0.5 / PRICE_SCALE), base]
    frame = _trades_at_prices(prices, step_ns=NS_30M // 4)
    bars = build_ohlcv_bars(frame, interval_ns=NS_30M)
    assert bars.height >= 1
    assert (bars["availability_ts"] == bars["bucket_end"]).all()
    assert (bars["availability_ts"] > bars["bucket_start"]).all()


def test_detect_h1_fvgs_available_after_formation() -> None:
    rows = []
    px = 100.0
    for i in range(6):
        start = i * NS_1H
        if i == 0:
            open_, high, low, close = 100.0, 100.5, 99.5, 100.2
        elif i == 1:
            open_, high, low, close = 100.2, 100.4, 100.0, 100.3
        elif i == 2:
            open_, high, low, close = 101.5, 102.0, 101.4, 101.8
        else:
            open_ = px
            high, low, close = px + 0.2, px - 0.2, px
        rows.append(
            {
                "bucket_start": start,
                "bucket_end": start + NS_1H,
                "availability_ts": start + NS_1H,
                "o": open_,
                "h": high,
                "l": low,
                "c": close,
                "volume": 10.0,
                "range": high - low,
            }
        )
        px = close
    h1 = pl.DataFrame(rows)
    fvgs = detect_h1_fvgs(h1)
    assert fvgs.height >= 1
    bull = fvgs.filter(pl.col("fvg_type") == "Bull")
    assert bull.height >= 1
    assert (bull["availability_ts"] > bull["formed_at"]).all()
    assert_availability_not_before_event(
        bull["formed_at"].to_numpy(),
        bull[AVAILABILITY_TS].to_numpy(),
    )


def test_failed_fvg_past_stable_when_future_perturbed() -> None:
    rng = np.random.default_rng(0)
    n = 160
    base = int(20_000 / PRICE_SCALE)
    prices = [base + int(rng.integers(-5, 6) * (1 / PRICE_SCALE)) for _ in range(n)]
    frame = _trades_at_prices(prices, step_ns=NS_30M // 3)
    baseline = failed_fvg_features(frame)

    cut_ts = int(baseline[AVAILABILITY_TS][baseline.height // 2])
    past_signal = baseline.filter(pl.col(AVAILABILITY_TS) <= cut_ts)["fail_fvg"].to_list()

    event_ts = frame["event_ts"].to_list()
    new_prices = frame["price"].to_list()
    for i, ts in enumerate(event_ts):
        if ts > cut_ts:
            new_prices[i] = base + int(rng.integers(50, 80) * (1 / PRICE_SCALE))
    perturbed = frame.with_columns(pl.Series("price", new_prices))
    after = failed_fvg_features(perturbed)
    after_past = after.filter(pl.col(AVAILABILITY_TS) <= cut_ts)["fail_fvg"].to_list()
    assert after_past == past_signal


def test_failed_fvg_signal_column_in_range() -> None:
    base = int(20_000 / PRICE_SCALE)
    prices = [base + (i % 7) * int(0.25 / PRICE_SCALE) for i in range(120)]
    frame = _trades_at_prices(prices, step_ns=NS_30M // 2)
    feats = failed_fvg_features(frame)
    assert "fail_fvg" in feats.columns
    assert set(feats["fail_fvg"].unique().to_list()).issubset({-1.0, 0.0, 1.0})
