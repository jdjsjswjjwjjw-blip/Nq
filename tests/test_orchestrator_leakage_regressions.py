"""System-level leakage regressions for high-priority audit findings."""

from __future__ import annotations

import polars as pl

from nq.research.orchestrator import PipelineConfig, _build_research_features
from tests.mbo_factory import Event, make_stream


def _market(prices: list[int], *, symbol: str, instrument_id: int) -> pl.DataFrame:
    events: list[Event] = []
    ts: list[int] = []
    seq: list[int] = []
    order_id = 1
    for i, price in enumerate(prices):
        base = i * 1_000
        events.extend(
            [
                ("A", "B", price - 1_000_000, 5, order_id),
                ("A", "A", price + 1_000_000, 5, order_id + 1),
                ("T", "B", price, 1, 0),
            ]
        )
        ts.extend([base, base + 1, base + 2])
        seq.extend([i * 3 + 1, i * 3 + 2, i * 3 + 3])
        order_id += 2
    return make_stream(
        events,
        symbol=symbol,
        instrument_id=instrument_id,
        event_ts=ts,
        sequence=seq,
    )


def test_nq_only_research_features_do_not_create_fake_mnq_evidence() -> None:
    nq = _market([20_000_000_000, 20_001_000_000, 20_002_000_000], symbol="NQ", instrument_id=1)
    cfg = PipelineConfig(
        cross_market_mode="nq_only",
        feature_mode="batch",
        include_failed_fvg=False,
        include_auction_vp=False,
        include_failed_breakout=False,
    )

    features = _build_research_features(nq, nq, cfg)

    forbidden_exact = {
        "lead_lag",
        "trap_setup",
        "divergence",
        "confirmation_failure",
        "nq_leads_corr",
        "mnq_leads_corr",
        "mnq_new_high",
        "mnq_new_low",
    }
    assert not any(col.startswith("mnq_") for col in features.columns)
    assert forbidden_exact.isdisjoint(features.columns)
