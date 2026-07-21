"""اختبارات تركيز فرضيات Volume Profile / Auction."""

from __future__ import annotations

from nq.core.determinism import make_generator
from nq.strategies.vp_auction import run_vp_auction_research
from tests.mbo_factory import make_stream


def _nq_stream(n: int = 400, *, seed: int = 11):
    rng = make_generator(seed)
    events = []
    ts = []
    seq = []
    price = 100
    for i in range(n):
        side = "B" if rng.random() > 0.45 else "A"
        price = max(90, min(110, price + int(rng.integers(-2, 3))))
        events.append(("T", side, price, int(rng.integers(1, 5)), 0))
        ts.append(i * 100)
        seq.append(i + 1)
    return make_stream(events, event_ts=ts, sequence=seq)


def test_run_vp_auction_research_uses_unified_features() -> None:
    nq = _nq_stream(500, seed=12)
    result = run_vp_auction_research(
        nq,
        max_rows=None,
        n_permutations=100,
        rng=make_generator(5),
    )
    assert "vp_balance" in result.features.columns
    assert "vp_imbalance" in result.features.columns
    assert "vp_balance" in result.signal_columns
    assert "fail_fvg" not in result.signal_columns


def test_run_vp_auction_research_produces_report() -> None:
    nq = _nq_stream(500, seed=13)
    result = run_vp_auction_research(
        nq,
        n_permutations=100,
        rng=make_generator(6),
    )
    md = result.unified.to_markdown()
    assert "قناة 1 — SSL" in md
    assert result.features.height > 0
