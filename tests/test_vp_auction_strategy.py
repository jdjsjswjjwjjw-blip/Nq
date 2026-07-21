"""اختبارات تركيز فرضيات Volume Profile / Auction."""

from __future__ import annotations

from nq.core.determinism import make_generator
from nq.strategies.vp_auction import run_vp_auction_research
from tests.test_coverage import _paired_streams


def test_run_vp_auction_research_uses_unified_features() -> None:
    nq, _mnq = _paired_streams(2500, seed=82)
    result = run_vp_auction_research(
        nq,
        n_permutations=100,
        rng=make_generator(5),
        quiet=True,
    )
    assert "vp_balance" in result.features.columns
    assert "vp_imbalance" in result.features.columns
    assert "vp_balance" in result.signal_columns
    assert "fail_fvg" not in result.signal_columns
    assert result.unified is not None


def test_run_vp_auction_research_produces_report() -> None:
    nq, _mnq = _paired_streams(2000, seed=83)
    result = run_vp_auction_research(
        nq,
        n_permutations=200,
        rng=make_generator(6),
        quiet=True,
    )
    md = result.unified.to_markdown()
    assert "قناة 1 — SSL" in md
    assert result.features.height > 0
