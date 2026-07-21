"""اختبارات مسار بحث Failed FVG عبر الخط الموحّد."""

from __future__ import annotations

from nq.core.determinism import make_generator
from nq.strategies.fail_fvg import run_fail_fvg_research
from tests.test_coverage import _paired_streams


def test_run_fail_fvg_research_uses_unified_features() -> None:
    nq, mnq = _paired_streams(2500, seed=80)
    result = run_fail_fvg_research(
        nq,
        mnq,
        n_permutations=100,
        rng=make_generator(0),
    )
    assert "fail_fvg" in result.features.columns
    assert "trap_setup" in result.features.columns or "lead_lag" in result.features.columns
    assert "fail_fvg" in result.signal_columns
    assert result.unified is not None


def test_run_fail_fvg_research_produces_report() -> None:
    nq, mnq = _paired_streams(2000, seed=81)
    result = run_fail_fvg_research(
        nq,
        mnq,
        n_permutations=200,
        rng=make_generator(1),
    )
    assert result.features.height > 0
    assert result.report is not None
