"""اختبارات مسار بحث Failed FVG."""

from __future__ import annotations

import numpy as np

from nq.contracts.mbo import PRICE_SCALE
from nq.simulation.fvg import NS_30M
from nq.strategies.fail_fvg import run_fail_fvg_research
from tests.mbo_factory import make_stream


def _synthetic_nq(n: int = 300) -> object:
    rng = np.random.default_rng(1)
    base = int(20_000 / PRICE_SCALE)
    prices = [base + int(rng.integers(-8, 9) * (0.25 / PRICE_SCALE)) for _ in range(n)]
    events = [("T", "B" if i % 2 == 0 else "A", p, 3, 0) for i, p in enumerate(prices)]
    ts = [i * (NS_30M // 4) for i in range(n)]
    return make_stream(events, event_ts=ts, sequence=list(range(1, n + 1)))


def test_run_fail_fvg_research_produces_report() -> None:
    result = run_fail_fvg_research(
        _synthetic_nq(),
        use_ssl_gate=False,
        n_permutations=200,
        rng=np.random.default_rng(0),
    )
    assert result.features.height > 0
    assert "fail_fvg" in result.signal_columns
    assert result.report.title.startswith("Failed FVG")


def test_run_fail_fvg_research_with_ssl_gate() -> None:
    result = run_fail_fvg_research(
        _synthetic_nq(400),
        use_ssl_gate=True,
        n_permutations=100,
        rng=np.random.default_rng(2),
    )
    assert "fail_fvg_ssl" in result.signal_columns
    assert "fail_fvg_ssl" in result.features.columns
