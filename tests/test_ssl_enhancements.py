"""اختبارات مولّد تعزيزات SSL السببية فوق Failed Breakout."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.strategies.breakout_hypothesis import (
    core_breakout_grid,
    search_fail_breakout_hypotheses,
)
from nq.strategies.ssl_enhancements import generate_ssl_enhancement_candidates
from tests.test_coverage import _paired_streams


def test_generate_ssl_enhancements_creates_columns() -> None:
    n = 120
    features = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "fail_breakout__base": [1.0 if i % 7 == 0 else (-1.0 if i % 11 == 0 else 0.0) for i in range(n)],
            "trap_setup": [1.0 if i % 5 == 0 else 0.0 for i in range(n)],
            "phase_balance": [1.0 if i % 3 == 0 else 0.0 for i in range(n)],
        }
    )
    rng = np.random.default_rng(0)
    embeddings = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "z0": rng.normal(size=n),
            "z1": rng.normal(size=n),
        }
    )
    out, cols, specs = generate_ssl_enhancement_candidates(
        features,
        embeddings,
        ["fail_breakout__base"],
    )
    assert len(cols) == len(specs) > 0
    assert all(c in out.columns for c in cols)
    assert any("__enh__ssl_abs_q" in c for c in cols)
    assert any("__enh__ssl_sign_" in c for c in cols)
    assert any("__enh__ctx_" in c for c in cols)
    # التعزيز لا يخلق إشارة من الصفر بدون أساس
    base = out["fail_breakout__base"].to_numpy()
    for c in cols:
        enh = out[c].to_numpy()
        assert np.all((enh == 0.0) | (np.sign(enh) == np.sign(base)) | (base == 0.0))


def test_ssl_quantile_gate_uses_past_only() -> None:
    """تغيير z المستقبلي لا يغيّر بوابة الصفوف الماضية."""
    n = 80
    features = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "sig": [1.0] * n,
        }
    )
    z = np.linspace(0.1, 2.0, n)
    emb1 = pl.DataFrame({AVAILABILITY_TS: list(range(n)), "z0": z, "z1": z})
    out1, cols, _ = generate_ssl_enhancement_candidates(
        features, emb1, ["sig"], include_context=False, include_sign_agree=False, quantiles=(0.7,)
    )
    z2 = z.copy()
    z2[60:] = 100.0
    emb2 = pl.DataFrame({AVAILABILITY_TS: list(range(n)), "z0": z2, "z1": z2})
    out2, _, _ = generate_ssl_enhancement_candidates(
        features, emb2, ["sig"], include_context=False, include_sign_agree=False, quantiles=(0.7,)
    )
    col = cols[0]
    past1 = out1.filter(pl.col(AVAILABILITY_TS) < 60).select(col)
    past2 = out2.filter(pl.col(AVAILABILITY_TS) < 60).select(col)
    assert past1.equals(past2)


def test_core_breakout_grid_small() -> None:
    assert 6 <= len(core_breakout_grid()) <= 12
    modes = {s.vol_mode for s in core_breakout_grid()}
    assert "bar" in modes and "cum" in modes
    assert "delta" in modes and "effort_result" in modes


def test_search_with_enhancements_runs() -> None:
    nq, mnq = _paired_streams(2200, seed=77)
    result = search_fail_breakout_hypotheses(
        nq,
        mnq,
        specs=core_breakout_grid()[:2],
        interval_ns=10_000,
        use_ssl_gate=True,
        enhance_with_ssl=True,
        n_splits=2,
        n_permutations=40,
        rng=make_generator(3),
        quiet=True,
    )
    assert result.report is not None
    assert len(result.candidate_columns) >= 2
    # إما تعزيزات أو على الأقل المرشّحون الأساس/البوابة
    assert result.features.height > 0
