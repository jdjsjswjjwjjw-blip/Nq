"""توظيف IC لمسار VP: محاذاة الإشارة بعد abs(IC) واستبعاد ميزات المستوى."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.strategies.fvg_hypothesis import walk_forward_select_hypotheses
from nq.strategies.vp_auction import (
    _VP_AUCTION_FOCUS,
    _VP_LEVEL_DISTANCE_FEATURES,
    _VP_REGIME_STATE_FEATURES,
)


def test_vp_ic_focus_excludes_level_and_regime_features() -> None:
    focus = set(_VP_AUCTION_FOCUS)
    assert focus.isdisjoint(_VP_LEVEL_DISTANCE_FEATURES)
    assert focus.isdisjoint(_VP_REGIME_STATE_FEATURES)
    assert "vp_rel_upper" not in focus
    assert "vp_look_fail" in focus
    assert "vp_absorb" in focus
    assert "vp_auction_setup" in focus


def test_wf_abs_ic_employs_sign_of_train_ic() -> None:
    """أفضل |IC| سالب على التدريب → OOS يُوظَّف بمقلوب الإشارة فيصير IC موجبًا."""
    n = 180
    rng = np.random.default_rng(7)
    # ابنِ عوائدًا أمامية صريحة ثم ادمجها في الأسعار حتى Spearman(signal, fwd) ≈ −1
    signal = rng.normal(size=n)
    noise = rng.normal(scale=0.02, size=n)
    # r[t] ≈ −signal[t] → IC سالب قوي
    fwd = -signal + noise
    prices = np.empty(n, dtype=np.float64)
    prices[0] = 100.0
    for t in range(n - 1):
        prices[t + 1] = prices[t] * (1.0 + fwd[t])
    weak = rng.normal(scale=0.01, size=n)
    features = pl.DataFrame(
        {
            "availability_ts": np.arange(n, dtype=np.int64) * 1_000_000_000,
            "nq_close": prices,
            "weak_noise": weak,
            "strong_inverse": signal,
        }
    )
    folds, oos_ic, _p, oos_n, best = walk_forward_select_hypotheses(
        features,
        ["weak_noise", "strong_inverse"],
        price_col="nq_close",
        horizon=1,
        n_splits=3,
        n_permutations=40,
        selection_aware_null=False,
        rng=rng,
    )
    assert best == "strong_inverse"
    assert folds.height >= 1
    assert "employed_sign" in folds.columns
    # اختيار بـ |IC| يعني train_ic سالب؛ التوظيف يقلب الإشارة
    assert float(folds["employed_sign"].mean()) < 0.0
    assert float(folds["train_ic"].mean()) < 0.0
    # IC الموظَّف خارج العينة يجب أن يكون موجبًا (بعد المحاذاة)
    assert oos_ic > 0.0
    assert float(folds["test_ic"].mean()) > 0.0
    assert oos_n > 0
