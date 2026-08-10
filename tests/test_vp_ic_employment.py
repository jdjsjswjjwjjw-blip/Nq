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
    n = 160
    rng = np.random.default_rng(7)
    # إشارة ترتبط عكسيًا بالعائد الأمامي (mean-revert / inverted label)
    noise = rng.normal(scale=0.05, size=n)
    inv = np.cumsum(rng.normal(size=n))
    # أسعار تتحرك عكس الإشارة → Spearman(signal, fwd) سالب وقوي
    prices = 100.0 + np.cumsum(-0.15 * np.diff(inv, prepend=inv[0])) + noise
    weak = rng.normal(size=n)
    features = pl.DataFrame(
        {
            "availability_ts": np.arange(n, dtype=np.int64) * 1_000_000_000,
            "nq_close": prices,
            "weak_noise": weak,
            "strong_inverse": inv,
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
    # اختيار بـ |IC| يعني train_ic غالبًا سالب؛ التوظيف يقلب الإشارة
    assert float(folds["employed_sign"].mean()) < 0.0
    assert float(folds["train_ic"].mean()) < 0.0
    # IC الموظَّف خارج العينة يجب أن يكون موجبًا (بعد المحاذاة)
    assert oos_ic > 0.0
    assert float(folds["test_ic"].mean()) > 0.0
    assert oos_n > 0
