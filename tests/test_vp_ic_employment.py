"""توظيف IC لمسار VP: محاذاة الإشارة بعد abs(IC) واستبعاد ميزات المستوى."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.deceptive_liquidity import DECEPTIVE_FEATURE_COLUMNS
from nq.simulation.market_truth import MarketTruthConfig, build_market_truth_frame
from nq.strategies.fvg_hypothesis import walk_forward_select_hypotheses
from nq.strategies.vp_auction import (
    _VP_AUCTION_FOCUS,
    _VP_LEVEL_DISTANCE_FEATURES,
    _VP_REGIME_STATE_FEATURES,
    _attach_oos_employed_signal,
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


def test_vp_level_features_are_not_directional_ic_candidates() -> None:
    """vp_rel_upper مسافة لـ VAH — لا يجوز أن تدخل بركة IC الاتجاهية."""
    assert "vp_rel_upper" in _VP_LEVEL_DISTANCE_FEATURES
    assert "vp_balance" in _VP_REGIME_STATE_FEATURES
    for col in ("vp_rel_upper", "vp_rel_mid", "vp_rel_lower", "vp_balance", "vp_imbalance"):
        assert col not in _VP_AUCTION_FOCUS


def test_oos_employed_signal_is_fold_specific_and_train_rows_are_zero() -> None:
    features = pl.DataFrame(
        {
            AVAILABILITY_TS: [10, 20, 30, 40, 50, 60],
            "sig_a": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "sig_b": [5.0, 4.0, 3.0, 2.0, 1.0, -1.0],
        }
    )
    folds = pl.DataFrame(
        {
            "fold": [0, 1],
            "selected": ["sig_a", "sig_b"],
            "train_ic": [0.4, -0.3],
            "test_ic": [0.2, 0.1],
            "employed_sign": [1.0, -1.0],
            "train_end_ts": [20, 40],
            "test_start_ts": [30, 50],
            "test_end_ts": [40, 60],
        }
    )

    out = _attach_oos_employed_signal(features, folds)

    assert out["vp_ic_employed"].to_list() == [0.0, 0.0, 2.0, 3.0, -1.0, 1.0]
    assert out["vp_ic_fold"].to_list() == [-1, -1, 0, 0, 1, 1]
    assert out["vp_ic_selected"].to_list() == ["", "", "sig_a", "sig_a", "sig_b", "sig_b"]


def test_market_truth_uses_exact_external_ic_thesis_not_close_vs_poc() -> None:
    def px(value: float) -> int:
        return round(value / PRICE_SCALE)

    times = [1, 2, 3, 4]
    states = pl.DataFrame(
        {
            AVAILABILITY_TS: times,
            BUCKET_START: times,
            BUCKET_END: times,
            # close فوق POC دائماً: القاعدة القديمة كانت ستنتج +1.
            "close": [px(101.0), px(101.0), px(100.0), px(100.0)],
            "poc": [px(99.0)] * 4,
            "vah": [px(102.0)] * 4,
            "val": [px(98.0)] * 4,
            "is_balanced": [False] * 4,
            "is_expansion": [True] * 4,
            "pullback_defended": [False] * 4,
        }
    )
    deco_values = {name: [0.0] * 4 for name in DECEPTIVE_FEATURE_COLUMNS}
    deco_values["real_liquidity_ratio"] = [1.0] * 4
    deco = pl.DataFrame(
        {
            AVAILABILITY_TS: times,
            BUCKET_START: times,
            BUCKET_END: times,
            **deco_values,
        }
    )
    thesis = pl.DataFrame(
        {
            AVAILABILITY_TS: times,
            # نبضة بيع عند t=2، ثم هبوط خلال الهولد إلى t=3.
            "vp_ic_employed": [0.0, -1.0, 0.0, 0.0],
        }
    )

    out = build_market_truth_frame(
        pl.DataFrame(),
        interval_ns=1,
        truth=MarketTruthConfig(
            hold_buckets=2,
            min_real_liquidity=0.0,
            max_deceptive_score=1.0,
            min_move_ticks=1.0,
        ),
        auction=states,
        deceptive_frame=deco,
        thesis_frame=thesis,
    )

    assert float(out["thesis_dir"][2]) == -1.0
    assert float(out["market_true"][2]) == 1.0
    assert float(out["entry_gate"][2]) == 1.0
