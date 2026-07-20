"""اختبارات الإخفاء الهيكلي (البُعد 6)."""

from __future__ import annotations

import numpy as np

from nq.models.masking_structural import structural_mask_sample
from nq.models.tick_stream import TICK_FEATURE_NAMES, MarketPhase, MaskPath


def _sample_window() -> np.ndarray:
    n_feat = len(TICK_FEATURE_NAMES)
    window = 3
    x = np.random.default_rng(0).normal(0, 1, (window, n_feat))
    x[:, TICK_FEATURE_NAMES.index("near_vah")] = [0, 1, 0]
    x[:, TICK_FEATURE_NAMES.index("nq_bid_size_log")] = 2.0
    return x


def test_standalone_balance_masks_liquidity() -> None:
    x = _sample_window()
    masked = structural_mask_sample(
        x,
        mask_path=int(MaskPath.STANDALONE),
        market_phase=int(MarketPhase.BALANCE),
    )
    assert bool(np.any(masked.mask))
    liq_idx = TICK_FEATURE_NAMES.index("nq_bid_size_log")
    assert bool(np.any(masked.mask[:, liq_idx]))


def test_cross_trap_masks_nq_not_mnq_flow() -> None:
    x = _sample_window()
    masked = structural_mask_sample(
        x,
        mask_path=int(MaskPath.CROSS_TRAP),
        market_phase=int(MarketPhase.NEUTRAL),
    )
    liq_idx = TICK_FEATURE_NAMES.index("nq_bid_size_log")
    flow_idx = TICK_FEATURE_NAMES.index("mnq_signed_vol")
    assert bool(np.any(masked.mask[:, liq_idx]))
    assert not bool(np.any(masked.mask[:, flow_idx]))
