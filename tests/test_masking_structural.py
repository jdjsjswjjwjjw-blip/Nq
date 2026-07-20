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
    x[:, TICK_FEATURE_NAMES.index("nq_vah_bid_liq_log")] = 2.0
    x[:, TICK_FEATURE_NAMES.index("nq_vah_ask_liq_log")] = 1.5
    return x


def test_standalone_balance_masks_book_levels_at_vah() -> None:
    x = _sample_window()
    masked = structural_mask_sample(
        x,
        mask_path=int(MaskPath.STANDALONE),
        market_phase=int(MarketPhase.BALANCE),
    )
    assert bool(np.any(masked.mask))
    vah_bid_idx = TICK_FEATURE_NAMES.index("nq_vah_bid_liq_log")
    top_bid_idx = TICK_FEATURE_NAMES.index("nq_bid_size_log")
    assert bool(np.any(masked.mask[:, vah_bid_idx]))
    assert not bool(np.any(masked.mask[:, top_bid_idx]))


def test_expansion_masks_trailing_book_depth() -> None:
    n_feat = len(TICK_FEATURE_NAMES)
    x = np.random.default_rng(1).normal(0, 1, (3, n_feat))
    x[:, TICK_FEATURE_NAMES.index("nq_trail_bid_liq_log")] = 3.0
    masked = structural_mask_sample(
        x,
        mask_path=int(MaskPath.STANDALONE),
        market_phase=int(MarketPhase.EXPANSION),
    )
    trail_idx = TICK_FEATURE_NAMES.index("nq_trail_bid_liq_log")
    assert bool(masked.mask[-1, trail_idx])
    assert not bool(masked.mask[0, trail_idx])


def test_cross_trap_masks_nq_book_boundary_not_mnq_flow() -> None:
    x = _sample_window()
    masked = structural_mask_sample(
        x,
        mask_path=int(MaskPath.CROSS_TRAP),
        market_phase=int(MarketPhase.NEUTRAL),
    )
    vah_idx = TICK_FEATURE_NAMES.index("nq_vah_bid_liq_log")
    flow_idx = TICK_FEATURE_NAMES.index("mnq_signed_vol")
    assert bool(np.any(masked.mask[:, vah_idx]))
    assert not bool(np.any(masked.mask[:, flow_idx]))
