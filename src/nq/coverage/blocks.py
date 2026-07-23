"""تعريف كتل ميزات المحاكيات (Simulator Feature Blocks).

تُستخدم في CRS و LORI و CER لربط العمى البنيوي بمحاكي محدد.
"""

from __future__ import annotations

from collections.abc import Sequence

#: كتل ميزات المحاكيات الافتراضية (من إطار البحث الموحّد).
DEFAULT_FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "order_flow": ("nq_delta", "mnq_delta"),
    "cross_market": (
        "lead_lag",
        "divergence",
        "trap_setup",
        "confirmation_failure",
        "nq_leads_corr",
        "mnq_leads_corr",
    ),
    "failed_fvg": (
        "fail_fvg",
        "effort_range_ratio",
        "effort_volume_ratio",
    ),
    "failed_breakout": (
        "fail_breakout",
        "fb_effort_range_ratio",
        "fb_effort_volume_ratio",
        "fb_effort_result_ratio",
        "fb_bar_volume",
        "fb_cum_volume",
        "fb_delta",
        "fb_cum_delta",
        "fb_vol_imbalance",
        "fb_absorption",
        "fb_entry_ref",
        "fb_break_level",
    ),
    "volume_profile_auction": (
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_close_in_value",
        "vp_in_value_frac",
        "vp_pullback_defense",
        "vp_poc_migration",
        "vp_flip_to_imbalance",
    ),
    "streaming_microstructure": (
        "phase_balance",
        "phase_expansion",
        "in_value_area",
        "near_vah",
        "near_val",
        "poc_dist_norm",
        "nq_spread_norm",
        "stream_vah_bid_liq",
        "stream_val_ask_liq",
    ),
    "price": ("nq_return", "mnq_return"),
}


def resolve_block_columns(
    frame_columns: Sequence[str],
    blocks: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[str]]:
    """يُرجع الأعمدة المتاحة فعليًا لكل كتلة."""
    mapping = blocks if blocks is not None else DEFAULT_FEATURE_BLOCKS
    available = set(frame_columns)
    resolved: dict[str, list[str]] = {}
    for name, cols in mapping.items():
        present = [c for c in cols if c in available]
        if present:
            resolved[name] = present
    return resolved
