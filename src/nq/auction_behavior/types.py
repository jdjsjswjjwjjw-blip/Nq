"""عقود مخرجات فهم سلوك المزاد (وصف احتمالي — بلا توصية صفقة)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BehaviorProbabilities:
    """استنتاجات احتمالية لحالة المزاد مع ثقة."""

    p_balanced: float
    p_imbalanced: float
    p_true_break: float
    p_false_break: float
    p_retest_success: float
    p_retest_fail: float
    p_expansion_continue: float
    p_return_to_value: float
    confidence: float
    n_samples: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BehaviorStateSnapshot:
    """لقطة حالة سوقية واحدة (سببية، عند ``availability_ts``)."""

    availability_ts: int
    liquidity_session: int
    is_balanced: float
    is_expansion: float
    close_in_value: float
    absorb: float
    look_fail: float
    fsm_break: float
    fsm_retest: float
    fsm_expand: float
    early_imbalance: float
    deceptive_score: float
    real_liquidity_ratio: float
    signal_quality: float
    auction_phase: str = ""
    asia_poc: float = 0.0
    asia_vah: float = 0.0
    asia_val: float = 0.0
    composite_poc: float = 0.0
    composite_vah: float = 0.0
    composite_val: float = 0.0
    projection_anchor_complete: float = 0.0
    projection_expansion_active: float = 0.0
    projection_value_transferred: float = 0.0
