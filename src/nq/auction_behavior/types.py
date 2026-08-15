"""عقود مخرجات فهم سلوك المزاد (وصف احتمالي — بلا توصية صفقة)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BehaviorProbabilities:
    """استنتاجات احتمالية لحالة المزاد مع ثقة.

    الأهداف الأساسية (expansion / rejection / repriced / residual) إن وُجدت
    تشكّل توزيعًا مشتركًا (مجموعها 1). الحقول القديمة الثنائية تبقى مستقلة
    أو من base-rate كخط أساس BSS وليست competing-risk.
    """

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
    p_expansion_accepting: float = 0.0
    p_rejection_return_to_asia: float = 0.0
    p_repriced_balance: float = 0.0
    p_residual: float = 0.0
    probability_source: str = "train_only_walk_forward_base_rates"
    probabilities_are_joint_distribution: bool = False
    n_oof_rows: int = 0


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
