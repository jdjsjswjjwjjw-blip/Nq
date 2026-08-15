"""جودة إشارة سلوكية — evidence / feature وليست احتمالًا معايرًا.

``signal_quality ∈ [0,1]`` يقيس قوة الدليل البنيوي/التدفقي المتاح الآن.
هذا **ليس** ``P(outcome | state)``؛ الاحتمال الشرطي المعاير يأتي من
``ConditionalModel`` ثم يُختبر بـ ECE/Brier على OOS فقط.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike

_ACTIVE_FLAG = 0.5


def attach_signal_quality(
    frame: pl.DataFrame,
    *,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يقيس قوة الكسر / جودة الريتست / توافق الأوردرفلو → ``signal_quality``.

    الناتج evidence ∈ [0, 1] — لا تُفسَّر كاحتمال سيناريو معاير.
    """
    if progress is not None:
        progress.op(f"attach_signal_quality bars={frame.height:,}")
    if frame.height == 0:
        return frame.with_columns(pl.lit(0.0).alias("signal_quality"))

    work = frame.sort(AVAILABILITY_TS)

    def _f(name: str, default: float = 0.0) -> pl.Expr:
        if name in work.columns:
            return pl.col(name).cast(pl.Float64).fill_null(default)
        return pl.lit(default)

    # قوة الكسر: نبضة كسر + اختلال/توسّع.
    break_strength = (
        (_f("vp_fsm_break").abs() > 0.0).cast(pl.Float64) * 0.35
        + (_f("vp_imbalance") > _ACTIVE_FLAG).cast(pl.Float64) * 0.15
        + (_f("vp_expansion") > _ACTIVE_FLAG).cast(pl.Float64) * 0.10
    )
    # جودة ريتست: ريتست مع امتصاص أو دفاع سحب.
    retest_q = (
        (_f("vp_fsm_retest").abs() > 0.0).cast(pl.Float64) * 0.20
        + (_f("vp_absorb").abs() > 0.0).cast(pl.Float64) * 0.10
        + (_f("vp_pullback_defense") > _ACTIVE_FLAG).cast(pl.Float64) * 0.05
    )
    # الثقة في السيولة تعدّل الدليل البنيوي ولا تُنشئ دليلاً من لا شيء.
    early_flow = (_f("vp_early_imbalance").abs() > 0.0).cast(pl.Float64) * 0.15
    rejection = (_f("vp_look_fail").abs() > 0.0).cast(pl.Float64) * 0.15
    projection_story = (
        _f("proj_expansion_testing") * 0.05
        + _f("proj_expansion_accepting") * 0.12
        + _f("proj_value_transferred") * 0.15
        + _f("proj_rejection_to_asia") * 0.10
    )
    path_confirm = _f("path_depth_confirm") * 0.22 + _f("path_change_progress") * 0.18
    evidence = (
        break_strength + retest_q + early_flow + rejection + projection_story + path_confirm
    ).clip(0.0, 1.0)
    liquidity_reliability = (
        _f("real_liquidity_ratio", 0.5).clip(0.0, 1.0) * 0.5
        + (1.0 - _f("deceptive_score", 0.5).clip(0.0, 1.0)) * 0.5
    )
    # دمج اختياري لأدلة الموثوقية الإحصائية إن وُجدت (ليست حكم حذف).
    if "rel_credibility" in work.columns:
        liquidity_reliability = 0.7 * liquidity_reliability + 0.3 * _f("rel_credibility", 0.5).clip(
            0.0, 1.0
        )
    quality = (evidence * (0.5 + 0.5 * liquidity_reliability)).clip(0.0, 1.0)
    return work.with_columns(
        quality.alias("signal_quality"),
        evidence.alias("signal_evidence"),
        pl.lit(False).alias("signal_quality_is_calibrated_probability"),
    )


def mean_confidence(frame: pl.DataFrame) -> float:
    """متوسط ``signal_quality`` على الإطار (evidence متوسط — ليس ECE)."""
    if frame.height == 0 or "signal_quality" not in frame.columns:
        return 0.0
    vals = frame["signal_quality"].to_numpy().astype(np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))
