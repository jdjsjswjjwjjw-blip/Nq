"""جودة إشارة سلوكية ودرجة ثقة (بلا تحجيم صفقة)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

_ACTIVE_FLAG = 0.5


def attach_signal_quality(frame: pl.DataFrame) -> pl.DataFrame:
    """يقيس قوة الكسر / جودة الريتست / توافق الأوردرفلو مع البروفايل → ``signal_quality``.

    كل المكوّنات من أعمدة سببية جاهزة؛ الناتج ∈ [0, 1].
    """
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
    # توافق أوردرفلو مبكر مع البروفايل + سيولة حقيقية منخفضة التضليل.
    flow_align = (
        (_f("vp_early_imbalance").abs() > 0.0).cast(pl.Float64) * 0.15
        + (_f("real_liquidity_ratio", 0.5).clip(0.0, 1.0) * 0.10)
        + ((1.0 - _f("deceptive_score", 0.0).clip(0.0, 1.0)) * 0.10)
    )
    quality = (break_strength + retest_q + flow_align).clip(0.0, 1.0)
    return work.with_columns(quality.alias("signal_quality"))


def mean_confidence(frame: pl.DataFrame) -> float:
    """متوسط ``signal_quality`` على الإطار."""
    if frame.height == 0 or "signal_quality" not in frame.columns:
        return 0.0
    vals = frame["signal_quality"].to_numpy().astype(np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))
