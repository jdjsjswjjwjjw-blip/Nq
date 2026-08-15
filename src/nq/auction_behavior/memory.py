"""ذاكرة سوقية أغنى: lags + نوافذ rolling سببية داخل المجموعة."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

_DEFAULT_LAGS = (1, 2, 3, 5)
_DEFAULT_ROLL = (3, 8)


def attach_causal_memory(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    lags: tuple[int, ...] = _DEFAULT_LAGS,
    group_col: str | None = None,
) -> pl.DataFrame:
    """يضيف ``shift(k)`` سببيًا، مع إعادة ضبط اختيارية عند حدود الجلسة/المجموعة."""
    if frame.height == 0:
        return frame
    work = frame.sort(AVAILABILITY_TS)
    exprs: list[pl.Expr] = []
    for col in columns:
        if col not in work.columns:
            continue
        base = pl.col(col).cast(pl.Float64)
        for lag in lags:
            if lag < 1:
                raise ValueError(f"lag must be >= 1, got {lag}")
            shifted = base.shift(lag)
            if group_col is not None:
                if group_col not in work.columns:
                    raise ValueError(f"group_col is missing: {group_col}")
                shifted = shifted.over(group_col)
            exprs.append(shifted.alias(f"{col}__lag{lag}"))
    if not exprs:
        return work
    return work.with_columns(exprs)


def attach_market_memory(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    lags: tuple[int, ...] = _DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = _DEFAULT_ROLL,
    group_col: str | None = None,
    event_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """ذاكرة كاملة: lags + متوسطات/مجاميع rolling ماضية + عدّادات أحداث.

    كل نافذة ``rolling_*`` تستخدم ``min_samples=1`` و``shift`` ضمني عبر أن
    المتوسط يُحسب على الصفوف حتى الحالي؛ لتجنب إدخال نفس صف الإشارة في
    بعض الاستخدامات نُزيح النتيجة بـ 1 بعد الـrolling (ماضي صارم).
    """
    work = attach_causal_memory(frame, columns=columns, lags=lags, group_col=group_col)
    if work.height == 0:
        return work
    work = work.sort(AVAILABILITY_TS)
    exprs: list[pl.Expr] = []
    for col in columns:
        if col not in work.columns:
            continue
        base = pl.col(col).cast(pl.Float64)
        for win in roll_windows:
            if win < 1:
                raise ValueError(f"roll window must be >= 1, got {win}")
            mean_expr = base.rolling_mean(window_size=win, min_samples=1)
            sum_expr = base.rolling_sum(window_size=win, min_samples=1)
            if group_col is not None:
                if group_col not in work.columns:
                    raise ValueError(f"group_col is missing: {group_col}")
                mean_expr = mean_expr.over(group_col)
                sum_expr = sum_expr.over(group_col)
            # ماضي صارم: لا تدخل قيمة الصف الحالي في ذاكرة القرار.
            exprs.append(mean_expr.shift(1).alias(f"{col}__rmean{win}"))
            exprs.append(sum_expr.shift(1).alias(f"{col}__rsum{win}"))
    events = event_columns or ()
    for col in events:
        if col not in work.columns:
            continue
        pulse = (pl.col(col).cast(pl.Float64).abs() > 0.0).cast(pl.Float64)
        for win in roll_windows:
            cnt = pulse.rolling_sum(window_size=win, min_samples=1)
            if group_col is not None:
                cnt = cnt.over(group_col)
            exprs.append(cnt.shift(1).alias(f"{col}__ecount{win}"))
    if not exprs:
        return work
    return work.with_columns(exprs)


def memory_feature_matrix(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
) -> np.ndarray:
    """مصفوفة رقمية للأعمدة المطلوبة (NaN→0) بترتيب الصفوف الحالي."""
    present = [c for c in columns if c in frame.columns]
    if not present:
        return np.zeros((frame.height, 0), dtype=np.float64)
    return frame.select(present).fill_null(0.0).to_numpy().astype(np.float64)


def list_memory_columns(frame: pl.DataFrame) -> list[str]:
    """أسماء أعمدة الذاكرة المولَّدة."""
    return [
        c
        for c in frame.columns
        if "__lag" in c or "__rmean" in c or "__rsum" in c or "__ecount" in c
    ]


__all__ = [
    "attach_causal_memory",
    "attach_market_memory",
    "list_memory_columns",
    "memory_feature_matrix",
]
