"""ذاكرة سوقية سببية: نوافذ ماضية فقط (بدون مستقبل)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

_DEFAULT_LAGS = (1, 2, 3, 5)


def attach_causal_memory(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...] | list[str],
    lags: tuple[int, ...] = _DEFAULT_LAGS,
) -> pl.DataFrame:
    """يضيف أعمدة ``col__lag{k}`` بـ ``shift(k)`` سببي داخل الإطار المرتّب زمنياً."""
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
            exprs.append(base.shift(lag).alias(f"{col}__lag{lag}"))
    if not exprs:
        return work
    return work.with_columns(exprs)


def memory_feature_matrix(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...] | list[str],
) -> np.ndarray:
    """مصفوفة رقمية للأعمدة المطلوبة (NaN→0) بترتيب الصفوف الحالي."""
    present = [c for c in columns if c in frame.columns]
    if not present:
        return np.zeros((frame.height, 0), dtype=np.float64)
    return frame.select(present).fill_null(0.0).to_numpy().astype(np.float64)
