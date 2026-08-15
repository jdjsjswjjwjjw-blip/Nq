"""هندسة خصائص بنيوية حول ``decision_*`` / HVN (سببية فقط).

كل المسافات تُقاس من إغلاق البرميل الحالي إلى حدود القرار المتأخرة.
لا تُستخدم ``vah/poc/val`` الحالية للحكم — فقط وصف اختياري منفصل.
"""

from __future__ import annotations

import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike

STRUCTURE_FEATURE_COLUMNS = (
    "struct_dist_vah_ticks",
    "struct_dist_val_ticks",
    "struct_dist_poc_ticks",
    "struct_va_width_ticks",
    "struct_close_in_value",
    "struct_above_vah",
    "struct_below_val",
    "struct_near_vah",
    "struct_near_val",
    "struct_near_poc",
    "struct_dist_asia_vah_ticks",
    "struct_dist_asia_val_ticks",
    "struct_dist_asia_poc_ticks",
    "struct_dist_asia_hvn_ticks",
    "struct_dist_composite_hvn_ticks",
    "struct_break_pressure",
    "struct_retest_pressure",
)

_NEAR_FRAC = 0.20
_EPS = 1e-12


def attach_structure_features(
    frame: pl.DataFrame,
    *,
    tick_size: float | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يضيف مسافات نسبية/تيكات عن حدود القرار ومرساة آسيا إن وُجدت."""
    if progress is not None:
        progress.op(f"attach_structure_features bars={frame.height:,}")
    if frame.height == 0:
        return frame.with_columns(pl.lit(0.0).alias(c) for c in STRUCTURE_FEATURE_COLUMNS)
    if AVAILABILITY_TS not in frame.columns:
        raise ValueError("frame requires availability_ts")

    work = frame.sort(AVAILABILITY_TS)
    tick = float(tick_size) if tick_size is not None else float(round(0.25 / PRICE_SCALE))
    if tick <= 0:
        raise ValueError("tick_size must be > 0")

    # أسعار القرار في وحدات السعر الخام (int) كما في auction states؛
    # إشارات vp_upper مُقاسة بالدولار — نفضّل decision_* إن وُجدت.
    has_decision = all(c in work.columns for c in ("decision_vah", "decision_val", "decision_poc"))
    if has_decision:
        vah = pl.col("decision_vah").cast(pl.Float64)
        val = pl.col("decision_val").cast(pl.Float64)
        poc = pl.col("decision_poc").cast(pl.Float64)
        close = (
            pl.col("close").cast(pl.Float64)
            if "close" in work.columns
            else pl.col("vp_mid").cast(pl.Float64) / float(PRICE_SCALE)
            if "vp_mid" in work.columns
            else pl.lit(None, dtype=pl.Float64)
        )
    else:
        # مسار إشارات فقط: vp_* بالدولار → حوّل لتيك عبر PRICE_SCALE ضمنيًا
        scale = float(PRICE_SCALE)
        vah = (
            pl.col("vp_upper").cast(pl.Float64) / scale
            if "vp_upper" in work.columns
            else pl.lit(None)
        )
        val = (
            pl.col("vp_lower").cast(pl.Float64) / scale
            if "vp_lower" in work.columns
            else pl.lit(None)
        )
        poc = (
            pl.col("vp_mid").cast(pl.Float64) / scale if "vp_mid" in work.columns else pl.lit(None)
        )
        close = pl.col("close").cast(pl.Float64) if "close" in work.columns else poc

    width = (vah - val).clip(lower_bound=tick)
    near = _NEAR_FRAC * width

    def _dist_ticks(level: pl.Expr) -> pl.Expr:
        return (close - level) / pl.lit(tick)

    exprs = [
        _dist_ticks(vah).fill_null(0.0).alias("struct_dist_vah_ticks"),
        _dist_ticks(val).fill_null(0.0).alias("struct_dist_val_ticks"),
        _dist_ticks(poc).fill_null(0.0).alias("struct_dist_poc_ticks"),
        (width / pl.lit(tick)).fill_null(0.0).alias("struct_va_width_ticks"),
        ((close >= val) & (close <= vah))
        .fill_null(False)
        .cast(pl.Float64)
        .alias("struct_close_in_value"),
        (close > vah).fill_null(False).cast(pl.Float64).alias("struct_above_vah"),
        (close < val).fill_null(False).cast(pl.Float64).alias("struct_below_val"),
        ((close - vah).abs() <= near).fill_null(False).cast(pl.Float64).alias("struct_near_vah"),
        ((close - val).abs() <= near).fill_null(False).cast(pl.Float64).alias("struct_near_val"),
        ((close - poc).abs() <= near).fill_null(False).cast(pl.Float64).alias("struct_near_poc"),
    ]

    # مرساة آسيا / HVN من الإسقاط إن وُجدت (متاحة فقط بعد اكتمالها asof خلفي).
    for src, dest in (
        ("asia_vah", "struct_dist_asia_vah_ticks"),
        ("asia_val", "struct_dist_asia_val_ticks"),
        ("asia_poc", "struct_dist_asia_poc_ticks"),
        ("asia_primary_hvn", "struct_dist_asia_hvn_ticks"),
        ("composite_primary_hvn", "struct_dist_composite_hvn_ticks"),
    ):
        if src in work.columns:
            exprs.append(_dist_ticks(pl.col(src).cast(pl.Float64)).fill_null(0.0).alias(dest))
        else:
            exprs.append(pl.lit(0.0).alias(dest))

    break_pulse = (
        pl.col("vp_fsm_break").abs().fill_null(0.0)
        if "vp_fsm_break" in work.columns
        else pl.lit(0.0)
    )
    retest_pulse = (
        pl.col("vp_fsm_retest").abs().fill_null(0.0)
        if "vp_fsm_retest" in work.columns
        else pl.lit(0.0)
    )
    exprs.extend(
        [
            (break_pulse * (1.0 / (1.0 + (close - vah).abs() / (width + _EPS))))
            .fill_null(0.0)
            .alias("struct_break_pressure"),
            (retest_pulse * (1.0 / (1.0 + (close - poc).abs() / (width + _EPS))))
            .fill_null(0.0)
            .alias("struct_retest_pressure"),
        ]
    )
    return work.with_columns(exprs)


__all__ = [
    "STRUCTURE_FEATURE_COLUMNS",
    "attach_structure_features",
]
