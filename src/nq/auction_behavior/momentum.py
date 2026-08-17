"""زخم / تدفق / انحراف عن القيمة — بجانب Volume Profile، لا بدلًا منه.

أربع خصائص تُحسب من براميل ``blended`` المعروفة عند ``t`` فقط:

* ``roc_10``: معدل تغير الإغلاق خلال 10 براميل.
* ``cvd_20``: مجموع دلتا التدفق لآخر 20 برميلًا (وكيل CVD من ``vp_of_delta``).
* ``distance_to_vwap``: (إغلاق − VWAP الجلسة) / ATR(14) للأيام المكتملة السابقة.
* ``range_width_ratio``: مدى آخر 10 براميل / ATR(14) اليومي السابق.

لا يُعاد تحميل MBO. الحجم الحقيقي غير موجود على ``blended``؛ VWAP يستخدم
``lf_arrival_intensity`` كوزن، وCVD يستخدم ``vp_of_delta * intensity``.
ATR اليومي لا يرى مدى اليوم الجاري — فقط تواريخ الجلسة المكتملة السابقة.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import session_date_from_ns
from nq.research.progress import ProgressLike

_EPS = 1e-12
_ROC_BARS = 10
_CVD_BARS = 20
_RANGE_BARS = 10
_ATR_DAYS = 14

MOMENTUM_FEATURE_COLUMNS = (
    "roc_10",
    "cvd_20",
    "distance_to_vwap",
    "range_width_ratio",
)

_HELPER_COLS = (
    "_mom_session_date",
    "_mom_group",
    "_atr14_prior",
)


def _session_dates(ts: Sequence[int]) -> list[str]:
    return [session_date_from_ns(int(t)) for t in ts]


def _zero_features(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.lit(0.0).alias(c) for c in MOMENTUM_FEATURE_COLUMNS)


def attach_momentum_features(
    frame: pl.DataFrame,
    *,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يلحق عائلة الزخم من OHLC + وكلاء التدفق على الإطار المرتّب سببيًا."""
    if progress is not None:
        progress.op(f"attach_momentum_features bars={frame.height:,}")
    if frame.height == 0:
        return _zero_features(frame)
    if AVAILABILITY_TS not in frame.columns:
        raise ValueError("frame requires availability_ts")

    work = frame.sort(AVAILABILITY_TS)
    dates = _session_dates([int(t) for t in work[AVAILABILITY_TS].to_list()])
    work = work.with_columns(pl.Series("_mom_session_date", dates, dtype=pl.Utf8))
    if "_behavior_story_run" in work.columns:
        work = work.with_columns(pl.col("_behavior_story_run").alias("_mom_group"))
    else:
        work = work.with_columns(pl.col("_mom_session_date").alias("_mom_group"))

    close = (
        pl.col("close").cast(pl.Float64)
        if "close" in work.columns
        else pl.lit(0.0, dtype=pl.Float64)
    )
    high = pl.col("high").cast(pl.Float64) if "high" in work.columns else close
    low = pl.col("low").cast(pl.Float64) if "low" in work.columns else close
    intensity = (
        pl.col("lf_arrival_intensity").cast(pl.Float64).fill_null(1.0).clip(lower_bound=0.0)
        if "lf_arrival_intensity" in work.columns
        else pl.lit(1.0, dtype=pl.Float64)
    )
    of_delta = (
        pl.col("vp_of_delta").cast(pl.Float64).fill_null(0.0)
        if "vp_of_delta" in work.columns
        else pl.lit(0.0, dtype=pl.Float64)
    )

    daily = (
        work.select(
            "_mom_session_date",
            high.alias("high"),
            low.alias("low"),
            close.alias("close"),
            AVAILABILITY_TS,
        )
        .sort(AVAILABILITY_TS)
        .group_by("_mom_session_date", maintain_order=True)
        .agg(
            pl.col("high").max().alias("_day_high"),
            pl.col("low").min().alias("_day_low"),
            pl.col("close").last().alias("_day_close"),
        )
        .sort("_mom_session_date")
        .with_columns(pl.col("_day_close").shift(1).alias("_prev_close"))
        .with_columns(
            pl.max_horizontal(
                pl.col("_day_high") - pl.col("_day_low"),
                (pl.col("_day_high") - pl.col("_prev_close")).abs().fill_null(0.0),
                (pl.col("_day_low") - pl.col("_prev_close")).abs().fill_null(0.0),
            ).alias("_tr")
        )
        .with_columns(
            pl.col("_tr")
            .shift(1)
            .rolling_mean(window_size=_ATR_DAYS, min_samples=1)
            .alias("_atr14_prior")
        )
        .select("_mom_session_date", "_atr14_prior")
    )
    work = work.join(daily, on="_mom_session_date", how="left")
    work = work.sort(["_mom_group", AVAILABILITY_TS])

    atr = pl.col("_atr14_prior").cast(pl.Float64).fill_null(0.0)
    close_lag = close.shift(_ROC_BARS).over("_mom_group")
    roc = (
        pl.when(close_lag.abs() > _EPS)
        .then((close - close_lag) / close_lag)
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("roc_10")
    )
    cvd = (
        (of_delta * intensity)
        .rolling_sum(window_size=_CVD_BARS, min_samples=1)
        .over("_mom_group")
        .fill_null(0.0)
        .alias("cvd_20")
    )
    weight = intensity
    vwap_num = (close * weight).cum_sum().over("_mom_session_date")
    vwap_den = weight.cum_sum().over("_mom_session_date")
    vwap = pl.when(vwap_den > _EPS).then(vwap_num / vwap_den).otherwise(close)
    dist = (
        pl.when(atr.abs() > _EPS)
        .then((close - vwap) / atr)
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("distance_to_vwap")
    )
    roll_high = high.rolling_max(window_size=_RANGE_BARS, min_samples=1).over("_mom_group")
    roll_low = low.rolling_min(window_size=_RANGE_BARS, min_samples=1).over("_mom_group")
    width = (
        pl.when(atr.abs() > _EPS)
        .then((roll_high - roll_low) / atr)
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("range_width_ratio")
    )
    work = work.with_columns(roc, cvd, dist, width).sort(AVAILABILITY_TS)
    drop = [c for c in _HELPER_COLS if c in work.columns]
    return work.drop(drop)


__all__ = [
    "MOMENTUM_FEATURE_COLUMNS",
    "attach_momentum_features",
]
