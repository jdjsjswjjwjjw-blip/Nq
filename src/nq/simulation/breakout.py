"""محاكي Failed Breakout السببي — من MBO فقط.

منطق الإشارة (عند إغلاق الشمعة فقط):

* جهد عالٍ: ``range`` و ``volume`` أعلى من متوسطات **ماضية** (shift(1)).
* كسر لأعلى ثم إغلاق تحت مستوى آخر N شموع مكتملة → SHORT.
* كسر لأسفل ثم إغلاق فوق المستوى → LONG.
* تأكيد اتجاه اختياري عبر SMA على إطار أعلى (asof خلفي).

إصلاح تسريب/وهم الدخول:

* الإشارة تُعلن عند ``availability_ts = bucket_end`` (إغلاق الشمعة).
* ``fail_breakout`` اتجاه فقط ∈ {-1,0,+1} — التقييم عبر مسار الألفا
  (mid أو bid/ask + slippage)، **وليس** ملء عند مستوى الكسر المثالي.
* ``fb_break_level`` مستوى الكسر الفاشل (مرجعي تحليلي).
* ``fb_entry_ref`` = إغلاق شمعة الإشارة (مرجع قابل للتنفيذ عند القرار).
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import SessionPhase, session_phase_from_ns
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.fvg import NS_1H, NS_30M, NS_PER_MIN, build_ohlcv_bars

SIGNAL_FB_SHORT: Final = -1.0
SIGNAL_FB_LONG: Final = 1.0

_DEFAULT_LOOKBACK = 5
_DEFAULT_ATR_WINDOW = 20
_DEFAULT_VOL_WINDOW = 20
_DEFAULT_SMA_PERIOD = 50
_DEFAULT_RANGE_MULT = 1.1
_DEFAULT_VOL_MULT = 1.2
_MIN_BARS = 30

_EMPTY_FB_SCHEMA: Final[dict[str, pl.DataType]] = {
    BUCKET_START: pl.Int64(),
    BUCKET_END: pl.Int64(),
    AVAILABILITY_TS: pl.Int64(),
    "fail_breakout": pl.Float64(),
    "fb_break_level": pl.Float64(),
    "fb_entry_ref": pl.Float64(),
    "fb_effort_range_ratio": pl.Float64(),
    "fb_effort_volume_ratio": pl.Float64(),
    "fb_risk_pts": pl.Float64(),
}


def _with_effort_baselines(
    bars: pl.DataFrame,
    *,
    atr_window: int,
    vol_window: int,
) -> pl.DataFrame:
    """متوسطات ماضية فقط — الشمعة الحالية لا تدخل خط الأساس."""
    return bars.sort(BUCKET_START).with_columns(
        pl.col("range")
        .shift(1)
        .rolling_mean(window_size=atr_window, min_samples=max(3, atr_window // 2))
        .alias("atr_past"),
        pl.col("volume")
        .shift(1)
        .rolling_mean(window_size=vol_window, min_samples=max(3, vol_window // 2))
        .alias("vol_sma_past"),
    )


def _sma_frame(higher: pl.DataFrame, *, period: int) -> pl.DataFrame:
    """SMA على إغلاق إطار أعلى؛ متاح عند إغلاق شمعة SMA فقط."""
    if higher.height == 0:
        return pl.DataFrame(
            schema={AVAILABILITY_TS: pl.Int64(), "sma": pl.Float64()}
        )
    return (
        higher.sort(BUCKET_START)
        .with_columns(
            pl.col("c")
            .shift(1)
            .rolling_mean(window_size=period, min_samples=max(5, period // 2))
            .alias("sma")
        )
        .select(pl.col(AVAILABILITY_TS), "sma")
        .drop_nulls("sma")
    )


def failed_breakout_from_bars(
    signal_bars: pl.DataFrame,
    *,
    trend_bars: pl.DataFrame | None = None,
    lookback: int = _DEFAULT_LOOKBACK,
    atr_window: int = _DEFAULT_ATR_WINDOW,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    range_mult: float = _DEFAULT_RANGE_MULT,
    vol_mult: float = _DEFAULT_VOL_MULT,
    sma_period: int = _DEFAULT_SMA_PERIOD,
    require_sma_filter: bool = True,
    rth_only: bool = True,
) -> pl.DataFrame:
    """يبني إشارة Failed Breakout من شموع مكتملة (سببي).

    Parameters
    ----------
    signal_bars:
        شموع الإطار التشغيلي (مثل 30m) من ``build_ohlcv_bars``.
    trend_bars:
        إطار أعلى لتأكيد SMA (مثل 1h). إن ``None`` يُعطَّل فلتر SMA.
    lookback:
        عدد الشموع **السابقة فقط** لمستوى المدى (لا تشمل الشمعة الحالية).
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if signal_bars.height < max(_MIN_BARS, lookback + atr_window):
        return pl.DataFrame(schema=_EMPTY_FB_SCHEMA)

    work = _with_effort_baselines(
        signal_bars, atr_window=atr_window, vol_window=vol_window
    )
    if require_sma_filter and trend_bars is not None and trend_bars.height > 0:
        sma = _sma_frame(trend_bars, period=sma_period)
        work = work.join_asof(
            sma.sort(AVAILABILITY_TS),
            on=AVAILABILITY_TS,
            strategy="backward",
        )
    else:
        work = work.with_columns(pl.lit(None).cast(pl.Float64()).alias("sma"))

    highs = work["h"].to_list()
    lows = work["l"].to_list()
    closes = work["c"].to_list()
    ranges = work["range"].to_list()
    volumes = work["volume"].to_list()
    atrs = work["atr_past"].to_list()
    vol_smas = work["vol_sma_past"].to_list()
    smas = work["sma"].to_list()
    starts = work[BUCKET_START].to_list()
    ends = work[BUCKET_END].to_list()
    avails = work[AVAILABILITY_TS].to_list()

    rows: list[dict[str, float | int]] = []
    for j in range(lookback, len(closes)):
        atr = atrs[j]
        vol_sma = vol_smas[j]
        if atr is None or vol_sma is None:
            continue
        atr_f = float(atr)
        vol_sma_f = float(vol_sma)
        if atr_f <= 0.0 or vol_sma_f <= 0.0:
            continue
        if float(ranges[j]) <= 0.0:
            continue

        avail = int(avails[j])
        if rth_only and session_phase_from_ns(avail) == int(SessionPhase.ETH):
            continue

        effort_r = float(ranges[j]) / atr_f
        effort_v = float(volumes[j]) / vol_sma_f
        if effort_r <= range_mult or effort_v <= vol_mult:
            continue

        # مدى الشموع السابقة فقط — بلا الشمعة الحالية
        prior_h = max(float(highs[k]) for k in range(j - lookback, j))
        prior_l = min(float(lows[k]) for k in range(j - lookback, j))
        h = float(highs[j])
        l = float(lows[j])
        c = float(closes[j])
        sma = smas[j]

        signal = 0.0
        level = 0.0
        risk = 0.0

        # كسر لأعلى وفشل → SHORT
        if h > prior_h and c < prior_h:
            if (not require_sma_filter) or (sma is not None and c < float(sma)):
                signal = SIGNAL_FB_SHORT
                level = prior_h
                risk = max(1.5 * 1.0, h - prior_h)  # نقاط سعر حقيقية (PRICE_SCALE already applied in bars)

        # كسر لأسفل وفشل → LONG
        if signal == 0.0 and l < prior_l and c > prior_l:
            if (not require_sma_filter) or (sma is not None and c > float(sma)):
                signal = SIGNAL_FB_LONG
                level = prior_l
                risk = max(1.5, prior_l - l)

        if signal == 0.0:
            continue

        rows.append(
            {
                BUCKET_START: int(starts[j]),
                BUCKET_END: int(ends[j]),
                AVAILABILITY_TS: avail,
                "fail_breakout": signal,
                "fb_break_level": level,
                # مرجع دخول قابل للتنفيذ عند القرار = إغلاق الشمعة (ليس مستوى الكسر)
                "fb_entry_ref": c,
                "fb_effort_range_ratio": effort_r,
                "fb_effort_volume_ratio": effort_v,
                "fb_risk_pts": float(risk),
            }
        )

    if not rows:
        return pl.DataFrame(schema=_EMPTY_FB_SCHEMA)
    return pl.DataFrame(rows).sort(AVAILABILITY_TS)


def failed_breakout_features(
    frame: pl.DataFrame,
    *,
    signal_interval_ns: int = NS_30M,
    trend_interval_ns: int = NS_1H,
    lookback: int = _DEFAULT_LOOKBACK,
    atr_window: int = _DEFAULT_ATR_WINDOW,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    range_mult: float = _DEFAULT_RANGE_MULT,
    vol_mult: float = _DEFAULT_VOL_MULT,
    sma_period: int = _DEFAULT_SMA_PERIOD,
    require_sma_filter: bool = True,
    rth_only: bool = True,
) -> pl.DataFrame:
    """يستخرج Failed Breakout من شريط MBO (صفقات → شموع → إشارة)."""
    signal_bars = build_ohlcv_bars(frame, interval_ns=signal_interval_ns)
    trend_bars = (
        build_ohlcv_bars(frame, interval_ns=trend_interval_ns)
        if require_sma_filter
        else None
    )
    return failed_breakout_from_bars(
        signal_bars,
        trend_bars=trend_bars,
        lookback=lookback,
        atr_window=atr_window,
        vol_window=vol_window,
        range_mult=range_mult,
        vol_mult=vol_mult,
        sma_period=sma_period,
        require_sma_filter=require_sma_filter,
        rth_only=rth_only,
    )


__all__ = [
    "NS_PER_MIN",
    "SIGNAL_FB_LONG",
    "SIGNAL_FB_SHORT",
    "failed_breakout_features",
    "failed_breakout_from_bars",
]
