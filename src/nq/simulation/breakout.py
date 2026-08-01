"""محاكي Failed Breakout سببي — من MBO فقط، بتركيز فوليوم.

منطق الإشارة (عند إغلاق الشمعة فقط):

* جهد سعري: ``range`` أعلى من متوسطات **ماضية** (shift(1)).
* جهد فوليوم (أوضاع فرضية):
  - ``bar``: حجم الشمعة / متوسط حجم ماضٍ.
  - ``cum``: حجم تراكمي لآخر N شموع / متوسط تراكمي ماضٍ.
  - ``delta``: |Δ| / متوسط |Δ| ماضٍ + اتفاق اتجاه العدوان مع فشل الكسر.
  - ``effort_result``: جهد حجم عالٍ مع نتيجة سعرية ضعيفة
    (امتصاص = volume/(range+ε) أعلى من ماضٍ) — جهد بلا نتيجة.
* أولوية الإشارة (``priority``):
  - ``structure_first``: الكسر الفاشل أولًا ثم بوابة فوليوم (الافتراضي التاريخي).
  - ``volume_first``: حدث الفوليوم يولّد المرشّح ثم بنية الكسر تؤكّد.
* أوضاع hold سببية عند الدخول (لا look-ahead للخروج):
  - ``none``: بلا شرط hold إضافي.
  - ``persist``: جهد حجم الشمعة السابقة مرتفع أيضًا (بناء/استمرار فوليوم).
  - ``absorption``: امتصاص عالٍ = حجم يُمسَك بلا نتيجة سعرية.
  - ``imbalance``: اختلال تدفق يتفق مع فشل الكسر.
* كسر لأعلى ثم إغلاق تحت مستوى آخر N شموع مكتملة → SHORT.
* كسر لأسفل ثم إغلاق فوق المستوى → LONG.
* تأكيد اتجاه اختياري عبر SMA على إطار أعلى (asof خلفي).

إصلاح تسريب/وهم الدخول:

* الإشارة تُعلن عند ``availability_ts = bucket_end`` (إغلاق الشمعة).
* ``fail_breakout`` اتجاه فقط ∈ {-1,0,+1} — التقييم عبر مسار الألفا.
* ``fb_break_level`` تحليلي؛ ``fb_entry_ref`` = إغلاق شمعة الإشارة.
* أفق الـ hold التنفيذي يُحدَّد عند التقييم (``--horizon``) — ليس داخل الميزة.
"""

from __future__ import annotations

from typing import Final, Literal

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import SessionPhase, session_phase_from_ns
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.fvg import NS_1H, NS_30M, NS_PER_MIN, build_ohlcv_bars

SIGNAL_FB_SHORT: Final = -1.0
SIGNAL_FB_LONG: Final = 1.0

VolMode = Literal["bar", "cum", "delta", "effort_result"]
SignalPriority = Literal["structure_first", "volume_first"]
HoldMode = Literal["none", "persist", "absorption", "imbalance"]

_DEFAULT_LOOKBACK = 5
_DEFAULT_ATR_WINDOW = 20
_DEFAULT_VOL_WINDOW = 20
_DEFAULT_CUM_WINDOW = 5
_DEFAULT_SMA_PERIOD = 50
_DEFAULT_RANGE_MULT = 1.1
_DEFAULT_VOL_MULT = 1.2
_DEFAULT_RESULT_MULT = 1.2
_DEFAULT_IMBALANCE_MIN = 0.15
_PERSIST_FRACTION = 0.85
_MIN_BARS = 30
_EPS = 1e-9

_EMPTY_FB_SCHEMA: Final[dict[str, pl.DataType]] = {
    BUCKET_START: pl.Int64(),
    BUCKET_END: pl.Int64(),
    AVAILABILITY_TS: pl.Int64(),
    "fail_breakout": pl.Float64(),
    "fb_break_level": pl.Float64(),
    "fb_entry_ref": pl.Float64(),
    "fb_effort_range_ratio": pl.Float64(),
    "fb_effort_volume_ratio": pl.Float64(),
    "fb_effort_result_ratio": pl.Float64(),
    "fb_bar_volume": pl.Float64(),
    "fb_cum_volume": pl.Float64(),
    "fb_delta": pl.Float64(),
    "fb_cum_delta": pl.Float64(),
    "fb_vol_imbalance": pl.Float64(),
    "fb_absorption": pl.Float64(),
    "fb_risk_pts": pl.Float64(),
}

# مرشّح قبل فلتر hold — عمود داخلي يُحذف من المخرج العام
_FB_EFFORT_PREV: Final = "fb_effort_volume_prev"
_CANDIDATE_SCHEMA: Final[dict[str, pl.DataType]] = {
    **_EMPTY_FB_SCHEMA,
    _FB_EFFORT_PREV: pl.Float64(),
}


def _ensure_flow_columns(bars: pl.DataFrame) -> pl.DataFrame:
    """يضمن أعمدة تدفّق على الشموع (للبيانات الاصطناعية بلا buy/sell)."""
    work = bars
    if "buy_volume" not in work.columns:
        work = work.with_columns(pl.lit(0.0).alias("buy_volume"))
    if "sell_volume" not in work.columns:
        work = work.with_columns(pl.lit(0.0).alias("sell_volume"))
    if "delta" not in work.columns:
        work = work.with_columns((pl.col("buy_volume") - pl.col("sell_volume")).alias("delta"))
    return work


def _with_volume_baselines(
    bars: pl.DataFrame,
    *,
    atr_window: int,
    vol_window: int,
    cum_window: int,
) -> pl.DataFrame:
    """متوسطات ماضية فقط — الشمعة الحالية لا تدخل خط الأساس."""
    work = _ensure_flow_columns(bars).sort(BUCKET_START)
    cum_w = max(1, cum_window)
    return work.with_columns(
        pl.col("range")
        .shift(1)
        .rolling_mean(window_size=atr_window, min_samples=max(3, atr_window // 2))
        .alias("atr_past"),
        pl.col("volume")
        .shift(1)
        .rolling_mean(window_size=vol_window, min_samples=max(3, vol_window // 2))
        .alias("vol_sma_past"),
        pl.col("volume").rolling_sum(window_size=cum_w, min_samples=1).alias("cum_volume"),
        pl.col("delta").cum_sum().alias("cum_delta"),
        (pl.col("volume") / (pl.col("range").abs() + _EPS)).alias("absorption"),
        pl.col("delta").abs().alias("abs_delta"),
    ).with_columns(
        pl.col("cum_volume")
        .shift(1)
        .rolling_mean(window_size=vol_window, min_samples=max(3, vol_window // 2))
        .alias("cum_vol_sma_past"),
        pl.col("abs_delta")
        .shift(1)
        .rolling_mean(window_size=vol_window, min_samples=max(3, vol_window // 2))
        .alias("abs_delta_sma_past"),
        pl.col("absorption")
        .shift(1)
        .rolling_mean(window_size=vol_window, min_samples=max(3, vol_window // 2))
        .alias("absorption_sma_past"),
    )


def _sma_frame(higher: pl.DataFrame, *, period: int) -> pl.DataFrame:
    """SMA على إغلاق إطار أعلى؛ متاح عند إغلاق شمعة SMA فقط."""
    if higher.height == 0:
        return pl.DataFrame(schema={AVAILABILITY_TS: pl.Int64(), "sma": pl.Float64()})
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


def _volume_gate(  # noqa: PLR0911
    *,
    mode: VolMode,
    effort_v: float,
    cum_effort: float,
    delta_effort: float,
    result_effort: float,
    delta: float,
    vol_mult: float,
    result_mult: float,
    signal_side: float,
) -> bool:
    """بوابة فوليوم سببية حسب وضع الفرضية."""
    if mode == "bar":
        return effort_v > vol_mult
    if mode == "cum":
        return cum_effort > vol_mult
    if mode == "delta":
        # فشل كسر صاعد مع عدوان شراء → SHORT؛ فشل هابط مع عدوان بيع → LONG
        if delta_effort <= vol_mult:
            return False
        if signal_side == SIGNAL_FB_SHORT:
            return delta > 0.0
        if signal_side == SIGNAL_FB_LONG:
            return delta < 0.0
        return False
    # mode == "effort_result"
    return effort_v > vol_mult and result_effort > result_mult


def apply_hold_mode_filter(
    candidates: pl.DataFrame,
    *,
    hold_mode: HoldMode = "none",
    vol_mult: float = _DEFAULT_VOL_MULT,
    result_mult: float = _DEFAULT_RESULT_MULT,
    imbalance_min: float = _DEFAULT_IMBALANCE_MIN,
) -> pl.DataFrame:
    """فلتر hold رخيص على مرشّحي مسح واحد — بلا إعادة مسح الشموع.

    المدخل قد يحتوي ``fb_effort_volume_prev``؛ المخرج يطابق ``_EMPTY_FB_SCHEMA``.
    """
    if hold_mode not in ("none", "persist", "absorption", "imbalance"):
        raise ValueError(f"unknown hold_mode: {hold_mode!r}")
    if candidates.height == 0:
        return pl.DataFrame(schema=_EMPTY_FB_SCHEMA)

    work = candidates
    if hold_mode == "persist":
        prev = pl.col(_FB_EFFORT_PREV) if _FB_EFFORT_PREV in work.columns else pl.lit(0.0)
        work = work.filter(prev > vol_mult * _PERSIST_FRACTION)
    elif hold_mode == "absorption":
        work = work.filter(pl.col("fb_effort_result_ratio") > result_mult)
    elif hold_mode == "imbalance":
        work = work.filter(
            (
                (pl.col("fail_breakout") == SIGNAL_FB_SHORT)
                & (pl.col("fb_vol_imbalance") > imbalance_min)
            )
            | (
                (pl.col("fail_breakout") == SIGNAL_FB_LONG)
                & (pl.col("fb_vol_imbalance") < -imbalance_min)
            )
        )

    drop = [c for c in work.columns if c not in _EMPTY_FB_SCHEMA]
    if drop:
        work = work.drop(drop)
    # أعمدة ناقصة؟ أعد ترتيباً مطابقاً للمخطط
    missing = [c for c in _EMPTY_FB_SCHEMA if c not in work.columns]
    if missing:
        work = work.with_columns([pl.lit(None).cast(_EMPTY_FB_SCHEMA[c]).alias(c) for c in missing])
    return work.select(list(_EMPTY_FB_SCHEMA)).sort(AVAILABILITY_TS)


def failed_breakout_candidates_from_bars(  # noqa: PLR0912, PLR0915
    signal_bars: pl.DataFrame,
    *,
    trend_bars: pl.DataFrame | None = None,
    lookback: int = _DEFAULT_LOOKBACK,
    atr_window: int = _DEFAULT_ATR_WINDOW,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    cum_window: int = _DEFAULT_CUM_WINDOW,
    range_mult: float = _DEFAULT_RANGE_MULT,
    vol_mult: float = _DEFAULT_VOL_MULT,
    result_mult: float = _DEFAULT_RESULT_MULT,
    vol_mode: VolMode = "bar",
    priority: SignalPriority = "structure_first",
    sma_period: int = _DEFAULT_SMA_PERIOD,
    require_sma_filter: bool = True,
    rth_only: bool = True,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """مسح مرشّحي Failed Breakout **بدون** فلتر hold (لإعادة استخدام المسح).

    يضيف ``fb_effort_volume_prev`` لتمكين ``apply_hold_mode_filter`` لاحقًا.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if vol_mode not in ("bar", "cum", "delta", "effort_result"):
        raise ValueError(f"unknown vol_mode: {vol_mode!r}")
    if priority not in ("structure_first", "volume_first"):
        raise ValueError(f"unknown priority: {priority!r}")
    if signal_bars.height < max(_MIN_BARS, lookback + atr_window):
        if progress is not None:
            progress.op(f"failed_breakout_candidates: تخطّي — شموع غير كافية ({signal_bars.height})")
        return pl.DataFrame(schema=_CANDIDATE_SCHEMA)

    work = _with_volume_baselines(
        signal_bars,
        atr_window=atr_window,
        vol_window=vol_window,
        cum_window=cum_window,
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
    deltas = work["delta"].to_list()
    cum_vols = work["cum_volume"].to_list()
    cum_deltas = work["cum_delta"].to_list()
    absorptions = work["absorption"].to_list()
    atrs = work["atr_past"].to_list()
    vol_smas = work["vol_sma_past"].to_list()
    cum_vol_smas = work["cum_vol_sma_past"].to_list()
    abs_delta_smas = work["abs_delta_sma_past"].to_list()
    absorption_smas = work["absorption_sma_past"].to_list()
    smas = work["sma"].to_list()
    starts = work[BUCKET_START].to_list()
    ends = work[BUCKET_END].to_list()
    avails = work[AVAILABILITY_TS].to_list()

    rows: list[dict[str, float | int]] = []
    n_scan = len(closes) - lookback
    if progress is not None:
        progress.op(
            f"failed_breakout_candidates: مسح {n_scan:,} شمعة · "
            f"mode={vol_mode} · priority={priority}"
        )
    for j in range(lookback, len(closes)):
        if progress is not None:
            done = j - lookback + 1
            progress.heartbeat(done, max(n_scan, 1), label="fb_bars")
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

        vol_j = float(volumes[j])
        range_j = float(ranges[j])
        delta_j = float(deltas[j]) if deltas[j] is not None else 0.0
        cum_vol_j = float(cum_vols[j]) if cum_vols[j] is not None else vol_j
        cum_delta_j = float(cum_deltas[j]) if cum_deltas[j] is not None else delta_j
        absorp_j = float(absorptions[j]) if absorptions[j] is not None else 0.0

        effort_r = range_j / atr_f
        effort_v = vol_j / vol_sma_f
        cum_sma = cum_vol_smas[j]
        cum_effort = (
            cum_vol_j / float(cum_sma) if cum_sma is not None and float(cum_sma) > 0 else 0.0
        )
        d_sma = abs_delta_smas[j]
        delta_effort = (
            abs(delta_j) / float(d_sma) if d_sma is not None and float(d_sma) > 0 else 0.0
        )
        a_sma = absorption_smas[j]
        result_effort = absorp_j / float(a_sma) if a_sma is not None and float(a_sma) > 0 else 0.0

        effort_v_prev = 0.0
        if j >= 1:
            prev_sma = vol_smas[j - 1]
            if prev_sma is not None and float(prev_sma) > 0.0:
                effort_v_prev = float(volumes[j - 1]) / float(prev_sma)

        range_ok = effort_r > range_mult
        prior_h = max(float(highs[k]) for k in range(j - lookback, j))
        prior_l = min(float(lows[k]) for k in range(j - lookback, j))
        h = float(highs[j])
        low_j = float(lows[j])
        c = float(closes[j])
        sma = smas[j]

        sma_ok_short = (not require_sma_filter) or (sma is not None and c < float(sma))
        sma_ok_long = (not require_sma_filter) or (sma is not None and c > float(sma))
        struct_short = h > prior_h and c < prior_h and sma_ok_short
        struct_long = low_j < prior_l and c > prior_l and sma_ok_long
        imbalance = delta_j / vol_j if vol_j > 0 else 0.0

        signal = 0.0
        level = 0.0
        risk = 0.0

        if priority == "structure_first" and not range_ok:
            continue

        for side, is_struct, lvl, risk_pts in (
            (SIGNAL_FB_SHORT, struct_short, prior_h, max(1.5, h - prior_h)),
            (SIGNAL_FB_LONG, struct_long, prior_l, max(1.5, prior_l - low_j)),
        ):
            if not is_struct:
                continue
            vol_ok = _volume_gate(
                mode=vol_mode,
                effort_v=effort_v,
                cum_effort=cum_effort,
                delta_effort=delta_effort,
                result_effort=result_effort,
                delta=delta_j,
                vol_mult=vol_mult,
                result_mult=result_mult,
                signal_side=side,
            )
            accepted = vol_ok and range_ok if priority == "volume_first" else vol_ok
            if accepted:
                signal = side
                level = lvl
                risk = risk_pts
                break

        if signal == 0.0:
            continue

        rows.append(
            {
                BUCKET_START: int(starts[j]),
                BUCKET_END: int(ends[j]),
                AVAILABILITY_TS: avail,
                "fail_breakout": signal,
                "fb_break_level": level,
                "fb_entry_ref": c,
                "fb_effort_range_ratio": effort_r,
                "fb_effort_volume_ratio": effort_v,
                "fb_effort_result_ratio": result_effort,
                "fb_bar_volume": vol_j,
                "fb_cum_volume": cum_vol_j,
                "fb_delta": delta_j,
                "fb_cum_delta": cum_delta_j,
                "fb_vol_imbalance": imbalance,
                "fb_absorption": absorp_j,
                "fb_risk_pts": float(risk),
                _FB_EFFORT_PREV: float(effort_v_prev),
            }
        )

    if not rows:
        return pl.DataFrame(schema=_CANDIDATE_SCHEMA)
    return pl.DataFrame(rows).sort(AVAILABILITY_TS)


def failed_breakout_from_bars(
    signal_bars: pl.DataFrame,
    *,
    trend_bars: pl.DataFrame | None = None,
    lookback: int = _DEFAULT_LOOKBACK,
    atr_window: int = _DEFAULT_ATR_WINDOW,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    cum_window: int = _DEFAULT_CUM_WINDOW,
    range_mult: float = _DEFAULT_RANGE_MULT,
    vol_mult: float = _DEFAULT_VOL_MULT,
    result_mult: float = _DEFAULT_RESULT_MULT,
    vol_mode: VolMode = "bar",
    priority: SignalPriority = "structure_first",
    hold_mode: HoldMode = "none",
    imbalance_min: float = _DEFAULT_IMBALANCE_MIN,
    sma_period: int = _DEFAULT_SMA_PERIOD,
    require_sma_filter: bool = True,
    rth_only: bool = True,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يبني إشارة Failed Breakout من شموع مكتملة (سببي + فوليوم).

    المسح مرة واحدة ثم ``apply_hold_mode_filter`` — نفس الأرقام، جاهز لكاش التجسيد.
    """
    if hold_mode not in ("none", "persist", "absorption", "imbalance"):
        raise ValueError(f"unknown hold_mode: {hold_mode!r}")
    candidates = failed_breakout_candidates_from_bars(
        signal_bars,
        trend_bars=trend_bars,
        lookback=lookback,
        atr_window=atr_window,
        vol_window=vol_window,
        cum_window=cum_window,
        range_mult=range_mult,
        vol_mult=vol_mult,
        result_mult=result_mult,
        vol_mode=vol_mode,
        priority=priority,
        sma_period=sma_period,
        require_sma_filter=require_sma_filter,
        rth_only=rth_only,
        progress=progress,
    )
    return apply_hold_mode_filter(
        candidates,
        hold_mode=hold_mode,
        vol_mult=vol_mult,
        result_mult=result_mult,
        imbalance_min=imbalance_min,
    )


def failed_breakout_features(
    frame: pl.DataFrame,
    *,
    signal_interval_ns: int = NS_30M,
    trend_interval_ns: int = NS_1H,
    lookback: int = _DEFAULT_LOOKBACK,
    atr_window: int = _DEFAULT_ATR_WINDOW,
    vol_window: int = _DEFAULT_VOL_WINDOW,
    cum_window: int = _DEFAULT_CUM_WINDOW,
    range_mult: float = _DEFAULT_RANGE_MULT,
    vol_mult: float = _DEFAULT_VOL_MULT,
    result_mult: float = _DEFAULT_RESULT_MULT,
    vol_mode: VolMode = "bar",
    priority: SignalPriority = "structure_first",
    hold_mode: HoldMode = "none",
    imbalance_min: float = _DEFAULT_IMBALANCE_MIN,
    sma_period: int = _DEFAULT_SMA_PERIOD,
    require_sma_filter: bool = True,
    rth_only: bool = True,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يستخرج Failed Breakout من شريط MBO (صفقات → شموع → إشارة فوليوم)."""
    if progress is not None:
        progress.op(
            f"failed_breakout_features: OHLCV signal={signal_interval_ns} · "
            f"trend={trend_interval_ns} · أحداث={frame.height:,} · "
            f"priority={priority} · hold={hold_mode}"
        )
    signal_bars = build_ohlcv_bars(frame, interval_ns=signal_interval_ns)
    if progress is not None:
        progress.op(f"OHLCV إشارة: {signal_bars.height:,} شمعة")
    trend_bars = (
        build_ohlcv_bars(frame, interval_ns=trend_interval_ns) if require_sma_filter else None
    )
    if progress is not None and trend_bars is not None:
        progress.op(f"OHLCV اتجاه: {trend_bars.height:,} شمعة")
    return failed_breakout_from_bars(
        signal_bars,
        trend_bars=trend_bars,
        lookback=lookback,
        atr_window=atr_window,
        vol_window=vol_window,
        cum_window=cum_window,
        range_mult=range_mult,
        vol_mult=vol_mult,
        result_mult=result_mult,
        vol_mode=vol_mode,
        priority=priority,
        hold_mode=hold_mode,
        imbalance_min=imbalance_min,
        sma_period=sma_period,
        require_sma_filter=require_sma_filter,
        rth_only=rth_only,
        progress=progress,
    )


__all__ = [
    "NS_PER_MIN",
    "SIGNAL_FB_LONG",
    "SIGNAL_FB_SHORT",
    "HoldMode",
    "SignalPriority",
    "VolMode",
    "apply_hold_mode_filter",
    "failed_breakout_candidates_from_bars",
    "failed_breakout_features",
    "failed_breakout_from_bars",
]
