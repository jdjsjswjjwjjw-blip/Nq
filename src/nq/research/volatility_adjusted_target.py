"""طبقة و: هدف/وقف مضبوطان بتقلب يومي سببي — قابلة للخلع.

ليست أداة تنفيذ حيّة حتى يصمد OOF. ليست 1:4 على VP وليست أقرب سيولة.

عند ``t``:
- ``daily_atr`` = متوسط المدى الحقيقي لآخر 14 **يومًا مكتملًا قبل** تاريخ الجلسة.
- فلتر: ``atr >= min_atr_pts`` (افتراضي 60 نقطة NQ).
- الهدف = الأقرب الأمامي بين هاي/لو آسيا و``entry ± 0.4 ATR``.
- الوقف = ``entry ∓ 0.2 ATR``.
- رفض إن RR < ``min_rr`` (افتراضي 2).

هاي آسيا = أقصى ``high`` (أو أدنى ``low``) لبراميل جلسة آسيا في نفس القصة،
معروف عند ``t``. اليوم الجاري لا يدخل ATR.

لا تغيّر Y العلمي. احذف هذا الملف + السكربت + ``_write_volatility_target`` للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession, session_date_from_ns
from nq.research.causal_entry import TICKS_PER_NQ_POINT, assert_not_raw_mbo_stream
from nq.research.hold_horizon import (
    causal_fires,
    jsonable,
    load_overlay_period_inputs,
    median_mean,
    oof_timestamps,
    walk_hold_windows,
)
from nq.research.progress import ProgressLike

LAYER_ID = "volatility_target"
_GROUP = "_behavior_story_run"
_SESSION_DATE = "_session_date"
_EPS = 1e-9
_ASIA = int(VpLiquiditySession.ASIA)
_PATH_COLS = (
    "close",
    "high",
    "low",
    "target_price",
    "stop_price",
    "direction",
)
PRICE_TICK = float(round(0.25 / PRICE_SCALE))


@dataclass(frozen=True, slots=True)
class VolatilityTargetConfig:
    """عتبات ثابتة غير مُقدَّرة على العينة. للأيام عالية التقلب فقط."""

    min_p: float = 0.5
    expansion_start_ticks: float = 16.0
    atr_days: int = 14
    min_atr_pts: float = 60.0
    target_atr_frac: float = 0.4
    stop_atr_frac: float = 0.2
    min_rr: float = 2.0
    use_asia_extreme: bool = True
    max_hold_bars: int = 120
    round_trip_cost_pts: float = 0.75
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class VolatilityTargetReport:
    trades: pl.DataFrame
    skipped: pl.DataFrame
    summaries: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _f64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    if frame.height == 0 or name not in frame.columns:
        return np.zeros(int(frame.height), dtype=np.float64)
    return frame[name].cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64, copy=False)


def _price_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    n = int(frame.height)
    if n == 0 or name not in frame.columns:
        return np.full(n, np.nan, dtype=np.float64)
    arr = frame[name].cast(pl.Float64).to_numpy().astype(np.float64, copy=True)
    missing = ~np.isfinite(arr) | (np.abs(arr) < _EPS)
    arr[missing] = np.nan
    return arr


def _i64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame[name].cast(pl.Int64).to_numpy().astype(np.int64, copy=False)


def infer_direction(
    *,
    break_dir: np.ndarray,
    close: np.ndarray,
    asia_vah: np.ndarray,
    asia_val: np.ndarray,
) -> np.ndarray:
    """+1 كسر أعلى آسيا، −1 أدنى VAL. لا نظرة أمامية."""
    out = np.sign(break_dir).astype(np.float64)
    unk = np.abs(out) < _EPS
    out[unk & np.isfinite(close) & np.isfinite(asia_vah) & (close > asia_vah)] = 1.0
    out[unk & np.isfinite(close) & np.isfinite(asia_val) & (close < asia_val)] = -1.0
    return out


def attach_session_dates(frame: pl.DataFrame, *, ts_col: str) -> pl.DataFrame:
    """تاريخ جلسة CME. إن وُجد العمود مسبقًا يُحترم (للاختبارات)."""
    if _SESSION_DATE in frame.columns:
        return frame
    if ts_col not in frame.columns:
        raise ValueError(f"{ts_col} required to attach session dates")
    dates = [session_date_from_ns(int(t)) for t in frame[ts_col].to_list()]
    return frame.with_columns(pl.Series(_SESSION_DATE, dates, dtype=pl.Utf8()))


def pts_to_price(pts: float, *, price_tick: float = PRICE_TICK) -> float:
    return float(pts) * float(TICKS_PER_NQ_POINT) * float(price_tick)


def price_to_pts(delta: float, *, price_tick: float = PRICE_TICK) -> float:
    return float(delta) / float(price_tick) / float(TICKS_PER_NQ_POINT)


def compute_daily_ranges(blended: pl.DataFrame) -> pl.DataFrame:
    """هاي/لو/إغلاق لكل تاريخ جلسة مكتمل. اليوم الجاري يُستثنى لاحقًا من ATR."""
    if blended.height == 0:
        return pl.DataFrame(
            {
                _SESSION_DATE: pl.Series(dtype=pl.Utf8()),
                "day_high": pl.Series(dtype=pl.Float64()),
                "day_low": pl.Series(dtype=pl.Float64()),
                "day_close": pl.Series(dtype=pl.Float64()),
            }
        )
    work = attach_session_dates(blended, ts_col=AVAILABILITY_TS).sort(
        [_SESSION_DATE, AVAILABILITY_TS]
    )
    return work.group_by(_SESSION_DATE, maintain_order=True).agg(
        pl.col("high").cast(pl.Float64).max().alias("day_high"),
        pl.col("low").cast(pl.Float64).min().alias("day_low"),
        pl.col("close").cast(pl.Float64).last().alias("day_close"),
    )


def causal_atr_pts(
    daily: pl.DataFrame,
    *,
    window: int,
    price_tick: float = PRICE_TICK,
) -> dict[str, float]:
    """ATR ليوم D من TR للأيام ``D-window … D-1`` فقط. لا يشمل مدى D."""
    if window < 1:
        raise ValueError("atr window must be >= 1")
    if daily.height == 0:
        return {}
    ordered = daily.sort(_SESSION_DATE)
    dates = ordered[_SESSION_DATE].to_list()
    high = _price_arr(ordered, "day_high")
    low = _price_arr(ordered, "day_low")
    close = _price_arr(ordered, "day_close")
    n = int(ordered.height)
    tr = np.full(n, np.nan, dtype=np.float64)
    span = high - low
    tr[0] = span[0]
    if n > 1:
        prev_c = close[:-1]
        h = high[1:]
        lo = low[1:]
        tr[1:] = np.nanmax(
            np.vstack([span[1:], np.abs(h - prev_c), np.abs(lo - prev_c)]),
            axis=0,
        )
    tr_pts = tr / float(price_tick) / float(TICKS_PER_NQ_POINT)
    out: dict[str, float] = {}
    for i, date in enumerate(dates):
        start = i - int(window)
        if start < 0:
            continue
        chunk = tr_pts[start:i]
        finite = chunk[np.isfinite(chunk)]
        if finite.size < int(window):
            continue
        out[str(date)] = float(np.mean(finite))
    return out


def asia_session_extremes(blended: pl.DataFrame, *, group_col: str = _GROUP) -> pl.DataFrame:
    """هاي/لو مدى آسيا لكل قصة — معروف عند أول بارميل لندن."""
    empty = pl.DataFrame(
        {
            group_col: pl.Series(dtype=pl.Int64()),
            "asia_session_high": pl.Series(dtype=pl.Float64()),
            "asia_session_low": pl.Series(dtype=pl.Float64()),
        }
    )
    if blended.height == 0 or group_col not in blended.columns:
        return empty
    if VP_LIQUIDITY_SESSION not in blended.columns:
        return empty
    asia = blended.filter(pl.col(VP_LIQUIDITY_SESSION).cast(pl.Int64) == _ASIA)
    if asia.height == 0:
        return empty
    return asia.group_by(group_col).agg(
        pl.col("high").cast(pl.Float64).max().alias("asia_session_high"),
        pl.col("low").cast(pl.Float64).min().alias("asia_session_low"),
    )


def _pick_target(
    *,
    close_px: float,
    direction: float,
    atr_pts: float,
    asia_high: float,
    asia_low: float,
    target_frac: float,
    use_asia: bool,
    price_tick: float,
) -> tuple[float, str]:
    vol_pts = float(target_frac) * float(atr_pts)
    vol_px = close_px + direction * pts_to_price(vol_pts, price_tick=price_tick)
    if not use_asia:
        return vol_px, "atr_extension"
    asia = asia_high if direction > 0.0 else asia_low
    if not np.isfinite(asia):
        return vol_px, "atr_extension"
    ahead = (asia - close_px) * direction
    if ahead <= 0.0:
        return vol_px, "atr_extension"
    if ahead <= abs(vol_px - close_px) + _EPS:
        return float(asia), "asia_session"
    return vol_px, "atr_extension"


def attach_volatility_at_t(
    entries: pl.DataFrame,
    _blended: pl.DataFrame,
    *,
    atr_by_date: Mapping[str, float],
    asia_ext: pl.DataFrame,
    config: VolatilityTargetConfig,
    price_tick: float = PRICE_TICK,
    group_col: str = _GROUP,
) -> pl.DataFrame:
    """يجمّد ATR/الهدف/الوقف عند ``t``. لا يقرأ مدى اليوم الجاري."""
    extra = {
        "direction": pl.Float64(),
        "daily_atr_pts": pl.Float64(),
        "target_name": pl.Utf8(),
        "target_price": pl.Float64(),
        "target_ticks": pl.Float64(),
        "risk_ticks": pl.Float64(),
        "stop_price": pl.Float64(),
        "rr_multiple": pl.Float64(),
        "vol_ok": pl.Boolean(),
        "skip_reason": pl.Utf8(),
    }
    n = int(entries.height)
    if n == 0:
        return entries.with_columns(
            pl.lit(None, dtype=dtype).alias(name) for name, dtype in extra.items()
        )
    work = attach_session_dates(entries, ts_col=SETUP_AVAILABILITY_TS)
    if asia_ext.height and group_col in asia_ext.columns:
        work = work.join(asia_ext, on=group_col, how="left")
    else:
        work = work.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("asia_session_high"),
            pl.lit(None, dtype=pl.Float64).alias("asia_session_low"),
        )
    close = _price_arr(work, "close")
    asia_vah = _price_arr(work, "asia_vah")
    asia_val = _price_arr(work, "asia_val")
    direction = infer_direction(
        break_dir=_f64_arr(work, "proj_break_direction"),
        close=close,
        asia_vah=asia_vah,
        asia_val=asia_val,
    )
    asia_high = _price_arr(work, "asia_session_high")
    asia_low = _price_arr(work, "asia_session_low")
    dates = work[_SESSION_DATE].to_list()
    atr = np.array([atr_by_date.get(str(d), float("nan")) for d in dates], dtype=np.float64)
    filled = _fill_vol_rows(
        close=close,
        direction=direction,
        atr=atr,
        asia_high=asia_high,
        asia_low=asia_low,
        config=config,
        price_tick=price_tick,
    )
    return work.with_columns(
        pl.Series("direction", direction),
        pl.Series("daily_atr_pts", atr),
        pl.Series("target_name", filled["target_name"], dtype=pl.Utf8()),
        pl.Series("target_price", filled["target_price"]),
        pl.Series("target_ticks", filled["target_ticks"]),
        pl.Series("risk_ticks", filled["risk_ticks"]),
        pl.Series("stop_price", filled["stop_price"]),
        pl.Series("rr_multiple", filled["rr"]),
        pl.Series("vol_ok", filled["ok"]),
        pl.Series("skip_reason", filled["skip"], dtype=pl.Utf8()),
    )


def _fill_vol_rows(
    *,
    close: np.ndarray,
    direction: np.ndarray,
    atr: np.ndarray,
    asia_high: np.ndarray,
    asia_low: np.ndarray,
    config: VolatilityTargetConfig,
    price_tick: float,
) -> dict[str, Any]:
    n = int(close.size)
    target_name: list[str | None] = [None] * n
    target_price = np.full(n, np.nan)
    target_ticks = np.full(n, np.nan)
    risk_ticks = np.full(n, np.nan)
    stop_price = np.full(n, np.nan)
    rr = np.full(n, np.nan)
    ok = np.zeros(n, dtype=np.bool_)
    skip: list[str | None] = [None] * n
    stop_frac = float(config.stop_atr_frac)
    min_atr = float(config.min_atr_pts)
    min_rr = float(config.min_rr)
    for i in range(n):
        if abs(float(direction[i])) < _EPS or not np.isfinite(close[i]):
            skip[i] = "no_direction"
            continue
        if not np.isfinite(atr[i]):
            skip[i] = "atr_unavailable"
            continue
        if atr[i] < min_atr:
            skip[i] = "atr_below_min"
            continue
        tgt_px, src = _pick_target(
            close_px=float(close[i]),
            direction=float(direction[i]),
            atr_pts=float(atr[i]),
            asia_high=float(asia_high[i]),
            asia_low=float(asia_low[i]),
            target_frac=config.target_atr_frac,
            use_asia=bool(config.use_asia_extreme),
            price_tick=price_tick,
        )
        stop_px = float(close[i]) - float(direction[i]) * pts_to_price(
            stop_frac * float(atr[i]), price_tick=price_tick
        )
        ahead_ticks = (tgt_px - float(close[i])) * float(direction[i]) / price_tick
        risk = abs(float(close[i]) - stop_px) / price_tick
        if not np.isfinite(ahead_ticks) or ahead_ticks < 1.0:
            skip[i] = "no_target_ahead"
            continue
        if not np.isfinite(risk) or risk < 1.0:
            skip[i] = "no_structure_stop"
            continue
        ratio = ahead_ticks / risk
        if ratio + 1e-12 < min_rr:
            skip[i] = "rr_below_min"
            continue
        target_name[i] = src
        target_price[i] = tgt_px
        target_ticks[i] = ahead_ticks
        risk_ticks[i] = risk
        stop_price[i] = stop_px
        rr[i] = ratio
        ok[i] = True
    return {
        "target_name": target_name,
        "target_price": target_price,
        "target_ticks": target_ticks,
        "risk_ticks": risk_ticks,
        "stop_price": stop_price,
        "rr": rr,
        "ok": ok,
        "skip": skip,
    }


def _bar_px(primary: np.ndarray | None, fallback: np.ndarray | None, k: int) -> float:
    if primary is not None and k < int(primary.size):
        raw = float(primary[k])
        if np.isfinite(raw) and abs(raw) > _EPS:
            return raw
    if fallback is not None and k < int(fallback.size):
        raw = float(fallback[k])
        if np.isfinite(raw) and abs(raw) > _EPS:
            return raw
    return float("nan")


def _hit_stop_or_take(
    *,
    direction: float,
    target_px: float,
    stop_px: float,
    close_px: float,
    high_px: float,
    low_px: float,
) -> str | None:
    if direction > 0.0:
        adverse = low_px if np.isfinite(low_px) else close_px
        favor = high_px if np.isfinite(high_px) else close_px
    else:
        adverse = high_px if np.isfinite(high_px) else close_px
        favor = low_px if np.isfinite(low_px) else close_px
    if np.isfinite(adverse) and (stop_px - adverse) * direction >= 0.0:
        return "stop"
    if np.isfinite(favor) and (favor - target_px) * direction >= 0.0:
        return "take"
    return None


def _decide_vol_exit(
    window: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
) -> tuple[int, str]:
    n = int(window["path_beyond_asia_ticks"].size)
    if n == 0:
        return 0, "empty"
    direction = float(scalars.get("direction_0", 0.0))
    target_px = float(scalars.get("target_price_0", float("nan")))
    stop_px = float(scalars.get("stop_price_0", float("nan")))
    use_price = np.isfinite(target_px) and np.isfinite(stop_px) and abs(direction) > _EPS
    if use_price:
        close_arr = window.get("close")
        high_arr = window.get("high")
        low_arr = window.get("low")
        for k in range(n):
            hit = _hit_stop_or_take(
                direction=direction,
                target_px=target_px,
                stop_px=stop_px,
                close_px=_bar_px(close_arr, None, k),
                high_px=_bar_px(high_arr, close_arr, k),
                low_px=_bar_px(low_arr, close_arr, k),
            )
            if hit is not None:
                return k, hit
    return max(n - 1, 0), "max_hold"


def _reason_counts(frame: pl.DataFrame, col: str) -> dict[str, int]:
    if frame.height == 0 or col not in frame.columns:
        return {}
    grouped = frame.group_by(col).agg(pl.len().cast(pl.Int64).alias("n")).sort(col)
    return {str(row[col]): int(row["n"]) for row in grouped.iter_rows(named=True)}


def _apply_basket_pnl(trades: pl.DataFrame, *, cost_pts: float) -> pl.DataFrame:
    if trades.height == 0:
        return trades
    return trades.with_columns(
        pl.when(pl.col("exit_reason") == "take")
        .then(pl.col("target_pts") - float(cost_pts))
        .when(pl.col("exit_reason") == "stop")
        .then(-pl.col("risk_pts") - float(cost_pts))
        .otherwise(pl.col("realized_beyond_pts") - float(cost_pts))
        .alias("net_pts")
    )


def _principles() -> tuple[str, ...]:
    return (
        "removable overlay — delete volatility_adjusted_target.py to remove",
        "not a live execution tool until OOF geometry holds; high-vol days only",
        "chart timeframe is unchanged; hold cap is N bars on the 30s blended clock",
        "science Y is unchanged; this is not a new label horizon",
        "daily ATR uses the previous 14 completed session dates, never the current day",
        "target is min(Asia session extreme ahead, 0.4 ATR); stop is 0.2 ATR",
        "min RR is a frozen gate, not estimated on OOF",
        "completed-wave peak / remaining-to-peak are never used",
        "holdout never scored; live_predictions never used",
    )


def _empty_diagnostics(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "empty": True,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "chart_timeframe_unchanged": True,
        "holdout_scored": False,
        "completed_wave_peak_not_used": True,
        "live_predictions_not_used": True,
        "not_live_execution": True,
        "high_vol_days_only": True,
        "current_day_range_not_in_atr": True,
        "principles": _principles(),
    }
    base.update(extra)
    return base


def run_volatility_target(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: VolatilityTargetConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> VolatilityTargetReport:
    """باسكت ATR + هاي آسيا بعد إطلاق OOF — بلا holdout وبلا 1:4."""
    cfg = config or VolatilityTargetConfig()
    if cfg.atr_days < 1:
        raise ValueError("atr_days must be >= 1")
    if cfg.min_atr_pts <= 0.0:
        raise ValueError("min_atr_pts must be > 0")
    if cfg.target_atr_frac <= 0.0 or cfg.stop_atr_frac <= 0.0:
        raise ValueError("ATR fractions must be > 0")
    if cfg.min_rr <= 0.0:
        raise ValueError("min_rr must be > 0")
    if cfg.max_hold_bars < 1:
        raise ValueError("max_hold_bars must be >= 1")
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    empty = VolatilityTargetReport(
        trades=pl.DataFrame(),
        skipped=pl.DataFrame(),
        summaries=pl.DataFrame(),
        diagnostics=_empty_diagnostics(),
    )
    fires = causal_fires(
        labeled,
        blended,
        predictions=predictions,
        oof_availability_ts=oof_availability_ts,
        holdout_cut_ts=holdout_cut_ts,
        min_p=cfg.min_p,
        expansion_start_ticks=cfg.expansion_start_ticks,
        holdout_months=cfg.holdout_months,
        progress=progress,
    )
    if fires.height == 0:
        return empty
    if progress is not None:
        progress.op("volatility_target daily ATR + Asia extremes")
    daily = compute_daily_ranges(blended)
    atr_map = causal_atr_pts(daily, window=cfg.atr_days)
    asia_ext = asia_session_extremes(blended)
    geo = attach_volatility_at_t(
        fires,
        blended,
        atr_by_date=atr_map,
        asia_ext=asia_ext,
        config=cfg,
    )
    traded = geo.filter(pl.col("vol_ok"))
    skipped = geo.filter(~pl.col("vol_ok"))
    summary_base: dict[str, Any] = {
        "layer_id": LAYER_ID,
        "n_fires": int(geo.height),
        "n_traded": 0,
        "n_skipped": int(skipped.height),
        "atr_days": int(cfg.atr_days),
        "min_atr_pts": float(cfg.min_atr_pts),
        "target_atr_frac": float(cfg.target_atr_frac),
        "stop_atr_frac": float(cfg.stop_atr_frac),
        "min_rr": float(cfg.min_rr),
        "use_asia_extreme": bool(cfg.use_asia_extreme),
        "max_hold_bars": int(cfg.max_hold_bars),
        "round_trip_cost_pts": float(cfg.round_trip_cost_pts),
        "take_n": 0,
        "stop_n": 0,
        "take_rate": None,
        "stop_rate": None,
        "net_pts_median": None,
        "net_pts_mean": None,
        "target_pts_median": None,
        "risk_pts_median": None,
        "rr_median": None,
        "atr_pts_median": None,
        "theoretical_be_rate": None,
    }
    if traded.height == 0:
        return VolatilityTargetReport(
            trades=pl.DataFrame(),
            skipped=skipped,
            summaries=pl.DataFrame([summary_base]),
            diagnostics=_empty_diagnostics(
                empty=False,
                n_fires=int(geo.height),
                n_traded=0,
                n_skipped=int(skipped.height),
                skip_reasons=_reason_counts(skipped, "skip_reason"),
                exit_reasons={},
                min_atr_pts=float(cfg.min_atr_pts),
                min_rr=float(cfg.min_rr),
                use_asia_extreme=bool(cfg.use_asia_extreme),
            ),
        )

    trades = walk_hold_windows(
        traded,
        blended,
        max_hold_bars=cfg.max_hold_bars,
        path_cols=_PATH_COLS,
        decide_exit=_decide_vol_exit,
        round_trip_cost_pts=cfg.round_trip_cost_pts,
    )
    extras = traded.with_row_index("entry_i").select(
        "entry_i",
        "target_name",
        "target_price",
        "target_ticks",
        "risk_ticks",
        "stop_price",
        "direction",
        "rr_multiple",
        "daily_atr_pts",
    )
    if trades.height:
        trades = trades.join(extras, on="entry_i", how="left")
        trades = trades.with_columns(
            (pl.col("target_ticks") / TICKS_PER_NQ_POINT).alias("target_pts"),
            (pl.col("risk_ticks") / TICKS_PER_NQ_POINT).alias("risk_pts"),
        )
        trades = _apply_basket_pnl(trades, cost_pts=cfg.round_trip_cost_pts)
    n_traded = int(trades.height)
    take_n = int((trades["exit_reason"] == "take").sum()) if n_traded else 0
    stop_n = int((trades["exit_reason"] == "stop").sum()) if n_traded else 0
    net_med, net_mean = median_mean(trades, "net_pts")
    tgt_med, tgt_mean = median_mean(trades, "target_pts")
    risk_med, risk_mean = median_mean(trades, "risk_pts")
    rr_med, rr_mean = median_mean(trades, "rr_multiple")
    atr_med, atr_mean = median_mean(trades, "daily_atr_pts")
    be = None if rr_med is None or rr_med <= 0.0 else float(1.0 / (1.0 + rr_med))
    summaries = pl.DataFrame(
        [
            {
                **summary_base,
                "n_traded": n_traded,
                "take_n": take_n,
                "stop_n": stop_n,
                "take_rate": (take_n / n_traded) if n_traded else None,
                "stop_rate": (stop_n / n_traded) if n_traded else None,
                "net_pts_median": net_med,
                "net_pts_mean": net_mean,
                "target_pts_median": tgt_med,
                "target_pts_mean": tgt_mean,
                "risk_pts_median": risk_med,
                "risk_pts_mean": risk_mean,
                "rr_median": rr_med,
                "rr_mean": rr_mean,
                "atr_pts_median": atr_med,
                "atr_pts_mean": atr_mean,
                "theoretical_be_rate": be,
            }
        ]
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "chart_timeframe_unchanged": True,
        "holdout_scored": False,
        "completed_wave_peak_not_used": True,
        "live_predictions_not_used": True,
        "not_live_execution": True,
        "high_vol_days_only": True,
        "current_day_range_not_in_atr": True,
        "hold_horizon_is_bars_not_label_window": True,
        "n_fires": int(geo.height),
        "n_traded": n_traded,
        "n_skipped": int(skipped.height),
        "skip_reasons": _reason_counts(skipped, "skip_reason"),
        "exit_reasons": _reason_counts(trades, "exit_reason"),
        "target_names": _reason_counts(trades, "target_name"),
        "take_rate": (take_n / n_traded) if n_traded else None,
        "stop_rate": (stop_n / n_traded) if n_traded else None,
        "net_pts_median": net_med,
        "net_pts_mean": net_mean,
        "target_pts_median": tgt_med,
        "risk_pts_median": risk_med,
        "rr_median": rr_med,
        "atr_pts_median": atr_med,
        "theoretical_be_rate": be,
        "min_atr_pts": float(cfg.min_atr_pts),
        "min_rr": float(cfg.min_rr),
        "use_asia_extreme": bool(cfg.use_asia_extreme),
        "principles": _principles(),
    }
    return VolatilityTargetReport(
        trades=trades,
        skipped=skipped,
        summaries=summaries,
        diagnostics=diagnostics,
    )


def run_volatility_target_grid(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    variants: Sequence[tuple[bool, float]] = ((True, 2.0), (True, 1.5), (False, 2.0)),
    config: VolatilityTargetConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """صمود مسبّق: مع/بدون هاي آسيا، وRR 2 و1.5 — ليس بحثًا عن الأفضل."""
    base = config or VolatilityTargetConfig()
    rows: list[pl.DataFrame] = []
    for use_asia, min_rr in variants:
        if progress is not None:
            progress.op(f"volatility_target asia={use_asia} min_rr={min_rr}")
        cfg = VolatilityTargetConfig(
            min_p=base.min_p,
            expansion_start_ticks=base.expansion_start_ticks,
            atr_days=base.atr_days,
            min_atr_pts=base.min_atr_pts,
            target_atr_frac=base.target_atr_frac,
            stop_atr_frac=base.stop_atr_frac,
            min_rr=float(min_rr),
            use_asia_extreme=bool(use_asia),
            max_hold_bars=base.max_hold_bars,
            round_trip_cost_pts=base.round_trip_cost_pts,
            holdout_months=base.holdout_months,
        )
        report = run_volatility_target(
            labeled,
            blended,
            config=cfg,
            oof_availability_ts=oof_availability_ts,
            holdout_cut_ts=holdout_cut_ts,
            predictions=predictions,
        )
        if report.summaries.height:
            rows.append(report.summaries)
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows, how="diagonal_relaxed")


def render_volatility_target_markdown(report: VolatilityTargetReport) -> str:
    d = report.diagnostics
    lines = [
        "# Volatility-adjusted target (removable layer F) — OOF test, high-vol days only",
        "",
        "Same causal OOF fire at `t`, same 30-second states. **Not** a chart",
        "timeframe change, **not** a new science Y, and **not** live execution",
        "until this geometry holds on OOF.",
        "Daily ATR is the mean true range of the previous 14 **completed**",
        "session dates. The current day's range is never used.",
        "Target = nearer of Asia session extreme still ahead and `0.4 ATR`.",
        "Stop = `0.2 ATR`. Skip if ATR < 60 pts or RR < 2.",
        "Delete this layer without touching science Y.",
        "",
        f"- layer_id={d.get('layer_id')} · removable={d.get('removable_layer')}",
        f"- not_live_execution={d.get('not_live_execution')} · "
        f"high_vol_days_only={d.get('high_vol_days_only')}",
        f"- current_day_range_not_in_atr={d.get('current_day_range_not_in_atr')}",
        f"- fires={d.get('n_fires')} · traded={d.get('n_traded')} · skipped={d.get('n_skipped')}",
        f"- take_rate={d.get('take_rate')} · stop_rate={d.get('stop_rate')}",
        f"- atr pts median={d.get('atr_pts_median')}",
        f"- target pts median={d.get('target_pts_median')} · "
        f"risk pts median={d.get('risk_pts_median')} · rr median={d.get('rr_median')}",
        f"- theoretical BE (from median rr)={d.get('theoretical_be_rate')}",
        f"- net median={d.get('net_pts_median')} · net mean={d.get('net_pts_mean')}",
        f"- skip_reasons={d.get('skip_reasons')}",
        f"- exit_reasons={d.get('exit_reasons')}",
        f"- target_names={d.get('target_names')}",
        "",
        "## Principles",
        "",
    ]
    for item in d.get("principles", ()):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_volatility_target_report(report: VolatilityTargetReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.trades.height:
        report.trades.write_parquet(out / "volatility_target_trades.parquet")
    if report.skipped.height:
        report.skipped.write_parquet(out / "volatility_target_skipped.parquet")
    if report.summaries.height:
        report.summaries.write_parquet(out / "volatility_target_summaries.parquet")
    payload = {
        "diagnostics": jsonable(report.diagnostics),
        "holdout_scored": False,
        "removable_layer": True,
        "chart_timeframe_unchanged": True,
        "does_not_modify_science_y": True,
        "not_live_execution": True,
        "high_vol_days_only": True,
    }
    (out / "volatility_target.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "VOLATILITY.md").write_text(render_volatility_target_markdown(report), encoding="utf-8")
    return out


def run_volatility_target_from_period_dir(
    period_dir: Path | str,
    *,
    config: VolatilityTargetConfig | None = None,
    progress: ProgressLike | None = None,
) -> VolatilityTargetReport:
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(period_dir)
    return run_volatility_target(
        labeled,
        blended,
        config=config,
        oof_availability_ts=oof_timestamps(oof),
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=progress,
    )


__all__ = [
    "LAYER_ID",
    "PRICE_TICK",
    "VolatilityTargetConfig",
    "VolatilityTargetReport",
    "asia_session_extremes",
    "attach_session_dates",
    "attach_volatility_at_t",
    "causal_atr_pts",
    "compute_daily_ranges",
    "infer_direction",
    "price_to_pts",
    "pts_to_price",
    "render_volatility_target_markdown",
    "run_volatility_target",
    "run_volatility_target_from_period_dir",
    "run_volatility_target_grid",
    "write_volatility_target_report",
]
