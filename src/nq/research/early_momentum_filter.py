"""طبقة ح: فلتر زخم مبكر عند ``t`` — قابلة للخلع.

ليست إعادة تعريف Y، وليست بوابة من موقع الموجة المكتملة.
فوق ``p ≥ 0.5`` وامتداد ظاهر، ثلاثة شروط سببية معروفة عند ``t``:

- ``early_momentum``: ``direction × (close_t − close_{t-5}) / london_atr ≥ 0.15``
- ``early_volume``: ``lf_arrival_intensity_t > 1.5 × mean(t-20 … t-1)``
  (لا عمود حجم شمعة على الحالات الممزوجة؛ الشدة هي وكيل MBO السببي)
- ``early_break``: السعر تجاوز هاي/لو مدى آسيا بأكثر من 2 نقطة (8 تكات)

``london_atr`` من هاي−لو لندن لآخر 14 جلسة مكتملة، بلا مدى اليوم.
الذروة المكتملة / ``wave_frac`` تُشخَّص بعد الواقعة ولا تُستخدم للدخول.

لا تغيّر Y العلمي. احذف هذا الملف + السكربت + ``_write_early_momentum`` للإزالة.
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
from nq.core.session import (
    VP_LIQUIDITY_SESSION,
    VpLiquiditySession,
    session_date_from_ns,
    vp_liquidity_session_bounds_ns,
)
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
from nq.research.wave_position import build_wave_geometry

LAYER_ID = "early_momentum_filter"
_GROUP = "_behavior_story_run"
_SESSION_DATE = "_session_date"
_EPS = 1e-9
_ASIA = int(VpLiquiditySession.ASIA)
_LONDON = int(VpLiquiditySession.LONDON)
_LONDON_MAX_BARS = 780
_VOLUME_COL = "lf_arrival_intensity"
_PATH_COLS = (
    "close",
    "high",
    "low",
    "target_price",
    "stop_price",
    "direction",
    "london_end_ts",
    VP_LIQUIDITY_SESSION,
)
PRICE_TICK = float(round(0.25 / PRICE_SCALE))


@dataclass(frozen=True, slots=True)
class EarlyMomentumConfig:
    """عتبات ثابتة غير مُقدَّرة على العينة. ليست كسر موجة مكتملة."""

    min_p: float = 0.5
    expansion_start_ticks: float = 16.0
    atr_days: int = 14
    momentum_bars: int = 5
    momentum_atr_frac: float = 0.15
    volume_bars: int = 20
    volume_multiple: float = 1.5
    break_pts: float = 2.0
    require_momentum: bool = True
    require_volume: bool = True
    require_break: bool = True
    target_atr_frac: float = 0.5
    stop_atr_frac: float = 0.2
    max_hold_bars: int = _LONDON_MAX_BARS
    round_trip_cost_pts: float = 0.75
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class EarlyMomentumReport:
    trades: pl.DataFrame
    skipped: pl.DataFrame
    fires: pl.DataFrame
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


def _path_by_story(path: pl.DataFrame, group_col: str) -> dict[Any, pl.DataFrame]:
    grouped: dict[Any, pl.DataFrame] = {}
    for key, group in path.group_by(group_col, maintain_order=True):
        story = key[0] if isinstance(key, tuple) else key
        grouped[story] = group
    return grouped


def infer_direction(
    *,
    break_dir: np.ndarray,
    close: np.ndarray,
    asia_vah: np.ndarray,
    asia_val: np.ndarray,
) -> np.ndarray:
    out = np.sign(break_dir).astype(np.float64)
    unk = np.abs(out) < _EPS
    out[unk & np.isfinite(close) & np.isfinite(asia_vah) & (close > asia_vah)] = 1.0
    out[unk & np.isfinite(close) & np.isfinite(asia_val) & (close < asia_val)] = -1.0
    return out


def attach_session_dates(frame: pl.DataFrame, *, ts_col: str) -> pl.DataFrame:
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


def compute_london_ranges(blended: pl.DataFrame) -> pl.DataFrame:
    empty = pl.DataFrame(
        {
            _SESSION_DATE: pl.Series(dtype=pl.Utf8()),
            "london_high": pl.Series(dtype=pl.Float64()),
            "london_low": pl.Series(dtype=pl.Float64()),
        }
    )
    if blended.height == 0 or VP_LIQUIDITY_SESSION not in blended.columns:
        return empty
    work = attach_session_dates(blended, ts_col=AVAILABILITY_TS).sort(
        [_SESSION_DATE, AVAILABILITY_TS]
    )
    london = work.filter(pl.col(VP_LIQUIDITY_SESSION).cast(pl.Int64) == _LONDON)
    if london.height == 0:
        return empty
    return london.group_by(_SESSION_DATE, maintain_order=True).agg(
        pl.col("high").cast(pl.Float64).max().alias("london_high"),
        pl.col("low").cast(pl.Float64).min().alias("london_low"),
    )


def causal_london_atr_pts(
    london_days: pl.DataFrame,
    *,
    window: int,
    price_tick: float = PRICE_TICK,
) -> dict[str, float]:
    if window < 1:
        raise ValueError("atr window must be >= 1")
    if london_days.height == 0:
        return {}
    ordered = london_days.sort(_SESSION_DATE)
    dates = ordered[_SESSION_DATE].to_list()
    high = _price_arr(ordered, "london_high")
    low = _price_arr(ordered, "london_low")
    span_pts = (high - low) / float(price_tick) / float(TICKS_PER_NQ_POINT)
    out: dict[str, float] = {}
    for i, date in enumerate(dates):
        start = i - int(window)
        if start < 0:
            continue
        chunk = span_pts[start:i]
        finite = chunk[np.isfinite(chunk)]
        if finite.size < int(window):
            continue
        out[str(date)] = float(np.mean(finite))
    return out


def asia_session_extremes(blended: pl.DataFrame, *, group_col: str = _GROUP) -> pl.DataFrame:
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


def _bar_index(ts: np.ndarray, t0: int) -> int:
    k = int(np.searchsorted(ts, t0, side="left"))
    if k < int(ts.size) and int(ts[k]) == int(t0):
        return k
    return -1


def _momentum_pts(
    close: np.ndarray,
    k: int,
    direction: float,
    bars: int,
    price_tick: float,
) -> float:
    if k < int(bars) or abs(direction) < _EPS:
        return float("nan")
    delta = (float(close[k]) - float(close[k - int(bars)])) * float(direction)
    if not np.isfinite(delta):
        return float("nan")
    return price_to_pts(delta, price_tick=price_tick)


def _volume_ratio(intensity: np.ndarray, k: int, bars: int) -> float:
    if k < int(bars):
        return float("nan")
    prior = intensity[k - int(bars) : k]
    finite = prior[np.isfinite(prior)]
    if finite.size < int(bars):
        return float("nan")
    avg = float(np.mean(finite))
    now = float(intensity[k])
    if not np.isfinite(now) or avg <= _EPS:
        return float("nan")
    return now / avg


def _break_pts(
    *,
    close_px: float,
    high_px: float,
    low_px: float,
    direction: float,
    asia_high: float,
    asia_low: float,
    price_tick: float,
) -> float:
    if abs(direction) < _EPS:
        return float("nan")
    if direction > 0.0:
        favor = high_px if np.isfinite(high_px) else close_px
        level = asia_high
    else:
        favor = low_px if np.isfinite(low_px) else close_px
        level = asia_low
    if not np.isfinite(favor) or not np.isfinite(level):
        return float("nan")
    return price_to_pts((favor - level) * direction, price_tick=price_tick)


def _first_skip(
    *,
    direction: float,
    atr_pts: float,
    mom_pts: float,
    vol_ratio: float,
    brk_pts: float,
    config: EarlyMomentumConfig,
) -> str | None:
    mom_need = float(config.momentum_atr_frac) * atr_pts if np.isfinite(atr_pts) else float("nan")
    checks: tuple[tuple[bool, str], ...] = (
        (abs(direction) < _EPS, "no_direction"),
        (not np.isfinite(atr_pts), "atr_unavailable"),
        (bool(config.require_momentum) and not np.isfinite(mom_pts), "momentum_unavailable"),
        (
            bool(config.require_momentum)
            and np.isfinite(mom_pts)
            and np.isfinite(mom_need)
            and mom_pts < mom_need - 1e-12,
            "momentum_below_min",
        ),
        (bool(config.require_volume) and not np.isfinite(vol_ratio), "volume_unavailable"),
        (
            bool(config.require_volume)
            and np.isfinite(vol_ratio)
            and vol_ratio <= float(config.volume_multiple) + 1e-12,
            "volume_below_min",
        ),
        (bool(config.require_break) and not np.isfinite(brk_pts), "break_unavailable"),
        (
            bool(config.require_break)
            and np.isfinite(brk_pts)
            and brk_pts + 1e-12 < float(config.break_pts),
            "break_below_min",
        ),
    )
    for failed, reason in checks:
        if failed:
            return reason
    return None


def attach_early_momentum_at_t(
    entries: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    atr_by_date: Mapping[str, float],
    asia_ext: pl.DataFrame,
    config: EarlyMomentumConfig,
    price_tick: float = PRICE_TICK,
    group_col: str = _GROUP,
) -> pl.DataFrame:
    """يجمّد فلاتر الزخم/الحجم/الكسر عند ``t``. بلا ذروة مكتملة."""
    extra = {
        "direction": pl.Float64(),
        "london_atr_pts": pl.Float64(),
        "momentum_pts": pl.Float64(),
        "momentum_atr_ratio": pl.Float64(),
        "volume_ratio": pl.Float64(),
        "break_pts": pl.Float64(),
        "printed_at_entry_pts": pl.Float64(),
        "target_price": pl.Float64(),
        "stop_price": pl.Float64(),
        "target_pts": pl.Float64(),
        "risk_pts": pl.Float64(),
        "london_end_ts": pl.Int64(),
        "early_ok": pl.Boolean(),
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
    close_e = _price_arr(work, "close")
    direction = infer_direction(
        break_dir=_f64_arr(work, "proj_break_direction"),
        close=close_e,
        asia_vah=_price_arr(work, "asia_vah"),
        asia_val=_price_arr(work, "asia_val"),
    )
    dates = work[_SESSION_DATE].to_list()
    atr = np.array([atr_by_date.get(str(d), float("nan")) for d in dates], dtype=np.float64)
    asia_high = _price_arr(work, "asia_session_high")
    asia_low = _price_arr(work, "asia_session_low")
    printed = np.maximum(
        _f64_arr(work, "path_beyond_asia_ticks"),
        _f64_arr(work, "path_extreme_ticks"),
    ) / float(TICKS_PER_NQ_POINT)
    path_cols = [group_col, AVAILABILITY_TS, "close", "high", "low"]
    if _VOLUME_COL in blended.columns:
        path_cols.append(_VOLUME_COL)
    stories = work[group_col].unique().to_list()
    path = (
        blended.filter(pl.col(group_col).is_in(stories))
        .select([c for c in path_cols if c in blended.columns])
        .sort([group_col, AVAILABILITY_TS])
    )
    grouped = _path_by_story(path, group_col)
    filled = _fill_early_rows(
        work=work,
        grouped=grouped,
        direction=direction,
        atr=atr,
        asia_high=asia_high,
        asia_low=asia_low,
        printed=printed,
        config=config,
        price_tick=price_tick,
        group_col=group_col,
    )
    return work.with_columns(
        pl.Series("direction", direction),
        pl.Series("london_atr_pts", atr),
        pl.Series("momentum_pts", filled["momentum_pts"]),
        pl.Series("momentum_atr_ratio", filled["momentum_atr_ratio"]),
        pl.Series("volume_ratio", filled["volume_ratio"]),
        pl.Series("break_pts", filled["break_pts"]),
        pl.Series("printed_at_entry_pts", printed),
        pl.Series("target_price", filled["target_price"]),
        pl.Series("stop_price", filled["stop_price"]),
        pl.Series("target_pts", filled["target_pts"]),
        pl.Series("risk_pts", filled["risk_pts"]),
        pl.Series("london_end_ts", filled["london_end_ts"], dtype=pl.Int64()),
        pl.Series("early_ok", filled["ok"]),
        pl.Series("skip_reason", filled["skip"], dtype=pl.Utf8()),
    )


def _fill_early_rows(
    *,
    work: pl.DataFrame,
    grouped: Mapping[Any, pl.DataFrame],
    direction: np.ndarray,
    atr: np.ndarray,
    asia_high: np.ndarray,
    asia_low: np.ndarray,
    printed: np.ndarray,
    config: EarlyMomentumConfig,
    price_tick: float,
    group_col: str,
) -> dict[str, Any]:
    n = int(work.height)
    momentum_pts = np.full(n, np.nan)
    mom_ratio = np.full(n, np.nan)
    vol_ratio = np.full(n, np.nan)
    brk_pts = np.full(n, np.nan)
    target_price = np.full(n, np.nan)
    stop_price = np.full(n, np.nan)
    target_pts = np.full(n, np.nan)
    risk_pts = np.full(n, np.nan)
    london_end_ts = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=np.bool_)
    skip: list[str | None] = [None] * n
    t0 = work[SETUP_AVAILABILITY_TS].cast(pl.Int64).to_list()
    stories = work[group_col].to_list()
    close_e = _price_arr(work, "close")

    for i in range(n):
        london_end_ts[i] = int(vp_liquidity_session_bounds_ns(int(t0[i]))[1])
        story_path = grouped.get(stories[i])
        if story_path is None or story_path.height == 0:
            skip[i] = "no_path"
            continue
        ts = story_path[AVAILABILITY_TS].cast(pl.Int64).to_numpy()
        k = _bar_index(ts, int(t0[i]))
        close = _price_arr(story_path, "close")
        high = _price_arr(story_path, "high")
        low = _price_arr(story_path, "low")
        intensity = (
            _f64_arr(story_path, _VOLUME_COL)
            if _VOLUME_COL in story_path.columns
            else np.full(int(story_path.height), np.nan)
        )
        if k < 0:
            skip[i] = "no_path"
            continue
        momentum_pts[i] = _momentum_pts(
            close, k, float(direction[i]), int(config.momentum_bars), price_tick
        )
        if np.isfinite(atr[i]) and atr[i] > _EPS:
            mom_ratio[i] = momentum_pts[i] / float(atr[i])
        vol_ratio[i] = _volume_ratio(intensity, k, int(config.volume_bars))
        brk_pts[i] = _break_pts(
            close_px=float(close[k]),
            high_px=float(high[k]),
            low_px=float(low[k]),
            direction=float(direction[i]),
            asia_high=float(asia_high[i]),
            asia_low=float(asia_low[i]),
            price_tick=price_tick,
        )
        reason = _first_skip(
            direction=float(direction[i]),
            atr_pts=float(atr[i]),
            mom_pts=float(momentum_pts[i]),
            vol_ratio=float(vol_ratio[i]),
            brk_pts=float(brk_pts[i]),
            config=config,
        )
        if reason is not None:
            skip[i] = reason
            continue
        tgt = float(config.target_atr_frac) * float(atr[i])
        risk = float(config.stop_atr_frac) * float(atr[i])
        target_pts[i] = tgt
        risk_pts[i] = risk
        target_price[i] = float(close_e[i]) + float(direction[i]) * pts_to_price(
            tgt, price_tick=price_tick
        )
        stop_price[i] = float(close_e[i]) - float(direction[i]) * pts_to_price(
            risk, price_tick=price_tick
        )
        ok[i] = True
    return {
        "momentum_pts": momentum_pts,
        "momentum_atr_ratio": mom_ratio,
        "volume_ratio": vol_ratio,
        "break_pts": brk_pts,
        "target_price": target_price,
        "stop_price": stop_price,
        "target_pts": target_pts,
        "risk_pts": risk_pts,
        "london_end_ts": london_end_ts,
        "ok": ok,
        "skip": skip,
        "printed": printed,
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


def _past_london(window: Mapping[str, np.ndarray], k: int, london_end: float) -> bool:
    ts_arr = window.get("availability_ts")
    if (
        ts_arr is not None
        and k < int(ts_arr.size)
        and np.isfinite(london_end)
        and london_end > 0.0
        and int(ts_arr[k]) >= int(london_end)
    ):
        return True
    sess_arr = window.get(VP_LIQUIDITY_SESSION)
    if sess_arr is None or k >= int(sess_arr.size):
        return False
    sess = float(sess_arr[k])
    return bool(np.isfinite(sess) and abs(sess) > _EPS and int(sess) != _LONDON)


def _decide_exit(
    window: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
) -> tuple[int, str]:
    n = int(window["path_beyond_asia_ticks"].size)
    if n == 0:
        return 0, "empty"
    direction = float(scalars.get("direction_0", 0.0))
    target_px = float(scalars.get("target_price_0", float("nan")))
    stop_px = float(scalars.get("stop_price_0", float("nan")))
    london_end = float(scalars.get("london_end_ts_0", float("nan")))
    use_price = np.isfinite(target_px) and np.isfinite(stop_px) and abs(direction) > _EPS
    close_arr = window.get("close")
    high_arr = window.get("high")
    low_arr = window.get("low")
    for k in range(n):
        if _past_london(window, k, london_end):
            return max(k - 1, 0), "london_end"
        if use_price:
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
    return max(n - 1, 0), "london_end"


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


def _attach_lookahead_remaining(
    fires: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    holdout_cut_ts: int | None,
) -> pl.DataFrame:
    """تشخيص نظرة أمامية — لا يُستخدم كفلتر."""
    if fires.height == 0:
        return fires.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("diag_wave_peak_pts"),
            pl.lit(None, dtype=pl.Float64).alias("diag_remaining_pts"),
        )
    waves = build_wave_geometry(blended, holdout_cut_ts=holdout_cut_ts)
    if waves.height == 0 or _GROUP not in waves.columns:
        return fires.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("diag_wave_peak_pts"),
            pl.lit(None, dtype=pl.Float64).alias("diag_remaining_pts"),
        )
    peaks = waves.select(
        _GROUP,
        (pl.col("wave_peak_ticks") / TICKS_PER_NQ_POINT).alias("diag_wave_peak_pts"),
    )
    joined = fires.join(peaks, on=_GROUP, how="left")
    return joined.with_columns(
        (pl.col("diag_wave_peak_pts") - pl.col("printed_at_entry_pts")).alias("diag_remaining_pts")
    )


def _principles() -> tuple[str, ...]:
    return (
        "removable overlay — delete early_momentum_filter.py to remove",
        "not a live execution tool until OOF geometry holds",
        "science Y is unchanged; this is not a new label horizon",
        "completed-wave 20% / wave_frac / remaining-to-peak are never entry filters",
        "London ATR uses prior completed London High-Low, never today",
        "volume proxy is lf_arrival_intensity vs the previous 20 bars (no candle volume)",
        "frozen basket after the filter is 0.5/0.2 London ATR, hold until London end",
        "holdout never scored; live_predictions never used",
    )


def _empty_diagnostics(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "empty": True,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "completed_wave_peak_not_used_as_filter": True,
        "wave_frac_not_used_as_entry_filter": True,
        "holdout_scored": False,
        "live_predictions_not_used": True,
        "not_live_execution": True,
        "volume_proxy": _VOLUME_COL,
        "principles": _principles(),
    }
    base.update(extra)
    return base


def _y_rate(frame: pl.DataFrame) -> float | None:
    if frame.height == 0 or "y" not in frame.columns:
        return None
    arr = _f64_arr(frame, "y")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def run_early_momentum(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: EarlyMomentumConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> EarlyMomentumReport:
    """فلتر زخم مبكر على إطلاقات OOF ثم باسكت ATR لندن مجمّد."""
    cfg = config or EarlyMomentumConfig()
    if cfg.momentum_bars < 1 or cfg.volume_bars < 1:
        raise ValueError("lookbacks must be >= 1")
    if cfg.momentum_atr_frac <= 0.0 or cfg.volume_multiple <= 0.0 or cfg.break_pts <= 0.0:
        raise ValueError("filter thresholds must be > 0")
    if cfg.atr_days < 1:
        raise ValueError("atr_days must be >= 1")
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    empty = EarlyMomentumReport(
        trades=pl.DataFrame(),
        skipped=pl.DataFrame(),
        fires=pl.DataFrame(),
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
        progress.op("early_momentum causal filters at t")
    atr_map = causal_london_atr_pts(compute_london_ranges(blended), window=cfg.atr_days)
    geo = attach_early_momentum_at_t(
        fires,
        blended,
        atr_by_date=atr_map,
        asia_ext=asia_session_extremes(blended),
        config=cfg,
    )
    geo = _attach_lookahead_remaining(geo, blended, holdout_cut_ts=holdout_cut_ts)
    traded = geo.filter(pl.col("early_ok"))
    skipped = geo.filter(~pl.col("early_ok"))
    summary_base: dict[str, Any] = {
        "layer_id": LAYER_ID,
        "n_fires": int(geo.height),
        "n_traded": 0,
        "n_skipped": int(skipped.height),
        "require_momentum": bool(cfg.require_momentum),
        "require_volume": bool(cfg.require_volume),
        "require_break": bool(cfg.require_break),
        "momentum_atr_frac": float(cfg.momentum_atr_frac),
        "volume_multiple": float(cfg.volume_multiple),
        "break_pts": float(cfg.break_pts),
        "y_rate_all": _y_rate(geo),
        "y_rate_traded": None,
        "printed_pts_median_all": median_mean(geo, "printed_at_entry_pts")[0],
        "printed_pts_median_traded": None,
        "diag_remaining_pts_median_all": median_mean(geo, "diag_remaining_pts")[0],
        "diag_remaining_pts_median_traded": None,
        "take_n": 0,
        "stop_n": 0,
        "take_rate": None,
        "stop_rate": None,
        "net_pts_median": None,
        "net_pts_mean": None,
        "target_pts_median": None,
        "risk_pts_median": None,
    }
    if traded.height == 0:
        return EarlyMomentumReport(
            trades=pl.DataFrame(),
            skipped=skipped,
            fires=geo,
            summaries=pl.DataFrame([summary_base]),
            diagnostics=_empty_diagnostics(
                empty=False,
                n_fires=int(geo.height),
                n_traded=0,
                n_skipped=int(skipped.height),
                skip_reasons=_reason_counts(skipped, "skip_reason"),
                funnel=_funnel(geo),
                printed_pts_median_all=summary_base["printed_pts_median_all"],
            ),
        )
    trades = walk_hold_windows(
        traded,
        blended,
        max_hold_bars=cfg.max_hold_bars,
        path_cols=_PATH_COLS,
        decide_exit=_decide_exit,
        round_trip_cost_pts=cfg.round_trip_cost_pts,
    )
    extras = traded.with_row_index("entry_i").select(
        "entry_i",
        "target_price",
        "stop_price",
        "target_pts",
        "risk_pts",
        "direction",
        "london_atr_pts",
        "printed_at_entry_pts",
        "momentum_pts",
        "volume_ratio",
        "break_pts",
        "diag_remaining_pts",
    )
    if trades.height:
        trades = trades.join(extras, on="entry_i", how="left")
        trades = _apply_basket_pnl(trades, cost_pts=cfg.round_trip_cost_pts)
    n_traded = int(trades.height)
    take_n = int((trades["exit_reason"] == "take").sum()) if n_traded else 0
    stop_n = int((trades["exit_reason"] == "stop").sum()) if n_traded else 0
    net_med, net_mean = median_mean(trades, "net_pts")
    tgt_med, _tgt_mean = median_mean(trades, "target_pts")
    risk_med, _risk_mean = median_mean(trades, "risk_pts")
    printed_med, _pmean = median_mean(traded, "printed_at_entry_pts")
    remain_med, _rmean = median_mean(traded, "diag_remaining_pts")
    summaries = pl.DataFrame(
        [
            {
                **summary_base,
                "n_traded": n_traded,
                "y_rate_traded": _y_rate(traded),
                "printed_pts_median_traded": printed_med,
                "diag_remaining_pts_median_traded": remain_med,
                "take_n": take_n,
                "stop_n": stop_n,
                "take_rate": (take_n / n_traded) if n_traded else None,
                "stop_rate": (stop_n / n_traded) if n_traded else None,
                "net_pts_median": net_med,
                "net_pts_mean": net_mean,
                "target_pts_median": tgt_med,
                "risk_pts_median": risk_med,
            }
        ]
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "completed_wave_peak_not_used_as_filter": True,
        "wave_frac_not_used_as_entry_filter": True,
        "holdout_scored": False,
        "live_predictions_not_used": True,
        "not_live_execution": True,
        "volume_proxy": _VOLUME_COL,
        "n_fires": int(geo.height),
        "n_traded": n_traded,
        "n_skipped": int(skipped.height),
        "skip_reasons": _reason_counts(skipped, "skip_reason"),
        "exit_reasons": _reason_counts(trades, "exit_reason"),
        "funnel": _funnel(geo),
        "y_rate_all": _y_rate(geo),
        "y_rate_traded": _y_rate(traded),
        "printed_pts_median_all": summary_base["printed_pts_median_all"],
        "printed_pts_median_traded": printed_med,
        "diag_remaining_pts_median_all": summary_base["diag_remaining_pts_median_all"],
        "diag_remaining_pts_median_traded": remain_med,
        "take_rate": (take_n / n_traded) if n_traded else None,
        "stop_rate": (stop_n / n_traded) if n_traded else None,
        "net_pts_median": net_med,
        "net_pts_mean": net_mean,
        "target_pts_median": tgt_med,
        "risk_pts_median": risk_med,
        "principles": _principles(),
    }
    return EarlyMomentumReport(
        trades=trades,
        skipped=skipped,
        fires=geo,
        summaries=summaries,
        diagnostics=diagnostics,
    )


def _funnel(geo: pl.DataFrame) -> dict[str, int]:
    if geo.height == 0:
        return {}
    return {
        "n_fires": int(geo.height),
        "n_pass": int(geo["early_ok"].sum()),
        **_reason_counts(geo.filter(~pl.col("early_ok")), "skip_reason"),
    }


def run_early_momentum_grid(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    variants: Sequence[tuple[bool, bool, bool]] = (
        (True, True, True),
        (True, False, True),
        (True, True, False),
        (True, False, False),
    ),
    config: EarlyMomentumConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """صمود مسبّق: إسقاط الحجم أو الكسر — ليس بحث عتبات."""
    base = config or EarlyMomentumConfig()
    rows: list[pl.DataFrame] = []
    for use_mom, use_vol, use_break in variants:
        if progress is not None:
            progress.op(f"early_momentum mom={use_mom} vol={use_vol} brk={use_break}")
        cfg = EarlyMomentumConfig(
            min_p=base.min_p,
            expansion_start_ticks=base.expansion_start_ticks,
            atr_days=base.atr_days,
            momentum_bars=base.momentum_bars,
            momentum_atr_frac=base.momentum_atr_frac,
            volume_bars=base.volume_bars,
            volume_multiple=base.volume_multiple,
            break_pts=base.break_pts,
            require_momentum=bool(use_mom),
            require_volume=bool(use_vol),
            require_break=bool(use_break),
            target_atr_frac=base.target_atr_frac,
            stop_atr_frac=base.stop_atr_frac,
            max_hold_bars=base.max_hold_bars,
            round_trip_cost_pts=base.round_trip_cost_pts,
            holdout_months=base.holdout_months,
        )
        report = run_early_momentum(
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


def render_early_momentum_markdown(report: EarlyMomentumReport) -> str:
    d = report.diagnostics
    lines = [
        "# Early-momentum entry filter (removable layer H) — OOF test",
        "",
        "Same causal OOF fire at `t`, same 30-second states. **Not** a new science Y.",
        "Completed-wave 20% / `wave_frac` / remaining-to-peak are **not** entry filters.",
        "Gates known at `t`: momentum vs London ATR, arrival-intensity burst,",
        "and a real Asia-session-extreme break (>2 pts).",
        "Frozen basket after the filter: 0.5 / 0.2 London ATR, hold until London end.",
        "Delete this layer without touching science Y.",
        "",
        f"- layer_id={d.get('layer_id')} · removable={d.get('removable_layer')}",
        f"- not_live_execution={d.get('not_live_execution')}",
        f"- wave_frac_not_used_as_entry_filter={d.get('wave_frac_not_used_as_entry_filter')}",
        f"- volume_proxy={d.get('volume_proxy')}",
        f"- fires={d.get('n_fires')} · traded={d.get('n_traded')} · skipped={d.get('n_skipped')}",
        f"- printed pts median all={d.get('printed_pts_median_all')} · "
        f"traded={d.get('printed_pts_median_traded')}",
        f"- diag remaining pts (look-ahead) all={d.get('diag_remaining_pts_median_all')} · "
        f"traded={d.get('diag_remaining_pts_median_traded')}",
        f"- y_rate all={d.get('y_rate_all')} · traded={d.get('y_rate_traded')}",
        f"- take_rate={d.get('take_rate')} · stop_rate={d.get('stop_rate')}",
        f"- target pts median={d.get('target_pts_median')} · "
        f"risk pts median={d.get('risk_pts_median')}",
        f"- net median={d.get('net_pts_median')} · net mean={d.get('net_pts_mean')}",
        f"- skip_reasons={d.get('skip_reasons')}",
        f"- exit_reasons={d.get('exit_reasons')}",
        f"- funnel={d.get('funnel')}",
        "",
        "## Principles",
        "",
    ]
    for item in d.get("principles", ()):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_early_momentum_report(report: EarlyMomentumReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.trades.height:
        report.trades.write_parquet(out / "early_momentum_trades.parquet")
    if report.skipped.height:
        report.skipped.write_parquet(out / "early_momentum_skipped.parquet")
    if report.summaries.height:
        report.summaries.write_parquet(out / "early_momentum_summaries.parquet")
    payload = {
        "diagnostics": jsonable(report.diagnostics),
        "holdout_scored": False,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "wave_frac_not_used_as_entry_filter": True,
        "not_live_execution": True,
    }
    (out / "early_momentum.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "EARLY_MOMENTUM.md").write_text(render_early_momentum_markdown(report), encoding="utf-8")
    return out


def run_early_momentum_from_period_dir(
    period_dir: Path | str,
    *,
    config: EarlyMomentumConfig | None = None,
    progress: ProgressLike | None = None,
) -> EarlyMomentumReport:
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(period_dir)
    return run_early_momentum(
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
    "EarlyMomentumConfig",
    "EarlyMomentumReport",
    "asia_session_extremes",
    "attach_early_momentum_at_t",
    "attach_session_dates",
    "causal_london_atr_pts",
    "compute_london_ranges",
    "infer_direction",
    "price_to_pts",
    "pts_to_price",
    "render_early_momentum_markdown",
    "run_early_momentum",
    "run_early_momentum_from_period_dir",
    "run_early_momentum_grid",
    "write_early_momentum_report",
]
