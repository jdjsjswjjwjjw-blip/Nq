"""طبقة هـ: باسكت هيكل — هدف سيولة عند ``t`` ووقف من قاع آخر البراميل.

ليست معادلة 1:4 على VP. الهدف = أقرب مستوى سيولة معروف عند ``t``
(قمة/قاع محلي مؤكَّد، VAH/VAL، POC سابق). الوقف = أسفل أدنى قاع
(أو أعلى أعلى قمة للبيع) خلال آخر ``lookback_bars`` بارميل، لا ``هدف/4``.

نسبة العائد/المخاطر ناتج الهيكل، ليست قيدًا. ليست أداة تنفيذ حيّة
حتى يصمد الاختبار على OOF.

لا تغيّر Y العلمي. احذف هذا الملف + السكربت + ``_write_structure_basket`` للإزالة.
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

LAYER_ID = "structure_basket"
_GROUP = "_behavior_story_run"
_EPS = 1e-9
_PATH_COLS = (
    "close",
    "high",
    "low",
    "target_price",
    "stop_price",
    "direction",
)
PRICE_TICK = float(round(0.25 / PRICE_SCALE))
DEFAULT_LOOKBACKS: tuple[int, ...] = (5, 8, 10)

# أقرب سيولة أمام الاتجاه. القمة المحلية أولًا ثم VAH ثم POC السابق.
LONG_LEVELS: tuple[str, ...] = (
    "local_high",
    "asia_vah",
    "decision_vah",
    "composite_vah",
    "asia_poc",
    "decision_poc",
    "composite_poc",
)
SHORT_LEVELS: tuple[str, ...] = (
    "local_low",
    "asia_val",
    "decision_val",
    "composite_val",
    "asia_poc",
    "decision_poc",
    "composite_poc",
)


@dataclass(frozen=True, slots=True)
class StructureBasketConfig:
    """عتبات ثابتة غير مُقدَّرة على العينة. الـ lookback في عائلة 5–10."""

    min_p: float = 0.5
    expansion_start_ticks: float = 16.0
    lookback_bars: int = 8
    min_stop_bars: int = 3
    stop_buffer_ticks: float = 1.0
    swing_radius: int = 2
    min_ahead_ticks: float = 16.0  # 4 نقاط — أرضية ضوضاء لا 1:4
    max_hold_bars: int = 120
    round_trip_cost_pts: float = 0.75
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class StructureBasketReport:
    trades: pl.DataFrame
    skipped: pl.DataFrame
    summaries: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _f64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    if frame.height == 0 or name not in frame.columns:
        return np.zeros(int(frame.height), dtype=np.float64)
    return frame[name].cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64, copy=False)


def _i64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame[name].cast(pl.Int64).to_numpy().astype(np.int64, copy=False)


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
    """+1 كسر أعلى آسيا، −1 أدنى VAL. لا نظرة أمامية."""
    out = np.sign(break_dir).astype(np.float64)
    unk = np.abs(out) < _EPS
    out[unk & np.isfinite(close) & np.isfinite(asia_vah) & (close > asia_vah)] = 1.0
    out[unk & np.isfinite(close) & np.isfinite(asia_val) & (close < asia_val)] = -1.0
    return out


def _swing_events(
    high: np.ndarray,
    low: np.ndarray,
    *,
    radius: int,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """قمم/قيعان مؤكَّدة: التأكيد عند ``i + radius`` (سببي عند t)."""
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    n = int(high.size)
    if radius < 1 or n < 2 * radius + 1:
        return highs, lows
    for i in range(radius, n - radius):
        h = float(high[i])
        lo = float(low[i])
        left_h = high[i - radius : i]
        right_h = high[i + 1 : i + radius + 1]
        left_l = low[i - radius : i]
        right_l = low[i + 1 : i + radius + 1]
        confirm = i + radius
        if (
            np.isfinite(h)
            and np.all(np.isfinite(left_h))
            and np.all(np.isfinite(right_h))
            and bool(np.all(h > left_h))
            and bool(np.all(h > right_h))
        ):
            highs.append((confirm, h))
        if (
            np.isfinite(lo)
            and np.all(np.isfinite(left_l))
            and np.all(np.isfinite(right_l))
            and bool(np.all(lo < left_l))
            and bool(np.all(lo < right_l))
        ):
            lows.append((confirm, lo))
    return highs, lows


def _nearest_ahead(
    *,
    close: float,
    direction: float,
    candidates: Sequence[tuple[str, float]],
    min_ahead_ticks: float,
    price_tick: float,
) -> tuple[str | None, float, float]:
    best_name: str | None = None
    best_price = float("nan")
    best_ahead = np.inf
    for name, price in candidates:
        if not np.isfinite(price) or not np.isfinite(close):
            continue
        ahead = (
            (price - close) / price_tick
            if direction > 0.0
            else (close - price) / price_tick
            if direction < 0.0
            else float("nan")
        )
        if np.isfinite(ahead) and ahead >= float(min_ahead_ticks) and ahead < best_ahead:
            best_ahead = float(ahead)
            best_name = name
            best_price = float(price)
    if best_name is None:
        return None, float("nan"), float("nan")
    return best_name, best_price, best_ahead


def _level_candidates(
    entries: pl.DataFrame,
    i: int,
    *,
    direction: float,
    local_high: float,
    local_low: float,
) -> list[tuple[str, float]]:
    names = LONG_LEVELS if direction > 0.0 else SHORT_LEVELS if direction < 0.0 else ()
    out: list[tuple[str, float]] = []
    for name in names:
        if name == "local_high":
            price = local_high
        elif name == "local_low":
            price = local_low
        elif name in entries.columns:
            raw = entries[name][i]
            price = float("nan") if raw is None else float(raw)
            if np.isfinite(price) and abs(price) < _EPS:
                price = float("nan")
        else:
            price = float("nan")
        out.append((name, price))
    return out


StoryCache = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, float]],
    list[tuple[int, float]],
]


def _validate_structure_cfg(
    *,
    lookback_bars: int,
    min_stop_bars: int,
    stop_buffer_ticks: float,
    swing_radius: int,
    min_ahead_ticks: float,
    price_tick: float,
) -> None:
    if lookback_bars < 1:
        raise ValueError("lookback_bars must be >= 1")
    if min_stop_bars < 1:
        raise ValueError("min_stop_bars must be >= 1")
    if stop_buffer_ticks < 0.0:
        raise ValueError("stop_buffer_ticks must be >= 0")
    if swing_radius < 1:
        raise ValueError("swing_radius must be >= 1")
    if min_ahead_ticks < 1.0:
        raise ValueError("min_ahead_ticks must be >= 1")
    if price_tick <= 0.0:
        raise ValueError("price_tick must be > 0")


def _story_cache(path: pl.DataFrame, *, radius: int) -> StoryCache:
    high_a = _price_arr(path, "high")
    low_a = _price_arr(path, "low")
    ts_a = _i64_arr(path, AVAILABILITY_TS)
    swings_h, swings_l = _swing_events(high_a, low_a, radius=radius)
    return ts_a, high_a, low_a, swings_h, swings_l


def _structure_stop(
    *,
    direction: float,
    close_px: float,
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    idx: int,
    buffer_ticks: float,
    price_tick: float,
) -> tuple[float, float]:
    if direction > 0.0:
        trough = float(np.nanmin(low[start : idx + 1]))
        if not np.isfinite(trough):
            return float("nan"), float("nan")
        stop_px = trough - float(buffer_ticks) * price_tick
        return stop_px, (close_px - stop_px) / price_tick
    peak = float(np.nanmax(high[start : idx + 1]))
    if not np.isfinite(peak):
        return float("nan"), float("nan")
    stop_px = peak + float(buffer_ticks) * price_tick
    return stop_px, (stop_px - close_px) / price_tick


def _nearest_confirmed_swing(
    swings: Sequence[tuple[int, float]],
    *,
    idx: int,
    close_px: float,
    direction: float,
    min_ahead_ticks: float,
    price_tick: float,
) -> float:
    best_price = float("nan")
    best_ahead = np.inf
    for confirm, price in swings:
        if confirm > idx:
            break
        ahead = (
            (price - close_px) / price_tick if direction > 0.0 else (close_px - price) / price_tick
        )
        if ahead >= float(min_ahead_ticks) and ahead < best_ahead:
            best_ahead = ahead
            best_price = price
    return best_price


def attach_structure_at_t(
    entries: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    lookback_bars: int,
    min_stop_bars: int,
    stop_buffer_ticks: float,
    swing_radius: int,
    min_ahead_ticks: float,
    price_tick: float = PRICE_TICK,
    group_col: str = _GROUP,
) -> pl.DataFrame:
    """يجمّد الهدف والوقف من الهيكل المعروف عند ``t``. لا ``هدف/4``."""
    extra = {
        "direction": pl.Float64(),
        "target_name": pl.Utf8(),
        "target_price": pl.Float64(),
        "target_ticks": pl.Float64(),
        "risk_ticks": pl.Float64(),
        "stop_price": pl.Float64(),
        "rr_multiple": pl.Float64(),
        "stop_lookback_bars": pl.Int64(),
        "structure_ok": pl.Boolean(),
        "skip_reason": pl.Utf8(),
    }
    n = int(entries.height)
    if n == 0:
        return entries.with_columns(
            pl.lit(None, dtype=dtype).alias(name) for name, dtype in extra.items()
        )
    _validate_structure_cfg(
        lookback_bars=lookback_bars,
        min_stop_bars=min_stop_bars,
        stop_buffer_ticks=stop_buffer_ticks,
        swing_radius=swing_radius,
        min_ahead_ticks=min_ahead_ticks,
        price_tick=price_tick,
    )

    close_e = _price_arr(entries, "close")
    asia_vah_e = _price_arr(entries, "asia_vah")
    asia_val_e = _price_arr(entries, "asia_val")
    break_dir = _f64_arr(entries, "proj_break_direction")
    direction = infer_direction(
        break_dir=break_dir, close=close_e, asia_vah=asia_vah_e, asia_val=asia_val_e
    )
    t0 = _i64_arr(entries, SETUP_AVAILABILITY_TS)
    story_vals = entries[group_col].to_list()
    need = [group_col, AVAILABILITY_TS, "close", "high", "low"]
    present = [c for c in need if c in blended.columns]
    stories = entries[group_col].unique().to_list()
    have_path = (
        bool(present) and group_col in blended.columns and AVAILABILITY_TS in blended.columns
    )
    path = (
        blended.filter(pl.col(group_col).is_in(stories))
        .select(present)
        .sort([group_col, AVAILABILITY_TS])
        if have_path
        else blended.head(0)
    )
    grouped = _path_by_story(path, group_col) if path.height else {}
    cache: dict[Any, StoryCache] = {}

    target_name: list[str | None] = [None] * n
    target_price = np.full(n, np.nan, dtype=np.float64)
    target_ticks = np.full(n, np.nan, dtype=np.float64)
    risk_ticks = np.full(n, np.nan, dtype=np.float64)
    stop_price = np.full(n, np.nan, dtype=np.float64)
    rr = np.full(n, np.nan, dtype=np.float64)
    used_lookback = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=np.bool_)
    skip: list[str | None] = [None] * n

    for i, story in enumerate(story_vals):
        if abs(float(direction[i])) < _EPS:
            skip[i] = "no_direction"
            continue
        story_path = grouped.get(story)
        if story_path is None or story_path.height == 0:
            skip[i] = "no_path"
            continue
        if story not in cache:
            cache[story] = _story_cache(story_path, radius=swing_radius)
        reason, name, tgt_px, ahead, risk, stop_px, rr_i, n_stop = _entry_structure(
            entries=entries,
            i=i,
            direction=float(direction[i]),
            close_px=float(close_e[i]),
            t0=int(t0[i]),
            cache=cache[story],
            lookback_bars=lookback_bars,
            min_stop_bars=min_stop_bars,
            stop_buffer_ticks=stop_buffer_ticks,
            min_ahead_ticks=min_ahead_ticks,
            price_tick=price_tick,
        )
        used_lookback[i] = int(n_stop)
        skip[i] = reason
        if reason is not None or name is None:
            continue
        target_name[i] = name
        target_price[i] = tgt_px
        target_ticks[i] = ahead
        risk_ticks[i] = risk
        stop_price[i] = stop_px
        rr[i] = rr_i
        ok[i] = True

    return entries.with_columns(
        pl.Series("direction", direction),
        pl.Series("target_name", target_name, dtype=pl.Utf8()),
        pl.Series("target_price", target_price),
        pl.Series("target_ticks", target_ticks),
        pl.Series("risk_ticks", risk_ticks),
        pl.Series("stop_price", stop_price),
        pl.Series("rr_multiple", rr),
        pl.Series("stop_lookback_bars", used_lookback),
        pl.Series("structure_ok", ok),
        pl.Series("skip_reason", skip, dtype=pl.Utf8()),
    )


def _entry_structure(
    *,
    entries: pl.DataFrame,
    i: int,
    direction: float,
    close_px: float,
    t0: int,
    cache: StoryCache,
    lookback_bars: int,
    min_stop_bars: int,
    stop_buffer_ticks: float,
    min_ahead_ticks: float,
    price_tick: float,
) -> tuple[str | None, str | None, float, float, float, float, float, int]:
    ts_a, high_a, low_a, swings_h, swings_l = cache
    idx = int(np.searchsorted(ts_a, t0, side="right") - 1)
    nan = float("nan")
    if idx < 0:
        return "no_path", None, nan, nan, nan, nan, nan, 0
    start = max(0, idx - int(lookback_bars) + 1)
    n_stop = idx - start + 1
    stop_px, risk = _structure_stop(
        direction=direction,
        close_px=close_px,
        high=high_a,
        low=low_a,
        start=start,
        idx=idx,
        buffer_ticks=stop_buffer_ticks,
        price_tick=price_tick,
    )
    if n_stop < int(min_stop_bars) or not np.isfinite(risk) or risk < 1.0:
        return "no_structure_stop", None, nan, nan, nan, nan, nan, n_stop
    local_high = _nearest_confirmed_swing(
        swings_h,
        idx=idx,
        close_px=close_px,
        direction=1.0,
        min_ahead_ticks=min_ahead_ticks,
        price_tick=price_tick,
    )
    local_low = _nearest_confirmed_swing(
        swings_l,
        idx=idx,
        close_px=close_px,
        direction=-1.0,
        min_ahead_ticks=min_ahead_ticks,
        price_tick=price_tick,
    )
    name, tgt_px, ahead = _nearest_ahead(
        close=close_px,
        direction=direction,
        candidates=_level_candidates(
            entries,
            i,
            direction=direction,
            local_high=local_high,
            local_low=local_low,
        ),
        min_ahead_ticks=min_ahead_ticks,
        price_tick=price_tick,
    )
    if name is None:
        return "no_level_ahead", None, nan, nan, nan, nan, nan, n_stop
    rr = ahead / risk if risk > 0.0 else nan
    return None, name, tgt_px, ahead, risk, stop_px, rr, n_stop


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


def _decide_structure_exit(
    window: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
) -> tuple[int, str]:
    n = int(window["path_beyond_asia_ticks"].size)
    if n == 0:
        return 0, "empty"
    direction = float(scalars.get("direction_0", 0.0))
    target_px = float(scalars.get("target_price_0", float("nan")))
    stop_px = float(scalars.get("stop_price_0", float("nan")))
    close_arr = window.get("close")
    high_arr = window.get("high")
    low_arr = window.get("low")
    use_price = np.isfinite(target_px) and np.isfinite(stop_px) and abs(direction) > _EPS
    if use_price:
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
    """PnL الباسكت: ملء الهدف/الوقف، لا بقايا مسار ما بعد آسيا."""
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


def _empty_diagnostics(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "empty": True,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "chart_timeframe_unchanged": True,
        "holdout_scored": False,
        "completed_wave_peak_not_used": True,
        "wave_frac_not_used": True,
        "live_predictions_not_used": True,
        "not_fixed_1_to_4": True,
        "stop_is_structure_low_not_target_over_rr": True,
        "not_live_execution": True,
        "levels_frozen_at_t": True,
        "principles": _principles(),
    }
    base.update(extra)
    return base


def _principles() -> tuple[str, ...]:
    return (
        "removable overlay — delete structure_basket.py to remove",
        "not a live execution tool; OOF geometry test only",
        "chart timeframe is unchanged; hold cap is N bars on the 30s blended clock",
        "science Y is unchanged; this is not a new label horizon",
        "target is nearest known liquidity at t: confirmed local swing, VAH/VAL, prior POC",
        "stop is below the lowest low (or above highest high) of the last 5–10 bars, not target/4",
        "reward/risk is an outcome of structure, not a constraint",
        "completed-wave peak / remaining-to-peak are never used",
        "holdout never scored; live_predictions never used",
    )


def run_structure_basket(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: StructureBasketConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> StructureBasketReport:
    """باسكت هيكل بعد إطلاق OOF — بلا 1:4 وبلا holdout."""
    cfg = config or StructureBasketConfig()
    if cfg.lookback_bars < 1:
        raise ValueError("lookback_bars must be >= 1")
    if cfg.min_ahead_ticks < 1.0:
        raise ValueError("min_ahead_ticks must be >= 1")
    if cfg.max_hold_bars < 1:
        raise ValueError("max_hold_bars must be >= 1")
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    empty = StructureBasketReport(
        trades=pl.DataFrame(),
        skipped=pl.DataFrame(),
        summaries=pl.DataFrame(),
        diagnostics=_empty_diagnostics(lookback_bars=int(cfg.lookback_bars)),
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
    geo = attach_structure_at_t(
        fires,
        blended,
        lookback_bars=cfg.lookback_bars,
        min_stop_bars=cfg.min_stop_bars,
        stop_buffer_ticks=cfg.stop_buffer_ticks,
        swing_radius=cfg.swing_radius,
        min_ahead_ticks=cfg.min_ahead_ticks,
    )
    traded = geo.filter(pl.col("structure_ok"))
    skipped = geo.filter(~pl.col("structure_ok"))
    summary_base = {
        "layer_id": LAYER_ID,
        "n_fires": int(geo.height),
        "n_traded": 0,
        "n_skipped": int(skipped.height),
        "lookback_bars": int(cfg.lookback_bars),
        "stop_buffer_ticks": float(cfg.stop_buffer_ticks),
        "min_ahead_ticks": float(cfg.min_ahead_ticks),
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
        "theoretical_be_rate": None,
    }
    if traded.height == 0:
        return StructureBasketReport(
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
                lookback_bars=int(cfg.lookback_bars),
            ),
        )

    trades = walk_hold_windows(
        traded,
        blended,
        max_hold_bars=cfg.max_hold_bars,
        path_cols=_PATH_COLS,
        decide_exit=_decide_structure_exit,
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
        "stop_lookback_bars",
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
        "wave_frac_not_used": True,
        "live_predictions_not_used": True,
        "not_fixed_1_to_4": True,
        "stop_is_structure_low_not_target_over_rr": True,
        "not_live_execution": True,
        "levels_frozen_at_t": True,
        "hold_horizon_is_bars_not_label_window": True,
        "lookback_bars": int(cfg.lookback_bars),
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
        "theoretical_be_rate": be,
        "principles": _principles(),
    }
    return StructureBasketReport(
        trades=trades,
        skipped=skipped,
        summaries=summaries,
        diagnostics=diagnostics,
    )


def run_structure_lookback_grid(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    config: StructureBasketConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """نفس العتبات على lookback 5/8/10 — ليس بحثًا عن الأفضل."""
    base = config or StructureBasketConfig()
    rows: list[pl.DataFrame] = []
    for lookback in lookbacks:
        if progress is not None:
            progress.op(f"structure_basket lookback={lookback}")
        cfg = StructureBasketConfig(
            min_p=base.min_p,
            expansion_start_ticks=base.expansion_start_ticks,
            lookback_bars=int(lookback),
            min_stop_bars=base.min_stop_bars,
            stop_buffer_ticks=base.stop_buffer_ticks,
            swing_radius=base.swing_radius,
            min_ahead_ticks=base.min_ahead_ticks,
            max_hold_bars=base.max_hold_bars,
            round_trip_cost_pts=base.round_trip_cost_pts,
            holdout_months=base.holdout_months,
        )
        report = run_structure_basket(
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


def render_structure_basket_markdown(report: StructureBasketReport) -> str:
    d = report.diagnostics
    lines = [
        "# Structure basket overlay (removable layer E) — OOF test, not live execution",
        "",
        "Same causal OOF fire at `t`, same 30-second states. **Not** a chart",
        "timeframe change, **not** a new science Y, and **not** a 1:4 VP basket.",
        "Target = nearest known liquidity at `t` (confirmed local swing, VAH/VAL,",
        "prior POC). Stop = below the lowest low of the last 5–10 bars",
        "(above the highest high for shorts), not `target / 4`.",
        "Reward/risk is whatever structure prints. Do not treat this as a live",
        "execution tool unless the OOF geometry holds.",
        "Delete this layer without touching science Y.",
        "",
        f"- layer_id={d.get('layer_id')} · removable={d.get('removable_layer')}",
        f"- not_live_execution={d.get('not_live_execution')}",
        f"- not_fixed_1_to_4={d.get('not_fixed_1_to_4')}",
        f"- lookback_bars={d.get('lookback_bars')}",
        f"- fires={d.get('n_fires')} · traded={d.get('n_traded')} · skipped={d.get('n_skipped')}",
        f"- take_rate={d.get('take_rate')} · stop_rate={d.get('stop_rate')}",
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


def write_structure_basket_report(report: StructureBasketReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.trades.height:
        report.trades.write_parquet(out / "structure_basket_trades.parquet")
    if report.skipped.height:
        report.skipped.write_parquet(out / "structure_basket_skipped.parquet")
    if report.summaries.height:
        report.summaries.write_parquet(out / "structure_basket_summaries.parquet")
    payload = {
        "diagnostics": jsonable(report.diagnostics),
        "holdout_scored": False,
        "removable_layer": True,
        "chart_timeframe_unchanged": True,
        "does_not_modify_science_y": True,
        "not_live_execution": True,
        "not_fixed_1_to_4": True,
    }
    (out / "structure_basket.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "STRUCTURE.md").write_text(render_structure_basket_markdown(report), encoding="utf-8")
    return out


def run_structure_basket_from_period_dir(
    period_dir: Path | str,
    *,
    config: StructureBasketConfig | None = None,
    progress: ProgressLike | None = None,
) -> StructureBasketReport:
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(period_dir)
    return run_structure_basket(
        labeled,
        blended,
        config=config,
        oof_availability_ts=oof_timestamps(oof),
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=progress,
    )


__all__ = [
    "DEFAULT_LOOKBACKS",
    "LAYER_ID",
    "LONG_LEVELS",
    "PRICE_TICK",
    "SHORT_LEVELS",
    "StructureBasketConfig",
    "StructureBasketReport",
    "attach_structure_at_t",
    "infer_direction",
    "render_structure_basket_markdown",
    "run_structure_basket",
    "run_structure_basket_from_period_dir",
    "run_structure_lookback_grid",
    "write_structure_basket_report",
]
