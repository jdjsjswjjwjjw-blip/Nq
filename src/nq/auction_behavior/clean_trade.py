"""صفقة نظيفة: هدف ديناميكي أكبر من التراجع، بوحدة ATR لندن.

``y_clean = 1`` إذا خلال 50 برميلًا (25 دقيقة على 30ث) بعد onset المسار:

* الامتداد ``MFE ≥ 0.15 × London_ATR``
* التراجع ``MAE < 0.08 × London_ATR``

ليس هدفًا ثابتًا بالنقاط، ولا يستبدل ``y_path_further_beyond`` /
``y_extend_5pts_25min`` / ``y_phase_extend``. ATR لندن سببي: مدى جلسات
لندن المكتملة السابقة فقط (انظر ``prior_london_atr14``).

تشخيص الخروج (أول لمس للهدف/الوقف) منفصل عن التسمية: لا يُدرَّب عليه
ولا يُصدَّر كطبقة تنفيذ حيّة.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.auction_behavior.phase_extend import prior_london_atr14
from nq.auction_behavior.realized_path import (
    _BEYOND,
    _BREAK,
    _ONSET,
    _RETEST,
    Y_CLEAN,
    _active,
    _binary_schema,
    _col_array,
    _onset_mask,
    _price_to_points,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_causal_order

CLEAN_TRADE_TARGETS = (Y_CLEAN,)
CLEAN_HORIZON_BARS = 50
CLEAN_TARGET_ATR_FRAC = 0.15
CLEAN_MAE_ATR_FRAC = 0.08
#: تكلفة ذهاب-إياب بالنقاط — تشخيص فقط، ليست جزءًا من Y.
CLEAN_ROUND_TRIP_COST_PTS = 0.75
CLEAN_OPERATING_P = 0.5
_EPS = 1e-12
_EXIT_TARGET = "target"
_EXIT_STOP = "stop"
_EXIT_TIME = "time"
_EXIT_NONE = "none"


def _infer_direction(
    *,
    i: int,
    brk_dir: np.ndarray,
    close_pts: np.ndarray,
    asia_vah: np.ndarray,
    asia_val: np.ndarray,
    has_asia: bool,
) -> float:
    direction = float(brk_dir[i])
    if abs(direction) >= _ONSET:
        return direction
    if not has_asia:
        return 0.0
    if close_pts[i] >= asia_vah[i] > 0.0:
        return 1.0
    if asia_val[i] > 0.0 and close_pts[i] <= asia_val[i]:
        return -1.0
    return 0.0


def _window_is_clean(
    *,
    direction: float,
    mfe_up: float,
    mae_up: float,
    mfe_down: float,
    mae_down: float,
    target_pts: float,
    stop_pts: float,
) -> bool:
    up = mfe_up >= target_pts and mae_up < stop_pts
    down = mfe_down >= target_pts and mae_down < stop_pts
    if direction > _ONSET:
        return bool(up)
    if direction < -_ONSET:
        return bool(down)
    return bool(up or down)


def _scan_clean_onsets(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    window: int,
    target_atr_frac: float,
    mae_atr_frac: float,
    group_col: str | None,
    progress: ProgressLike | None,
) -> list[dict[str, object]]:
    """يمسح onsets: تسمية النافذة + أول لمس هدف/وقف (للتشخيص)."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if target_atr_frac <= 0.0 or mae_atr_frac <= 0.0:
        raise ValueError("ATR fractions must be > 0")
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return []
    if group_col is not None and group_col not in frame.columns:
        raise ValueError(f"group_col is missing: {group_col}")

    work = frame.sort(AVAILABILITY_TS)
    n = work.height
    ts = work[AVAILABILITY_TS].to_numpy().astype(np.int64)
    assert_causal_order(ts)
    groups = (
        work[group_col].fill_null(-1).to_numpy().astype(np.int64)
        if group_col is not None
        else np.zeros(n, dtype=np.int64)
    )
    beyond = _col_array(work, _BEYOND, n)
    brk = _active(_col_array(work, _BREAK, n))
    retest = _active(_col_array(work, _RETEST, n))
    onset = _onset_mask(beyond, brk, retest, groups)
    close_pts = np.array(
        [_price_to_points(v) for v in _col_array(work, "close", n)], dtype=np.float64
    )
    high_pts = np.array(
        [_price_to_points(v) for v in _col_array(work, "high", n)], dtype=np.float64
    )
    low_pts = np.array([_price_to_points(v) for v in _col_array(work, "low", n)], dtype=np.float64)
    atr_raw = prior_london_atr14(work)
    atr_pts = np.array([_price_to_points(float(v)) for v in atr_raw], dtype=np.float64)
    brk_dir = _col_array(work, "proj_break_direction", n)
    asia_vah = np.array(
        [_price_to_points(v) for v in _col_array(work, "asia_vah", n)], dtype=np.float64
    )
    asia_val = np.array(
        [_price_to_points(v) for v in _col_array(work, "asia_val", n)], dtype=np.float64
    )
    has_asia = "asia_vah" in work.columns and "asia_val" in work.columns

    if progress is not None:
        progress.op(f"clean-trade bars={n:,} window={window}")
    rows: list[dict[str, object]] = []
    for i in range(n):
        if progress is not None:
            progress.heartbeat(i + 1, n, label="clean-trade")
        if not onset[i]:
            continue
        direction = _infer_direction(
            i=i,
            brk_dir=brk_dir,
            close_pts=close_pts,
            asia_vah=asia_vah,
            asia_val=asia_val,
            has_asia=has_asia,
        )
        entry = float(close_pts[i])
        atr = float(atr_pts[i])
        target_pts = float(target_atr_frac) * atr
        stop_pts = float(mae_atr_frac) * atr
        visible = 0
        last_j = i
        max_high = entry
        min_low = entry
        path_mae = 0.0
        path_mfe = 0.0
        exit_reason = _EXIT_NONE
        exit_px = entry
        for j in range(i + 1, min(n, i + window + 1)):
            if groups[j] != groups[i]:
                break
            visible += 1
            last_j = j
            hi = float(high_pts[j])
            lo = float(low_pts[j])
            max_high = max(max_high, hi)
            min_low = min(min_low, lo)
            if exit_reason != _EXIT_NONE:
                continue
            if direction > _ONSET:
                hit_stop = lo <= entry - stop_pts
                hit_target = hi >= entry + target_pts
                path_mae = max(path_mae, entry - lo)
                path_mfe = max(path_mfe, hi - entry)
                if hit_stop:
                    exit_reason = _EXIT_STOP
                    exit_px = entry - stop_pts
                elif hit_target:
                    exit_reason = _EXIT_TARGET
                    exit_px = entry + target_pts
            elif direction < -_ONSET:
                hit_stop = hi >= entry + stop_pts
                hit_target = lo <= entry - target_pts
                path_mae = max(path_mae, hi - entry)
                path_mfe = max(path_mfe, entry - lo)
                if hit_stop:
                    exit_reason = _EXIT_STOP
                    exit_px = entry + stop_pts
                elif hit_target:
                    exit_reason = _EXIT_TARGET
                    exit_px = entry - target_pts
            else:
                path_mae = max(path_mae, entry - lo, hi - entry)
                path_mfe = max(path_mfe, hi - entry, entry - lo)
        window_complete = visible >= int(window)
        if exit_reason == _EXIT_NONE:
            exit_reason = _EXIT_TIME if window_complete else _EXIT_NONE
            exit_px = float(close_pts[last_j])
        mfe_up = max(0.0, max_high - entry)
        mae_up = max(0.0, entry - min_low)
        mfe_down = max(0.0, entry - min_low)
        mae_down = max(0.0, max_high - entry)
        if direction > _ONSET:
            window_mfe, window_mae = mfe_up, mae_up
            pnl_gross = exit_px - entry
        elif direction < -_ONSET:
            window_mfe, window_mae = mfe_down, mae_down
            pnl_gross = entry - exit_px
        else:
            window_mfe, window_mae = max(mfe_up, mfe_down), max(mae_up, mae_down)
            pnl_gross = 0.0
        atr_ok = atr > _EPS
        resolved = window_complete and atr_ok
        y = (
            1.0
            if resolved
            and _window_is_clean(
                direction=direction,
                mfe_up=mfe_up,
                mae_up=mae_up,
                mfe_down=mfe_down,
                mae_down=mae_down,
                target_pts=target_pts,
                stop_pts=stop_pts,
            )
            else 0.0
        )
        rows.append(
            {
                "ts_i": int(ts[i]),
                "ts_last": int(ts[last_j]),
                "group": int(groups[i]),
                "horizon": int(last_j - i),
                "direction": float(direction),
                "atr_pts": atr,
                "target_pts": target_pts,
                "stop_pts": stop_pts,
                "window_mfe_pts": float(window_mfe),
                "window_mae_pts": float(window_mae),
                "path_mfe_pts": float(path_mfe),
                "path_mae_pts": float(path_mae),
                "exit_reason": exit_reason,
                "pnl_gross_pts": float(pnl_gross),
                "y": float(y),
                "label_status": "resolved" if resolved else "censored",
                "tradeable": abs(direction) >= _ONSET and atr_ok,
            }
        )
    if progress is not None:
        progress.op(f"clean-trade onsets={len(rows):,}")
    return rows


def build_clean_trade_outcomes(
    frame: pl.DataFrame,
    *,
    window: int = CLEAN_HORIZON_BARS,
    target_atr_frac: float = CLEAN_TARGET_ATR_FRAC,
    mae_atr_frac: float = CLEAN_MAE_ATR_FRAC,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """هل كانت الصفقة نظيفة: امتداد ATR مع MAE أصغر منه خلال 50 برميلًا؟"""
    schema = _binary_schema()
    scanned = _scan_clean_onsets(
        frame,
        window=window,
        target_atr_frac=target_atr_frac,
        mae_atr_frac=mae_atr_frac,
        group_col=group_col,
        progress=progress,
    )
    if not scanned:
        return pl.DataFrame(schema=schema)
    out_rows = [
        {
            SETUP_AVAILABILITY_TS: row["ts_i"],
            OUTCOME_AVAILABLE_TS: row["ts_last"],
            "outcome_name": Y_CLEAN,
            "y": row["y"],
            "horizon_bars": row["horizon"],
            "group_id": row["group"],
            "label_status": row["label_status"],
        }
        for row in scanned
    ]
    return pl.DataFrame(out_rows, schema=schema)


def simulate_clean_exits(
    frame: pl.DataFrame,
    *,
    window: int = CLEAN_HORIZON_BARS,
    target_atr_frac: float = CLEAN_TARGET_ATR_FRAC,
    mae_atr_frac: float = CLEAN_MAE_ATR_FRAC,
    round_trip_cost_pts: float = CLEAN_ROUND_TRIP_COST_PTS,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """أول لمس لهدف/وقف ATR — تشخيص، ليس تسمية تدريب."""
    schema = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        "direction": pl.Float64(),
        "atr_pts": pl.Float64(),
        "target_pts": pl.Float64(),
        "stop_pts": pl.Float64(),
        "window_mfe_pts": pl.Float64(),
        "window_mae_pts": pl.Float64(),
        "path_mfe_pts": pl.Float64(),
        "path_mae_pts": pl.Float64(),
        "exit_reason": pl.Utf8(),
        "pnl_gross_pts": pl.Float64(),
        "pnl_net_pts": pl.Float64(),
        "y_clean": pl.Float64(),
        "label_status": pl.Utf8(),
        "tradeable": pl.Boolean(),
        "horizon_bars": pl.Int64(),
    }
    scanned = _scan_clean_onsets(
        frame,
        window=window,
        target_atr_frac=target_atr_frac,
        mae_atr_frac=mae_atr_frac,
        group_col=group_col,
        progress=progress,
    )
    if not scanned:
        return pl.DataFrame(schema=schema)
    cost = float(round_trip_cost_pts)
    rows = [
        {
            SETUP_AVAILABILITY_TS: row["ts_i"],
            OUTCOME_AVAILABLE_TS: row["ts_last"],
            "direction": row["direction"],
            "atr_pts": row["atr_pts"],
            "target_pts": row["target_pts"],
            "stop_pts": row["stop_pts"],
            "window_mfe_pts": row["window_mfe_pts"],
            "window_mae_pts": row["window_mae_pts"],
            "path_mfe_pts": row["path_mfe_pts"],
            "path_mae_pts": row["path_mae_pts"],
            "exit_reason": row["exit_reason"],
            "pnl_gross_pts": row["pnl_gross_pts"],
            "pnl_net_pts": float(cast(float, row["pnl_gross_pts"])) - cost,
            "y_clean": row["y"],
            "label_status": row["label_status"],
            "tradeable": row["tradeable"],
            "horizon_bars": row["horizon"],
        }
        for row in scanned
    ]
    return pl.DataFrame(rows, schema=schema)


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def summarize_clean_oof(
    exits: pl.DataFrame,
    oof: pl.DataFrame,
    *,
    min_p: float = CLEAN_OPERATING_P,
) -> dict[str, Any]:
    """إطلاقات OOF عند ``p ≥ min_p`` — نقطة تشغيل مُعلنة، ليست معايرة على OOF."""
    empty = {
        "n_labeled": 0,
        "n_oof": 0,
        "n_fires": 0,
        "n_skipped_no_direction": 0,
        "win_rate": float("nan"),
        "mean_path_mae_pts": float("nan"),
        "mean_path_mfe_pts": float("nan"),
        "mean_window_mae_pts": float("nan"),
        "mean_net_pts": float("nan"),
        "sum_net_pts": 0.0,
        "mean_gross_pts": float("nan"),
        "n_target": 0,
        "n_stop": 0,
        "n_time": 0,
        "operating_p": float(min_p),
        "round_trip_cost_pts": CLEAN_ROUND_TRIP_COST_PTS,
        "holdout_touched": False,
        "thresholds_tuned_on_oof": False,
        "is_live_overlay": False,
    }
    if exits.height == 0 or oof.height == 0:
        return empty
    if "outcome_name" not in oof.columns:
        return empty
    oof_clean = oof.filter(pl.col("outcome_name") == Y_CLEAN)
    if oof_clean.height == 0:
        return empty
    p_col = "p_cal" if "p_cal" in oof_clean.columns else "p_hat"
    scored = oof_clean.select(SETUP_AVAILABILITY_TS, p_col).rename({p_col: "p"})
    joined = exits.join(scored, on=SETUP_AVAILABILITY_TS, how="inner")
    if joined.height == 0:
        return {**empty, "n_labeled": int(exits.height)}
    fires = joined.filter(
        (pl.col("p") >= float(min_p)) & pl.col("tradeable") & (pl.col("label_status") == "resolved")
    )
    skipped = joined.filter((pl.col("p") >= float(min_p)) & (~pl.col("tradeable")))
    n_fires = int(fires.height)
    wins = fires.filter(pl.col("exit_reason") == _EXIT_TARGET) if n_fires else fires
    return {
        "n_labeled": int(exits.height),
        "n_oof": int(joined.height),
        "n_fires": n_fires,
        "n_skipped_no_direction": int(skipped.height),
        "win_rate": (float(wins.height) / n_fires) if n_fires else float("nan"),
        "mean_path_mae_pts": _finite_mean(fires["path_mae_pts"].to_list())
        if n_fires
        else float("nan"),
        "mean_path_mfe_pts": _finite_mean(fires["path_mfe_pts"].to_list())
        if n_fires
        else float("nan"),
        "mean_window_mae_pts": (
            _finite_mean(fires["window_mae_pts"].to_list()) if n_fires else float("nan")
        ),
        "mean_net_pts": _finite_mean(fires["pnl_net_pts"].to_list()) if n_fires else float("nan"),
        "sum_net_pts": float(np.sum(np.asarray(fires["pnl_net_pts"].to_list(), dtype=np.float64)))
        if n_fires
        else 0.0,
        "mean_gross_pts": _finite_mean(fires["pnl_gross_pts"].to_list())
        if n_fires
        else float("nan"),
        "n_target": int(fires.filter(pl.col("exit_reason") == _EXIT_TARGET).height)
        if n_fires
        else 0,
        "n_stop": int(fires.filter(pl.col("exit_reason") == _EXIT_STOP).height) if n_fires else 0,
        "n_time": int(fires.filter(pl.col("exit_reason") == _EXIT_TIME).height) if n_fires else 0,
        "operating_p": float(min_p),
        "round_trip_cost_pts": CLEAN_ROUND_TRIP_COST_PTS,
        "holdout_touched": False,
        "thresholds_tuned_on_oof": False,
        "is_live_overlay": False,
    }


__all__ = [
    "CLEAN_HORIZON_BARS",
    "CLEAN_MAE_ATR_FRAC",
    "CLEAN_OPERATING_P",
    "CLEAN_ROUND_TRIP_COST_PTS",
    "CLEAN_TARGET_ATR_FRAC",
    "CLEAN_TRADE_TARGETS",
    "Y_CLEAN",
    "build_clean_trade_outcomes",
    "simulate_clean_exits",
    "summarize_clean_oof",
]
