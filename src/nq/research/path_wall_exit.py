"""وقف يدوي خلف أقوى جدار عند بدايات مسار المزاد — وصف، ليست overlay.

الكون = نفس ``_onset_mask`` لرأس ``y_path_further_beyond`` على براميل 30ث.
الاتجاه سببي عند الإعداد (``proj_break_direction`` أو الإغلاق مقابل آسيا VA).
الوقف = أكبر حجم معلّق على جهة الإبطال من دفتر **ذلك اليوم فقط** حتى ``t``.
الأفق = 50 برميل × 30ث (25د) كما في ``y_extend_5pts_25min``. لا اختيار بـ ``y``.
``p_y_path_further_beyond`` OOF يُوسَم إن وُجد ولا يُفلتر الكون ولا يُضبط.
الخروج يبقى يدويًا. ليست قفلًا وليست إشارة فشل كسر. أيلول–كانون 2025 لا تُمس.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.auction_behavior.realized_path import (
    EXTEND_HORIZON_BARS,
    _active,
    _col_array,
    _onset_mask,
)
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import session_date_from_ns
from nq.orderbook.book import OrderBook
from nq.research.cross_nq_mnq import MNQ_MULT
from nq.research.failed_break_diag import (
    WALL_SEARCH_MIN_PTS,
    _advance_book,
    _book_event_arrays,
    _search_pts,
    strongest_wall_stop,
)
from nq.research.failed_break_flow import load_idrive_day
from nq.research.failed_breakout import (
    ATR_WINDOW,
    _fmt_ts,
    _json_num,
    _median,
)
from nq.research.mbo_sequence_mlp import HOLDOUT_START_DATE, resolve_idrive_mbo
from nq.simulation.fvg import NS_30M

LAYER_ID: Final = "path_wall_exit"
Side = Literal["short", "long"]
P_PATH_MIN: Final = 0.5
GROUP_COL: Final = "_behavior_story_run"
_BEYOND: Final = "path_beyond_asia_ticks"
_BREAK: Final = "vp_fsm_break"
_RETEST: Final = "vp_fsm_retest"
_ONSET: Final = 0.5
_FIXED_POINT_FLOOR: Final = 1.0 / float(PRICE_SCALE)
_MAX_ERRORS: Final = 12
_EPS: Final = 1e-12

_SETUP_SCHEMA: Final[dict[str, pl.DataType]] = {
    "setup_ts": pl.Int64(),
    "clock": pl.Utf8(),
    "day_id": pl.Utf8(),
    "side": pl.Utf8(),
    "entry": pl.Float64(),
    "sl": pl.Float64(),
    "wall_px": pl.Float64(),
    "wall_sz": pl.Int64(),
    "search_pts": pl.Float64(),
    "atr": pl.Float64(),
    "risk_pts": pl.Float64(),
    "mae_pts": pl.Float64(),
    "mfe_pts": pl.Float64(),
    "hit_sl": pl.Boolean(),
    "exit_reason": pl.Utf8(),
    "horizon_bars_seen": pl.Int64(),
    "time_pnl_pts": pl.Float64(),
    "p_path": pl.Float64(),
    "oof_ge_half": pl.Boolean(),
    "path_beyond_asia_ticks": pl.Float64(),
}


def _empty() -> dict[str, Any]:
    return {
        "layer": LAYER_ID,
        "not_fb_lock": True,
        "not_overlay": True,
        "exits_remain_manual": True,
        "not_path_tick_overlay": True,
        "not_mbo_book_concat": True,
        "wall_book": "single_day_until_t",
        "universe": "path_onset_mask",
        "horizon_bars": EXTEND_HORIZON_BARS,
        "p_path_min_predeclared": P_PATH_MIN,
        "p_path_not_tuned": True,
        "nq_tape": "unavailable_idrive_mnq_only",
        "holdout_start": HOLDOUT_START_DATE,
        "point_value_usd": MNQ_MULT,
        "n_days": 0,
        "n_skipped_holdout": 0,
        "n_skipped_missing_blended": 0,
        "n_skipped_no_mbo": 0,
        "n_skipped_error": 0,
        "n_onsets": 0,
        "n_directed": 0,
        "n_skipped_no_side": 0,
        "n_with_wall": 0,
        "n_no_wall": 0,
        "n_oof_scored": 0,
        "n_oof_ge_half": 0,
        "not_new_idrive_scan": False,
        "oof_source": "per_day_if_present",
        "errors": [],
    }


def _px_to_points(px: float) -> float:
    value = float(px)
    if abs(value) >= _FIXED_POINT_FLOOR:
        return value * float(PRICE_SCALE)
    return value


def _arr_points(frame: pl.DataFrame, name: str, n: int) -> NDArray[np.float64]:
    raw = _col_array(frame, name, n)
    return np.asarray([_px_to_points(float(v)) for v in raw], dtype=np.float64)


def _join_oof(work: pl.DataFrame, oof: pl.DataFrame | None) -> pl.DataFrame:
    if "p_y_path_further_beyond" in work.columns:
        return work
    empty = work.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_y_path_further_beyond"))
    if oof is None or oof.height == 0 or "p_y_path_further_beyond" not in oof.columns:
        return empty
    ts_col = AVAILABILITY_TS if AVAILABILITY_TS in oof.columns else SETUP_AVAILABILITY_TS
    if ts_col not in oof.columns:
        return empty
    pcol = oof.select(pl.col(ts_col).alias(AVAILABILITY_TS), "p_y_path_further_beyond")
    return work.join(pcol, on=AVAILABILITY_TS, how="left")


def _continuation_side(
    *,
    brk_dir: float,
    close_pts: float,
    asia_vah: float,
    asia_val: float,
    has_asia: bool,
) -> Side | None:
    direction = float(brk_dir)
    if abs(direction) < _ONSET and has_asia:
        if close_pts >= asia_vah > 0.0:
            direction = 1.0
        elif asia_val > 0.0 and close_pts <= asia_val:
            direction = -1.0
        else:
            direction = 0.0
    if direction > _ONSET:
        return "long"
    if direction < -_ONSET:
        return "short"
    return None


def _30m_ranges(
    ts: NDArray[np.int64],
    high: NDArray[np.float64],
    low: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    by: dict[int, tuple[float, float]] = {}
    for stamp, h, lo in zip(ts, high, low, strict=True):
        bucket = int(stamp) // NS_30M
        prev = by.get(bucket)
        if prev is None:
            by[bucket] = (float(h), float(lo))
        else:
            by[bucket] = (max(prev[0], float(h)), min(prev[1], float(lo)))
    keys = np.asarray(sorted(by), dtype=np.int64)
    rng = np.asarray([by[int(k)][0] - by[int(k)][1] for k in keys], dtype=np.float64)
    return keys, rng


def _atr_before(
    buckets: NDArray[np.int64],
    ranges: NDArray[np.float64],
    stamp: int,
    window: int,
) -> float:
    if buckets.size == 0:
        return float("nan")
    cur = int(stamp) // NS_30M
    hi = int(np.searchsorted(buckets, cur, side="left"))
    if hi <= 0:
        return float("nan")
    lo = max(0, hi - max(1, int(window)))
    chunk = ranges[lo:hi]
    if chunk.size == 0:
        return float("nan")
    return float(np.mean(chunk))


def _forward_path(
    *,
    side: Side,
    entry: float,
    sl: float,
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    groups: NDArray[np.int64],
    start_i: int,
    horizon: int,
) -> tuple[float, float, bool, int, float, str]:
    mae = 0.0
    mfe = 0.0
    seen = 0
    last = float(entry)
    hit = False
    n = int(high.size)
    has_sl = math.isfinite(sl)
    for j in range(start_i + 1, min(n, start_i + int(horizon) + 1)):
        if int(groups[j]) != int(groups[start_i]):
            break
        seen += 1
        h = float(high[j])
        lo = float(low[j])
        last = float(close[j])
        if side == "long":
            mae = max(mae, entry - lo)
            mfe = max(mfe, h - entry)
            if has_sl and lo <= sl:
                hit = True
                last = float(sl)
                break
        else:
            mae = max(mae, h - entry)
            mfe = max(mfe, entry - lo)
            if has_sl and h >= sl:
                hit = True
                last = float(sl)
                break
    pnl = (last - entry) if side == "long" else (entry - last)
    reason = "sl" if hit else "time"
    return float(mae), float(mfe), hit, int(seen), float(pnl), reason


def scan_path_wall_day(  # noqa: PLR0915
    blended: pl.DataFrame,
    mbo: pl.DataFrame | None = None,
    oof: pl.DataFrame | None = None,
    *,
    horizon_bars: int = EXTEND_HORIZON_BARS,
    atr_window: int = ATR_WINDOW,
    point_value: float = MNQ_MULT,
    day_id: str = "",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """بدايات المسار في يوم واحد + جدار حتى ``t``. ليست إشارة."""

    diag = _empty()
    diag["point_value_usd"] = float(point_value)
    diag["horizon_bars"] = int(horizon_bars)
    need = [AVAILABILITY_TS, "high", "low", "close"]
    if blended.height == 0 or any(c not in blended.columns for c in need):
        return pl.DataFrame(schema=_SETUP_SCHEMA), diag
    work = _join_oof(blended.sort(AVAILABILITY_TS), oof)
    n = work.height
    ts = np.asarray(work[AVAILABILITY_TS].to_numpy(), dtype=np.int64)
    high = _arr_points(work, "high", n)
    low = _arr_points(work, "low", n)
    close = _arr_points(work, "close", n)
    beyond = _col_array(work, _BEYOND, n)
    brk = _active(_col_array(work, _BREAK, n))
    retest = _active(_col_array(work, _RETEST, n))
    brk_dir = _col_array(work, "proj_break_direction", n)
    asia_vah = _arr_points(work, "asia_vah", n)
    asia_val = _arr_points(work, "asia_val", n)
    has_asia = "asia_vah" in work.columns and "asia_val" in work.columns
    groups = (
        work[GROUP_COL].fill_null(-1).to_numpy().astype(np.int64)
        if GROUP_COL in work.columns
        else np.zeros(n, dtype=np.int64)
    )
    p_arr = np.asarray(work["p_y_path_further_beyond"].to_numpy(), dtype=np.float64)
    onset = _onset_mask(beyond, brk, retest, groups)
    buckets, ranges = _30m_ranges(ts, high, low)
    book = OrderBook()
    book_avail, _, book_act, book_side, book_px, book_sz, book_oid = (
        _book_event_arrays(mbo)
        if mbo is not None and mbo.height
        else (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            [],
            [],
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    )
    book_i = 0
    rows: list[dict[str, Any]] = []
    n_onsets = 0
    n_no_side = 0
    for i in range(n):
        if not onset[i]:
            continue
        n_onsets += 1
        side = _continuation_side(
            brk_dir=float(brk_dir[i]),
            close_pts=float(close[i]),
            asia_vah=float(asia_vah[i]),
            asia_val=float(asia_val[i]),
            has_asia=has_asia,
        )
        if side is None:
            n_no_side += 1
            continue
        stamp = int(ts[i])
        entry = float(close[i])
        atr = _atr_before(buckets, ranges, stamp, int(atr_window))
        search = _search_pts(atr) if math.isfinite(atr) else WALL_SEARCH_MIN_PTS
        if book_avail.size:
            book_i = _advance_book(
                book,
                cursor=book_i,
                avail=book_avail,
                actions=book_act,
                sides=book_side,
                prices=book_px,
                sizes=book_sz,
                oids=book_oid,
                stamp=stamp,
            )
        found = strongest_wall_stop(book, side=side, entry=entry, search_pts=float(search))
        sl = float("nan")
        wall_px = float("nan")
        wall_sz = 0
        risk = float("nan")
        if found is not None:
            sl, wall_px, wall_sz = found
            if (side == "long" and entry <= sl) or (side == "short" and entry >= sl):
                sl = float("nan")
                wall_px = float("nan")
                wall_sz = 0
            else:
                risk = abs(entry - sl)
        mae, mfe, hit, seen, pnl, reason = _forward_path(
            side=side,
            entry=entry,
            sl=sl,
            high=high,
            low=low,
            close=close,
            groups=groups,
            start_i=i,
            horizon=int(horizon_bars),
        )
        p_path = float(p_arr[i])
        oof_ok = bool(math.isfinite(p_path) and p_path >= P_PATH_MIN)
        rows.append(
            {
                "setup_ts": stamp,
                "clock": _fmt_ts(stamp),
                "day_id": day_id,
                "side": side,
                "entry": entry,
                "sl": sl,
                "wall_px": wall_px,
                "wall_sz": int(wall_sz),
                "search_pts": float(search),
                "atr": float(atr),
                "risk_pts": risk,
                "mae_pts": mae,
                "mfe_pts": mfe,
                "hit_sl": bool(hit),
                "exit_reason": reason if math.isfinite(sl) else "time",
                "horizon_bars_seen": seen,
                "time_pnl_pts": pnl,
                "p_path": p_path,
                "oof_ge_half": oof_ok,
                "path_beyond_asia_ticks": float(beyond[i]),
            }
        )
    table = pl.DataFrame(rows, schema=_SETUP_SCHEMA) if rows else pl.DataFrame(schema=_SETUP_SCHEMA)
    packed = _pack(table, diag)
    packed["n_onsets"] = n_onsets
    packed["n_skipped_no_side"] = n_no_side
    packed["n_directed"] = table.height
    return table, packed


def _share(flags: list[bool]) -> float:
    if not flags:
        return float("nan")
    return float(sum(1 for x in flags if x)) / float(len(flags))


def _subset_stats(table: pl.DataFrame) -> dict[str, Any]:
    if table.height == 0:
        return {
            "n": 0,
            "n_with_wall": 0,
            "median_mae_pts": float("nan"),
            "median_risk_pts": float("nan"),
            "median_mfe_pts": float("nan"),
            "share_hit_sl": float("nan"),
            "median_mae_unstopped": float("nan"),
            "median_mae_over_risk": float("nan"),
            "median_wall_sz": float("nan"),
            "median_time_pnl_pts": float("nan"),
        }
    wall = table.filter(pl.col("wall_sz") > 0)
    mae = [float(x) for x in table["mae_pts"].to_list()]
    risk = [float(x) for x in wall["risk_pts"].to_list()] if wall.height else []
    hits = [bool(x) for x in wall["hit_sl"].to_list()] if wall.height else []
    unstopped = (
        [float(x) for x, h in zip(wall["mae_pts"].to_list(), hits, strict=True) if not h]
        if wall.height
        else []
    )
    ratio = [
        float(m) / float(r)
        for m, r in zip(wall["mae_pts"].to_list(), wall["risk_pts"].to_list(), strict=True)
        if math.isfinite(float(r)) and float(r) > _EPS
    ]
    return {
        "n": table.height,
        "n_with_wall": wall.height,
        "median_mae_pts": _median(mae),
        "median_risk_pts": _median(risk),
        "median_mfe_pts": _median([float(x) for x in table["mfe_pts"].to_list()]),
        "share_hit_sl": _share(hits),
        "median_mae_unstopped": _median(unstopped),
        "median_mae_over_risk": _median(ratio),
        "median_wall_sz": _median([float(x) for x in wall["wall_sz"].to_list()])
        if wall.height
        else float("nan"),
        "median_time_pnl_pts": _median([float(x) for x in table["time_pnl_pts"].to_list()]),
    }


def _pack(table: pl.DataFrame, diag: dict[str, Any]) -> dict[str, Any]:
    diag["n_directed"] = table.height
    diag["n_with_wall"] = int(table.filter(pl.col("wall_sz") > 0).height) if table.height else 0
    diag["n_no_wall"] = int(table.height - diag["n_with_wall"])
    p = []
    if table.height:
        for x in table["p_path"].to_list():
            if x is None:
                continue
            p.append(float(x))
    diag["n_oof_scored"] = sum(1 for x in p if math.isfinite(x))
    diag["n_oof_ge_half"] = int(table.filter(pl.col("oof_ge_half")).height) if table.height else 0
    all_stats = _subset_stats(table)
    wall_stats = _subset_stats(table.filter(pl.col("wall_sz") > 0)) if table.height else all_stats
    oof_stats = (
        _subset_stats(table.filter(pl.col("oof_ge_half"))) if table.height else _subset_stats(table)
    )
    diag["all"] = all_stats
    diag["with_wall"] = wall_stats
    diag["oof_ge_half"] = oof_stats
    diag["median_mae_pts"] = wall_stats["median_mae_pts"]
    diag["median_risk_pts"] = wall_stats["median_risk_pts"]
    diag["share_hit_sl"] = wall_stats["share_hit_sl"]
    diag["median_mae_unstopped"] = wall_stats["median_mae_unstopped"]
    diag["median_wall_sz"] = wall_stats["median_wall_sz"]
    return diag


def _scan_one_year_day(
    day: Path,
    mbo_root: Path | str,
    *,
    horizon_bars: int,
    point_value: float,
    log: Callable[[str], None] | None,
) -> tuple[pl.DataFrame, dict[str, Any]] | str:
    mbo_path = resolve_idrive_mbo(mbo_root, day.name)
    if mbo_path is None:
        if log is not None:
            log(f"day {day.name} no IDrive MBO")
        return "no_mbo"
    if log is not None:
        log(f"day {day.name} {mbo_path.name}")
    blended = pl.read_parquet(day / "blended.parquet")
    _trades, mbo = load_idrive_day(mbo_path, full_mbo=True)
    oof_path = day / "oof_predictions.parquet"
    oof = pl.read_parquet(oof_path) if oof_path.is_file() else None
    return scan_path_wall_day(
        blended,
        mbo,
        oof,
        horizon_bars=horizon_bars,
        point_value=point_value,
        day_id=day.name,
    )


def scan_year_path_wall(
    year_dir: Path | str,
    mbo_root: Path | str,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
    horizon_bars: int = EXTEND_HORIZON_BARS,
    point_value: float = MNQ_MULT,
    log: Callable[[str], None] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """أيام ``auction_behavior_year`` قبل holdout + دفتر IDrive لذلك اليوم فقط."""

    root = Path(year_dir)
    diag = _empty()
    diag["point_value_usd"] = float(point_value)
    diag["horizon_bars"] = int(horizon_bars)
    diag["holdout_start"] = holdout_start
    if not root.is_dir():
        return pl.DataFrame(schema=_SETUP_SCHEMA), diag
    days = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:4].isdigit())
    tables: list[pl.DataFrame] = []
    counts = {"days": 0, "hold": 0, "skip": 0, "no_mbo": 0, "err": 0, "onsets": 0, "no_side": 0}
    errors: list[str] = []
    for day in days:
        if day.name >= holdout_start:
            counts["hold"] += 1
            continue
        if not (day / "blended.parquet").is_file():
            counts["skip"] += 1
            continue
        try:
            result = _scan_one_year_day(
                day,
                mbo_root,
                horizon_bars=horizon_bars,
                point_value=point_value,
                log=log,
            )
        except (ValueError, OSError) as exc:
            counts["err"] += 1
            if len(errors) < _MAX_ERRORS:
                errors.append(f"{day.name}: {exc}")
            continue
        if isinstance(result, str):
            counts["no_mbo"] += 1
            continue
        table, day_diag = result
        counts["days"] += 1
        counts["onsets"] += int(day_diag.get("n_onsets") or 0)
        counts["no_side"] += int(day_diag.get("n_skipped_no_side") or 0)
        if table.height:
            tables.append(table)
    stacked = pl.concat(tables, how="vertical") if tables else pl.DataFrame(schema=_SETUP_SCHEMA)
    packed = _pack(stacked, diag)
    packed["n_days"] = counts["days"]
    packed["n_skipped_holdout"] = counts["hold"]
    packed["n_skipped_missing_blended"] = counts["skip"]
    packed["n_skipped_no_mbo"] = counts["no_mbo"]
    packed["n_skipped_error"] = counts["err"]
    packed["n_onsets"] = counts["onsets"]
    packed["n_skipped_no_side"] = counts["no_side"]
    packed["n_directed"] = stacked.height
    packed["errors"] = errors
    packed["holdout_start"] = holdout_start
    return stacked, packed


def attach_period_path_oof(
    table: pl.DataFrame,
    oof: pl.DataFrame,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """يلحق ``p_y_path_further_beyond`` OOF على صفوف الجدار الموجودة. بلا مسح جديد.

    لا يصفّي الكون بـ ``p`` ولا بـ ``y``. أيلول–كانون يُحذف من OOF فقط.
    """

    diag = _empty() if diagnostics is None else dict(diagnostics)
    diag["not_new_idrive_scan"] = True
    diag["oof_source"] = "period_join"
    diag["holdout_start"] = holdout_start
    if table.height == 0:
        return table, _pack(table, diag)
    scored = oof
    if scored.height and "eligible_for_backtest" in scored.columns:
        scored = scored.filter(pl.col("eligible_for_backtest"))
    if scored.height and "prediction_is_oof" in scored.columns:
        scored = scored.filter(pl.col("prediction_is_oof"))
    ts_col = AVAILABILITY_TS if AVAILABILITY_TS in scored.columns else SETUP_AVAILABILITY_TS
    if (
        scored.height == 0
        or ts_col not in scored.columns
        or "p_y_path_further_beyond" not in scored.columns
    ):
        return table, _pack(table, diag)
    days = [session_date_from_ns(int(t)) for t in scored[ts_col].to_list()]
    scored = scored.with_columns(pl.Series("_oof_day", days, dtype=pl.Utf8))
    scored = scored.filter(pl.col("_oof_day") < holdout_start)
    pcol = scored.select(
        pl.col(ts_col).alias("setup_ts"),
        pl.col("p_y_path_further_beyond").alias("_p_join"),
    ).unique(subset=["setup_ts"], keep="first")
    work = table
    if "p_path" in work.columns:
        work = work.drop("p_path")
    if "oof_ge_half" in work.columns:
        work = work.drop("oof_ge_half")
    work = work.join(pcol, on="setup_ts", how="left")
    p_join = pl.col("_p_join")
    work = work.with_columns(
        p_join.cast(pl.Float64).fill_null(float("nan")).alias("p_path"),
        (p_join.is_finite() & (p_join >= P_PATH_MIN)).fill_null(False).alias("oof_ge_half"),
    ).drop("_p_join")
    packed = _pack(work, diag)
    packed["not_new_idrive_scan"] = True
    packed["oof_source"] = "period_join"
    return work, packed


def write_path_wall_exit_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "path_wall_exit.parquet")
    packed = {k: _json_num(v) if isinstance(v, float) else v for k, v in dict(diagnostics).items()}
    (out / "summary.json").write_text(
        json.dumps(packed, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    wall = diagnostics.get("with_wall") or {}
    oof = diagnostics.get("oof_ge_half") or {}
    lines = [
        "# Path-setup wall-stop (manual-exit hint, not a lock)",
        "",
        "Universe is the same path-onset mask as `y_path_further_beyond`, not failed-break "
        "entries and not realized `y`. Direction is causal at setup. Stop is behind the "
        "strongest resting wall on the invalidation side from that day's book up to t. "
        f"Horizon is {diagnostics.get('horizon_bars')} x 30s bars. OOF "
        "`p_y_path_further_beyond` is tagged when present (threshold "
        f"{diagnostics.get('p_path_min_predeclared')}, not tuned) and does not select the "
        "universe. Exits stay manual. Not an overlay. Not a live tick model. "
        f"Holdout {diagnostics.get('holdout_start')} is not scanned. "
        "NQ tape is not in IDrive MES_MBO. "
        "Period OOF may be joined onto an existing wall table "
        f"(not_new_idrive_scan={diagnostics.get('not_new_idrive_scan')}, "
        f"oof_source={diagnostics.get('oof_source')}); that is not a new backtest.",
        "",
        (
            f"days={diagnostics.get('n_days')} "
            f"holdout_skipped={diagnostics.get('n_skipped_holdout')} "
            f"no_mbo={diagnostics.get('n_skipped_no_mbo')} "
            f"onsets={diagnostics.get('n_onsets')} "
            f"directed={diagnostics.get('n_directed')} "
            f"no_side={diagnostics.get('n_skipped_no_side')} "
            f"with_wall={diagnostics.get('n_with_wall')} "
            f"no_wall={diagnostics.get('n_no_wall')} "
            f"oof_scored={diagnostics.get('n_oof_scored')} "
            f"oof_ge_half={diagnostics.get('n_oof_ge_half')}."
        ),
        "",
        "## With wall (descriptive MAE, not a system)",
        "",
        (
            f"n={wall.get('n') if isinstance(wall, dict) else None} "
            f"with_wall={wall.get('n_with_wall') if isinstance(wall, dict) else None} "
            f"median_mae_pts={wall.get('median_mae_pts') if isinstance(wall, dict) else None} "
            f"median_risk_pts={wall.get('median_risk_pts') if isinstance(wall, dict) else None} "
            f"share_hit_sl={wall.get('share_hit_sl') if isinstance(wall, dict) else None} "
            f"median_mae_unstopped="
            f"{wall.get('median_mae_unstopped') if isinstance(wall, dict) else None} "
            f"median_mae_over_risk="
            f"{wall.get('median_mae_over_risk') if isinstance(wall, dict) else None} "
            f"median_wall_sz={wall.get('median_wall_sz') if isinstance(wall, dict) else None}."
        ),
        "",
        "## OOF p_path >= 0.5 subset (predeclared, not tuned)",
        "",
        f"n={oof.get('n') if isinstance(oof, dict) else None} "
        f"with_wall={oof.get('n_with_wall') if isinstance(oof, dict) else None} "
        f"median_mae_pts={oof.get('median_mae_pts') if isinstance(oof, dict) else None} "
        f"share_hit_sl={oof.get('share_hit_sl') if isinstance(oof, dict) else None} "
        f"median_risk_pts={oof.get('median_risk_pts') if isinstance(oof, dict) else None}.",
        "",
        "This does not replace manual exits and is not a locked pattern.",
        "",
    ]
    (out / "PATH_WALL_EXIT.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "LAYER_ID",
    "P_PATH_MIN",
    "attach_period_path_oof",
    "scan_path_wall_day",
    "scan_year_path_wall",
    "write_path_wall_exit_report",
]
