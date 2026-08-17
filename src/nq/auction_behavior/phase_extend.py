"""امتداد هيكلي (طور) — ليس هدفًا رقميًا ثابتًا بالنقاط.

``y_phase_extend = 1`` إذا استمر المسار 15 برميلًا (7.5 دقائق على 30ث)
دون كسر نقطة الدخول بأكثر من ``0.1 × London_ATR``، وحقق حدًا أدنى
``0.2 × London_ATR`` — بغض النظر عن عدد النقاط.

ATR لندن سببي: مدى جلسة ``[03:00, 09:30)`` ET لتواريخ الجلسة المكتملة
السابقة فقط. مدى لندن لليوم الجاري لا يدخل التعريف.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.auction_behavior.realized_path import (
    _BEYOND,
    _BREAK,
    _ONSET,
    _RETEST,
    Y_PHASE_EXTEND,
    _active,
    _binary_schema,
    _col_array,
    _onset_mask,
    _price_to_points,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import (
    VP_LIQUIDITY_SESSION,
    VpLiquiditySession,
    session_date_from_ns,
    vp_liquidity_session_from_ns,
)
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_causal_order

PHASE_EXTEND_TARGETS = (Y_PHASE_EXTEND,)
PHASE_HORIZON_BARS = 15
PHASE_EXPAND_ATR_FRAC = 0.2
PHASE_GIVEBACK_ATR_FRAC = 0.1
_LONDON_ATR_DAYS = 14
_LONDON = int(VpLiquiditySession.LONDON)
_EPS = 1e-12


def _session_dates(ts: Sequence[int]) -> list[str]:
    return [session_date_from_ns(int(t)) for t in ts]


def prior_london_atr14(frame: pl.DataFrame) -> np.ndarray:
    """ATR(14) لندن لكل برميل بوحدة السعر الخام — أيام لندن السابقة فقط."""
    n = frame.height
    if n == 0 or AVAILABILITY_TS not in frame.columns:
        return np.zeros(0, dtype=np.float64)
    work = frame.sort(AVAILABILITY_TS)
    dates = _session_dates([int(t) for t in work[AVAILABILITY_TS].to_list()])
    work = work.with_columns(pl.Series("_phase_session_date", dates, dtype=pl.Utf8))
    if VP_LIQUIDITY_SESSION in work.columns:
        london = work.filter(pl.col(VP_LIQUIDITY_SESSION).cast(pl.Int64) == _LONDON)
    else:
        codes = [vp_liquidity_session_from_ns(int(t)) for t in work[AVAILABILITY_TS].to_list()]
        work = work.with_columns(pl.Series("_phase_sess", codes, dtype=pl.Int64))
        london = work.filter(pl.col("_phase_sess") == _LONDON)
    if london.height == 0 or "high" not in work.columns or "low" not in work.columns:
        return np.zeros(n, dtype=np.float64)
    close_expr = (
        pl.col("close").cast(pl.Float64)
        if "close" in london.columns
        else pl.col("high").cast(pl.Float64)
    )
    daily = (
        london.sort(AVAILABILITY_TS)
        .group_by("_phase_session_date", maintain_order=True)
        .agg(
            pl.col("high").cast(pl.Float64).max().alias("_lon_high"),
            pl.col("low").cast(pl.Float64).min().alias("_lon_low"),
            close_expr.last().alias("_lon_close"),
        )
        .sort("_phase_session_date")
        .with_columns(pl.col("_lon_close").shift(1).alias("_prev_close"))
        .with_columns(
            pl.max_horizontal(
                pl.col("_lon_high") - pl.col("_lon_low"),
                (pl.col("_lon_high") - pl.col("_prev_close")).abs().fill_null(0.0),
                (pl.col("_lon_low") - pl.col("_prev_close")).abs().fill_null(0.0),
            ).alias("_tr")
        )
        .with_columns(
            pl.col("_tr")
            .shift(1)
            .rolling_mean(window_size=_LONDON_ATR_DAYS, min_samples=1)
            .alias("_london_atr14_prior")
        )
        .select("_phase_session_date", "_london_atr14_prior")
    )
    joined = work.join(daily, on="_phase_session_date", how="left")
    return joined["_london_atr14_prior"].fill_null(0.0).to_numpy().astype(np.float64)


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


def _held_expansion(
    *,
    direction: float,
    close: float,
    max_high: float,
    min_low: float,
    atr: float,
    expand_frac: float,
    giveback_frac: float,
) -> bool:
    if atr <= _EPS:
        return False
    expand = float(expand_frac) * atr
    give = float(giveback_frac) * atr
    up = max_high >= close + expand and min_low >= close - give
    down = min_low <= close - expand and max_high <= close + give
    if direction > _ONSET:
        return bool(up)
    if direction < -_ONSET:
        return bool(down)
    return bool(up or down)


def build_phase_extend_outcomes(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    window: int = PHASE_HORIZON_BARS,
    expand_atr_frac: float = PHASE_EXPAND_ATR_FRAC,
    giveback_atr_frac: float = PHASE_GIVEBACK_ATR_FRAC,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """هل استمر طور الامتداد 15 برميلًا مع حد ATR لندن الديناميكي؟

    النافذة كاملة ``t+1..t+window`` لازمة للحسم: العودة داخل النافذة تلغي
    الامتداد حتى لو تحقق الحد الأدنى مبكرًا. ATR اليوم الجاري غير مستخدم.
    """
    schema = _binary_schema()
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if expand_atr_frac <= 0.0 or giveback_atr_frac <= 0.0:
        raise ValueError("ATR fractions must be > 0")
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return pl.DataFrame(schema=schema)
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
        progress.op(f"phase-extend bars={n:,} window={window}")
    out_rows: list[dict[str, object]] = []
    for i in range(n):
        if progress is not None:
            progress.heartbeat(i + 1, n, label="phase-extend")
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
        visible = 0
        last_j = i
        max_high = close_pts[i]
        min_low = close_pts[i]
        for j in range(i + 1, min(n, i + window + 1)):
            if groups[j] != groups[i]:
                break
            visible += 1
            last_j = j
            max_high = max(max_high, high_pts[j])
            min_low = min(min_low, low_pts[j])
        window_complete = visible >= int(window)
        atr = float(atr_pts[i])
        if not window_complete or atr <= _EPS:
            status = "censored"
            y = 0.0
        else:
            status = "resolved"
            y = (
                1.0
                if _held_expansion(
                    direction=direction,
                    close=close_pts[i],
                    max_high=max_high,
                    min_low=min_low,
                    atr=atr,
                    expand_frac=expand_atr_frac,
                    giveback_frac=giveback_atr_frac,
                )
                else 0.0
            )
        out_rows.append(
            {
                SETUP_AVAILABILITY_TS: int(ts[i]),
                OUTCOME_AVAILABLE_TS: int(ts[last_j]),
                "outcome_name": Y_PHASE_EXTEND,
                "y": y,
                "horizon_bars": int(last_j - i),
                "group_id": int(groups[i]),
                "label_status": status,
            }
        )
    out = pl.DataFrame(out_rows, schema=schema) if out_rows else pl.DataFrame(schema=schema)
    if progress is not None:
        progress.op(f"phase_extend rows={out.height:,}")
    return out


__all__ = [
    "PHASE_EXPAND_ATR_FRAC",
    "PHASE_EXTEND_TARGETS",
    "PHASE_GIVEBACK_ATR_FRAC",
    "PHASE_HORIZON_BARS",
    "Y_PHASE_EXTEND",
    "build_phase_extend_outcomes",
    "prior_london_atr14",
]
