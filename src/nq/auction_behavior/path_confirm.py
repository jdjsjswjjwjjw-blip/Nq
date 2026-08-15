"""تأكيد مسار لندن مقابل مرساة آسيا — قياس مستمر، بلا بوابات IF.

آسيا تُفهم كتوزيع ثابت. كل التتبع في لندن على فريم الإشارة (30ث):
كسر حدود VAH/VAL/POC، تصحيح بسيط بعد ما يتثبت المسار، والعمق هو دليل
نجاح/فشل الهجرة. لا ريتست كامل للمنطقة، ولا عتبة ``if`` للقبول.
"""

from __future__ import annotations

import polars as pl

from nq.auction_behavior.level_flow import LEVEL_FLOW_COLUMNS
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.research.progress import ProgressLike

_EPS = 1e-12
_TICK = float(round(0.25 / PRICE_SCALE))
_LONDON = int(VpLiquiditySession.LONDON)

PATH_CONFIRM_COLUMNS = (
    "path_beyond_asia_ticks",
    "path_extreme_ticks",
    "path_correction_ticks",
    "path_held_frac",
    "path_inside_asia_va",
    "path_depth_follow",
    "path_depth_defend",
    "path_depth_confirm",
    "path_change_progress",
    "path_change_fail",
)


def _col(frame: pl.DataFrame, name: str, default: float = 0.0) -> pl.Expr:
    if name in frame.columns:
        return pl.col(name).cast(pl.Float64).fill_null(default)
    return pl.lit(default, dtype=pl.Float64)


def attach_path_depth_confirmation(
    frame: pl.DataFrame,
    *,
    tick_size: float | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يلحق أدلة مسار/عمق مستمرة على براميل لندن فقط.

    كل عمود ناتج حساب متجهي: ``relu`` / نسبة / ``tanh``. لا فرع ``if``
    على ريتست أو تداخل VA أو عدد تيكات.
    """
    if progress is not None:
        progress.op(f"attach_path_depth_confirmation bars={frame.height:,}")
    if frame.height == 0:
        return frame.with_columns(pl.lit(0.0).alias(c) for c in PATH_CONFIRM_COLUMNS)
    if AVAILABILITY_TS not in frame.columns:
        raise ValueError("frame requires availability_ts")

    tick = float(tick_size) if tick_size is not None else _TICK
    if tick <= 0:
        raise ValueError("tick_size must be > 0")

    work = frame.sort(AVAILABILITY_TS)
    london = (_col(work, VP_LIQUIDITY_SESSION, -1.0) == float(_LONDON)).cast(pl.Float64)
    close = _col(work, "close")
    vah = _col(work, "asia_vah")
    val = _col(work, "asia_val")
    width = (vah - val).clip(lower_bound=tick)
    beyond_up = london * ((close - vah) / tick).clip(lower_bound=0.0)
    beyond_down = london * ((val - close) / tick).clip(lower_bound=0.0)
    inside_span = pl.min_horizontal(vah - close, close - val)
    inside = london * (inside_span / (0.5 * width)).clip(0.0, 1.0)

    if "_liquidity_run" in work.columns:
        extreme_up = beyond_up.cum_max().over("_liquidity_run")
        extreme_down = beyond_down.cum_max().over("_liquidity_run")
    else:
        extreme_up = beyond_up.cum_max()
        extreme_down = beyond_down.cum_max()
    mass = extreme_up + extreme_down
    w_up = extreme_up / (mass + _EPS)
    w_down = extreme_down / (mass + _EPS)
    beyond = beyond_up * w_up + beyond_down * w_down
    extreme = extreme_up * w_up + extreme_down * w_down
    correction = (extreme - beyond).clip(lower_bound=0.0)
    held = beyond / (extreme + _EPS)

    follow = (
        _col(work, "lf_liquidity_migration")
        + _col(work, "lf_break_level_trade_intensity")
        + _col(work, "lf_liquidity_withdrawal") * (beyond / (beyond + 1.0))
        + _col(work, "lf_near_vah_cancel_ratio") * w_up
        + _col(work, "lf_near_val_cancel_ratio") * w_down
        + _col(work, "lf_near_hvn_cancel_ratio") * (beyond / (beyond + 1.0))
    )
    defend = (
        _col(work, "lf_refill_rate") * inside
        + _col(work, "lf_absorption_proxy").abs() * inside
        + _col(work, "lf_queue_survival_rate") * inside
        + _col(work, "lf_near_vah_add_intensity") * w_up * inside
        + _col(work, "lf_near_val_add_intensity") * w_down * inside
        + _col(work, "lf_near_poc_add_intensity") * inside
    )
    confirm = (0.5 + 0.5 * (follow - defend).tanh()).clip(0.0, 1.0) * london
    mig = _col(work, "proj_poc_shift_ticks").abs().tanh()
    outside = _col(work, "proj_outside_volume_share").clip(0.0, 1.0)
    overlap = _col(work, "proj_va_overlap").clip(0.0, 1.0)
    progress_score = (london * confirm * held * (0.5 * mig + 0.5 * outside)).clip(0.0, 1.0)
    fail = (london * inside * (1.0 - confirm) * overlap).clip(0.0, 1.0)

    out = work.with_columns(
        beyond.alias("path_beyond_asia_ticks"),
        extreme.alias("path_extreme_ticks"),
        correction.alias("path_correction_ticks"),
        held.clip(0.0, 1.0).alias("path_held_frac"),
        inside.alias("path_inside_asia_va"),
        follow.alias("path_depth_follow"),
        defend.alias("path_depth_defend"),
        confirm.alias("path_depth_confirm"),
        progress_score.alias("path_change_progress"),
        fail.alias("path_change_fail"),
    )
    if progress is not None:
        progress.op(f"path_depth_confirmation done bars={out.height:,}")
    missing_lf = [c for c in LEVEL_FLOW_COLUMNS if c not in work.columns]
    if missing_lf and progress is not None:
        progress.op(f"path confirm: {len(missing_lf)} level_flow cols defaulted to 0")
    return out


__all__ = [
    "PATH_CONFIRM_COLUMNS",
    "attach_path_depth_confirmation",
]
