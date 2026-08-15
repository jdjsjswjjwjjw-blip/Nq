"""ذاكرة سوقية أغنى: lags + rolling سببي + تسلسل أحداث + dwell / migrations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike

_DEFAULT_LAGS = (1, 2, 3, 5)
_DEFAULT_ROLL = (3, 8)
_ACTIVE = 0.5

SEQUENCE_MEMORY_COLUMNS = (
    "mem_time_since_break",
    "mem_time_since_retest",
    "mem_time_since_absorb",
    "mem_break_test_count",
    "mem_retest_test_count",
    "mem_dwell_inside_value",
    "mem_dwell_above_vah",
    "mem_dwell_below_val",
    "mem_visit_order_break_retest",
    "mem_poc_migration_abs",
    "mem_hvn_migration_abs",
    "mem_va_migration_abs",
    "mem_poc_migration_speed",
    "mem_value_transfer_gradual",
    "mem_bars_since_london_open_proxy",
)


def attach_causal_memory(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    lags: tuple[int, ...] = _DEFAULT_LAGS,
    group_col: str | None = None,
) -> pl.DataFrame:
    """يضيف ``shift(k)`` سببيًا، مع إعادة ضبط اختيارية عند حدود الجلسة/المجموعة."""
    if frame.height == 0:
        return frame
    work = frame.sort(AVAILABILITY_TS)
    exprs: list[pl.Expr] = []
    for col in columns:
        if col not in work.columns:
            continue
        base = pl.col(col).cast(pl.Float64)
        for lag in lags:
            if lag < 1:
                raise ValueError(f"lag must be >= 1, got {lag}")
            shifted = base.shift(lag)
            if group_col is not None:
                if group_col not in work.columns:
                    raise ValueError(f"group_col is missing: {group_col}")
                shifted = shifted.over(group_col)
            exprs.append(shifted.alias(f"{col}__lag{lag}"))
    if not exprs:
        return work
    return work.with_columns(exprs)


def attach_market_memory(  # noqa: PLR0912
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    lags: tuple[int, ...] = _DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = _DEFAULT_ROLL,
    group_col: str | None = None,
    event_columns: Sequence[str] | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """ذاكرة كاملة: lags + متوسطات/مجاميع rolling ماضية + عدّادات أحداث.

    كل نافذة ``rolling_*`` تستخدم ``min_samples=1`` و``shift`` ضمني عبر أن
    المتوسط يُحسب على الصفوف حتى الحالي؛ لتجنب إدخال نفس صف الإشارة في
    بعض الاستخدامات نُزيح النتيجة بـ 1 بعد الـrolling (ماضي صارم).
    """
    if progress is not None:
        progress.op(f"attach_market_memory cols={len(columns)} lags={lags}")
    work = attach_causal_memory(frame, columns=columns, lags=lags, group_col=group_col)
    if work.height == 0:
        return work
    work = work.sort(AVAILABILITY_TS)
    exprs: list[pl.Expr] = []
    for col in columns:
        if col not in work.columns:
            continue
        base = pl.col(col).cast(pl.Float64)
        for win in roll_windows:
            if win < 1:
                raise ValueError(f"roll window must be >= 1, got {win}")
            mean_expr = base.rolling_mean(window_size=win, min_samples=1)
            sum_expr = base.rolling_sum(window_size=win, min_samples=1)
            if group_col is not None:
                if group_col not in work.columns:
                    raise ValueError(f"group_col is missing: {group_col}")
                mean_expr = mean_expr.over(group_col)
                sum_expr = sum_expr.over(group_col)
                # ماضي صارم داخل المجموعة فقط — لا وراثة من قصة سابقة.
                exprs.append(mean_expr.shift(1).over(group_col).alias(f"{col}__rmean{win}"))
                exprs.append(sum_expr.shift(1).over(group_col).alias(f"{col}__rsum{win}"))
            else:
                exprs.append(mean_expr.shift(1).alias(f"{col}__rmean{win}"))
                exprs.append(sum_expr.shift(1).alias(f"{col}__rsum{win}"))
    events = event_columns or ()
    for col in events:
        if col not in work.columns:
            continue
        pulse = (pl.col(col).cast(pl.Float64).abs() > 0.0).cast(pl.Float64)
        for win in roll_windows:
            cnt = pulse.rolling_sum(window_size=win, min_samples=1)
            if group_col is not None:
                if group_col not in work.columns:
                    raise ValueError(f"group_col is missing: {group_col}")
                cnt = cnt.over(group_col)
                exprs.append(cnt.shift(1).over(group_col).alias(f"{col}__ecount{win}"))
            else:
                exprs.append(cnt.shift(1).alias(f"{col}__ecount{win}"))
    if not exprs:
        if progress is not None:
            progress.op("market_memory: no extra rolling columns")
        return work
    out = work.with_columns(exprs)
    if progress is not None:
        progress.op(f"market_memory rows={out.height:,} extra_cols={len(exprs)}")
    return out


def attach_sequence_memory(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """تسلسل/dwell/هجرة: السوق عند t ليس مجرد لقطة سعر.

    كل المقاييس سببية (ماضي فقط عبر cum/shift). لا تستخدم نتائج مستقبلية.
    """
    if progress is not None:
        progress.op(f"attach_sequence_memory bars={frame.height:,}")
    if frame.height == 0:
        return frame.with_columns(pl.lit(0.0).alias(c) for c in SEQUENCE_MEMORY_COLUMNS)

    work = frame.sort(AVAILABILITY_TS)
    n = work.height

    def _pulse(name: str) -> np.ndarray:
        if name not in work.columns:
            return np.zeros(n, dtype=np.float64)
        return (np.abs(work[name].fill_null(0.0).to_numpy().astype(np.float64)) > _ACTIVE).astype(
            np.float64
        )

    def _f(name: str) -> np.ndarray:
        if name not in work.columns:
            return np.zeros(n, dtype=np.float64)
        return work[name].fill_null(0.0).to_numpy().astype(np.float64)

    groups = (
        work[group_col].fill_null(-1).to_numpy().astype(np.int64)
        if group_col is not None and group_col in work.columns
        else np.zeros(n, dtype=np.int64)
    )

    break_p = _pulse("vp_fsm_break")
    retest_p = _pulse("vp_fsm_retest")
    absorb_p = _pulse("vp_absorb")
    inside = (
        _pulse("struct_close_in_value")
        if "struct_close_in_value" in work.columns
        else _pulse("vp_close_in_value")
    )
    above = _pulse("struct_above_vah")
    below = _pulse("struct_below_val")
    if not above.any() and not below.any():
        # احتياطي عبر اتجاه كسر الإسقاط عندما لا تتوفر أعلام البنية
        above = ((1.0 - inside) * (_f("proj_break_direction") > 0)).astype(np.float64)
        below = ((1.0 - inside) * (_f("proj_break_direction") < 0)).astype(np.float64)

    poc_shift = np.abs(_f("proj_poc_shift_ticks"))
    hvn_shift = np.abs(_f("proj_hvn_shift_ticks"))
    va_shift = np.abs(_f("proj_va_center_shift_ticks"))
    mig_speed = np.abs(_f("proj_migration_speed_ticks"))

    time_break = np.zeros(n, dtype=np.float64)
    time_retest = np.zeros(n, dtype=np.float64)
    time_absorb = np.zeros(n, dtype=np.float64)
    cnt_break = np.zeros(n, dtype=np.float64)
    cnt_retest = np.zeros(n, dtype=np.float64)
    dwell_in = np.zeros(n, dtype=np.float64)
    dwell_above = np.zeros(n, dtype=np.float64)
    dwell_below = np.zeros(n, dtype=np.float64)
    visit_order = np.zeros(n, dtype=np.float64)
    bars_london = np.zeros(n, dtype=np.float64)
    gradual = np.zeros(n, dtype=np.float64)

    last_b = last_r = last_a = -10_000
    cb = cr = 0.0
    din = dab = dbl = 0.0
    last_event = 0.0  # 1=break, 2=retest
    story_i = 0
    prev_g = groups[0] if n else 0
    for i in range(n):
        if progress is not None:
            progress.heartbeat(i + 1, n, label="sequence-memory")
        if groups[i] != prev_g:
            last_b = last_r = last_a = -10_000
            cb = cr = 0.0
            din = dab = dbl = 0.0
            last_event = 0.0
            story_i = 0
            prev_g = groups[i]
        # قيم ماضية صارمة للقرار: حدّث العدادات بعد قراءة "منذ"
        time_break[i] = float(i - last_b) if last_b >= 0 else float(story_i)
        time_retest[i] = float(i - last_r) if last_r >= 0 else float(story_i)
        time_absorb[i] = float(i - last_a) if last_a >= 0 else float(story_i)
        cnt_break[i] = cb
        cnt_retest[i] = cr
        dwell_in[i] = din
        dwell_above[i] = dab
        dwell_below[i] = dbl
        visit_order[i] = last_event
        bars_london[i] = float(story_i)
        # تدريجي vs سريع: سرعة منخفضة مع إزاحة كبيرة → تدريجي
        if poc_shift[i] > 1.0:
            gradual[i] = float(1.0 / (1.0 + mig_speed[i]))
        else:
            gradual[i] = 0.0

        if break_p[i] > 0:
            last_b = i
            cb += 1.0
            last_event = 1.0
        if retest_p[i] > 0:
            last_r = i
            cr += 1.0
            last_event = 2.0
        if absorb_p[i] > 0:
            last_a = i
        if inside[i] > 0:
            din += 1.0
            dab = 0.0
            dbl = 0.0
        elif above[i] > 0:
            dab += 1.0
            din = 0.0
            dbl = 0.0
        elif below[i] > 0:
            dbl += 1.0
            din = 0.0
            dab = 0.0
        story_i += 1

    out = work.with_columns(
        pl.Series("mem_time_since_break", time_break),
        pl.Series("mem_time_since_retest", time_retest),
        pl.Series("mem_time_since_absorb", time_absorb),
        pl.Series("mem_break_test_count", cnt_break),
        pl.Series("mem_retest_test_count", cnt_retest),
        pl.Series("mem_dwell_inside_value", dwell_in),
        pl.Series("mem_dwell_above_vah", dwell_above),
        pl.Series("mem_dwell_below_val", dwell_below),
        pl.Series("mem_visit_order_break_retest", visit_order),
        pl.Series("mem_poc_migration_abs", poc_shift),
        pl.Series("mem_hvn_migration_abs", hvn_shift),
        pl.Series("mem_va_migration_abs", va_shift),
        pl.Series("mem_poc_migration_speed", mig_speed),
        pl.Series("mem_value_transfer_gradual", gradual),
        pl.Series("mem_bars_since_london_open_proxy", bars_london),
    )
    if progress is not None:
        progress.op(f"sequence_memory done bars={out.height:,}")
    return out


def memory_feature_matrix(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
) -> np.ndarray:
    """مصفوفة رقمية للأعمدة المطلوبة (NaN→0) بترتيب الصفوف الحالي."""
    present = [c for c in columns if c in frame.columns]
    if not present:
        return np.zeros((frame.height, 0), dtype=np.float64)
    return frame.select(present).fill_null(0.0).to_numpy().astype(np.float64)


def list_memory_columns(frame: pl.DataFrame) -> list[str]:
    """أسماء أعمدة الذاكرة المولَّدة."""
    return [
        c
        for c in frame.columns
        if "__lag" in c
        or "__rmean" in c
        or "__rsum" in c
        or "__ecount" in c
        or c.startswith("mem_")
    ]


__all__ = [
    "SEQUENCE_MEMORY_COLUMNS",
    "attach_causal_memory",
    "attach_market_memory",
    "attach_sequence_memory",
    "list_memory_columns",
    "memory_feature_matrix",
]
