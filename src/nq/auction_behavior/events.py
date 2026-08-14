"""محرّك أحداث سلوكية من أعمدة المزاد/FSM (نبضات غير sticky)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

#: أحداث مخرَجة — أسماء ثابتة للطبقة الاحتمالية.
BEHAVIOR_EVENT_COLUMNS = (
    "evt_breakout",
    "evt_failed_breakout",
    "evt_retest_success",
    "evt_retest_fail",
    "evt_accept_expansion",
    "evt_reject_value",
    "evt_absorb_buy",
    "evt_absorb_sell",
    "evt_reload_bid",
    "evt_reload_ask",
)

#: عتبة اعتبار عمود ثنائي/علم «نشط» (أعمدة VP ∈ {0,1} تقريبًا).
_ACTIVE_FLAG = 0.5


def build_behavior_events(frame: pl.DataFrame) -> pl.DataFrame:
    """يحوّل نبضات VP/FSM/امتصاص إلى أحداث مفهومة (سببية من الأعمدة الجاهزة).

    يعتمد على مخرجات ``auction_signals_from_states`` (+ أعمدة خام اختيارية).
    لا يعيد حساب حدود VA — الحدود المتأخرة ``decision_*`` سبق تضمينها في المصدر.
    """
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        empty = {c: pl.Series(c, [], dtype=pl.Float64) for c in BEHAVIOR_EVENT_COLUMNS}
        return pl.DataFrame({AVAILABILITY_TS: pl.Series([], dtype=pl.Int64), **empty})

    work = frame.sort(AVAILABILITY_TS)

    def _col(name: str) -> pl.Expr:
        if name in work.columns:
            return pl.col(name).cast(pl.Float64).fill_null(0.0)
        return pl.lit(0.0)

    breakout = _col("vp_fsm_break").abs() > 0.0
    look_fail = _col("vp_look_fail")
    absorb = _col("vp_absorb")
    retest = _col("vp_fsm_retest")
    expand = _col("vp_fsm_expand").abs() > 0.0
    fr_accept = _col("vp_fr_accepted_expansion").abs() > 0.0
    fr_exit = _col("vp_fr_exit").abs() > 0.0
    close_in = _col("vp_close_in_value") > _ACTIVE_FLAG

    # ريتست ناجح: نبضة ريتست ثم توسّع لاحق في نفس الصف أو قبول FR؛ فاشل: look_fail بعد كسر.
    retest_ok = (retest.abs() > 0.0) & (expand | fr_accept)
    retest_bad = (retest.abs() > 0.0) & (look_fail.abs() > 0.0) & (~expand)

    return work.select(
        pl.col(AVAILABILITY_TS),
        breakout.cast(pl.Float64).alias("evt_breakout"),
        ((look_fail.abs() > 0.0) | fr_exit).cast(pl.Float64).alias("evt_failed_breakout"),
        retest_ok.cast(pl.Float64).alias("evt_retest_success"),
        retest_bad.cast(pl.Float64).alias("evt_retest_fail"),
        fr_accept.cast(pl.Float64).alias("evt_accept_expansion"),
        ((look_fail.abs() > 0.0) & close_in).cast(pl.Float64).alias("evt_reject_value"),
        (absorb > 0.0).cast(pl.Float64).alias("evt_absorb_buy"),
        (absorb < 0.0).cast(pl.Float64).alias("evt_absorb_sell"),
        # إعادة تحميل: امتصاص مع بقاء داخل القيمة (سيولة تُعاد عند الحد).
        ((absorb > 0.0) & close_in).cast(pl.Float64).alias("evt_reload_bid"),
        ((absorb < 0.0) & close_in).cast(pl.Float64).alias("evt_reload_ask"),
    )


def event_rate(events: pl.DataFrame, column: str) -> float:
    """نسبة حدوث حدث على الإطار (0 إن فارغ)."""
    if events.height == 0 or column not in events.columns:
        return 0.0
    vals = events[column].to_numpy().astype(np.float64)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals > 0.0))
