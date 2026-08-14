"""محرّك أحداث سلوكية من أعمدة المزاد/FSM (نبضات غير sticky)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

#: أحداث مخرَجة — أسماء ثابتة للطبقة الاحتمالية.
BEHAVIOR_EVENT_COLUMNS = (
    "evt_breakout",
    "evt_true_break",
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


def build_behavior_events(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    outcome_window: int = 8,
    group_col: str | None = None,
) -> pl.DataFrame:
    """يحوّل نبضات VP/FSM/امتصاص إلى أحداث مفهومة (سببية من الأعمدة الجاهزة).

    يعتمد على مخرجات ``auction_signals_from_states`` (+ أعمدة خام اختيارية).
    لا يعيد حساب حدود VA — الحدود المتأخرة ``decision_*`` سبق تضمينها في المصدر.
    """
    if outcome_window < 1:
        raise ValueError(f"outcome_window must be >= 1, got {outcome_window}")
    if group_col is not None and frame.height > 0 and group_col not in frame.columns:
        raise ValueError(f"group_col is missing: {group_col}")
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        empty = {c: pl.Series(c, [], dtype=pl.Float64) for c in BEHAVIOR_EVENT_COLUMNS}
        return pl.DataFrame({AVAILABILITY_TS: pl.Series([], dtype=pl.Int64), **empty})

    work = frame.sort(AVAILABILITY_TS)

    def _col(name: str) -> pl.Expr:
        if name in work.columns:
            return pl.col(name).cast(pl.Float64).fill_null(0.0)
        return pl.lit(0.0)

    def _array(name: str) -> np.ndarray:
        if name not in work.columns:
            return np.zeros(work.height, dtype=np.float64)
        return work[name].fill_null(0.0).to_numpy().astype(np.float64)

    breakout_a = _array("vp_fsm_break")
    look_fail_a = _array("vp_look_fail")
    retest_a = _array("vp_fsm_retest")
    expand_a = _array("vp_fsm_expand")
    fr_accept_a = _array("vp_fr_accepted_expansion")
    fr_exit_a = _array("vp_fr_exit")
    groups = (
        work[group_col].to_numpy()
        if group_col is not None
        else np.zeros(work.height, dtype=np.int64)
    )

    # النتائج تُنبض عند الصف الذي أصبحت فيه معروفة، لا عند صف الكسر/الريتست القديم.
    true_break = np.zeros(work.height, dtype=np.float64)
    false_break = np.zeros(work.height, dtype=np.float64)
    retest_ok = np.zeros(work.height, dtype=np.float64)
    retest_bad = np.zeros(work.height, dtype=np.float64)
    pending_break = -1
    pending_retest = -1
    for i in range(work.height):
        if i > 0 and groups[i] != groups[i - 1]:
            pending_break = -1
            pending_retest = -1

        accepted_now = bool(
            abs(fr_accept_a[i]) > 0.0 or abs(fr_exit_a[i]) > 0.0 or abs(expand_a[i]) > 0.0
        )
        failed_now = bool(abs(look_fail_a[i]) > 0.0)

        if pending_break >= 0:
            age = i - pending_break
            if age >= 1 and accepted_now:
                true_break[i] = 1.0
                pending_break = -1
            elif (age >= 1 and failed_now) or age > outcome_window:
                false_break[i] = 1.0
                pending_break = -1

        if pending_retest >= 0:
            age = i - pending_retest
            if age >= 1 and accepted_now:
                retest_ok[i] = 1.0
                pending_retest = -1
            elif (age >= 1 and failed_now) or age > outcome_window:
                retest_bad[i] = 1.0
                pending_retest = -1

        # FR exit/accept هو قبول توسّع في لحظته، وليس failed breakout.
        if abs(fr_accept_a[i]) > 0.0 or abs(fr_exit_a[i]) > 0.0:
            true_break[i] = 1.0
        # نبدأ السياق بعد تسوية النتائج الحالية حتى يبقى "لاحقًا" صارمًا.
        if abs(breakout_a[i]) > 0.0 and true_break[i] == 0.0:
            pending_break = i
        if abs(retest_a[i]) > 0.0:
            pending_retest = i

    breakout = _col("vp_fsm_break").abs() > 0.0
    look_fail = _col("vp_look_fail")
    absorb = _col("vp_absorb")
    fr_accept = _col("vp_fr_accepted_expansion").abs() > 0.0
    fr_exit = _col("vp_fr_exit").abs() > 0.0
    close_in = _col("vp_close_in_value") > _ACTIVE_FLAG

    return work.select(
        pl.col(AVAILABILITY_TS),
        breakout.cast(pl.Float64).alias("evt_breakout"),
        pl.Series("evt_true_break", true_break),
        pl.Series("evt_failed_breakout", false_break),
        pl.Series("evt_retest_success", retest_ok),
        pl.Series("evt_retest_fail", retest_bad),
        (fr_accept | fr_exit).cast(pl.Float64).alias("evt_accept_expansion"),
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
