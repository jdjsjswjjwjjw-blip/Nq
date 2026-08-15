"""عقود نتائج سلوكية مع ``outcome_available_ts`` (سببية صارمة).

القاعدة:
  * ``setup_availability_ts`` — متى صارت حالة الإعداد معروفة (ميزات القرار).
  * ``outcome_available_ts`` — متى صارت نتيجة الإعداد معروفة (لا قبلها).
  * دائمًا: ``outcome_available_ts >= setup_availability_ts``.
  * التدريب لا يستخدم صفًا قبل أن يصبح ``outcome_available_ts`` داخل نافذة القطار.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.validation.leakage import assert_availability_not_before_event, assert_causal_order

OUTCOME_AVAILABLE_TS = "outcome_available_ts"
SETUP_AVAILABILITY_TS = "setup_availability_ts"

#: مخرجات احتمالية أساسية للنموذج الشرطي.
OUTCOME_TARGETS = (
    "y_true_break",
    "y_false_break",
    "y_retest_success",
    "y_retest_fail",
    "y_expansion_continue",
    "y_return_to_value",
)

_ACTIVE = 0.5


@dataclass(frozen=True, slots=True)
class OutcomeSpec:
    """تعريف نتيجة واحدة: محفّز إعداد + محكّات حل لاحقة."""

    name: str
    trigger_col: str
    success_cols: tuple[str, ...]
    fail_cols: tuple[str, ...]
    window: int


_DEFAULT_SPECS: tuple[OutcomeSpec, ...] = (
    OutcomeSpec(
        name="y_true_break",
        trigger_col="vp_fsm_break",
        success_cols=("vp_fsm_expand", "vp_fr_accepted_expansion", "vp_fr_exit"),
        fail_cols=("vp_look_fail",),
        window=8,
    ),
    OutcomeSpec(
        name="y_false_break",
        trigger_col="vp_fsm_break",
        success_cols=("vp_look_fail",),
        fail_cols=("vp_fsm_expand", "vp_fr_accepted_expansion", "vp_fr_exit"),
        window=8,
    ),
    OutcomeSpec(
        name="y_retest_success",
        trigger_col="vp_fsm_retest",
        success_cols=("vp_fsm_expand", "vp_fr_accepted_expansion"),
        fail_cols=("vp_look_fail",),
        window=8,
    ),
    OutcomeSpec(
        name="y_retest_fail",
        trigger_col="vp_fsm_retest",
        success_cols=("vp_look_fail",),
        fail_cols=("vp_fsm_expand", "vp_fr_accepted_expansion"),
        window=8,
    ),
    OutcomeSpec(
        name="y_expansion_continue",
        trigger_col="vp_fsm_expand",
        success_cols=(
            "vp_fr_accepted_expansion",
            "proj_expansion_accepting",
            "proj_value_transferred",
        ),
        fail_cols=("vp_look_fail", "proj_rejection_to_asia"),
        window=8,
    ),
    OutcomeSpec(
        name="y_return_to_value",
        trigger_col="vp_look_fail",
        success_cols=("vp_close_in_value", "proj_rejection_to_asia"),
        fail_cols=("vp_fsm_expand", "proj_value_transferred"),
        window=8,
    ),
)


def _active(arr: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(arr) > _ACTIVE, dtype=bool)


def _col_array(frame: pl.DataFrame, name: str, n: int) -> np.ndarray:
    if name not in frame.columns:
        return np.zeros(n, dtype=np.float64)
    raw = frame[name].fill_null(0.0).to_list()
    out = np.zeros(n, dtype=np.float64)
    for i, value in enumerate(raw[:n]):
        out[i] = float(value) if value is not None else 0.0
    return out


def build_labeled_outcomes(  # noqa: PLR0912
    frame: pl.DataFrame,
    *,
    outcome_window: int = 8,
    group_col: str | None = None,
    specs: tuple[OutcomeSpec, ...] | None = None,
) -> pl.DataFrame:
    """يبني جدول إعداد→نتيجة مع طابعي الإعداد والإتاحة.

    كل صف = إعداد واحد لنوع نتيجة واحد. لا يُلصق ``y`` على صفوف المستقبل
    في إطار الميزات الخام؛ الجدول منفصل ليُدمَج فقط بعد التحقق الزمني.
    """
    schema = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        "outcome_name": pl.Utf8(),
        "y": pl.Float64(),
        "horizon_bars": pl.Int64(),
        "group_id": pl.Int64(),
    }
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return pl.DataFrame(schema=schema)
    if outcome_window < 1:
        raise ValueError(f"outcome_window must be >= 1, got {outcome_window}")

    work = frame.sort(AVAILABILITY_TS)
    n = work.height
    ts = work[AVAILABILITY_TS].to_numpy().astype(np.int64)
    assert_causal_order(ts)
    groups = (
        work[group_col].fill_null(-1).to_numpy().astype(np.int64)
        if group_col is not None and group_col in work.columns
        else np.zeros(n, dtype=np.int64)
    )
    active_specs = (
        specs
        if specs is not None
        else tuple(
            OutcomeSpec(
                name=s.name,
                trigger_col=s.trigger_col,
                success_cols=s.success_cols,
                fail_cols=s.fail_cols,
                window=int(outcome_window),
            )
            for s in _DEFAULT_SPECS
        )
    )

    rows: list[dict[str, object]] = []
    for spec in active_specs:
        trigger = _active(_col_array(work, spec.trigger_col, n))
        success = np.zeros(n, dtype=bool)
        for col in spec.success_cols:
            success |= _active(_col_array(work, col, n))
        fail = np.zeros(n, dtype=bool)
        for col in spec.fail_cols:
            fail |= _active(_col_array(work, col, n))

        for i in range(n):
            if not trigger[i]:
                continue
            # النتيجة تُحسم في صف لاحق فقط (عمر >= 1) — لا نفس برميل الإعداد.
            resolved = False
            for j in range(i + 1, min(n, i + spec.window + 1)):
                if groups[j] != groups[i]:
                    break
                if success[j] and not fail[j]:
                    rows.append(
                        {
                            SETUP_AVAILABILITY_TS: int(ts[i]),
                            OUTCOME_AVAILABLE_TS: int(ts[j]),
                            "outcome_name": spec.name,
                            "y": 1.0,
                            "horizon_bars": int(j - i),
                            "group_id": int(groups[i]),
                        }
                    )
                    resolved = True
                    break
                if fail[j] and not success[j]:
                    rows.append(
                        {
                            SETUP_AVAILABILITY_TS: int(ts[i]),
                            OUTCOME_AVAILABLE_TS: int(ts[j]),
                            "outcome_name": spec.name,
                            "y": 0.0,
                            "horizon_bars": int(j - i),
                            "group_id": int(groups[i]),
                        }
                    )
                    resolved = True
                    break
            if not resolved:
                # نافذة انتهت بلا حسم → فشل افتراضي عند آخر صف مرئي في النافذة/المجموعة
                end = i
                for j in range(i + 1, min(n, i + spec.window + 1)):
                    if groups[j] != groups[i]:
                        break
                    end = j
                if end > i:
                    rows.append(
                        {
                            SETUP_AVAILABILITY_TS: int(ts[i]),
                            OUTCOME_AVAILABLE_TS: int(ts[end]),
                            "outcome_name": spec.name,
                            "y": 0.0,
                            "horizon_bars": int(end - i),
                            "group_id": int(groups[i]),
                        }
                    )

    out = pl.DataFrame(rows) if rows else pl.DataFrame(schema=schema)
    if out.height:
        assert_availability_not_before_event(
            out[SETUP_AVAILABILITY_TS].to_numpy(),
            out[OUTCOME_AVAILABLE_TS].to_numpy(),
        )
        if bool(
            np.any(out[OUTCOME_AVAILABLE_TS].to_numpy() < out[SETUP_AVAILABILITY_TS].to_numpy())
        ):
            raise AssertionError("outcome_available_ts before setup_availability_ts")
    return out


def attach_outcome_availability_guard(
    features: pl.DataFrame,
    outcomes: pl.DataFrame,
) -> pl.DataFrame:
    """يربط كل إعداد بميزاته عند ``setup_availability_ts`` فقط (join exact)."""
    if features.height == 0 or outcomes.height == 0:
        return outcomes
    if AVAILABILITY_TS not in features.columns:
        raise ValueError("features require availability_ts")
    feat = features.sort(AVAILABILITY_TS)
    # exact join: الميزات المتاحة عند لحظة الإعداد — لا asof أمامي.
    return outcomes.join(
        feat,
        left_on=SETUP_AVAILABILITY_TS,
        right_on=AVAILABILITY_TS,
        how="inner",
    )


def filter_outcomes_known_by(outcomes: pl.DataFrame, *, asof_ts: int) -> pl.DataFrame:
    """صفوف النتائج التي أصبحت معروفة عند/قبل ``asof_ts`` (للتدريب فقط)."""
    if outcomes.height == 0 or OUTCOME_AVAILABLE_TS not in outcomes.columns:
        return outcomes
    return outcomes.filter(pl.col(OUTCOME_AVAILABLE_TS) <= int(asof_ts))


__all__ = [
    "OUTCOME_AVAILABLE_TS",
    "OUTCOME_TARGETS",
    "SETUP_AVAILABILITY_TS",
    "OutcomeSpec",
    "attach_outcome_availability_guard",
    "build_labeled_outcomes",
    "filter_outcomes_known_by",
]
