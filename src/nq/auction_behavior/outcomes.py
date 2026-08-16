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
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_availability_not_before_event, assert_causal_order

OUTCOME_AVAILABLE_TS = "outcome_available_ts"
SETUP_AVAILABILITY_TS = "setup_availability_ts"

#: فئات «أول انتقال» من ``expansion_testing`` — توزيع مشترك مجموعه 1.
#: ``no_transition`` = نافذة مكتملة بلا أي انتقال (فئة صريحة، ليست فشلًا صامتًا).
FIRST_TRANSITION_CLASSES = (
    "expansion_accepting",
    "rejection_return_to_asia",
    "repriced_balance",
    "no_transition",
)
FIRST_TRANSITION_CLASS_COL = "first_transition_class"

#: أعمدة الإسقاط التي تحسم فئة أول انتقال (بترتيب الفئات أعلاه).
_FIRST_TRANSITION_SOURCES = (
    ("expansion_accepting", "proj_expansion_accepting"),
    ("rejection_return_to_asia", "proj_rejection_to_asia"),
    ("repriced_balance", "proj_repriced_balance"),
)

#: أهداف الإسقاط الأساسية (Asia→London) — أولوية التعلم الشرطي.
PRIMARY_OUTCOME_TARGETS = (
    "y_expansion_accepting",
    "y_rejection_return_to_asia",
    "y_repriced_balance",
)

#: مخرجات احتمالية كاملة للنموذج الشرطي (أساسي + أحداث VP/FSM).
OUTCOME_TARGETS = (
    *PRIMARY_OUTCOME_TARGETS,
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
    """تعريف نتيجة واحدة: محفّز إعداد + محكّات حل لاحقة.

    ``trigger_cols`` إن وُجدت تُفعّل الإعداد عند أي عمود نشط؛ وإلا ``trigger_col``.
    """

    name: str
    trigger_col: str
    success_cols: tuple[str, ...]
    fail_cols: tuple[str, ...]
    window: int
    trigger_cols: tuple[str, ...] | None = None


_DEFAULT_SPECS: tuple[OutcomeSpec, ...] = (
    # —— أهداف الإسقاط الصارمة: prediction عند onset · outcome_available_ts لاحقًا ——
    OutcomeSpec(
        name="y_expansion_accepting",
        trigger_col="proj_expansion_testing",
        success_cols=("proj_expansion_accepting",),
        fail_cols=("proj_rejection_to_asia", "proj_repriced_balance"),
        window=30,
    ),
    OutcomeSpec(
        name="y_rejection_return_to_asia",
        trigger_col="proj_expansion_testing",
        trigger_cols=("proj_expansion_testing", "proj_expansion_accepting"),
        success_cols=("proj_rejection_to_asia",),
        fail_cols=("proj_repriced_balance", "proj_value_transferred"),
        window=30,
    ),
    OutcomeSpec(
        name="y_repriced_balance",
        trigger_col="proj_expansion_testing",
        trigger_cols=("proj_expansion_testing", "proj_expansion_accepting"),
        success_cols=("proj_repriced_balance",),
        fail_cols=("proj_rejection_to_asia",),
        window=30,
    ),
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


def build_labeled_outcomes(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    outcome_window: int = 8,
    group_col: str | None = None,
    specs: tuple[OutcomeSpec, ...] | None = None,
    progress: ProgressLike | None = None,
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
        "label_status": pl.Utf8(),
    }
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return pl.DataFrame(schema=schema)
    if outcome_window < 1:
        raise ValueError(f"outcome_window must be >= 1, got {outcome_window}")
    if group_col is not None and group_col not in frame.columns:
        raise ValueError(f"group_col is missing: {group_col}")

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
                window=int(outcome_window)
                if s.name not in PRIMARY_OUTCOME_TARGETS
                else max(int(outcome_window), s.window),
                trigger_cols=s.trigger_cols,
            )
            for s in _DEFAULT_SPECS
        )
    )

    rows: list[dict[str, object]] = []
    n_specs = len(active_specs)
    if progress is not None:
        progress.op(f"build_labeled_outcomes bars={n:,} specs={n_specs}")
    for spec_i, spec in enumerate(active_specs, start=1):
        if progress is not None:
            progress.heartbeat(spec_i, n_specs, label="outcome-specs", force=True)
            progress.op(f"outcome {spec.name} ({spec_i}/{n_specs})")
        trig_names = spec.trigger_cols if spec.trigger_cols else (spec.trigger_col,)
        trigger = np.zeros(n, dtype=bool)
        for col in trig_names:
            trigger |= _active(_col_array(work, col, n))
        # onset داخل كل مجموعة: الصف الأول من القصة previous=False دائمًا
        onset = np.zeros(n, dtype=bool)
        for i in range(n):
            if progress is not None:
                progress.heartbeat(i + 1, n, label=f"onset-{spec.name}")
            if not trigger[i]:
                continue
            if i == 0 or groups[i] != groups[i - 1] or not trigger[i - 1]:
                onset[i] = True
        success = np.zeros(n, dtype=bool)
        for col in spec.success_cols:
            success |= _active(_col_array(work, col, n))
        fail = np.zeros(n, dtype=bool)
        for col in spec.fail_cols:
            fail |= _active(_col_array(work, col, n))

        for i in range(n):
            if progress is not None:
                progress.heartbeat(i + 1, n, label=f"outcome-{spec.name}")
            if not onset[i]:
                continue
            # كم صفًا لاحقًا داخل المجموعة قبل انقطاع القصة؟
            visible = 0
            last_j = i
            for j in range(i + 1, min(n, i + spec.window + 1)):
                if groups[j] != groups[i]:
                    break
                visible += 1
                last_j = j
            window_complete = visible >= int(spec.window)

            resolved = False
            for j in range(i + 1, min(n, i + spec.window + 1)):
                if groups[j] != groups[i]:
                    break
                if success[j] and fail[j]:
                    # حدثان متعارضان في نفس البرميل لا يملكان ترتيبًا داخليًا
                    # يمكن إثباته من الإطار المجمّع؛ لا نحوّلهما إلى فشل صامت.
                    rows.append(
                        {
                            SETUP_AVAILABILITY_TS: int(ts[i]),
                            OUTCOME_AVAILABLE_TS: int(ts[j]),
                            "outcome_name": spec.name,
                            "y": float("nan"),
                            "horizon_bars": int(j - i),
                            "group_id": int(groups[i]),
                            "label_status": "ambiguous",
                        }
                    )
                    resolved = True
                    break
                if success[j]:
                    rows.append(
                        {
                            SETUP_AVAILABILITY_TS: int(ts[i]),
                            OUTCOME_AVAILABLE_TS: int(ts[j]),
                            "outcome_name": spec.name,
                            "y": 1.0,
                            "horizon_bars": int(j - i),
                            "group_id": int(groups[i]),
                            "label_status": "resolved",
                        }
                    )
                    resolved = True
                    break
                if fail[j]:
                    rows.append(
                        {
                            SETUP_AVAILABILITY_TS: int(ts[i]),
                            OUTCOME_AVAILABLE_TS: int(ts[j]),
                            "outcome_name": spec.name,
                            "y": 0.0,
                            "horizon_bars": int(j - i),
                            "group_id": int(groups[i]),
                            "label_status": "resolved",
                        }
                    )
                    resolved = True
                    break
            if not resolved:
                # نافذة غير مكتملة → right-censored (لا تُحسب فشلًا في التدريب/التقييم)
                if not window_complete:
                    rows.append(
                        {
                            SETUP_AVAILABILITY_TS: int(ts[i]),
                            OUTCOME_AVAILABLE_TS: int(ts[last_j]),
                            "outcome_name": spec.name,
                            "y": float("nan"),
                            "horizon_bars": int(last_j - i),
                            "group_id": int(groups[i]),
                            "label_status": "censored",
                        }
                    )
                    continue
                # نافذة مكتملة بلا حسم → فشل محسم عند آخر صف في النافذة
                rows.append(
                    {
                        SETUP_AVAILABILITY_TS: int(ts[i]),
                        OUTCOME_AVAILABLE_TS: int(ts[last_j]),
                        "outcome_name": spec.name,
                        "y": 0.0,
                        "horizon_bars": int(last_j - i),
                        "group_id": int(groups[i]),
                        "label_status": "resolved",
                    }
                )

    out = pl.DataFrame(rows) if rows else pl.DataFrame(schema=schema)
    if progress is not None:
        progress.op(f"labeled outcomes rows={out.height:,}")
    if out.height:
        # التحقق الزمني على الصفوف المحسومة فقط (censored قد يحمل NaN في y)
        known = out.filter(pl.col("label_status") == "resolved")
        if known.height:
            assert_availability_not_before_event(
                known[SETUP_AVAILABILITY_TS].to_numpy(),
                known[OUTCOME_AVAILABLE_TS].to_numpy(),
            )
            if bool(
                np.any(
                    known[OUTCOME_AVAILABLE_TS].to_numpy() < known[SETUP_AVAILABILITY_TS].to_numpy()
                )
            ):
                raise AssertionError("outcome_available_ts before setup_availability_ts")
    return out


def build_first_transition_outcomes(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    window: int = 30,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """توزيع مشترك للنتائج المتنافسة: أول انتقال بعد onset ``expansion_testing``.

    كل إعداد يُنسب إلى **فئة واحدة بالضبط** (أو censored عند نافذة ناقصة):
    أول ظهور لأي من قبول/رفض/إعادة تسعير داخل النافذة يحسم الفئة عند لحظة
    ظهوره (``outcome_available_ts``). نافذة مكتملة بلا انتقال = فئة صريحة
    ``no_transition`` تُحسم عند نهاية النافذة. ظهور فئتين في نفس البرميل =
    ``ambiguous`` (لا ترتيب داخلي مثبت) ولا يدخل التدريب.
    """
    schema = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        FIRST_TRANSITION_CLASS_COL: pl.Utf8(),
        "horizon_bars": pl.Int64(),
        "group_id": pl.Int64(),
        "label_status": pl.Utf8(),
    }
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return pl.DataFrame(schema=schema)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
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
    trigger = _active(_col_array(work, "proj_expansion_testing", n))
    class_active = {
        name: _active(_col_array(work, col, n)) for name, col in _FIRST_TRANSITION_SOURCES
    }

    rows: list[dict[str, object]] = []
    if progress is not None:
        progress.op(f"build_first_transition_outcomes bars={n:,} window={window}")
    for i in range(n):
        if progress is not None:
            progress.heartbeat(i + 1, n, label="first-transition")
        if not trigger[i]:
            continue
        if i > 0 and groups[i] == groups[i - 1] and trigger[i - 1]:
            continue  # onset فقط — لا إعداد جديد لكل بار داخل نفس الاختبار

        visible = 0
        last_j = i
        resolved_class: str | None = None
        resolved_j = -1
        ambiguous = False
        for j in range(i + 1, min(n, i + window + 1)):
            if groups[j] != groups[i]:
                break
            visible += 1
            last_j = j
            hits = [name for name, active in class_active.items() if active[j]]
            if len(hits) > 1:
                ambiguous = True
                resolved_j = j
                break
            if len(hits) == 1:
                resolved_class = hits[0]
                resolved_j = j
                break

        if ambiguous:
            rows.append(
                {
                    SETUP_AVAILABILITY_TS: int(ts[i]),
                    OUTCOME_AVAILABLE_TS: int(ts[resolved_j]),
                    FIRST_TRANSITION_CLASS_COL: None,
                    "horizon_bars": int(resolved_j - i),
                    "group_id": int(groups[i]),
                    "label_status": "ambiguous",
                }
            )
            continue
        if resolved_class is not None:
            rows.append(
                {
                    SETUP_AVAILABILITY_TS: int(ts[i]),
                    OUTCOME_AVAILABLE_TS: int(ts[resolved_j]),
                    FIRST_TRANSITION_CLASS_COL: resolved_class,
                    "horizon_bars": int(resolved_j - i),
                    "group_id": int(groups[i]),
                    "label_status": "resolved",
                }
            )
            continue
        if visible >= window:
            rows.append(
                {
                    SETUP_AVAILABILITY_TS: int(ts[i]),
                    OUTCOME_AVAILABLE_TS: int(ts[last_j]),
                    FIRST_TRANSITION_CLASS_COL: "no_transition",
                    "horizon_bars": int(last_j - i),
                    "group_id": int(groups[i]),
                    "label_status": "resolved",
                }
            )
        else:
            rows.append(
                {
                    SETUP_AVAILABILITY_TS: int(ts[i]),
                    OUTCOME_AVAILABLE_TS: int(ts[last_j]),
                    FIRST_TRANSITION_CLASS_COL: None,
                    "horizon_bars": int(last_j - i),
                    "group_id": int(groups[i]),
                    "label_status": "censored",
                }
            )

    out = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    if out.height:
        known = out.filter(pl.col("label_status") == "resolved")
        if known.height:
            assert_availability_not_before_event(
                known[SETUP_AVAILABILITY_TS].to_numpy(),
                known[OUTCOME_AVAILABLE_TS].to_numpy(),
            )
    if progress is not None:
        progress.op(f"first_transition rows={out.height:,}")
    return out


def filter_resolved_outcomes(outcomes: pl.DataFrame) -> pl.DataFrame:
    """يستبعد right-censored من التدريب والتقييم الكمي."""
    if outcomes.height == 0:
        return outcomes
    if "label_status" not in outcomes.columns:
        return outcomes
    return outcomes.filter(pl.col("label_status") == "resolved")


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
    if feat[AVAILABILITY_TS].n_unique() != feat.height:
        raise ValueError(
            "features require unique availability_ts for exact outcome join; "
            "duplicate timestamps would multiply labeled setups"
        )
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
    known = outcomes.filter(pl.col(OUTCOME_AVAILABLE_TS) <= int(asof_ts))
    return filter_resolved_outcomes(known)


__all__ = [
    "FIRST_TRANSITION_CLASSES",
    "FIRST_TRANSITION_CLASS_COL",
    "OUTCOME_AVAILABLE_TS",
    "OUTCOME_TARGETS",
    "PRIMARY_OUTCOME_TARGETS",
    "SETUP_AVAILABILITY_TS",
    "OutcomeSpec",
    "attach_outcome_availability_guard",
    "build_first_transition_outcomes",
    "build_labeled_outcomes",
    "filter_outcomes_known_by",
    "filter_resolved_outcomes",
]
