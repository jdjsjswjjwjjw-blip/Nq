"""أول انتقال متحقق بعد حالة مرصودة — ليس تصنيف سيناريو جاهز.

محرّك السيناريو (``expansion_testing`` / ``accepting`` / ``rejection`` /
``repriced_balance``) يبقى **وصفًا/ملمحًا** في المتجه، لا الـY الوحيدة.

الهدف هنا: ``State(t) → أول حركة هندسية/نبضية خلال الأفق``.
عدم وجود ريتست معلومة، لا فشل سيناريو. المسار الكامل سلسلة لاحقة؛ هذه الطبقة
تتعلّم الخطوة التالية فقط.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import (
    FIRST_TRANSITION_CLASS_COL,
    FIRST_TRANSITION_CLASSES,
    OUTCOME_AVAILABLE_TS,
    PRIMARY_OUTCOME_TARGETS,
    SETUP_AVAILABILITY_TS,
    build_first_transition_outcomes,
)
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_causal_order

_ACTIVE = 0.5


class _SetupScan(TypedDict):
    i: int
    ts_i: int
    ts_last: int
    ts_resolved: int
    group: int
    visible: int
    window: int
    resolved_class: str | None
    ambiguous: bool
    seen: frozenset[str]
    horizon: int


def _active(arr: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(arr) > _ACTIVE, dtype=bool)


def _col_array(frame: pl.DataFrame, name: str, n: int) -> np.ndarray:
    if name not in frame.columns:
        return np.zeros(n, dtype=np.float64)
    raw = frame[name].fill_null(0.0).to_list()
    out = np.zeros(n, dtype=np.float64)
    for i, value in enumerate(raw[:n]):
        out[i] = float(value)
    return out


#: أول انتقال مرصود داخل الأفق — ليست فصول الإسقاط الجاهزة.
REALIZED_NEXT_PATH_CLASSES = (
    "further_beyond_asia",
    "return_inside_asia_va",
    "value_built",
    "value_migrated",
    "continue_direction",
    "reverse_path",
    "no_material_change",
)

REALIZED_PATH_BINARY_TARGETS = (
    "y_path_further_beyond",
    "y_path_return_inside",
    "y_path_value_migrated",
    "y_path_value_built",
    "y_path_continue",
    "y_path_reverse",
)

#: امتداد 5 نقاط NQ خلال 25 دقيقة (50 برميل × 30ث) — بجانب ثنائيات المسار، لا بدلًا منها.
Y_EXTEND_5PTS_25MIN = "y_extend_5pts_25min"
EXTEND_HORIZON_TARGETS = (Y_EXTEND_5PTS_25MIN,)
EXTEND_HORIZON_BARS = 50
EXTEND_HORIZON_POINTS = 5.0
_FIXED_POINT_FLOOR = 1.0 / float(PRICE_SCALE)

_BEYOND = "path_beyond_asia_ticks"
_INSIDE = "path_inside_asia_va"
_POC_STEP = "proj_poc_step_ticks"
_LOOK_FAIL = "vp_look_fail"
_EXPAND = "vp_fsm_expand"
_BREAK = "vp_fsm_break"
_RETEST = "vp_fsm_retest"
_BALANCE = "vp_balance"
_ABSORB = "vp_absorb"
_CLOSE_IN_VALUE = "vp_close_in_value"
_FURTHER_TICKS = 1.0
_POC_TICKS = 1.0
_ONSET = 0.5


def _competing_schema() -> dict[str, pl.DataType]:
    return {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        FIRST_TRANSITION_CLASS_COL: pl.Utf8(),
        "horizon_bars": pl.Int64(),
        "group_id": pl.Int64(),
        "label_status": pl.Utf8(),
    }


def _binary_schema() -> dict[str, pl.DataType]:
    return {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        OUTCOME_AVAILABLE_TS: pl.Int64(),
        "outcome_name": pl.Utf8(),
        "y": pl.Float64(),
        "horizon_bars": pl.Int64(),
        "group_id": pl.Int64(),
        "label_status": pl.Utf8(),
    }


def _onset_mask(
    beyond: np.ndarray,
    brk: np.ndarray,
    retest: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """إعداد عندما يبدأ خروج/كسر/ريتست متحقق — ليس عند ``proj_expansion_testing``."""
    n = beyond.size
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        beyond_on = beyond[i] > _ONSET
        brk_on = brk[i]
        retest_on = retest[i]
        if not (beyond_on or brk_on or retest_on):
            continue
        if i > 0 and groups[i] == groups[i - 1]:
            prev_beyond = beyond[i - 1] > _ONSET
            prev_brk = brk[i - 1]
            prev_retest = retest[i - 1]
            if beyond_on and prev_beyond and not brk_on and not retest_on:
                continue
            if brk_on and prev_brk:
                continue
            if retest_on and prev_retest and not beyond_on and not brk_on:
                continue
        out[i] = True
    return out


def _hits_at(
    *,
    j: int,
    i: int,
    beyond: np.ndarray,
    inside: np.ndarray,
    poc_step: np.ndarray,
    look_fail: np.ndarray,
    expand: np.ndarray,
    built: np.ndarray,
) -> list[str]:
    """حركات مرصودة في البرميل ``j`` بعد إعداد ``i`` — بلا فصول إسقاط."""
    hits: list[str] = []
    started_outside = beyond[i] > _ONSET
    returned = started_outside and inside[j] > _ONSET
    retraced = started_outside and (beyond[j] <= beyond[i] - _FURTHER_TICKS) and not returned
    poc_move = abs(float(poc_step[j])) >= _POC_TICKS
    if beyond[j] >= beyond[i] + _FURTHER_TICKS:
        hits.append("further_beyond_asia")
    if returned:
        hits.append("return_inside_asia_va")
    if retraced or look_fail[j]:
        hits.append("reverse_path")
    if poc_move:
        hits.append("value_migrated")
    if expand[j]:
        hits.append("continue_direction")
    if (
        (not started_outside)
        and inside[j] > _ONSET
        and (not poc_move)
        and abs(float(beyond[j] - beyond[i])) < _FURTHER_TICKS
        and built[j]
    ):
        hits.append("value_built")
    return hits


def _scan_setups(
    frame: pl.DataFrame,
    *,
    window: int,
    group_col: str | None,
    progress: ProgressLike | None,
    label: str,
) -> list[_SetupScan]:
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
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
    inside = _col_array(work, _INSIDE, n)
    poc_step = _col_array(work, _POC_STEP, n)
    look_fail = _active(_col_array(work, _LOOK_FAIL, n))
    expand = _active(_col_array(work, _EXPAND, n))
    brk = _active(_col_array(work, _BREAK, n))
    retest = _active(_col_array(work, _RETEST, n))
    built = (
        _active(_col_array(work, _BALANCE, n))
        | _active(_col_array(work, _ABSORB, n))
        | _active(_col_array(work, _CLOSE_IN_VALUE, n))
    )
    onset = _onset_mask(beyond, brk, retest, groups)

    rows: list[_SetupScan] = []
    if progress is not None:
        progress.op(f"{label} bars={n:,} window={window}")
    for i in range(n):
        if progress is not None:
            progress.heartbeat(i + 1, n, label=label)
        if not onset[i]:
            continue
        visible = 0
        last_j = i
        resolved_class: str | None = None
        resolved_j = -1
        ambiguous = False
        seen: set[str] = set()
        for j in range(i + 1, min(n, i + window + 1)):
            if groups[j] != groups[i]:
                break
            visible += 1
            last_j = j
            hits = _hits_at(
                j=j,
                i=i,
                beyond=beyond,
                inside=inside,
                poc_step=poc_step,
                look_fail=look_fail,
                expand=expand,
                built=built,
            )
            for name in hits:
                seen.add(name)
            if resolved_class is None and not ambiguous:
                if len(hits) > 1:
                    ambiguous = True
                    resolved_j = j
                elif len(hits) == 1:
                    resolved_class = hits[0]
                    resolved_j = j
        rows.append(
            {
                "i": i,
                "ts_i": int(ts[i]),
                "ts_last": int(ts[last_j]),
                "ts_resolved": int(ts[resolved_j]) if resolved_j >= 0 else int(ts[last_j]),
                "group": int(groups[i]),
                "visible": int(visible),
                "window": int(window),
                "resolved_class": resolved_class,
                "ambiguous": bool(ambiguous),
                "seen": frozenset(seen),
                "horizon": int((resolved_j if resolved_j >= 0 else last_j) - i),
            }
        )
    return rows


def build_realized_next_path_outcomes(
    frame: pl.DataFrame,
    *,
    window: int = 30,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """أول انتقال مرصود بعد إعداد حدث فعلًا — softmax لاحقًا مجموعه 1.

    لا يستخدم فصول الإسقاط كـY. ``proj_*`` تبقى ملامح. نافذة ناقصة = censored.
    حركتان في نفس البرميل = ambiguous. كسر بلا ريتست إعداد صالح.
    """
    schema = _competing_schema()
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return pl.DataFrame(schema=schema)
    if group_col is not None and group_col not in frame.columns:
        raise ValueError(f"group_col is missing: {group_col}")
    setups = _scan_setups(
        frame, window=window, group_col=group_col, progress=progress, label="realized-next-path"
    )
    out_rows: list[dict[str, object]] = []
    for row in setups:
        if row["ambiguous"]:
            out_rows.append(
                {
                    SETUP_AVAILABILITY_TS: row["ts_i"],
                    OUTCOME_AVAILABLE_TS: row["ts_resolved"],
                    FIRST_TRANSITION_CLASS_COL: None,
                    "horizon_bars": row["horizon"],
                    "group_id": row["group"],
                    "label_status": "ambiguous",
                }
            )
            continue
        if row["resolved_class"] is not None:
            out_rows.append(
                {
                    SETUP_AVAILABILITY_TS: row["ts_i"],
                    OUTCOME_AVAILABLE_TS: row["ts_resolved"],
                    FIRST_TRANSITION_CLASS_COL: row["resolved_class"],
                    "horizon_bars": row["horizon"],
                    "group_id": row["group"],
                    "label_status": "resolved",
                }
            )
            continue
        if int(row["visible"]) >= int(row["window"]):
            out_rows.append(
                {
                    SETUP_AVAILABILITY_TS: row["ts_i"],
                    OUTCOME_AVAILABLE_TS: row["ts_last"],
                    FIRST_TRANSITION_CLASS_COL: "no_material_change",
                    "horizon_bars": row["horizon"],
                    "group_id": row["group"],
                    "label_status": "resolved",
                }
            )
        else:
            out_rows.append(
                {
                    SETUP_AVAILABILITY_TS: row["ts_i"],
                    OUTCOME_AVAILABLE_TS: row["ts_last"],
                    FIRST_TRANSITION_CLASS_COL: None,
                    "horizon_bars": row["horizon"],
                    "group_id": row["group"],
                    "label_status": "censored",
                }
            )
    out = pl.DataFrame(out_rows, schema=schema) if out_rows else pl.DataFrame(schema=schema)
    if progress is not None:
        progress.op(f"realized_next_path rows={out.height:,}")
    return out


def build_realized_path_binary_outcomes(
    frame: pl.DataFrame,
    *,
    window: int = 30,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """ثنائيات مسار: هل الحركة الهندسية حصلت داخل النافذة (لا: أي فصل إسقاط)."""
    schema = _binary_schema()
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return pl.DataFrame(schema=schema)
    if group_col is not None and group_col not in frame.columns:
        raise ValueError(f"group_col is missing: {group_col}")
    setups = _scan_setups(
        frame, window=window, group_col=group_col, progress=progress, label="realized-path-binary"
    )
    name_to_class = {
        "y_path_further_beyond": "further_beyond_asia",
        "y_path_return_inside": "return_inside_asia_va",
        "y_path_value_migrated": "value_migrated",
        "y_path_value_built": "value_built",
        "y_path_continue": "continue_direction",
        "y_path_reverse": "reverse_path",
    }
    out_rows: list[dict[str, object]] = []
    for row in setups:
        window_complete = int(row["visible"]) >= int(row["window"])
        for outcome_name, class_name in name_to_class.items():
            happened = class_name in row["seen"]
            if happened:
                status = "resolved"
                y = 1.0
            elif window_complete:
                status = "resolved"
                y = 0.0
            else:
                status = "censored"
                y = 0.0
            out_rows.append(
                {
                    SETUP_AVAILABILITY_TS: row["ts_i"],
                    OUTCOME_AVAILABLE_TS: row["ts_resolved"] if happened else row["ts_last"],
                    "outcome_name": outcome_name,
                    "y": y,
                    "horizon_bars": row["horizon"],
                    "group_id": row["group"],
                    "label_status": status,
                }
            )
    out = pl.DataFrame(out_rows, schema=schema) if out_rows else pl.DataFrame(schema=schema)
    if progress is not None:
        progress.op(f"realized_path_binary rows={out.height:,}")
    return out


def _price_to_points(px: float) -> float:
    """إغلاق ``blended`` ثابت-النقطة → نقاط NQ؛ الاختبارات الصغيرة تبقى بالدولار."""
    value = float(px)
    if abs(value) >= _FIXED_POINT_FLOOR:
        return value * float(PRICE_SCALE)
    return value


def build_extend_horizon_outcomes(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    window: int = EXTEND_HORIZON_BARS,
    extend_points: float = EXTEND_HORIZON_POINTS,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """هل امتد السعر ``extend_points`` نقطة خلال ``window`` برميل بعد onset المسار؟

    النافذة تنظر إلى ``t+1..t+window`` داخل نفس القصة فقط. الاتجاه من
    ``proj_break_direction`` إن وُجد، وإلا من موقع الإغلاق مقابل آسيا VA،
    وإلا أي اتجاه. نافذة ناقصة بلا إصابة = censored.
    """
    schema = _binary_schema()
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if extend_points <= 0.0:
        raise ValueError(f"extend_points must be > 0, got {extend_points}")
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
    brk_dir = _col_array(work, "proj_break_direction", n)
    asia_vah = np.array(
        [_price_to_points(v) for v in _col_array(work, "asia_vah", n)], dtype=np.float64
    )
    asia_val = np.array(
        [_price_to_points(v) for v in _col_array(work, "asia_val", n)], dtype=np.float64
    )
    has_asia = "asia_vah" in work.columns and "asia_val" in work.columns

    if progress is not None:
        progress.op(f"extend-horizon bars={n:,} window={window} pts={extend_points}")
    out_rows: list[dict[str, object]] = []
    for i in range(n):
        if progress is not None:
            progress.heartbeat(i + 1, n, label="extend-horizon")
        if not onset[i]:
            continue
        direction = float(brk_dir[i])
        if abs(direction) < _ONSET and has_asia:
            if close_pts[i] >= asia_vah[i] > 0.0:
                direction = 1.0
            elif asia_val[i] > 0.0 and close_pts[i] <= asia_val[i]:
                direction = -1.0
            else:
                direction = 0.0
        up_level = close_pts[i] + float(extend_points)
        down_level = close_pts[i] - float(extend_points)
        visible = 0
        last_j = i
        hit_j = -1
        for j in range(i + 1, min(n, i + window + 1)):
            if groups[j] != groups[i]:
                break
            visible += 1
            last_j = j
            went_up = high_pts[j] >= up_level
            went_down = low_pts[j] <= down_level
            if direction > _ONSET:
                hit = went_up
            elif direction < -_ONSET:
                hit = went_down
            else:
                hit = went_up or went_down
            if hit:
                hit_j = j
                break
        window_complete = visible >= int(window)
        if hit_j >= 0:
            status = "resolved"
            y = 1.0
            out_ts = int(ts[hit_j])
            horizon = int(hit_j - i)
        elif window_complete:
            status = "resolved"
            y = 0.0
            out_ts = int(ts[last_j])
            horizon = int(last_j - i)
        else:
            status = "censored"
            y = 0.0
            out_ts = int(ts[last_j])
            horizon = int(last_j - i)
        out_rows.append(
            {
                SETUP_AVAILABILITY_TS: int(ts[i]),
                OUTCOME_AVAILABLE_TS: out_ts,
                "outcome_name": Y_EXTEND_5PTS_25MIN,
                "y": y,
                "horizon_bars": horizon,
                "group_id": int(groups[i]),
                "label_status": status,
            }
        )
    out = pl.DataFrame(out_rows, schema=schema) if out_rows else pl.DataFrame(schema=schema)
    if progress is not None:
        progress.op(f"extend_horizon rows={out.height:,}")
    return out


def concat_path_and_horizon_binaries(
    frame: pl.DataFrame,
    *,
    path_window: int,
    extend_window: int = EXTEND_HORIZON_BARS,
    extend_points: float = EXTEND_HORIZON_POINTS,
    group_col: str | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """ثنائيات المسار + أفق الامتداد — نفس إعدادات onset، نافذتان مستقلتان."""
    path_binaries = build_realized_path_binary_outcomes(
        frame, window=path_window, group_col=group_col, progress=progress
    )
    horizon = build_extend_horizon_outcomes(
        frame,
        window=extend_window,
        extend_points=extend_points,
        group_col=group_col,
        progress=progress,
    )
    parts = [part for part in (path_binaries, horizon) if part.height]
    if not parts:
        return path_binaries
    if len(parts) == 1:
        return parts[0]
    return pl.concat(parts, how="diagonal_relaxed")


VP_REALIZED_OUTCOME_TARGETS = (
    "y_true_break",
    "y_false_break",
    "y_retest_success",
    "y_retest_fail",
    "y_expansion_continue",
    "y_return_to_value",
)


def competing_family_spec(family: str) -> tuple[str, tuple[str, ...]]:
    """``(family, classes)`` — ``assumed_scripts`` يبقى للتشخيص فقط."""
    key = str(family)
    if key == "assumed_scripts":
        return key, FIRST_TRANSITION_CLASSES
    if key != "realized_path":
        raise ValueError(f"unknown competing_family {family!r}")
    return key, REALIZED_NEXT_PATH_CLASSES


def science_outcome_targets(*, include_assumed_scripts: bool) -> tuple[str, ...]:
    """أهداف الثنائيات: مسار متحقق + نبض VP + أفق الامتداد. القالب الجاهز اختياري."""

    scripts = PRIMARY_OUTCOME_TARGETS if include_assumed_scripts else ()
    return (
        *scripts,
        *VP_REALIZED_OUTCOME_TARGETS,
        *REALIZED_PATH_BINARY_TARGETS,
        *EXTEND_HORIZON_TARGETS,
    )


def build_competing_outcomes_for_family(
    frame: pl.DataFrame,
    *,
    family: str,
    window: int,
    group_col: str | None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يبني تسميات الرأس المتنافس حسب عائلة الهدف."""
    key, _classes = competing_family_spec(family)
    if key == "assumed_scripts":
        return build_first_transition_outcomes(
            frame, window=window, group_col=group_col, progress=progress
        )
    return build_realized_next_path_outcomes(
        frame, window=window, group_col=group_col, progress=progress
    )


__all__ = [
    "EXTEND_HORIZON_BARS",
    "EXTEND_HORIZON_POINTS",
    "EXTEND_HORIZON_TARGETS",
    "REALIZED_NEXT_PATH_CLASSES",
    "REALIZED_PATH_BINARY_TARGETS",
    "VP_REALIZED_OUTCOME_TARGETS",
    "Y_EXTEND_5PTS_25MIN",
    "build_competing_outcomes_for_family",
    "build_extend_horizon_outcomes",
    "build_realized_next_path_outcomes",
    "build_realized_path_binary_outcomes",
    "competing_family_spec",
    "concat_path_and_horizon_binaries",
    "science_outcome_targets",
]
