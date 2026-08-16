"""تقدير احتمالات سلوك المزاد بلا تداول أو هدف PnL."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from nq.auction_behavior.events import event_rate
from nq.auction_behavior.types import BehaviorProbabilities
from nq.contracts.temporal import AVAILABILITY_TS
from nq.models.splitting import purged_walk_forward_split
from nq.research.progress import ProgressLike
from nq.validation.leakage import assert_temporal_split

_ACTIVE_FLAG = 0.5


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _rate(frame: pl.DataFrame, col: str) -> float:
    if frame.height == 0 or col not in frame.columns:
        return 0.0
    vals = frame[col].to_numpy().astype(np.float64)
    return float(np.mean(vals > _ACTIVE_FLAG)) if vals.size else 0.0


def _brier(frame: pl.DataFrame, column: str, prediction: float) -> float:
    if frame.height == 0 or column not in frame.columns:
        return 0.0
    y = (frame[column].fill_null(0.0).to_numpy().astype(np.float64) > _ACTIVE_FLAG).astype(
        np.float64
    )
    return float(np.mean(np.square(y - prediction))) if y.size else 0.0


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    total = sum(weight for _, weight in values)
    if total <= 0:
        return 0.0
    return float(sum(value * weight for value, weight in values) / total)


_TARGETS: tuple[tuple[str, str, Callable[[pl.DataFrame, str], float]], ...] = (
    ("balanced", "vp_balance", _rate),
    ("imbalanced", "vp_imbalance", _rate),
    ("true_break", "evt_true_break", event_rate),
    ("false_break", "evt_failed_breakout", event_rate),
    ("retest_success", "evt_retest_success", event_rate),
    ("retest_fail", "evt_retest_fail", event_rate),
    ("expansion_continue", "evt_expansion_continue", event_rate),
    ("return_to_value", "evt_return_to_value", event_rate),
)


def _descriptive_probabilities(work: pl.DataFrame) -> BehaviorProbabilities:
    """معدلات وصفية على كل الإطار (بما فيه المستقبل) — ليست تنبؤات OOS.

    ``confidence=0.0`` إلزامي: لا يوجد أي تحقق خارج العينة هنا، وأي ثقة
    موجبة قد تُغري باستخدام هذه المعدلات كتوقعات مدرَّبة (خداع ذاتي).
    """
    rates = {name: fn(work, column) for name, column, fn in _TARGETS}
    return BehaviorProbabilities(
        p_balanced=_clip01(rates["balanced"]),
        p_imbalanced=_clip01(rates["imbalanced"]),
        p_true_break=_clip01(rates["true_break"]),
        p_false_break=_clip01(rates["false_break"]),
        p_retest_success=_clip01(rates["retest_success"]),
        p_retest_fail=_clip01(rates["retest_fail"]),
        p_expansion_continue=_clip01(rates["expansion_continue"]),
        p_return_to_value=_clip01(rates["return_to_value"]),
        confidence=0.0,
        n_samples=int(work.height),
        detail=(
            "insufficient folds — full-sample descriptive rates only "
            "(includes future; not OOS forecasts; confidence forced to 0)"
        ),
    )


def estimate_behavior_probabilities(  # noqa: PLR0915
    blended: pl.DataFrame,
    events: pl.DataFrame,
    *,
    n_splits: int = 3,
    embargo: int = 0,
    purge_samples: int = 1,
    min_train_size: int = 8,
    progress: ProgressLike | None = None,
) -> tuple[BehaviorProbabilities, pl.DataFrame]:
    """يعيد توقعات base-rate من التدريب فقط ومقاييس تحقق OOS منفصلة.

    هذه المرحلة لا تدّعي نموذجًا شرطيًا لكل حالة. لكل طيّة يُقدَّر المعدّل من
    ``train`` فقط، ثم يُقارن بنتيجة ``test``. لذلك لا تدخل نتيجة الاختبار في
    الاحتمال المعلن، وتبقى معدلات OOS أعمدة تحقق لا تنبؤات.
    """
    if progress is not None:
        progress.op(f"estimate_behavior_probabilities bars={blended.height:,}")
    if blended.height == 0 or AVAILABILITY_TS not in blended.columns:
        empty = BehaviorProbabilities(
            p_balanced=0.0,
            p_imbalanced=0.0,
            p_true_break=0.0,
            p_false_break=0.0,
            p_retest_success=0.0,
            p_retest_fail=0.0,
            p_expansion_continue=0.0,
            p_return_to_value=0.0,
            confidence=0.0,
            n_samples=0,
            detail="empty frame",
        )
        return empty, pl.DataFrame()

    work = blended.join(events, on=AVAILABILITY_TS, how="left").sort(AVAILABILITY_TS)
    event_cols = [c for c in events.columns if c != AVAILABILITY_TS and c in work.columns]
    if event_cols:
        work = work.with_columns(pl.col(c).fill_null(0.0) for c in event_cols)

    times = work[AVAILABILITY_TS].to_numpy().astype(np.int64)
    try:
        folds = purged_walk_forward_split(
            times,
            n_splits=max(1, int(n_splits)),
            embargo=max(0, int(embargo)),
            purge_samples=max(0, int(purge_samples)),
            min_train_size=max(1, int(min_train_size)),
        )
    except ValueError:
        folds = []
    if not folds:
        if progress is not None:
            progress.op("base-rate: insufficient folds — descriptive rates")
        return _descriptive_probabilities(work), pl.DataFrame()

    fold_rows: list[dict[str, float | int]] = []
    forecasts: dict[str, list[tuple[float, int]]] = {name: [] for name, _, _ in _TARGETS}
    total_oos = 0
    n_folds = len(folds)
    if progress is not None:
        progress.op(f"base-rate folds={n_folds}")
    for fold_i, fold in enumerate(folds):
        if progress is not None:
            progress.heartbeat(fold_i + 1, n_folds, label="base-rate-folds", force=True)
        assert_temporal_split(times[fold.train_idx], times[fold.test_idx], embargo=float(embargo))
        train = work[fold.train_idx]
        test = work[fold.test_idx]
        if progress is not None:
            progress.op(
                f"base-rate fold {fold_i + 1}/{n_folds} train={train.height:,} test={test.height:,}"
            )
        test_n = int(test.height)
        total_oos += test_n
        row: dict[str, float | int] = {
            "fold": fold_i,
            "train_n": int(train.height),
            "test_n": test_n,
            "train_end_ts": int(times[fold.train_idx].max()),
            "test_start_ts": int(times[fold.test_idx].min()),
            "test_end_ts": int(times[fold.test_idx].max()),
        }
        calibration: list[float] = []
        briers: list[float] = []
        for name, column, fn in _TARGETS:
            predicted_rate = _clip01(fn(train, column))
            realized = _clip01(fn(test, column))
            forecasts[name].append((predicted_rate, test_n))
            row[f"train_p_{name}"] = predicted_rate
            row[f"oos_{name}_rate"] = realized
            row[f"oos_{name}_brier"] = _brier(test, column, predicted_rate)
            calibration.append(abs(predicted_rate - realized))
            briers.append(float(row[f"oos_{name}_brier"]))
        row["calibration_mae"] = float(np.mean(calibration))
        row["brier_mean"] = float(np.mean(briers))
        fold_rows.append(row)

    aggregate_predictions = {name: _weighted_mean(values) for name, values in forecasts.items()}
    calibration_mae = _weighted_mean(
        [(float(row["calibration_mae"]), int(row["test_n"])) for row in fold_rows]
    )
    support = min(1.0, total_oos / float(max(30, min_train_size * max(1, len(folds)))))
    oos_evidence = _weighted_mean(
        [
            (
                max(
                    float(row["oos_true_break_rate"]),
                    float(row["oos_false_break_rate"]),
                    float(row["oos_retest_success_rate"]),
                    float(row["oos_retest_fail_rate"]),
                    float(row["oos_expansion_continue_rate"]),
                    float(row["oos_return_to_value_rate"]),
                ),
                int(row["test_n"]),
            )
            for row in fold_rows
        ]
    )
    confidence = _clip01((1.0 - calibration_mae) * support * np.sqrt(oos_evidence))
    probs = BehaviorProbabilities(
        p_balanced=_clip01(aggregate_predictions["balanced"]),
        p_imbalanced=_clip01(aggregate_predictions["imbalanced"]),
        p_true_break=_clip01(aggregate_predictions["true_break"]),
        p_false_break=_clip01(aggregate_predictions["false_break"]),
        p_retest_success=_clip01(aggregate_predictions["retest_success"]),
        p_retest_fail=_clip01(aggregate_predictions["retest_fail"]),
        p_expansion_continue=_clip01(aggregate_predictions["expansion_continue"]),
        p_return_to_value=_clip01(aggregate_predictions["return_to_value"]),
        confidence=confidence,
        n_samples=total_oos,
        detail=(
            f"purged walk-forward train-only base-rate forecasts; folds={len(folds)}; "
            "OOS outcomes used for calibration only; not state-conditional"
        ),
    )
    return probs, pl.DataFrame(fold_rows)
