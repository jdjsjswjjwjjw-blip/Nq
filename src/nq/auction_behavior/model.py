"""نموذج احتمالي تجريبي لسلوك المزاد (بدون RL وبدون هدف PnL)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.auction_behavior.events import event_rate
from nq.auction_behavior.quality import mean_confidence
from nq.auction_behavior.types import BehaviorProbabilities
from nq.contracts.temporal import AVAILABILITY_TS
from nq.models.splitting import purged_walk_forward_split
from nq.validation.leakage import assert_temporal_split

_ACTIVE_FLAG = 0.5


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _rate(frame: pl.DataFrame, col: str) -> float:
    if frame.height == 0 or col not in frame.columns:
        return 0.0
    vals = frame[col].to_numpy().astype(np.float64)
    return float(np.mean(vals > _ACTIVE_FLAG)) if vals.size else 0.0


def estimate_behavior_probabilities(
    blended: pl.DataFrame,
    events: pl.DataFrame,
    *,
    n_splits: int = 3,
    embargo: int = 0,
    purge_samples: int = 1,
    min_train_size: int = 8,
) -> tuple[BehaviorProbabilities, pl.DataFrame]:
    """يقدّر احتمالات السلوك على طيّات purged؛ القياس من OOS فقط.

    التدريب يجمع معدّلات الأحداث شرطيًا على توازن/اختلال؛ الاختبار يقيس
    توافق المعدّلات مع تحقّق الأحداث خارج العينة.
    """
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
    for c in events.columns:
        if c != AVAILABILITY_TS and c in work.columns:
            work = work.with_columns(pl.col(c).fill_null(0.0))

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

    fold_rows: list[dict[str, float | int]] = []
    oos_true_break: list[float] = []
    oos_false_break: list[float] = []
    oos_retest_ok: list[float] = []
    oos_retest_bad: list[float] = []
    oos_expand: list[float] = []
    oos_return: list[float] = []
    oos_bal: list[float] = []
    oos_imb: list[float] = []

    for fold_i, fold in enumerate(folds):
        assert_temporal_split(
            times[fold.train_idx],
            times[fold.test_idx],
            embargo=float(embargo),
        )
        train = work[fold.train_idx]
        test = work[fold.test_idx]
        # احتمالات تجريبية من التدريب فقط (تُقارن بمعدّل OOS للمعايرة)
        train_p_true = event_rate(train, "evt_breakout") * (
            1.0 - event_rate(train, "evt_failed_breakout")
        )
        # تحقّق OOS
        fold_rows.append(
            {
                "fold": fold_i,
                "train_n": int(train.height),
                "test_n": int(test.height),
                "train_p_true_break": train_p_true,
                "oos_break_rate": event_rate(test, "evt_breakout"),
                "oos_failed_break_rate": event_rate(test, "evt_failed_breakout"),
            }
        )
        oos_true_break.append(event_rate(test, "evt_breakout"))
        oos_false_break.append(event_rate(test, "evt_failed_breakout"))
        oos_retest_ok.append(event_rate(test, "evt_retest_success"))
        oos_retest_bad.append(event_rate(test, "evt_retest_fail"))
        oos_expand.append(event_rate(test, "evt_accept_expansion"))
        oos_return.append(event_rate(test, "evt_reject_value"))
        oos_bal.append(_rate(test, "vp_balance"))
        oos_imb.append(_rate(test, "vp_imbalance"))

    if not folds:
        # عيّنة صغيرة: تقدير وصفي كامل مع وسم صريح — ليس ادّعاء OOS.
        probs = BehaviorProbabilities(
            p_balanced=_clip01(_rate(work, "vp_balance")),
            p_imbalanced=_clip01(_rate(work, "vp_imbalance")),
            p_true_break=_clip01(event_rate(work, "evt_breakout")),
            p_false_break=_clip01(event_rate(work, "evt_failed_breakout")),
            p_retest_success=_clip01(event_rate(work, "evt_retest_success")),
            p_retest_fail=_clip01(event_rate(work, "evt_retest_fail")),
            p_expansion_continue=_clip01(event_rate(work, "evt_accept_expansion")),
            p_return_to_value=_clip01(event_rate(work, "evt_reject_value")),
            confidence=_clip01(mean_confidence(work) * 0.5),
            n_samples=int(work.height),
            detail="insufficient folds — descriptive rates only (not OOS claims)",
        )
        return probs, pl.DataFrame(fold_rows)

    conf = mean_confidence(work)
    probs = BehaviorProbabilities(
        p_balanced=_clip01(float(np.mean(oos_bal))),
        p_imbalanced=_clip01(float(np.mean(oos_imb))),
        p_true_break=_clip01(float(np.mean(oos_true_break))),
        p_false_break=_clip01(float(np.mean(oos_false_break))),
        p_retest_success=_clip01(float(np.mean(oos_retest_ok))),
        p_retest_fail=_clip01(float(np.mean(oos_retest_bad))),
        p_expansion_continue=_clip01(float(np.mean(oos_expand))),
        p_return_to_value=_clip01(float(np.mean(oos_return))),
        confidence=_clip01(conf),
        n_samples=int(work.height),
        detail=f"purged OOS folds={len(folds)} · confidence=mean(signal_quality)",
    )
    return probs, pl.DataFrame(fold_rows)
