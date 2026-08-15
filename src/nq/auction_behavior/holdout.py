"""Holdout نهائي مجمّد — يُقاس مرة واحدة بعد قفل التطوير."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import evaluate_calibration
from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.validation.leakage import assert_temporal_split

_HOLDOUT_FRAC_MIN = 0.05
_HOLDOUT_FRAC_MAX = 0.5
_MIN_ROWS_FOR_SPLIT = 2


@dataclass(frozen=True, slots=True)
class FrozenHoldout:
    """تقسيم تطوير / Holdout مجمّد."""

    develop: pl.DataFrame
    holdout: pl.DataFrame
    cut_ts: int
    holdout_frac: float
    frozen: bool = True
    touched: bool = False


@dataclass(frozen=True, slots=True)
class HoldoutEvaluation:
    """نتيجة تقييم واحد على الـholdout بعد القفل."""

    n: int
    brier: float
    ece: float
    mae: float
    evaluated: bool
    detail: str = ""


def carve_frozen_holdout(
    labeled: pl.DataFrame,
    *,
    holdout_frac: float = 0.2,
    ts_col: str = SETUP_AVAILABILITY_TS,
) -> FrozenHoldout:
    """يعزل الذيل الزمني كـholdout نهائي؛ التطوير لا يراه."""
    if not _HOLDOUT_FRAC_MIN <= holdout_frac <= _HOLDOUT_FRAC_MAX:
        raise ValueError(
            f"holdout_frac must be in [{_HOLDOUT_FRAC_MIN}, {_HOLDOUT_FRAC_MAX}], "
            f"got {holdout_frac}"
        )
    if labeled.height == 0 or ts_col not in labeled.columns:
        return FrozenHoldout(
            develop=labeled,
            holdout=labeled.head(0),
            cut_ts=-1,
            holdout_frac=holdout_frac,
            frozen=True,
            touched=False,
        )
    ordered = labeled.sort(ts_col)
    n = ordered.height
    cut = int(n * (1.0 - holdout_frac))
    cut = min(max(cut, 1), n - 1) if n >= _MIN_ROWS_FOR_SPLIT else n
    provisional = ordered.head(cut)
    if provisional.height == 0:
        return FrozenHoldout(
            develop=ordered,
            holdout=ordered.head(0),
            cut_ts=-1,
            holdout_frac=float(holdout_frac),
            frozen=True,
            touched=False,
        )
    max_ts = provisional[ts_col].max()
    cut_ts = -1 if max_ts is None else int(np.asarray(max_ts).item())
    # فصل صارم بالطابع: التطوير <= cut_ts · الـholdout > cut_ts (لا تداخل نفس اللحظة)
    develop = ordered.filter(pl.col(ts_col) <= cut_ts)
    holdout = ordered.filter(pl.col(ts_col) > cut_ts)
    if holdout.height == 0 and n >= _MIN_ROWS_FOR_SPLIT:
        # ذيل صفوف بنفس الطابع — انقل آخر طابع فريد للـholdout
        uniq = ordered[ts_col].unique(maintain_order=True)
        if uniq.len() >= _MIN_ROWS_FOR_SPLIT:
            cut_ts = int(np.asarray(uniq[-2]).item())
            develop = ordered.filter(pl.col(ts_col) <= cut_ts)
            holdout = ordered.filter(pl.col(ts_col) > cut_ts)
    if develop.height and holdout.height:
        assert_temporal_split(
            develop[ts_col].to_numpy(),
            holdout[ts_col].to_numpy(),
            embargo=0.0,
        )
    return FrozenHoldout(
        develop=develop,
        holdout=holdout,
        cut_ts=cut_ts,
        holdout_frac=float(holdout_frac),
        frozen=True,
        touched=False,
    )


def mark_holdout_touched(holdout: FrozenHoldout) -> FrozenHoldout:
    """يُعلَّم أن الـholdout لُمِس — للتشخيص فقط؛ التقييم النهائي يجب أن يرفض اللمس المتكرر."""
    return FrozenHoldout(
        develop=holdout.develop,
        holdout=holdout.holdout,
        cut_ts=holdout.cut_ts,
        holdout_frac=holdout.holdout_frac,
        frozen=holdout.frozen,
        touched=True,
    )


def evaluate_frozen_holdout_once(
    holdout: FrozenHoldout,
    scored_holdout: pl.DataFrame,
    *,
    allow_retouch: bool = False,
) -> tuple[HoldoutEvaluation, FrozenHoldout]:
    """تقييم واحد؛ يرفض إعادة اللمس ما لم يُصرَّح."""
    if holdout.touched and not allow_retouch:
        raise RuntimeError(
            "frozen holdout already touched — refuse repeated evaluation "
            "(indirect training on holdout)"
        )
    rep = evaluate_calibration(scored_holdout)
    evaluation = HoldoutEvaluation(
        n=rep.n,
        brier=rep.brier,
        ece=rep.ece,
        mae=rep.mae,
        evaluated=True,
        detail="single frozen holdout evaluation",
    )
    return evaluation, mark_holdout_touched(holdout)


__all__ = [
    "FrozenHoldout",
    "HoldoutEvaluation",
    "carve_frozen_holdout",
    "evaluate_frozen_holdout_once",
    "mark_holdout_touched",
]
