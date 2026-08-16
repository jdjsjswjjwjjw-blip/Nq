"""Holdout نهائي مجمّد — يُقاس مرة واحدة بعد قفل التطوير."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.auction_behavior.calibration import evaluate_calibration
from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.auction_behavior.walk_forward import month_key_from_ns, unique_month_keys
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
    holdout_months: int | None = None


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
    holdout_months: int | None = None,
    ts_col: str = SETUP_AVAILABILITY_TS,
) -> FrozenHoldout:
    """يعزل الذيل الزمني كـholdout نهائي؛ التطوير لا يراه.

    ``holdout_months``: آخر N شهور تقويمية (بروتوكول السنة 4/4/4).
    وإلا يُستخدم ``holdout_frac`` على صفوف الإعداد.
    """
    if holdout_months is not None:
        return carve_frozen_holdout_by_months(
            labeled, n_holdout_months=int(holdout_months), ts_col=ts_col
        )
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


def carve_frozen_holdout_by_months(
    labeled: pl.DataFrame,
    *,
    n_holdout_months: int,
    ts_col: str = SETUP_AVAILABILITY_TS,
) -> FrozenHoldout:
    """Holdout = آخر ``n_holdout_months`` شهور إعداد؛ بلا إعادة بناء للداتا."""
    if n_holdout_months < 1:
        raise ValueError(f"n_holdout_months must be >= 1, got {n_holdout_months}")
    if labeled.height == 0 or ts_col not in labeled.columns:
        return FrozenHoldout(
            develop=labeled,
            holdout=labeled.head(0),
            cut_ts=-1,
            holdout_frac=0.0,
            frozen=True,
            touched=False,
            holdout_months=int(n_holdout_months),
        )
    ordered = labeled.sort(ts_col)
    months = unique_month_keys(ordered, ts_col=ts_col)
    if len(months) < n_holdout_months + 1:
        raise ValueError(
            f"need at least {n_holdout_months + 1} distinct setup months "
            f"to carve {n_holdout_months}-month holdout; got {len(months)}: {list(months)}"
        )
    holdout_keys = months[-n_holdout_months:]
    holdout_set = set(holdout_keys)
    row_months = [month_key_from_ns(int(t)) for t in ordered[ts_col].to_list()]
    tagged = ordered.with_columns(pl.Series("_block_month", row_months, dtype=pl.Utf8))
    develop = tagged.filter(~pl.col("_block_month").is_in(list(holdout_set))).drop("_block_month")
    holdout = tagged.filter(pl.col("_block_month").is_in(list(holdout_set))).drop("_block_month")
    if develop.height == 0 or holdout.height == 0:
        raise ValueError(
            "month holdout carve produced an empty develop or holdout "
            f"(develop={develop.height}, holdout={holdout.height}, months={list(months)})"
        )
    max_ts = develop[ts_col].max()
    cut_ts = -1 if max_ts is None else int(np.asarray(max_ts).item())
    assert_temporal_split(
        develop[ts_col].to_numpy(),
        holdout[ts_col].to_numpy(),
        embargo=0.0,
    )
    return FrozenHoldout(
        develop=develop,
        holdout=holdout,
        cut_ts=cut_ts,
        holdout_frac=float(n_holdout_months) / float(len(months)),
        frozen=True,
        touched=False,
        holdout_months=int(n_holdout_months),
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
        holdout_months=holdout.holdout_months,
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
    "carve_frozen_holdout_by_months",
    "evaluate_frozen_holdout_once",
    "mark_holdout_touched",
]
