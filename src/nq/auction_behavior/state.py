"""تمثيل حالة سوقية واحدة للنموذج الاحتمالي."""

from __future__ import annotations

import numpy as np
import polars as pl

from nq.auction_behavior.projection import PROJECTION_NUMERIC_COLUMNS
from nq.auction_behavior.types import BehaviorStateSnapshot
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION

STATE_FEATURE_COLUMNS = (
    "vp_balance",
    "vp_imbalance",
    "vp_expansion",
    "vp_close_in_value",
    "vp_absorb",
    "vp_look_fail",
    "vp_fsm_break",
    "vp_fsm_retest",
    "vp_fsm_expand",
    "vp_early_imbalance",
    "vp_liquidity_session",
    "deceptive_score",
    "real_liquidity_ratio",
    "signal_quality",
    *PROJECTION_NUMERIC_COLUMNS,
)


def attach_state_vector(frame: pl.DataFrame) -> pl.DataFrame:
    """يضمن وجود أعمدة الحالة الرقمية (يملأ الغائب بأصفار)."""
    work = frame
    for col in STATE_FEATURE_COLUMNS:
        if col not in work.columns:
            work = work.with_columns(pl.lit(0.0).alias(col))
        else:
            work = work.with_columns(pl.col(col).cast(pl.Float64).fill_null(0.0))
    return work


def latest_state_snapshot(frame: pl.DataFrame) -> BehaviorStateSnapshot | None:
    """آخر صف سببي كـلقطة حالة."""
    if frame.height == 0 or AVAILABILITY_TS not in frame.columns:
        return None
    row = frame.sort(AVAILABILITY_TS).tail(1)
    sess = (
        int(row[VP_LIQUIDITY_SESSION][0])
        if VP_LIQUIDITY_SESSION in row.columns
        else int(row["vp_liquidity_session"][0])
        if "vp_liquidity_session" in row.columns
        else 0
    )

    def _v(name: str) -> float:
        if name not in row.columns:
            return 0.0
        val = row[name][0]
        return 0.0 if val is None else float(val)

    def _s(name: str) -> str:
        if name not in row.columns or row[name][0] is None:
            return ""
        return str(row[name][0])

    return BehaviorStateSnapshot(
        availability_ts=int(row[AVAILABILITY_TS][0]),
        liquidity_session=sess,
        is_balanced=_v("vp_balance"),
        is_expansion=_v("vp_expansion"),
        close_in_value=_v("vp_close_in_value"),
        absorb=_v("vp_absorb"),
        look_fail=_v("vp_look_fail"),
        fsm_break=_v("vp_fsm_break"),
        fsm_retest=_v("vp_fsm_retest"),
        fsm_expand=_v("vp_fsm_expand"),
        early_imbalance=_v("vp_early_imbalance"),
        deceptive_score=_v("deceptive_score"),
        real_liquidity_ratio=_v("real_liquidity_ratio"),
        signal_quality=_v("signal_quality"),
        auction_phase=_s("auction_phase"),
        asia_poc=_v("asia_poc"),
        asia_vah=_v("asia_vah"),
        asia_val=_v("asia_val"),
        composite_poc=_v("composite_poc"),
        composite_vah=_v("composite_vah"),
        composite_val=_v("composite_val"),
        projection_anchor_complete=_v("proj_anchor_complete"),
        projection_expansion_active=_v("proj_expansion_active"),
        projection_value_transferred=_v("proj_value_transferred"),
    )


def state_matrix(frame: pl.DataFrame) -> np.ndarray:
    """مصفوفة خصائص الحالة للطيّات."""
    present = [c for c in STATE_FEATURE_COLUMNS if c in frame.columns]
    if not present:
        return np.zeros((frame.height, 0), dtype=np.float64)
    return frame.select(present).fill_null(0.0).to_numpy().astype(np.float64)
