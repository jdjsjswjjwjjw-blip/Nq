"""تحقق وتعميم لفهم سلوك المزاد (OOS + فحوص تسريب)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.validation.leakage import assert_availability_not_before_event, assert_causal_order


@dataclass(frozen=True, slots=True)
class BehaviorValidationReport:
    """نتيجة تحقق المرحلة الأولى."""

    ok: bool
    n_rows: int
    n_folds: int
    causal_ok: bool
    decision_bounds_present: bool
    no_trade_outputs: bool
    detail: str


_TRADE_FORBIDDEN = (
    "edge_pnl",
    "entry_gate",
    "edge_entry",
    "edge_stop",
    "edge_target",
    "position_size",
)


def validate_behavior_frame(
    frame: pl.DataFrame,
    *,
    fold_df: pl.DataFrame | None = None,
    event_ts_col: str | None = None,
) -> BehaviorValidationReport:
    """يفحص أن الإطار سببي وخالٍ من مخرجات تداول، وأن ``decision_*`` موجودة عند التوفر."""
    n = int(frame.height)
    if n == 0:
        return BehaviorValidationReport(
            ok=True,
            n_rows=0,
            n_folds=0 if fold_df is None else int(fold_df.height),
            causal_ok=True,
            decision_bounds_present=False,
            no_trade_outputs=True,
            detail="empty",
        )

    causal_ok = True
    detail_parts: list[str] = []
    if AVAILABILITY_TS in frame.columns:
        try:
            assert_causal_order(frame[AVAILABILITY_TS].to_numpy())
        except Exception as exc:
            causal_ok = False
            detail_parts.append(f"availability_order:{exc}")
    if event_ts_col and event_ts_col in frame.columns and AVAILABILITY_TS in frame.columns:
        try:
            assert_availability_not_before_event(
                frame[event_ts_col].to_numpy(),
                frame[AVAILABILITY_TS].to_numpy(),
            )
        except Exception as exc:
            causal_ok = False
            detail_parts.append(f"availability_vs_event:{exc}")

    decision_ok = all(
        c in frame.columns for c in ("decision_vah", "decision_val", "decision_poc")
    ) or all(c in frame.columns for c in ("vp_upper", "vp_lower", "vp_mid"))
    # vp_* من auction_signals مبنية على decision_* داخليًا.

    trade_leak = [c for c in _TRADE_FORBIDDEN if c in frame.columns]
    no_trade = len(trade_leak) == 0
    if not no_trade:
        detail_parts.append(f"forbidden_trade_cols={trade_leak}")

    n_folds = 0 if fold_df is None else int(fold_df.height)
    ok = causal_ok and no_trade
    if ok:
        detail_parts.append("behavior_validation_passed")
    return BehaviorValidationReport(
        ok=ok,
        n_rows=n,
        n_folds=n_folds,
        causal_ok=causal_ok,
        decision_bounds_present=decision_ok,
        no_trade_outputs=no_trade,
        detail="; ".join(detail_parts),
    )


def calibration_error(
    predicted_rate: float,
    realized_rate: float,
) -> float:
    """خطأ معايرة بسيط |p̂ − p|."""
    return float(abs(float(predicted_rate) - float(realized_rate)))


def mean_absolute_calibration(fold_df: pl.DataFrame) -> float:
    """متوسط |train_p − oos_rate| إن وُجدت الأعمدة."""
    if fold_df.height == 0:
        return 0.0
    if "train_p_true_break" not in fold_df.columns or "oos_break_rate" not in fold_df.columns:
        return 0.0
    a = fold_df["train_p_true_break"].to_numpy().astype(np.float64)
    b = fold_df["oos_break_rate"].to_numpy().astype(np.float64)
    return float(np.mean(np.abs(a - b)))
