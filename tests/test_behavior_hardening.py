"""اختبارات مضادة لإصلاحات التسريب العلمي (censoring / memory / OOF / PSI / …)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior.conditional import (
    MODEL_STATUS_INSUFFICIENT,
    MODEL_STATUS_SINGLE_CLASS,
    fit_conditional_models,
    select_feature_names_by_family,
)
from nq.auction_behavior.drift import _psi_1d
from nq.auction_behavior.level_flow import _order_lifecycle_by_bucket
from nq.auction_behavior.memory import attach_market_memory
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    OutcomeSpec,
    build_labeled_outcomes,
    filter_resolved_outcomes,
)
from nq.auction_behavior.walk_forward import build_time_folds_for_frame
from nq.contracts.mbo import MboAction
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.simulation.common import BUCKET_START


@pytest.mark.leakage
def test_censored_not_counted_as_failure() -> None:
    """نافذة غير مكتملة → censored وليس y=0."""
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3],
            "group_id": [1, 1, 1],
            "vp_fsm_break": [1.0, 0.0, 0.0],
            "vp_fsm_expand": [0.0, 0.0, 0.0],
            "vp_look_fail": [0.0, 0.0, 0.0],
            "vp_fr_accepted_expansion": [0.0, 0.0, 0.0],
            "vp_fr_exit": [0.0, 0.0, 0.0],
        }
    )
    out = build_labeled_outcomes(
        frame,
        outcome_window=5,
        group_col="group_id",
        specs=(
            OutcomeSpec(
                name="y_true_break",
                trigger_col="vp_fsm_break",
                success_cols=("vp_fsm_expand",),
                fail_cols=("vp_look_fail",),
                window=5,
            ),
        ),
    )
    assert out.height >= 1
    assert (out["label_status"] == "censored").all()
    resolved = filter_resolved_outcomes(out)
    assert resolved.height == 0


@pytest.mark.leakage
def test_onset_resets_at_new_story_group() -> None:
    """trigger نشط عبر حدود القصص ما زال يُنشئ setup في القصة الجديدة."""
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(1, 13)),
            "group_id": [1] * 6 + [2] * 6,
            "vp_fsm_break": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            "vp_fsm_expand": [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            "vp_look_fail": [0.0] * 12,
            "vp_fr_accepted_expansion": [0.0] * 12,
            "vp_fr_exit": [0.0] * 12,
        }
    )
    out = build_labeled_outcomes(
        frame,
        outcome_window=4,
        group_col="group_id",
        specs=(
            OutcomeSpec(
                name="y_true_break",
                trigger_col="vp_fsm_break",
                success_cols=("vp_fsm_expand",),
                fail_cols=("vp_look_fail",),
                window=4,
            ),
        ),
    )
    setups = out.filter(pl.col("label_status") == "resolved")[SETUP_AVAILABILITY_TS].to_list()
    # onset في المجموعة 1 عند ts=2 وفي المجموعة 2 عند ts=7
    assert 2 in setups
    assert 7 in setups


@pytest.mark.leakage
def test_memory_shift_does_not_leak_across_stories() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3, 4, 5, 6],
            "_story": [1, 1, 1, 2, 2, 2],
            "vp_fsm_break": [10.0, 20.0, 30.0, 0.0, 0.0, 0.0],
        }
    )
    out = attach_market_memory(
        frame,
        columns=["vp_fsm_break"],
        lags=(1,),
        roll_windows=(2,),
        group_col="_story",
        event_columns=["vp_fsm_break"],
    )
    # أول صف في القصة 2 يجب ألا يرث rmean/rsum من القصة 1
    row = out.filter(pl.col(AVAILABILITY_TS) == 4)
    assert row["vp_fsm_break__rmean2"][0] is None or (
        row["vp_fsm_break__rmean2"][0] != 15.0
        and (
            row["vp_fsm_break__rmean2"][0] is None or float(row["vp_fsm_break__rmean2"][0]) != 15.0
        )
    )
    # polars shift over group → null في أول صف
    assert row["vp_fsm_break__rmean2"][0] is None
    assert row["vp_fsm_break__rsum2"][0] is None


@pytest.mark.leakage
def test_folds_unique_setup_no_duplicate_oos() -> None:
    # نفس setup_ts يظهر في عدة outcomes
    rows = []
    for t in range(20):
        for name in ("y_a", "y_b"):
            rows.append(
                {
                    SETUP_AVAILABILITY_TS: t,
                    OUTCOME_AVAILABLE_TS: t + 1,
                    "outcome_name": name,
                    "y": 1.0 if t % 2 == 0 else 0.0,
                    "label_status": "resolved",
                }
            )
    frame = pl.DataFrame(rows)
    folds = build_time_folds_for_frame(
        frame, n_splits=3, embargo=0, purge_samples=1, min_train_size=4
    )
    assert folds
    seen: set[int] = set()
    for sf in folds:
        test = frame[sf.test_idx]
        ts = set(int(x) for x in test[SETUP_AVAILABILITY_TS].to_list())
        assert not (ts & seen)
        seen |= ts
        train = frame[sf.train_idx]
        train_ts = set(int(x) for x in train[SETUP_AVAILABILITY_TS].to_list())
        assert not (train_ts & ts)


def test_psi_detects_far_shift() -> None:
    ref = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 5)
    cmp = np.array([100.0, 100.5, 101.0] * 10)
    psi = _psi_1d(ref, cmp, n_bins=5)
    assert psi > 0.5


def test_family_feature_selection_includes_level_flow_and_reliability() -> None:
    n = 40
    cols: dict[str, object] = {AVAILABILITY_TS: list(range(n))}
    for c in (
        "vp_balance",
        "vp_imbalance",
        "struct_dist_vah_ticks",
        "struct_dist_val_ticks",
        "proj_poc_shift_ticks",
        "proj_va_overlap",
        "proj_expansion_testing",
        "mem_time_since_break",
        "lf_arrival_intensity",
        "lf_absorption_proxy",
        "lf_near_vah_cancel_ratio",
        "rel_credibility",
        "rel_evidence_strength",
        "signal_quality",
    ):
        cols[c] = np.linspace(0.0, 1.0, n) + (hash(c) % 7) * 0.01
    # أضف أعمدة حالة كثيرة ثابتة/متشابهة لتضغط السقف قديمًا
    for i in range(60):
        cols[f"junk_{i}"] = np.zeros(n)
    frame = pl.DataFrame(cols)
    names = select_feature_names_by_family(frame, max_features=64)
    assert any(c.startswith("lf_") for c in names)
    assert any(c.startswith("rel_") for c in names)
    assert "junk_0" not in names


def test_insufficient_support_returns_nan_not_half() -> None:
    labeled = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1, 2, 3],
            OUTCOME_AVAILABLE_TS: [2, 3, 4],
            "outcome_name": ["y_true_break"] * 3,
            "y": [0.0, 0.0, 0.0],
            "label_status": ["resolved"] * 3,
            "f1": [0.1, 0.2, 0.3],
        }
    )
    model = fit_conditional_models(
        labeled,
        feature_names=("f1",),
        outcomes=("y_true_break",),
        train_end_ts=10,
        min_train=8,
        min_pos=3,
        min_neg=3,
    )
    assert model.status["y_true_break"] in {
        MODEL_STATUS_INSUFFICIENT,
        MODEL_STATUS_SINGLE_CLASS,
    }
    p = model.predict_proba(labeled, "y_true_break")
    assert np.all(np.isnan(p))


def test_order_lifecycle_spans_buckets() -> None:
    events = pl.DataFrame(
        {
            EVENT_TS: [10, 25],
            BUCKET_START: [0, 20],
            "order_id": [7, 7],
            "action": [MboAction.ADD.value, MboAction.CANCEL.value],
            "size": [5.0, 5.0],
        }
    )
    life = _order_lifecycle_by_bucket(events)
    assert life.height == 1
    assert int(life[BUCKET_START][0]) == 20
    assert float(life["lf_mean_order_lifetime_ns"][0]) == 15.0
