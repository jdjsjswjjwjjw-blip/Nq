"""اختبارات مضادة لإصلاحات التسريب العلمي (censoring / memory / OOF / PSI / …)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior.calibration import (
    apply_calibrators_by_outcome,
    apply_calibrators_to_state_predictions,
    brier_skill_score,
    fit_platt_calibrators_by_outcome,
)
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
    attach_outcome_availability_guard,
    build_labeled_outcomes,
    filter_resolved_outcomes,
)
from nq.auction_behavior.science import _calibration_tail_split
from nq.auction_behavior.walk_forward import (
    build_expanding_month_folds,
    build_time_folds_for_frame,
)
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
def test_last_row_setup_is_emitted_as_censored() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1],
            "group_id": [1],
            "trigger": [1.0],
            "success": [0.0],
            "fail": [0.0],
        }
    )
    out = build_labeled_outcomes(
        frame,
        group_col="group_id",
        specs=(OutcomeSpec("y_x", "trigger", ("success",), ("fail",), 3),),
    )
    assert out.height == 1
    assert out["label_status"][0] == "censored"
    assert int(out[OUTCOME_AVAILABLE_TS][0]) == 1


@pytest.mark.leakage
def test_ambiguous_same_bar_outcome_is_not_a_failure() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2],
            "group_id": [1, 1],
            "trigger": [1.0, 0.0],
            "success": [0.0, 1.0],
            "fail": [0.0, 1.0],
        }
    )
    out = build_labeled_outcomes(
        frame,
        group_col="group_id",
        specs=(OutcomeSpec("y_x", "trigger", ("success",), ("fail",), 1),),
    )
    assert out["label_status"][0] == "ambiguous"
    assert np.isnan(float(out["y"][0]))
    assert filter_resolved_outcomes(out).height == 0


def test_outcome_contract_rejects_missing_group_and_duplicate_feature_time() -> None:
    frame = pl.DataFrame({AVAILABILITY_TS: [1], "trigger": [1.0]})
    with pytest.raises(ValueError, match="group_col is missing"):
        build_labeled_outcomes(
            frame,
            group_col="missing_story",
            specs=(OutcomeSpec("y_x", "trigger", (), (), 1),),
        )

    outcomes = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1],
            OUTCOME_AVAILABLE_TS: [2],
            "outcome_name": ["y_x"],
            "y": [1.0],
            "label_status": ["resolved"],
        }
    )
    duplicated = pl.DataFrame({AVAILABILITY_TS: [1, 1], "feature": [0.1, 0.2]})
    with pytest.raises(ValueError, match="unique availability_ts"):
        attach_outcome_availability_guard(duplicated, outcomes)


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


def test_small_feature_budget_is_balanced_across_families() -> None:
    n = 30
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "proj_poc_shift_ticks": np.arange(n),
            "struct_dist_vah_ticks": np.arange(n) * 2,
            "mem_time_since_break": np.arange(n) * 3,
            "lf_arrival_intensity": np.arange(n) * 4,
            "rel_credibility": np.arange(n) * 5,
            "signal_quality": np.arange(n) * 6,
        }
    )
    names = select_feature_names_by_family(frame, max_features=5)
    assert len(names) == 5
    assert any(name.startswith("proj_") for name in names)
    assert any(name.startswith("struct_") for name in names)
    assert any(name.startswith("lf_") for name in names)
    assert any(name.startswith("rel_") for name in names)


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


def test_order_lifecycle_partial_cancel_trade_refill_and_clear_semantics() -> None:
    events = pl.DataFrame(
        {
            EVENT_TS: [10, 12, 14, 16, 18, 20, 21, 22],
            BUCKET_START: [0, 0, 0, 0, 0, 20, 20, 20],
            "order_id": [7, 7, 0, 7, 7, 8, 0, 8],
            "action": [
                MboAction.ADD.value,
                MboAction.CANCEL.value,
                MboAction.TRADE.value,
                MboAction.MODIFY.value,
                MboAction.CANCEL.value,
                MboAction.ADD.value,
                MboAction.CLEAR.value,
                MboAction.CANCEL.value,
            ],
            "price": [100.0] * 8,
            "size": [10.0, 3.0, 1.0, 12.0, 12.0, 4.0, 0.0, 4.0],
            "_near_level": [True] * 8,
        }
    )
    life = _order_lifecycle_by_bucket(events)
    # partial cancel لا يغلق؛ الإغلاق الحقيقي عند t=18. أمر 8 صار censored بفعل CLEAR.
    assert life.height == 1
    assert float(life["lf_mean_order_lifetime_ns"][0]) == 8.0
    assert float(life["lf_queue_survival_rate"][0]) == 1.0
    assert float(life["lf_partial_exec_rate"][0]) == 1.0
    assert float(life["lf_refill_rate"][0]) == 1.0


def test_order_lifecycle_tracks_near_add_after_level_moves() -> None:
    events = pl.DataFrame(
        {
            EVENT_TS: [10, 25],
            BUCKET_START: [0, 20],
            "order_id": [9, 9],
            "action": [MboAction.ADD.value, MboAction.CANCEL.value],
            "price": [100.0, 100.0],
            "size": [5.0, 5.0],
            "_near_level": [True, False],
        }
    )
    life = _order_lifecycle_by_bucket(events)
    assert life.height == 1
    assert int(life[BUCKET_START][0]) == 20


@pytest.mark.leakage
def test_calibration_tail_split_never_divides_setup_timestamp() -> None:
    rows = []
    for ts in range(10):
        for outcome in ("y_a", "y_b"):
            rows.append(
                {
                    SETUP_AVAILABILITY_TS: ts,
                    OUTCOME_AVAILABLE_TS: ts + 1,
                    "outcome_name": outcome,
                    "y": float(ts % 2),
                    "label_status": "resolved",
                }
            )
    fit, cal = _calibration_tail_split(pl.DataFrame(rows), frac=0.3)
    assert set(fit[SETUP_AVAILABILITY_TS].to_list()).isdisjoint(
        cal[SETUP_AVAILABILITY_TS].to_list()
    )
    assert max(fit[SETUP_AVAILABILITY_TS]) < min(cal[SETUP_AVAILABILITY_TS])


def test_calibration_is_per_outcome_and_reaches_state_predictions() -> None:
    scored = pl.DataFrame(
        {
            "outcome_name": ["y_a"] * 20 + ["y_b"] * 20,
            "y": ([0.0, 1.0] * 10) + ([1.0, 0.0] * 10),
            "p_hat": ([0.05, 0.20] * 10) + ([0.80, 0.95] * 10),
        }
    )
    calibrators = fit_platt_calibrators_by_outcome(scored, min_samples=10)
    assert set(calibrators) == {"y_a", "y_b"}
    for calibrator in calibrators.values():
        transformed = calibrator.transform(np.array([0.05, 0.5, 0.95]))
        assert np.all((transformed > 0.0) & (transformed < 1.0))
        assert np.max(np.abs(np.array([calibrator.a, calibrator.b]))) < 20.0
    calibrated = apply_calibrators_by_outcome(scored, calibrators)
    assert "p_cal" in calibrated.columns
    states = pl.DataFrame({AVAILABILITY_TS: [1], "p_y_a": [0.05], "p_y_b": [0.05]})
    state_cal = apply_calibrators_to_state_predictions(states, calibrators)
    assert float(state_cal["p_y_a"][0]) != pytest.approx(float(state_cal["p_y_b"][0]))


def test_brier_skill_uses_past_training_baseline() -> None:
    y = np.array([1.0, 1.0, 1.0, 0.0])
    p = np.array([0.7, 0.7, 0.7, 0.7])
    train_baseline = np.full(4, 0.2)
    explicit = brier_skill_score(y, p, train_baseline)
    test_rate_baseline = brier_skill_score(y, p)
    assert explicit != pytest.approx(test_rate_baseline)


@pytest.mark.leakage
def test_month_folds_apply_setup_level_purge() -> None:
    def _ns(year: int, month: int, day: int) -> int:
        return int(dt.datetime(year, month, day, tzinfo=dt.UTC).timestamp() * 1e9)

    times = [_ns(2025, 1, d) for d in range(1, 7)] + [_ns(2025, 2, d) for d in range(1, 4)]
    frame = pl.DataFrame({SETUP_AVAILABILITY_TS: times})
    folds = build_expanding_month_folds(frame, min_train_months=1, purge_samples=2)
    assert len(folds) == 1
    train_times = frame[folds[0].train_idx][SETUP_AVAILABILITY_TS].to_list()
    assert max(train_times) == times[3]
