"""اختبارات رأس المخاطر المتنافسة + دراسة الـ ablation (سببية + صحة كمية)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior.ablation import (
    ABLATION_STACKS,
    _select_stack_features,
    _stack_feature_groups,
    run_behavior_ablation,
    stack_feature_columns,
)
from nq.auction_behavior.calibration import log_loss_score, roc_auc
from nq.auction_behavior.competing import (
    COMPETING_STATUS_INSUFFICIENT,
    calibrate_competing_temperature,
    evaluate_competing_scores,
    fit_competing_risk_model,
    score_competing_risk_model,
)
from nq.auction_behavior.outcomes import (
    FIRST_TRANSITION_CLASS_COL,
    FIRST_TRANSITION_CLASSES,
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    build_first_transition_outcomes,
)
from nq.auction_behavior.realized_path import REALIZED_NEXT_PATH_CLASSES
from nq.auction_behavior.science import ScienceConfig, run_behavior_science
from nq.contracts.temporal import AVAILABILITY_TS
from nq.validation.leakage import LeakageError
from tests.realized_path_factory import path_bar_fields, path_kind_for_index

_HOUR_NS = 3_600 * 1_000_000_000


def _ts(value: object) -> int:
    assert value is not None
    return int(float(str(value)))


def _episode_frame() -> pl.DataFrame:
    """قصص اصطناعية: onset اختبار ثم انتقال تقوده ميزة ``vp_imbalance``.

    كل قصة 8 بارات: testing في البار 0–1، الانتقال في البار 2
    (accepting إذا vp_imbalance=1، وإلا rejection)، وقصص بلا انتقال إطلاقًا.
    """
    rows: list[dict[str, float | int]] = []
    ts = 0
    for episode in range(60):
        kind = episode % 3  # 0=accepting · 1=rejection · 2=no transition
        imbalance = 1.0 if kind == 0 else 0.0
        for bar in range(8):
            testing = 1.0 if bar in (0, 1) else 0.0
            accepting = 1.0 if (kind == 0 and bar == 2) else 0.0
            rejection = 1.0 if (kind == 1 and bar == 2) else 0.0
            rows.append(
                {
                    AVAILABILITY_TS: ts,
                    "_behavior_story_run": episode,
                    "proj_expansion_testing": testing,
                    "proj_expansion_accepting": accepting,
                    "proj_rejection_to_asia": rejection,
                    "proj_repriced_balance": 0.0,
                    "vp_imbalance": imbalance,
                    "vp_balance": 1.0 - imbalance,
                    "struct_dist_vah_ticks": float(bar) - 3.0,
                    "lf_arrival_intensity": float((episode * 7 + bar) % 5),
                    "rel_credibility": float((episode + bar) % 3) / 2.0,
                    "mem_time_since_break": float(bar),
                    "vp_imbalance__lag1": imbalance if bar > 0 else 0.0,
                }
            )
            ts += _HOUR_NS
    return pl.DataFrame(rows)


def _realized_path_frame() -> pl.DataFrame:
    """قصص اصطناعية: كسر متحقق بلا ريتست ثم أول انتقال هندسي.

    ``proj_*`` تبقى ملامح. الـY هو المسار التالي، وليس فصل الإسقاط.
    """
    rows: list[dict[str, float | int]] = []
    ts = 0
    for episode in range(60):
        kind = path_kind_for_index(episode)
        imbalance = 1.0 if kind in {"further_beyond_asia", "continue_direction"} else 0.0
        for bar in range(8):
            testing = 1.0 if bar in (0, 1) else 0.0
            rows.append(
                {
                    AVAILABILITY_TS: ts,
                    "_behavior_story_run": episode,
                    "proj_expansion_testing": testing,
                    "proj_expansion_accepting": 0.0,
                    "proj_rejection_to_asia": 0.0,
                    "proj_repriced_balance": 0.0,
                    "vp_imbalance": imbalance,
                    "struct_dist_vah_ticks": float(bar) - 3.0,
                    "lf_arrival_intensity": float((episode * 7 + bar) % 5),
                    "rel_credibility": float((episode + bar) % 3) / 2.0,
                    "mem_time_since_break": float(bar),
                    "vp_imbalance__lag1": imbalance if bar > 0 else 0.0,
                    **path_bar_fields(kind, bar),
                }
            )
            ts += _HOUR_NS
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# تسميات أول انتقال
# ---------------------------------------------------------------------------


def test_first_transition_labels_classes_and_censoring() -> None:
    frame = _episode_frame()
    labels = build_first_transition_outcomes(frame, window=5, group_col="_behavior_story_run")
    assert labels.height == 60  # onset واحد لكل قصة
    resolved = labels.filter(pl.col("label_status") == "resolved")
    assert resolved.height == 60
    counts = {
        str(k[0] if isinstance(k, tuple) else k): part.height
        for k, part in resolved.group_by(FIRST_TRANSITION_CLASS_COL)
    }
    assert counts == {
        "expansion_accepting": 20,
        "rejection_return_to_asia": 20,
        "no_transition": 20,
    }
    # النتيجة تُتاح عند بار الانتقال (بعد الإعداد)، لا قبله
    assert (resolved[OUTCOME_AVAILABLE_TS] > resolved[SETUP_AVAILABILITY_TS]).all()


def test_first_transition_truncated_window_is_censored_not_no_transition() -> None:
    frame = _episode_frame().head(8 * 2 + 3)  # القصة الثالثة مقطوعة بعد onset مباشرة
    labels = build_first_transition_outcomes(frame, window=5, group_col="_behavior_story_run")
    last = labels.tail(1)
    assert last["label_status"][0] == "censored"
    assert last[FIRST_TRANSITION_CLASS_COL][0] is None


def test_first_transition_same_bar_conflict_is_ambiguous() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [0, 1, 2, 3, 4, 5],
            "proj_expansion_testing": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "proj_expansion_accepting": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "proj_rejection_to_asia": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "proj_repriced_balance": [0.0] * 6,
        }
    )
    labels = build_first_transition_outcomes(frame, window=4)
    assert labels.height == 1
    assert labels["label_status"][0] == "ambiguous"


# ---------------------------------------------------------------------------
# النموذج متعدد الفئات
# ---------------------------------------------------------------------------


def _competing_labeled() -> pl.DataFrame:
    frame = _episode_frame()
    labels = build_first_transition_outcomes(frame, window=5, group_col="_behavior_story_run")
    features = frame.select(AVAILABILITY_TS, "vp_imbalance", "vp_balance", "struct_dist_vah_ticks")
    return labels.join(
        features, left_on=SETUP_AVAILABILITY_TS, right_on=AVAILABILITY_TS, how="inner"
    )


def test_competing_model_probabilities_sum_to_one_and_learn_state() -> None:
    labeled = _competing_labeled()
    cut = _ts(labeled[SETUP_AVAILABILITY_TS].quantile(0.7))
    model = fit_competing_risk_model(
        labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= cut),
        feature_names=("vp_imbalance", "struct_dist_vah_ticks"),
        train_end_ts=cut,
        min_train=10,
        min_class_count=2,
    )
    assert model.is_usable()
    test = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) > cut)
    probs = model.predict_proba(test)
    assert probs.shape == (test.height, len(FIRST_TRANSITION_CLASSES))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    # الحالة تقود التنبؤ: imbalance=1 → قبول أرجح من الرفض والعكس
    idx_acc = FIRST_TRANSITION_CLASSES.index("expansion_accepting")
    idx_rej = FIRST_TRANSITION_CLASSES.index("rejection_return_to_asia")
    imb = test["vp_imbalance"].to_numpy()
    assert float(np.mean(probs[imb > 0.5, idx_acc])) > float(np.mean(probs[imb > 0.5, idx_rej]))
    assert float(np.mean(probs[imb < 0.5, idx_rej])) > float(np.mean(probs[imb < 0.5, idx_acc]))


def test_competing_model_insufficient_support_returns_nan_not_uniform() -> None:
    labeled = _competing_labeled().head(5)
    model = fit_competing_risk_model(
        labeled,
        feature_names=("vp_imbalance",),
        train_end_ts=_ts(labeled[SETUP_AVAILABILITY_TS].max()),
        min_train=24,
    )
    assert model.status == COMPETING_STATUS_INSUFFICIENT
    probs = model.predict_proba(labeled)
    assert np.all(np.isnan(probs))


@pytest.mark.leakage
def test_competing_scoring_rejects_test_inside_train_window() -> None:
    labeled = _competing_labeled()
    train_end = _ts(labeled[SETUP_AVAILABILITY_TS].max())
    model = fit_competing_risk_model(
        labeled,
        feature_names=("vp_imbalance",),
        train_end_ts=train_end,
        min_train=10,
        min_class_count=2,
    )
    with pytest.raises(LeakageError):
        score_competing_risk_model(model, labeled)  # كل الإعدادات ≤ train_end


def test_competing_temperature_calibration_bounded_and_preserves_sum() -> None:
    labeled = _competing_labeled()
    cut = _ts(labeled[SETUP_AVAILABILITY_TS].quantile(0.6))
    model = fit_competing_risk_model(
        labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= cut),
        feature_names=("vp_imbalance",),
        train_end_ts=cut,
        min_train=10,
        min_class_count=2,
    )
    tail = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= cut).tail(15)
    calibrated = calibrate_competing_temperature(model, tail)
    assert 0.25 <= calibrated.temperature <= 4.0
    probs = calibrated.predict_proba(labeled.tail(10))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_evaluate_competing_scores_known_values() -> None:
    classes = FIRST_TRANSITION_CLASSES
    scored = pl.DataFrame(
        {
            FIRST_TRANSITION_CLASS_COL: ["expansion_accepting", "no_transition"],
            "p_first_expansion_accepting": [1.0, 0.0],
            "p_first_rejection_return_to_asia": [0.0, 0.0],
            "p_first_repriced_balance": [0.0, 0.0],
            "p_first_no_transition": [0.0, 1.0],
        }
    )
    metrics = evaluate_competing_scores(scored, classes=classes)
    assert metrics["n"] == 2.0
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# مقاييس ثنائية جديدة
# ---------------------------------------------------------------------------


def test_roc_auc_and_log_loss_known_values() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert np.isnan(roc_auc(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4])))
    expected = -float(np.mean(np.log([0.9, 0.8, 0.8, 0.9])))
    assert log_loss_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# دراسة الـ ablation
# ---------------------------------------------------------------------------


def test_stack_feature_columns_cumulative_and_reject_unknown() -> None:
    frame = _episode_frame()
    previous: set[str] = set()
    for stack in ABLATION_STACKS:
        cols = set(stack_feature_columns(frame, stack))
        assert previous.issubset(cols)
        previous = cols
    assert "lf_arrival_intensity" not in stack_feature_columns(frame, "plus_memory")
    assert "lf_arrival_intensity" in stack_feature_columns(frame, "plus_mbo_flow")
    assert "rel_credibility" not in stack_feature_columns(frame, "plus_mbo_flow")
    assert "rel_credibility" in stack_feature_columns(frame, "plus_reliability")
    with pytest.raises(ValueError, match="unknown ablation stack"):
        stack_feature_columns(frame, "nope")


def test_round_robin_selection_guarantees_new_family_representation() -> None:
    frame = _episode_frame()
    groups = _stack_feature_groups(frame, "plus_reliability")
    chosen = _select_stack_features(frame, groups, max_features=6)
    assert any(c.startswith("rel_") for c in chosen)
    assert any(c.startswith("lf_") for c in chosen)


def test_run_behavior_ablation_same_folds_and_untouched_holdout() -> None:
    frame = _realized_path_frame()
    cfg = ScienceConfig(
        outcome_window=5,
        competing_window=5,
        n_splits=3,
        min_train_size=8,
        holdout_frac=0.2,
        use_month_folds=False,
        competing_min_train=10,
        competing_min_class=2,
    )
    report = run_behavior_ablation(frame, config=cfg)
    assert report.n_folds >= 1
    assert report.diagnostics["holdout_untouched"] is True
    assert report.diagnostics["identical_folds_and_labels_across_stacks"] is True
    assert report.diagnostics["n_holdout_excluded"] >= 1
    assert report.frame.height > 0
    # نفس عدد صفوف OOS لكل ستاك داخل الهدف الواحد — نفس الطيّات والتسميات
    for _key, part in report.frame.group_by("outcome_name"):
        assert part["n_oof"].n_unique() == 1
        assert set(part["stack"].to_list()) == set(ABLATION_STACKS)
    # رأس المخاطر المتنافسة يعمل لكل ستاك بنفس عدد OOS
    assert report.competing_frame.height == len(ABLATION_STACKS)
    assert report.competing_frame["n_oof"].n_unique() == 1


# ---------------------------------------------------------------------------
# تكامل العلم: OOF شرطي + متنافس
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_science_competing_head_oof_is_causal_and_sums_to_one() -> None:
    frame = _realized_path_frame()
    cfg = ScienceConfig(
        outcome_window=5,
        competing_window=5,
        n_splits=3,
        min_train_size=8,
        holdout_frac=0.2,
        use_month_folds=False,
        competing_min_train=10,
        competing_min_class=2,
    )
    report = run_behavior_science(frame, config=cfg)
    diag = report.diagnostics
    assert diag["competing_risk_enabled"] is True
    assert diag["competing_family"] == "realized_path"
    assert diag["scenario_labels_are_features_not_exclusive_y"] is True
    assert diag["competing_probabilities_sum_to_one"] is True
    assert diag["n_competing_labeled"] > 0
    assert "competing_brier" in report.fold_frame.columns
    assert "log_loss" in report.fold_frame.columns
    assert "auc" in report.fold_frame.columns
    oof = report.competing_oof_predictions
    assert oof.height > 0
    assert oof["prediction_is_oof"].all()
    assert oof["eligible_for_backtest"].all()
    assert (oof["model_train_end_ts"] < oof[AVAILABILITY_TS]).all()
    pcols = [f"p_first_{c}" for c in REALIZED_NEXT_PATH_CLASSES]
    assert all(c in oof.columns for c in pcols)
    assert "p_first_expansion_accepting" not in oof.columns
    probs = oof.select(pcols).to_numpy()
    finite = np.all(np.isfinite(probs), axis=1)
    assert finite.any()
    np.testing.assert_allclose(probs[finite].sum(axis=1), 1.0, atol=1e-9)
    live = report.competing_live_predictions
    assert live.height == frame.height
    assert not live["eligible_for_backtest"].any()
