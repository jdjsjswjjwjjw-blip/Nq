"""نموذج شرطي competing-risk + ablation عائلات — سببية وsimplex."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nq.auction_behavior.ablation import (
    ABLATION_SPECS,
    AblationFoldSlice,
    ablation_nested_families,
    run_binary_feature_ablation,
    run_feature_ablation,
)
from nq.auction_behavior.calibration import log_loss, roc_auc
from nq.auction_behavior.competing import (
    COMPETING_CLASS_NAMES,
    RESIDUAL_CLASS,
    assert_features_are_state_only,
    competing_rows_sum_to_one,
    evaluate_competing_scores,
    fit_competing_risk,
    pivot_competing_labels,
    predict_competing_at_states,
    score_competing_risk,
)
from nq.auction_behavior.conditional import select_feature_names_by_family
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    OUTCOME_TARGETS,
    PRIMARY_OUTCOME_TARGETS,
    SETUP_AVAILABILITY_TS,
)
from nq.auction_behavior.pipeline import (
    BehaviorConfig,
    probabilities_from_science,
    run_auction_behavior_analysis,
)
from nq.auction_behavior.science import ScienceConfig, run_behavior_science
from nq.auction_behavior.types import BehaviorProbabilities
from nq.contracts.temporal import AVAILABILITY_TS
from tests.test_auction_behavior import _dense_trade_stream


def _labeled_setup(
    setup_ts: int,
    *,
    expansion: float,
    rejection: float,
    repriced: float,
    feat: float,
    available_at: int | None = None,
) -> list[dict[str, object]]:
    avail = int(available_at if available_at is not None else setup_ts + 10)
    rows = []
    for name, y in zip(
        PRIMARY_OUTCOME_TARGETS,
        (expansion, rejection, repriced),
        strict=True,
    ):
        rows.append(
            {
                SETUP_AVAILABILITY_TS: setup_ts,
                OUTCOME_AVAILABLE_TS: avail,
                AVAILABILITY_TS: setup_ts,
                "outcome_name": name,
                "y": y,
                "label_status": "resolved",
                "group_id": 0,
                "feat_a": feat,
                "feat_b": 1.0 - feat,
            }
        )
    return rows


def test_competing_assignment_exclusive_residual_and_conflict() -> None:
    rows: list[dict[str, object]] = []
    rows.extend(_labeled_setup(1, expansion=1.0, rejection=0.0, repriced=0.0, feat=0.9))
    rows.extend(_labeled_setup(2, expansion=0.0, rejection=0.0, repriced=0.0, feat=0.1))
    rows.extend(_labeled_setup(3, expansion=1.0, rejection=1.0, repriced=0.0, feat=0.8))
    pivoted = pivot_competing_labels(pl.DataFrame(rows))
    by_ts = {
        int(ts): name
        for ts, name in zip(pivoted[SETUP_AVAILABILITY_TS], pivoted["class_name"], strict=False)
    }
    assert by_ts[1] == "y_expansion_accepting"
    assert by_ts[2] == RESIDUAL_CLASS
    assert by_ts[3] == "y_expansion_accepting"
    assert bool(pivoted.filter(pl.col(SETUP_AVAILABILITY_TS) == 3)["conflict"][0]) is True


def test_censored_setup_is_excluded_from_competing() -> None:
    rows = _labeled_setup(1, expansion=1.0, rejection=0.0, repriced=0.0, feat=0.2)
    rows[1]["label_status"] = "censored"
    rows[1]["y"] = float("nan")
    pivoted = pivot_competing_labels(pl.DataFrame(rows))
    assert pivoted.height == 0


def test_softmax_rows_sum_to_one_and_ignore_future_labels() -> None:
    rows: list[dict[str, object]] = []
    for i in range(40):
        exp = 1.0 if i % 2 == 0 else 0.0
        rows.extend(
            _labeled_setup(
                i,
                expansion=exp,
                rejection=1.0 - exp,
                repriced=0.0,
                feat=0.9 if exp else 0.1,
                available_at=i + 1,
            )
        )
    labeled = pivot_competing_labels(pl.DataFrame(rows))
    train = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= 19)
    test = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) >= 21)
    model = fit_competing_risk(
        train,
        feature_names=("feat_a", "feat_b"),
        train_end_ts=20,
        l2=0.5,
        min_train=8,
        min_class=1,
    )
    assert model.is_usable()
    probs = model.predict_proba(test)
    assert competing_rows_sum_to_one(probs)
    assert np.all(probs >= 0.0)
    scored = score_competing_risk(model, test)
    metrics = evaluate_competing_scores(scored)
    assert metrics["n"] == float(test.height)
    assert np.isfinite(metrics["log_loss"])
    assert np.isfinite(metrics["brier"])
    with pytest.raises(AssertionError, match="labels/probs"):
        assert_features_are_state_only(("feat_a", "y_expansion_accepting"))


def test_ablation_families_are_nested_and_schema_stable() -> None:
    assert ablation_nested_families()
    assert [spec.name for spec in ABLATION_SPECS] == [
        "A_volume_profile_price",
        "B_plus_dynamic_asia_london",
        "C_plus_memory",
        "D_plus_mbo",
        "E_plus_reliability",
    ]
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(20)),
            "vp_balance": np.linspace(0, 1, 20),
            "struct_dist_vah_ticks": np.arange(20),
            "signal_quality": np.linspace(0.2, 0.8, 20),
            "proj_poc_shift_ticks": np.arange(20) * 2,
            "path_depth_confirm": np.linspace(0, 1, 20),
            "lf_arrival_intensity": np.arange(20) * 3,
            "rel_credibility": np.linspace(0.1, 0.9, 20),
        }
    )
    only_a = select_feature_names_by_family(
        frame, max_features=32, include_families=("state", "structure", "quality")
    )
    full = select_feature_names_by_family(
        frame,
        max_features=32,
        include_families=(
            "state",
            "structure",
            "quality",
            "projection",
            "path",
            "memory_roll",
            "sequence",
            "level_flow",
            "reliability",
        ),
    )
    assert not any(name.startswith("lf_") or name.startswith("rel_") for name in only_a)
    assert any(name.startswith("lf_") for name in full)
    assert any(name.startswith("rel_") for name in full)


def test_ablation_runner_returns_five_specs() -> None:
    rows: list[dict[str, object]] = []
    for i in range(30):
        exp = 1.0 if i % 2 == 0 else 0.0
        rows.extend(
            _labeled_setup(
                i,
                expansion=exp,
                rejection=1.0 - exp,
                repriced=0.0,
                feat=0.8 if exp else 0.2,
                available_at=i + 3,
            )
        )
    labeled = pl.DataFrame(rows).with_columns(
        pl.col("feat_a").alias("vp_balance"),
        pl.col("feat_b").alias("struct_dist_vah_ticks"),
    )
    train = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) < 20)
    test = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) >= 20)
    table = run_feature_ablation(
        [
            AblationFoldSlice(
                fold=1,
                segment="test",
                train_end_ts=19,
                train=train,
                test=test,
            )
        ],
        max_features=8,
        min_train=6,
        min_class=1,
    )
    assert table.height == 5
    assert set(table["spec"].to_list()) == {spec.name for spec in ABLATION_SPECS}
    binary = run_binary_feature_ablation(
        [
            AblationFoldSlice(
                fold=1,
                segment="test",
                train_end_ts=19,
                train=train,
                test=test,
            )
        ],
        max_features=8,
        min_train=6,
        min_pos=1,
        min_neg=1,
    )
    assert binary.height >= 5
    assert "__pooled__" in binary["outcome_name"].to_list()
    ok_rows = binary.filter(pl.col("status") == "ok")
    assert ok_rows.height > 0
    assert (ok_rows["n_oof"] > 0).all()


def test_binary_ablation_skips_unscored_outcomes_and_uses_all_targets() -> None:
    """لا تُحسب صفوف n=0 / Brier=0 كدليل OOS؛ الأهداف الثانوية تُجرَّب إن وُجدت."""
    rows: list[dict[str, object]] = []
    for i in range(36):
        exp = 1.0 if i % 2 == 0 else 0.0
        rows.extend(
            _labeled_setup(
                i,
                expansion=exp,
                rejection=1.0 - exp,
                repriced=0.0,
                feat=0.85 if exp else 0.15,
                available_at=i + 2,
            )
        )
        rows.append(
            {
                SETUP_AVAILABILITY_TS: i,
                OUTCOME_AVAILABLE_TS: i + 2,
                AVAILABILITY_TS: i,
                "outcome_name": "y_retest_success",
                "y": 1.0 if i % 3 == 0 else 0.0,
                "label_status": "resolved",
                "group_id": 0,
                "feat_a": 0.7 if i % 3 == 0 else 0.2,
                "feat_b": 0.3,
            }
        )
    labeled = pl.DataFrame(rows).with_columns(
        pl.col("feat_a").alias("vp_balance"),
        pl.col("feat_b").alias("struct_dist_vah_ticks"),
    )
    train = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) < 24)
    test = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) >= 24)
    table = run_binary_feature_ablation(
        [
            AblationFoldSlice(
                fold=1,
                segment="test",
                train_end_ts=23,
                train=train,
                test=test,
            )
        ],
        outcomes=OUTCOME_TARGETS,
        max_features=8,
        min_train=6,
        min_pos=1,
        min_neg=1,
    )
    names = set(table["outcome_name"].to_list())
    assert "y_retest_success" in names
    assert "y_false_break" not in names
    fake_ok = table.filter((pl.col("status") == "ok") & (pl.col("n_oof") <= 0))
    assert fake_ok.height == 0
    retest = table.filter(pl.col("outcome_name") == "y_retest_success")
    assert retest.height == 5
    assert (retest["n_oof"] > 0).all()


def test_log_loss_and_auc_bounds() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert log_loss(y, p) < log_loss(y, np.full(4, 0.5))
    assert roc_auc(y, p) == pytest.approx(1.0)


def test_pipeline_headline_is_conditional_when_science_runs() -> None:
    frame = _dense_trade_stream(n_bars=140, bar_ns=200)
    result = run_auction_behavior_analysis(
        frame,
        config=BehaviorConfig(
            profile_interval_ns=1000,
            signal_interval_ns=200,
            fixed_range=False,
            include_deceptive_scores=False,
            include_level_flow=True,
            include_reliability_evidence=True,
            include_asia_london_projection=True,
            include_science=True,
            evaluate_holdout=False,
            n_splits=3,
            min_train_size=8,
            holdout_frac=0.2,
        ),
    )
    assert result.validation.ok
    assert result.science is not None
    assert result.science.holdout.touched is False
    diag = result.science.diagnostics
    assert "competing_risk_softmax_joint" in diag["science_steps"]
    assert "feature_family_ablation" in diag["science_steps"]
    assert diag["holdout_touched"] is False
    assert result.diagnostics["causality"]["holdout_evaluation"] == "explicit_opt_in_only"
    if result.live_predictions.height:
        assert result.live_predictions["eligible_for_backtest"].unique().to_list() == [False]
    if result.oof_predictions.height:
        assert result.oof_predictions["eligible_for_backtest"].unique().to_list() == [True]
        if "competing_mass" in result.oof_predictions.columns:
            mass = result.oof_predictions["competing_mass"].drop_nulls().to_numpy()
            if mass.size:
                assert np.allclose(mass, 1.0, atol=1e-5)
    if result.probabilities.probability_source == "state_conditional_oof":
        joint = (
            result.probabilities.p_expansion_accepting
            + result.probabilities.p_rejection_return_to_asia
            + result.probabilities.p_repriced_balance
            + result.probabilities.p_residual
        )
        if result.probabilities.probabilities_are_joint_distribution:
            assert joint == pytest.approx(1.0, abs=1e-6)


def test_probabilities_from_science_prefers_oof_not_base_rate() -> None:
    oof = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2],
            "p_y_true_break": [0.2, 0.4],
            "p_y_expansion_accepting": [0.5, 0.5],
            "p_y_rejection_return_to_asia": [0.2, 0.2],
            "p_y_repriced_balance": [0.2, 0.2],
            "p_residual": [0.1, 0.1],
        }
    )
    competing = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1, 2],
            "class_id": [0, 0],
            "p_y_expansion_accepting": [0.6, 0.4],
            "p_y_rejection_return_to_asia": [0.2, 0.2],
            "p_y_repriced_balance": [0.1, 0.2],
            "p_residual": [0.1, 0.2],
        }
    )
    fallback = BehaviorProbabilities(
        p_balanced=0.9,
        p_imbalanced=0.1,
        p_true_break=0.99,
        p_false_break=0.0,
        p_retest_success=0.0,
        p_retest_fail=0.0,
        p_expansion_continue=0.0,
        p_return_to_value=0.0,
        confidence=0.5,
        n_samples=10,
        detail="base-rate",
        probability_source="train_only_walk_forward_base_rates",
    )
    science = run_behavior_science(
        pl.DataFrame({AVAILABILITY_TS: []}),
        config=ScienceConfig(evaluate_holdout=False),
    )
    science = science.__class__(
        labeled=science.labeled,
        feature_names=science.feature_names,
        fold_frame=science.fold_frame,
        fold_scores=science.fold_scores,
        conditional_oof_predictions=oof,
        calibration_by_outcome=science.calibration_by_outcome,
        drift_summary=science.drift_summary,
        stability=science.stability,
        holdout=science.holdout,
        holdout_eval=science.holdout_eval,
        final_model=science.final_model,
        final_calibrators=science.final_calibrators,
        live_model_predictions=science.live_model_predictions,
        state_predictions=science.state_predictions,
        competing_fold_scores=competing,
        ablation=science.ablation,
        binary_ablation=science.binary_ablation,
        final_competing=science.final_competing,
        diagnostics={"probabilities_are_joint_distribution": True},
    )
    probs = probabilities_from_science(science, fallback)
    assert probs.probability_source == "state_conditional_oof"
    assert probs.p_true_break == pytest.approx(0.3)
    assert probs.p_true_break != fallback.p_true_break
    mass = (
        probs.p_expansion_accepting
        + probs.p_rejection_return_to_asia
        + probs.p_repriced_balance
        + probs.p_residual
    )
    assert mass == pytest.approx(1.0)
    assert probs.probabilities_are_joint_distribution is True


def test_live_competing_predictions_do_not_use_future_y() -> None:
    rows: list[dict[str, object]] = []
    for i in range(24):
        rows.extend(
            _labeled_setup(
                i,
                expansion=1.0 if i % 2 == 0 else 0.0,
                rejection=0.0 if i % 2 == 0 else 1.0,
                repriced=0.0,
                feat=0.85 if i % 2 == 0 else 0.15,
                available_at=i + 1,
            )
        )
    labeled = pivot_competing_labels(pl.DataFrame(rows))
    model = fit_competing_risk(
        labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= 15),
        feature_names=("feat_a", "feat_b"),
        train_end_ts=16,
        min_train=8,
        min_class=1,
    )
    assert model.is_usable()
    states = pl.DataFrame(
        {
            AVAILABILITY_TS: [16, 17],
            "feat_a": [0.85, 0.15],
            "feat_b": [0.15, 0.85],
            "y": [1.0, 0.0],
        }
    )
    preds = predict_competing_at_states(
        model,
        states,
        prediction_is_oof=False,
        eligible_for_backtest=False,
    )
    assert "y" not in preds.columns
    assert preds["eligible_for_backtest"].unique().to_list() == [False]
    assert competing_rows_sum_to_one(
        np.column_stack([preds[f"p_{name}"].to_numpy() for name in COMPETING_CLASS_NAMES])
    )
