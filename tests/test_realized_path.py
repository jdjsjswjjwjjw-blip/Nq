"""أول انتقال متحقق — السيناريو ملمح، ليس الـY."""

from __future__ import annotations

import polars as pl
import pytest

from nq.auction_behavior.outcomes import (
    FIRST_TRANSITION_CLASS_COL,
    OUTCOME_AVAILABLE_TS,
    PRIMARY_OUTCOME_TARGETS,
    SETUP_AVAILABILITY_TS,
    build_first_transition_outcomes,
)
from nq.auction_behavior.realized_path import (
    REALIZED_NEXT_PATH_CLASSES,
    REALIZED_PATH_BINARY_TARGETS,
    build_realized_next_path_outcomes,
    build_realized_path_binary_outcomes,
    competing_family_spec,
    science_outcome_targets,
)
from nq.auction_behavior.science import ScienceConfig
from nq.contracts.temporal import AVAILABILITY_TS
from nq.validation.leakage import assert_availability_not_before_event
from tests.realized_path_factory import path_bar_fields, path_kind_for_index


def _path_frame() -> pl.DataFrame:
    """كسر بلا ريتست ثم ابتعاد عن آسيا — إعداد صالح في النظام الجديد."""
    n = 12
    return pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "path_beyond_asia_ticks": [0.0, 2.0, 2.0, 4.0, 5.0, 5.0, 6.0, 6.0, 3.0, 0.4, 0.2, 0.0],
            "path_inside_asia_va": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "proj_poc_step_ticks": [0.0] * n,
            "vp_look_fail": [0.0] * n,
            "vp_fsm_expand": [0.0] * n,
            "vp_fsm_break": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "vp_fsm_retest": [0.0] * n,
            "vp_balance": [0.0] * n,
            "vp_absorb": [0.0] * n,
            "vp_close_in_value": [0.0] * n,
            "_behavior_story_run": [1] * n,
        }
    )


def test_science_default_does_not_fit_assumed_scripts() -> None:
    cfg = ScienceConfig()
    assert cfg.competing_family == "realized_path"
    assert cfg.include_assumed_script_outcomes is False
    names = science_outcome_targets(include_assumed_scripts=False)
    for script in PRIMARY_OUTCOME_TARGETS:
        assert script not in names
    assert "y_extend_5pts_25min" in names
    assert "y_path_further_beyond" in names
    assert "y_phase_extend" in names
    assert "y_clean" in names
    family, classes = competing_family_spec(cfg.competing_family)
    assert family == "realized_path"
    assert classes == REALIZED_NEXT_PATH_CLASSES
    assert "expansion_accepting" not in classes


def test_break_without_retest_is_a_valid_setup() -> None:
    labels = build_realized_next_path_outcomes(
        _path_frame(), window=5, group_col="_behavior_story_run"
    )
    resolved = labels.filter(pl.col("label_status") == "resolved")
    assert resolved.height >= 1
    assert "further_beyond_asia" in resolved[FIRST_TRANSITION_CLASS_COL].to_list()


def test_next_transition_is_not_scenario_phase() -> None:
    labels = build_realized_next_path_outcomes(_path_frame(), window=5)
    classes = set(
        c for c in labels[FIRST_TRANSITION_CLASS_COL].drop_nulls().to_list() if c is not None
    )
    assert classes.isdisjoint(
        {"expansion_testing", "expansion_accepting", "rejection_return_to_asia", "repriced_balance"}
    )


def test_realized_path_respects_availability_order() -> None:
    labels = build_realized_next_path_outcomes(_path_frame(), window=6)
    assert labels.height >= 1
    assert_availability_not_before_event(
        labels[SETUP_AVAILABILITY_TS].to_numpy(),
        labels[OUTCOME_AVAILABLE_TS].to_numpy(),
    )


def test_incomplete_window_is_censored_not_no_change() -> None:
    short = _path_frame().head(3)
    labels = build_realized_next_path_outcomes(short, window=8)
    assert labels.filter(pl.col("label_status") == "censored").height >= 1
    assert labels.filter(pl.col(FIRST_TRANSITION_CLASS_COL) == "no_material_change").height == 0


def test_path_binaries_do_not_use_script_names() -> None:
    binaries = build_realized_path_binary_outcomes(_path_frame(), window=5)
    names = set(binaries["outcome_name"].to_list())
    assert names == set(REALIZED_PATH_BINARY_TARGETS)
    assert names.isdisjoint(set(PRIMARY_OUTCOME_TARGETS))


def test_assumed_script_engine_still_exists_for_annotation() -> None:
    """محرّك السيناريو لم يُحذف — فقط لم يعد الـY الافتراضي."""
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(8)),
            "proj_expansion_testing": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "proj_expansion_accepting": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "proj_rejection_to_asia": [0.0] * 8,
            "proj_repriced_balance": [0.0] * 8,
        }
    )
    scripts = build_first_transition_outcomes(frame, window=5)
    assert scripts.height >= 1
    family, classes = competing_family_spec("assumed_scripts")
    assert family == "assumed_scripts"
    assert "expansion_accepting" in classes


def test_factory_stories_mix_realized_path_classes_not_scenarios() -> None:
    rows: list[dict[str, float | int]] = []
    ts = 0
    for episode in range(6):
        kind = path_kind_for_index(episode)
        for bar in range(8):
            rows.append(
                {
                    AVAILABILITY_TS: ts,
                    "_behavior_story_run": episode,
                    **path_bar_fields(kind, bar),
                }
            )
            ts += 1
    labels = build_realized_next_path_outcomes(
        pl.DataFrame(rows), window=5, group_col="_behavior_story_run"
    )
    resolved = labels.filter(pl.col("label_status") == "resolved")
    classes = set(resolved[FIRST_TRANSITION_CLASS_COL].to_list())
    assert set(path_kind_for_index(i) for i in range(6)).issubset(classes)
    assert classes.isdisjoint(
        {"expansion_testing", "expansion_accepting", "rejection_return_to_asia", "repriced_balance"}
    )


def test_unknown_competing_family_rejected() -> None:
    with pytest.raises(ValueError, match="unknown competing_family"):
        competing_family_spec("full_path_screenplay")
