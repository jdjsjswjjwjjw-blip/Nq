"""بروتوكول السنة: 4 أشهر تدريب · 4 walk-forward · 4 holdout — بلا إعادة بناء."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from nq.auction_behavior.holdout import carve_frozen_holdout
from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.auction_behavior.science import ScienceConfig
from nq.auction_behavior.walk_forward import (
    expanding_min_train_months,
    month_key_from_ns,
    unique_month_keys,
)
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.research.behavior_period import (
    default_period_science_config,
    run_behavior_period_science,
)
from nq.validation.leakage import assert_temporal_split
from tests.realized_path_factory import path_bar_fields, path_kind_for_index

_HOUR_NS = 3_600 * 1_000_000_000


def _month_ns(year: int, month: int, day: int = 10, hour: int = 8) -> int:
    stamp = dt.datetime(year, month, day, hour, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def _year_blended(*, year: int = 2025, episodes_per_month: int = 6, bars: int = 8) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    story = 0
    for month in range(1, 13):
        ts = _month_ns(year, month)
        for episode in range(episodes_per_month):
            path_kind = path_kind_for_index(story)
            imbalance = 1.0 if path_kind in {"further_beyond_asia", "continue_direction"} else 0.0
            for bar in range(bars):
                testing = 1.0 if bar in (0, 1) else 0.0
                rows.append(
                    {
                        AVAILABILITY_TS: ts,
                        VP_LIQUIDITY_SESSION: int(VpLiquiditySession.LONDON),
                        "_behavior_story_run": story,
                        "proj_expansion_testing": testing,
                        "proj_expansion_accepting": 0.0,
                        "proj_rejection_to_asia": 0.0,
                        "proj_repriced_balance": 0.0,
                        "vp_imbalance": imbalance,
                        "struct_dist_vah_ticks": float(bar) - 3.0,
                        "lf_arrival_intensity": float((month * 7 + episode + bar) % 5),
                        "rel_credibility": float((episode + bar) % 3) / 2.0,
                        "mem_time_since_break": float(bar),
                        **path_bar_fields(path_kind, bar),
                    }
                )
                ts += _HOUR_NS
            story += 1
    return pl.DataFrame(rows)


def test_expanding_min_train_months_locks_four_wf_windows() -> None:
    assert expanding_min_train_months(8, min_train_months=4, walk_forward_months=4) == 4
    assert expanding_min_train_months(8, min_train_months=1, walk_forward_months=None) == 1
    with pytest.raises(ValueError, match="need >= 4 train"):
        expanding_min_train_months(5, min_train_months=4, walk_forward_months=4)


def test_carve_holdout_last_four_calendar_months() -> None:
    frame = _year_blended()
    pack = carve_frozen_holdout(frame, holdout_months=4, ts_col=AVAILABILITY_TS)
    assert pack.holdout_months == 4
    assert unique_month_keys(pack.develop, ts_col=AVAILABILITY_TS) == (
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
    )
    assert unique_month_keys(pack.holdout, ts_col=AVAILABILITY_TS) == (
        "2025-09",
        "2025-10",
        "2025-11",
        "2025-12",
    )
    assert_temporal_split(
        pack.develop[AVAILABILITY_TS].to_numpy(),
        pack.holdout[AVAILABILITY_TS].to_numpy(),
        embargo=0.0,
    )


def test_carve_holdout_months_rejects_short_span() -> None:
    frame = _year_blended().head(20)
    with pytest.raises(ValueError, match="distinct setup months"):
        carve_frozen_holdout(frame, holdout_months=4, ts_col=AVAILABILITY_TS)


def test_year_protocol_is_four_four_four_without_touching_holdout() -> None:
    blended = _year_blended()
    cfg = default_period_science_config()
    assert cfg.min_train_months == 4
    assert cfg.walk_forward_months == 4
    assert cfg.holdout_months == 4
    assert cfg.evaluate_holdout is False
    report = run_behavior_period_science(
        blended=blended,
        day_ids=tuple(f"2025-{m:02d}-10" for m in range(1, 13)),
        config=cfg,
        include_ablation=False,
    )
    science = report.science
    assert science.diagnostics["holdout_touched"] is False
    assert science.diagnostics["competing_family"] == "realized_path"
    assert science.diagnostics["scenario_labels_are_features_not_exclusive_y"] is True
    assert science.diagnostics["include_assumed_script_outcomes"] is False
    assert science.diagnostics["holdout_months"] == 4
    assert science.diagnostics["walk_forward_months"] == 4
    assert science.diagnostics["min_train_months_used"] == 4
    assert science.diagnostics["n_folds"] == 4
    assert report.diagnostics["book_not_reconstructed"] is True
    assert report.diagnostics["raw_mbo_not_loaded"] is True

    labeled = science.labeled.filter(pl.col("label_status") == "resolved")
    develop_months = unique_month_keys(
        labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= science.holdout.cut_ts),
        ts_col=SETUP_AVAILABILITY_TS,
    )
    holdout_months = unique_month_keys(
        labeled.filter(pl.col(SETUP_AVAILABILITY_TS) > science.holdout.cut_ts),
        ts_col=SETUP_AVAILABILITY_TS,
    )
    assert develop_months == (
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
    )
    assert holdout_months == ("2025-09", "2025-10", "2025-11", "2025-12")

    test_segments = science.fold_frame["segment"].to_list()
    assert test_segments == [
        "month->2025-05",
        "month->2025-06",
        "month->2025-07",
        "month->2025-08",
    ]
    oof = science.conditional_oof_predictions
    assert oof.height >= 1
    oof_months = {month_key_from_ns(int(t)) for t in oof[AVAILABILITY_TS].to_list()}
    assert oof_months <= {"2025-05", "2025-06", "2025-07", "2025-08"}
    leaked = oof.filter(pl.col(AVAILABILITY_TS) <= pl.col("model_train_end_ts"))
    assert leaked.height == 0
    assert_temporal_split(
        labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= science.holdout.cut_ts)[
            SETUP_AVAILABILITY_TS
        ].to_numpy(),
        labeled.filter(pl.col(SETUP_AVAILABILITY_TS) > science.holdout.cut_ts)[
            SETUP_AVAILABILITY_TS
        ].to_numpy(),
        embargo=0.0,
    )


def test_day_science_defaults_are_unchanged() -> None:
    cfg = ScienceConfig()
    assert cfg.holdout_frac == 0.2
    assert cfg.holdout_months is None
    assert cfg.min_train_months == 1
    assert cfg.walk_forward_months is None
    assert cfg.competing_family == "realized_path"
    assert cfg.include_assumed_script_outcomes is False
    assert cfg.max_features == 68
    assert cfg.extend_horizon_bars == 50
    assert cfg.extend_points == 5.0
