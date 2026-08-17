"""ميكانيكا الامتداد: سبق الحجم/العمق، تسلسل المزاد، حماية المركز."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.expansion_mechanics import (
    DEPTH_LEAD_CLASS_COL,
    LEAD_CLASS_COL,
    SEQUENCE_CLASS_COL,
    ExpansionMechanicsConfig,
    assert_not_raw_mbo_stream,
    classify_auction_sequence,
    classify_depth_price_lead,
    classify_volume_price_lead,
    run_expansion_mechanics,
    run_expansion_mechanics_from_period_dir,
    write_expansion_mechanics_report,
)

_HOUR_NS = 3_600 * 1_000_000_000


def _month_ns(year: int, month: int, day: int = 10) -> int:
    stamp = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def _row(
    *,
    ts: int,
    y: float,
    outcome: str,
    vol: float,
    beyond: float,
    follow: float,
    defend: float = 0.1,
    vol_lag: float | None = None,
    beyond_lag: float | None = None,
    follow_lag: float | None = None,
    balance_lag: float = 0.0,
    imb_lag: float = 0.0,
    held: float = 0.0,
    poc: float = 0.0,
    migration: float = 0.0,
    absorb: float = 0.0,
    testing: float = 0.0,
    accepting: float = 0.0,
) -> dict[str, float | int | str]:
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: ts + _HOUR_NS,
        "outcome_name": outcome,
        "y": y,
        "label_status": "resolved",
        "proj_outside_volume_share": vol,
        "path_beyond_asia_ticks": beyond,
        "path_depth_follow": follow,
        "path_depth_defend": defend,
        "path_held_frac": held,
        "proj_poc_shift_ticks": poc,
        "lf_liquidity_migration": migration,
        "lf_absorption_proxy": absorb,
        "proj_expansion_testing": testing,
        "proj_expansion_accepting": accepting,
        "proj_outside_volume_share__lag5": vol if vol_lag is None else vol_lag,
        "path_beyond_asia_ticks__lag5": beyond if beyond_lag is None else beyond_lag,
        "path_depth_follow__lag5": follow if follow_lag is None else follow_lag,
        "vp_balance__lag5": balance_lag,
        "vp_imbalance__lag3": imb_lag,
        "vp_fsm_break__lag3": 0.0,
        "proj_expansion_testing__lag3": 0.0,
        "vp_balance": 0.0,
        "vp_imbalance": 0.0,
        "vp_fsm_break": 0.0,
        "vp_fsm_expand": 0.0,
    }


def test_volume_lead_is_not_price_lead() -> None:
    frame = pl.DataFrame(
        [
            _row(
                ts=1,
                y=1.0,
                outcome="y_path_further_beyond",
                vol=0.7,
                beyond=8.0,
                follow=2.0,
                vol_lag=0.6,
                beyond_lag=0.0,
            ),
            _row(
                ts=2,
                y=1.0,
                outcome="y_path_further_beyond",
                vol=0.7,
                beyond=8.0,
                follow=2.0,
                vol_lag=0.0,
                beyond_lag=6.0,
            ),
            _row(
                ts=3,
                y=1.0,
                outcome="y_path_further_beyond",
                vol=0.7,
                beyond=8.0,
                follow=2.0,
                vol_lag=0.6,
                beyond_lag=6.0,
            ),
            _row(
                ts=4,
                y=0.0,
                outcome="y_path_further_beyond",
                vol=0.05,
                beyond=1.0,
                follow=0.4,
                vol_lag=0.0,
                beyond_lag=0.0,
            ),
        ]
    )
    labeled = classify_volume_price_lead(frame)
    classes = labeled[LEAD_CLASS_COL].to_list()
    assert classes == ["volume_lead", "price_lead", "both_already", "neither"]


def test_depth_lead_uses_follow_lag_not_current() -> None:
    frame = pl.DataFrame(
        [
            _row(
                ts=1,
                y=1.0,
                outcome="y_path_further_beyond",
                vol=0.2,
                beyond=8.0,
                follow=3.0,
                follow_lag=2.0,
                beyond_lag=0.0,
            )
        ]
    )
    out = classify_depth_price_lead(frame)
    assert out[DEPTH_LEAD_CLASS_COL].to_list() == ["depth_lead"]


def test_balance_imbalance_expansion_sequence() -> None:
    frame = pl.DataFrame(
        [
            _row(
                ts=1,
                y=1.0,
                outcome="y_path_further_beyond",
                vol=0.6,
                beyond=6.0,
                follow=2.0,
                beyond_lag=0.0,
                balance_lag=1.0,
                imb_lag=1.0,
                testing=1.0,
            )
        ]
    )
    out = classify_auction_sequence(frame)
    assert out[SEQUENCE_CLASS_COL].to_list() == ["balance_imbalance_expansion"]


def test_already_expanding_when_price_was_already_out() -> None:
    frame = pl.DataFrame(
        [
            _row(
                ts=1,
                y=1.0,
                outcome="y_path_further_beyond",
                vol=0.6,
                beyond=10.0,
                follow=2.0,
                beyond_lag=8.0,
                balance_lag=0.0,
                imb_lag=0.0,
            )
        ]
    )
    out = classify_auction_sequence(frame)
    assert out[SEQUENCE_CLASS_COL].to_list() == ["already_expanding"]


def test_missing_lags_do_not_crash_and_mark_fallback() -> None:
    frame = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1, 2],
            OUTCOME_AVAILABLE_TS: [2, 3],
            "outcome_name": ["y_path_further_beyond", "y_path_further_beyond"],
            "y": [1.0, 0.0],
            "label_status": ["resolved", "resolved"],
            "proj_outside_volume_share": [0.7, 0.05],
            "path_beyond_asia_ticks": [8.0, 1.0],
            "path_depth_follow": [2.0, 0.4],
        }
    )
    report = run_expansion_mechanics(
        frame, config=ExpansionMechanicsConfig(holdout_months=None, n_permutations=31)
    )
    assert report.diagnostics["lag_classification_uses_contemporaneous_fallback"] is True
    assert report.diagnostics["holdout_scored"] is False
    assert report.lead_lag.height >= 1


def test_holdout_months_are_excluded_and_never_scored() -> None:
    rows: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        holdout = month >= 9
        rows.append(
            _row(
                ts=ts,
                y=1.0 if holdout else 0.0,
                outcome="y_path_further_beyond",
                vol=0.8 if holdout else 0.05,
                beyond=12.0 if holdout else 0.5,
                follow=3.0 if holdout else 0.2,
                vol_lag=0.7 if holdout else 0.0,
                beyond_lag=0.0,
                follow_lag=0.2,
            )
        )
        rows.append(
            _row(
                ts=ts + 1,
                y=0.0,
                outcome="y_path_reverse",
                vol=0.05,
                beyond=0.5,
                follow=0.2,
                vol_lag=0.0,
                beyond_lag=0.0,
            )
        )
    labeled = pl.DataFrame(rows)
    report = run_expansion_mechanics(
        labeled, config=ExpansionMechanicsConfig(holdout_months=4, n_permutations=31)
    )
    assert report.diagnostics["holdout_scored"] is False
    assert report.diagnostics["holdout_excluded"] is True
    assert report.diagnostics["holdout_n_rows"] >= 4
    vol = report.lead_lag.filter(
        (pl.col("scope") == "develop") & (pl.col(LEAD_CLASS_COL) == "volume_lead")
    )
    assert vol.height == 1
    assert int(vol["n"][0]) == 0


def test_oof_scope_is_primary_when_timestamps_provided() -> None:
    rows: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        if month >= 9:
            y, vol_lag, beyond_lag = 1.0, 0.8, 0.0
        elif month >= 5:
            y, vol_lag, beyond_lag = 1.0, 0.6, 6.0
        else:
            y, vol_lag, beyond_lag = 0.0, 0.0, 0.0
        rows.append(
            _row(
                ts=ts,
                y=y,
                outcome="y_path_further_beyond",
                vol=0.7 if y else 0.05,
                beyond=8.0 if y else 0.5,
                follow=2.0 if y else 0.2,
                vol_lag=vol_lag,
                beyond_lag=beyond_lag,
                held=0.5 if y else 0.0,
                poc=10.0 if y else 0.0,
            )
        )
    labeled = pl.DataFrame(rows)
    oof_ts = [_month_ns(2025, month) for month in (5, 6, 7, 8)]
    report = run_expansion_mechanics(
        labeled,
        config=ExpansionMechanicsConfig(holdout_months=4, n_permutations=31),
        oof_availability_ts=oof_ts,
    )
    assert report.diagnostics["primary_scope"] == "oof_develop"
    assert report.diagnostics["holdout_scored"] is False
    both = report.lead_lag.filter(
        (pl.col("scope") == "oof_develop") & (pl.col(LEAD_CLASS_COL) == "both_already")
    )
    assert both.height == 1
    assert int(both["n"][0]) == 4
    assert float(both["pos_rate"][0]) == 1.0


def test_protection_follow_higher_on_successful_expansion() -> None:
    rows: list[dict[str, float | int | str]] = []
    ts = _month_ns(2025, 3)
    for i in range(12):
        success = i < 6
        rows.append(
            _row(
                ts=ts + i,
                y=1.0 if success else 0.0,
                outcome="y_path_further_beyond",
                vol=0.7,
                beyond=10.0,
                follow=2.5 if success else 0.4,
                defend=0.05 if success else 0.4,
                held=0.6 if success else 0.05,
                vol_lag=0.6,
                beyond_lag=8.0,
                follow_lag=2.0 if success else 0.3,
                poc=12.0 if success else 0.2,
                migration=0.8 if success else 0.2,
                absorb=0.1 if success else 0.8,
            )
        )
    labeled = pl.DataFrame(rows)
    report = run_expansion_mechanics(
        labeled, config=ExpansionMechanicsConfig(holdout_months=None, n_permutations=63)
    )
    follow = report.protection.filter(
        (pl.col("feature") == "path_depth_follow") & (pl.col("already_expanded"))
    )
    assert follow.height >= 1
    assert float(follow["mean_pos"][0]) > float(follow["mean_neg"][0])
    defend = report.protection.filter(
        (pl.col("feature") == "path_depth_defend") & (pl.col("already_expanded"))
    )
    assert defend.height >= 1
    assert float(defend["mean_pos"][0]) < float(defend["mean_neg"][0])


def test_shuffling_y_does_not_change_lead_class() -> None:
    frame = pl.DataFrame(
        [
            _row(
                ts=1,
                y=1.0,
                outcome="y_path_further_beyond",
                vol=0.7,
                beyond=8.0,
                follow=2.0,
                vol_lag=0.6,
                beyond_lag=0.0,
            ),
            _row(
                ts=2,
                y=0.0,
                outcome="y_path_further_beyond",
                vol=0.1,
                beyond=8.0,
                follow=0.5,
                vol_lag=0.0,
                beyond_lag=6.0,
            ),
        ]
    )
    a = classify_volume_price_lead(frame)[LEAD_CLASS_COL].to_list()
    shuffled = frame.with_columns(pl.col("y").reverse().alias("y"))
    b = classify_volume_price_lead(shuffled)[LEAD_CLASS_COL].to_list()
    assert a == b


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame(
        {
            SETUP_AVAILABILITY_TS: [1],
            "order_id": [1],
            "action": ["A"],
            "outcome_name": ["y_path_further_beyond"],
            "y": [1.0],
        }
    )
    with pytest.raises(ValueError, match="refuses raw MBO"):
        assert_not_raw_mbo_stream(raw)
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_expansion_mechanics(raw, config=ExpansionMechanicsConfig(holdout_months=None))


def test_write_report_and_period_dir_roundtrip(tmp_path: Path) -> None:
    labeled = pl.DataFrame(
        [
            _row(
                ts=_month_ns(2025, m),
                y=float(m % 2),
                outcome="y_path_further_beyond",
                vol=0.6,
                beyond=8.0,
                follow=2.0,
                vol_lag=0.5,
                beyond_lag=6.0,
                held=0.4,
                poc=8.0,
            )
            for m in range(1, 9)
        ]
    )
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    oof = pl.DataFrame(
        {
            AVAILABILITY_TS: [_month_ns(2025, m) for m in (5, 6, 7, 8)],
            "eligible_for_backtest": [True] * 4,
        }
    )
    oof.write_parquet(period / "oof_predictions.parquet")
    (period / "summary.json").write_text(
        json.dumps(
            {
                "diagnostics": {
                    "science": {
                        "holdout_cut_ts": _month_ns(2025, 8),
                        "holdout_touched": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    report = run_expansion_mechanics_from_period_dir(
        period, config=ExpansionMechanicsConfig(holdout_months=None, n_permutations=31)
    )
    out = tmp_path / "mech"
    write_expansion_mechanics_report(report, out)
    assert (out / "EXPANSION.md").is_file()
    assert (out / "expansion_mechanics.json").is_file()
    text = (out / "EXPANSION.md").read_text(encoding="utf-8")
    assert "Holdout never scored" in text
    assert "No MBO reload" in text
    assert report.diagnostics["holdout_scored"] is False
    assert report.diagnostics["primary_scope"] == "oof_develop"
