"""مرحلة 2: تجميع أيام ثم علم واحد — ليس متوسط احتمالات يومية."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    build_labeled_outcomes,
)
from nq.auction_behavior.science import ScienceConfig
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.research.behavior_period import (
    assert_labels_do_not_cross_period_days,
    assert_labels_do_not_cross_session_dates,
    load_period_blended,
    remint_period_story_runs,
    run_behavior_period_science,
    write_behavior_period_report,
)

_HOUR_NS = 3_600 * 1_000_000_000
# 2026-08-03 08:00 UTC ≈ Asia for that session date; next calendar day later.
_DAY1_START = 1_785_744_000_000_000_000  # 2026-08-03 08:00:00 UTC
_DAY2_START = 1_785_830_400_000_000_000  # 2026-08-04 08:00:00 UTC


def _day_bars(start_ts: int, *, story_run: int, n: int = 8, resolve: bool) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    for bar in range(n):
        testing = 1.0 if bar in (0, 1) else 0.0
        accepting = 1.0 if (resolve and bar == 2) else 0.0
        rows.append(
            {
                AVAILABILITY_TS: start_ts + bar * _HOUR_NS,
                VP_LIQUIDITY_SESSION: int(VpLiquiditySession.ASIA),
                "_behavior_story_run": story_run,
                "proj_expansion_testing": testing,
                "proj_expansion_accepting": accepting,
                "proj_rejection_to_asia": 0.0,
                "proj_repriced_balance": 0.0,
                "vp_imbalance": 1.0 if resolve else 0.0,
                "vp_balance": 0.0 if resolve else 1.0,
                "struct_dist_vah_ticks": float(bar),
                "lf_arrival_intensity": float(bar),
                "rel_credibility": 0.5,
                "mem_time_since_break": float(bar),
            }
        )
    return pl.DataFrame(rows)


def test_remint_prevents_cross_day_label_window() -> None:
    """قصة يوم 1 غير محسومة + انتقال يوم 2 بنفس رقم القصة المحلي.

    بدون إعادة الترقيم تمتد النافذة عبر منتصف الليل. بعدها تبقى التسمية داخل اليوم.
    """
    day1 = _day_bars(_DAY1_START, story_run=1, resolve=False)
    day2 = _day_bars(_DAY2_START, story_run=1, resolve=True)
    naive = pl.concat([day1, day2]).sort(AVAILABILITY_TS)
    naive_labels = build_labeled_outcomes(naive, outcome_window=8, group_col="_behavior_story_run")
    leaked = naive_labels.filter(
        (pl.col(SETUP_AVAILABILITY_TS) < _DAY2_START)
        & (pl.col(OUTCOME_AVAILABLE_TS) >= _DAY2_START)
    )
    assert leaked.height >= 1

    reminted = remint_period_story_runs(naive)
    assert reminted["_behavior_story_run"].n_unique() == 2
    assert reminted["_liquidity_run"].n_unique() == 2
    safe_labels = build_labeled_outcomes(
        reminted, outcome_window=8, group_col="_behavior_story_run"
    )
    assert_labels_do_not_cross_session_dates(safe_labels)
    assert_labels_do_not_cross_period_days(safe_labels, reminted)
    crossed = safe_labels.filter(
        (pl.col(SETUP_AVAILABILITY_TS) < _DAY2_START)
        & (pl.col(OUTCOME_AVAILABLE_TS) >= _DAY2_START)
    )
    assert crossed.height == 0


def test_load_period_blended_isolates_same_cme_session_split_across_day_files(
    tmp_path: Path,
) -> None:
    """18:00 ET يوم 1 و 01:00 ET يوم 2 نفس تاريخ جلسة CME، لكن ملفي مرحلة-1 منفصلين.

    بدون ``_period_day_id`` تندمج القصة عبر منتصف الليل رغم أن كل دفتر أُعيد بناؤه وحده.
    """
    # 2026-08-03 22:00 UTC = 18:00 EDT → آسيا، تاريخ الجلسة 2026-08-04
    asia_open = 1_785_794_400_000_000_000
    day1 = _day_bars(asia_open, story_run=1, n=4, resolve=False)
    # 2026-08-04 05:00 UTC = 01:00 EDT → ما زالت آسيا لنفس تاريخ الجلسة
    after_midnight = 1_785_819_600_000_000_000
    day2 = _day_bars(after_midnight, story_run=1, n=4, resolve=True)
    dir_a = tmp_path / "2026-08-03"
    dir_b = tmp_path / "2026-08-04"
    dir_a.mkdir()
    dir_b.mkdir()
    day1.write_parquet(dir_a / "blended.parquet")
    day2.write_parquet(dir_b / "blended.parquet")

    naive = pl.concat([day1, day2]).sort(AVAILABILITY_TS)
    naive_labels = build_labeled_outcomes(naive, outcome_window=8, group_col="_behavior_story_run")
    leaked = naive_labels.filter(
        (pl.col(SETUP_AVAILABILITY_TS) < after_midnight)
        & (pl.col(OUTCOME_AVAILABLE_TS) >= after_midnight)
    )
    assert leaked.height >= 1

    pooled, ids = load_period_blended([dir_a, dir_b])
    assert ids == ("2026-08-03", "2026-08-04")
    assert pooled["_behavior_story_run"].n_unique() == 2
    assert pooled["_liquidity_run"].n_unique() == 2
    safe_labels = build_labeled_outcomes(pooled, outcome_window=8, group_col="_behavior_story_run")
    assert_labels_do_not_cross_session_dates(safe_labels)
    assert_labels_do_not_cross_period_days(safe_labels, pooled)
    crossed = safe_labels.filter(
        (pl.col(SETUP_AVAILABILITY_TS) < after_midnight)
        & (pl.col(OUTCOME_AVAILABLE_TS) >= after_midnight)
    )
    assert crossed.height == 0


def test_load_period_blended_rejects_duplicate_availability_ts(tmp_path: Path) -> None:
    day_a = tmp_path / "2026-08-03"
    day_b = tmp_path / "2026-08-04"
    day_a.mkdir()
    day_b.mkdir()
    frame = _day_bars(_DAY1_START, story_run=1, resolve=True)
    frame.write_parquet(day_a / "blended.parquet")
    frame.write_parquet(day_b / "blended.parquet")
    with pytest.raises(ValueError, match="duplicate availability_ts"):
        load_period_blended([day_a, day_b])


def test_period_science_is_pooled_walk_forward_not_daily_average(tmp_path: Path) -> None:
    rows: list[dict[str, float | int]] = []
    ts = _DAY1_START
    for episode in range(48):
        kind = episode % 3
        imbalance = 1.0 if kind == 0 else 0.0
        for bar in range(8):
            testing = 1.0 if bar in (0, 1) else 0.0
            accepting = 1.0 if (kind == 0 and bar == 2) else 0.0
            rejection = 1.0 if (kind == 1 and bar == 2) else 0.0
            rows.append(
                {
                    AVAILABILITY_TS: ts,
                    VP_LIQUIDITY_SESSION: int(VpLiquiditySession.LONDON),
                    "_behavior_story_run": 1,  # متعمّد: تصادم محلي عبر الحلقات
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
    blended = pl.DataFrame(rows)
    cfg = ScienceConfig(
        outcome_window=5,
        competing_window=5,
        n_splits=3,
        min_train_size=8,
        holdout_frac=0.2,
        use_month_folds=False,
        evaluate_holdout=False,
        competing_min_train=10,
        competing_min_class=2,
    )
    report = run_behavior_period_science(
        blended=blended,
        day_ids=("2026-08-03", "2026-08-04"),
        config=cfg,
        include_ablation=True,
    )
    assert report.diagnostics["pooled_not_averaged_daily_probabilities"] is True
    assert report.diagnostics["phase1_day_files_not_rejoined"] is True
    assert report.blended["_behavior_story_run"].n_unique() > 1
    assert report.diagnostics["live_predictions_eligible_for_backtest"] is False
    assert report.diagnostics["oof_predictions_eligible_for_backtest"] is True
    assert report.science.conditional_oof_predictions.height >= 1
    assert report.science.diagnostics["holdout_touched"] is False
    assert report.ablation is not None
    assert report.ablation.diagnostics["holdout_untouched"] is True
    assert_labels_do_not_cross_session_dates(report.science.labeled)
    assert_labels_do_not_cross_period_days(report.science.labeled, report.blended)

    out = tmp_path / "period"
    write_behavior_period_report(report, out)
    assert (out / "summary.json").is_file()
    assert (out / "PERIOD.md").is_file()
    assert (out / "oof_predictions.parquet").is_file()
    text = (out / "PERIOD.md").read_text(encoding="utf-8")
    assert "Not a mean of per-day probabilities" in text
