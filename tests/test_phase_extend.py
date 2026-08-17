"""طور امتداد هيكلي: 15 برميلًا + ATR لندن السابق، بلا هدف بالنقاط الثابتة."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.auction_behavior.phase_extend import (
    PHASE_EXPAND_ATR_FRAC,
    PHASE_GIVEBACK_ATR_FRAC,
    PHASE_HORIZON_BARS,
    Y_PHASE_EXTEND,
    build_phase_extend_outcomes,
    prior_london_atr14,
)
from nq.auction_behavior.realized_path import science_outcome_targets
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.validation.leakage import assert_availability_not_before_event

_ET = ZoneInfo("America/New_York")
_LONDON = int(VpLiquiditySession.LONDON)


def _ns(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    stamp = dt.datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _london_story(
    *,
    day: int,
    n: int,
    close: float,
    high: float,
    low: float,
    story: int,
    onset_at: int | None = None,
    dip_low: float | None = None,
    dip_bar: int | None = None,
) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    for bar in range(n):
        is_onset = onset_at is not None and bar == onset_at
        bar_low = dip_low if dip_bar is not None and bar == dip_bar else low
        bar_high = high if bar > (onset_at or -1) else close
        rows.append(
            {
                AVAILABILITY_TS: _ns(2025, 6, day, 4, bar),
                VP_LIQUIDITY_SESSION: _LONDON,
                "close": close,
                "high": bar_high if bar > (onset_at or -1) else close,
                "low": bar_low if bar > (onset_at or -1) else close,
                "path_beyond_asia_ticks": 2.0
                if (onset_at is not None and bar >= onset_at)
                else 0.0,
                "vp_fsm_break": 1.0 if is_onset else 0.0,
                "vp_fsm_retest": 0.0,
                "proj_break_direction": 1.0,
                "_behavior_story_run": story,
            }
        )
    return pl.DataFrame(rows)


def test_science_keeps_numeric_horizon_and_adds_phase() -> None:
    names = science_outcome_targets(include_assumed_scripts=False)
    assert "y_path_further_beyond" in names
    assert "y_extend_5pts_25min" in names
    assert Y_PHASE_EXTEND in names
    assert PHASE_HORIZON_BARS == 15
    assert PHASE_EXPAND_ATR_FRAC == 0.2
    assert PHASE_GIVEBACK_ATR_FRAC == 0.1


def test_london_atr_excludes_current_london_session() -> None:
    prior = _london_story(day=2, n=3, close=110.0, high=150.0, low=100.0, story=0)
    # Force OHLC on prior day: range 50
    prior = prior.with_columns(
        pl.lit(150.0).alias("high"),
        pl.lit(100.0).alias("low"),
        pl.lit(120.0).alias("close"),
    )
    today = _london_story(day=3, n=3, close=200.0, high=600.0, low=200.0, story=1)
    today = today.with_columns(
        pl.lit(600.0).alias("high"),
        pl.lit(200.0).alias("low"),
        pl.lit(250.0).alias("close"),
    )
    frame = pl.concat([prior, today])
    atr = prior_london_atr14(frame)
    assert atr[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert atr[3:] == pytest.approx([50.0, 50.0, 50.0])


def test_phase_extend_true_when_structure_holds() -> None:
    prior = _london_story(day=2, n=3, close=120.0, high=150.0, low=100.0, story=0)
    prior = prior.with_columns(
        pl.lit(150.0).alias("high"),
        pl.lit(100.0).alias("low"),
        pl.lit(120.0).alias("close"),
    )
    today = _london_story(
        day=3,
        n=16,
        close=200.0,
        high=211.0,
        low=198.0,
        story=1,
        onset_at=0,
    )
    labels = build_phase_extend_outcomes(
        pl.concat([prior, today]), window=15, group_col="_behavior_story_run"
    )
    resolved = labels.filter(pl.col("label_status") == "resolved")
    assert resolved.height == 1
    assert float(resolved["y"][0]) == 1.0
    assert_availability_not_before_event(
        resolved[SETUP_AVAILABILITY_TS].to_numpy(),
        resolved[OUTCOME_AVAILABLE_TS].to_numpy(),
    )


def test_phase_extend_false_when_entry_is_given_back() -> None:
    prior = _london_story(day=2, n=3, close=120.0, high=150.0, low=100.0, story=0)
    prior = prior.with_columns(
        pl.lit(150.0).alias("high"),
        pl.lit(100.0).alias("low"),
        pl.lit(120.0).alias("close"),
    )
    today = _london_story(
        day=3,
        n=16,
        close=200.0,
        high=211.0,
        low=198.0,
        story=1,
        onset_at=0,
        dip_low=194.0,
        dip_bar=8,
    )
    labels = build_phase_extend_outcomes(
        pl.concat([prior, today]), window=15, group_col="_behavior_story_run"
    )
    resolved = labels.filter(pl.col("label_status") == "resolved")
    assert resolved.height == 1
    assert float(resolved["y"][0]) == 0.0


def test_phase_extend_false_when_expansion_is_short() -> None:
    prior = _london_story(day=2, n=3, close=120.0, high=150.0, low=100.0, story=0)
    prior = prior.with_columns(
        pl.lit(150.0).alias("high"),
        pl.lit(100.0).alias("low"),
        pl.lit(120.0).alias("close"),
    )
    today = _london_story(
        day=3,
        n=16,
        close=200.0,
        high=205.0,
        low=198.0,
        story=1,
        onset_at=0,
    )
    labels = build_phase_extend_outcomes(
        pl.concat([prior, today]), window=15, group_col="_behavior_story_run"
    )
    assert float(labels.filter(pl.col("label_status") == "resolved")["y"][0]) == 0.0


def test_phase_extend_incomplete_window_is_censored() -> None:
    prior = _london_story(day=2, n=3, close=120.0, high=150.0, low=100.0, story=0)
    prior = prior.with_columns(
        pl.lit(150.0).alias("high"),
        pl.lit(100.0).alias("low"),
        pl.lit(120.0).alias("close"),
    )
    today = _london_story(
        day=3,
        n=6,
        close=200.0,
        high=211.0,
        low=198.0,
        story=1,
        onset_at=0,
    )
    labels = build_phase_extend_outcomes(
        pl.concat([prior, today]), window=15, group_col="_behavior_story_run"
    )
    assert labels.filter(pl.col("label_status") == "censored").height == 1
    assert labels.filter(pl.col("label_status") == "resolved").height == 0
