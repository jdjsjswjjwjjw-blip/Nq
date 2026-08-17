"""زخم سببي بجانب VP: ROC / CVD / VWAP / مدى، بلا تسريب."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from nq.auction_behavior.conditional import select_feature_names_by_family
from nq.auction_behavior.momentum import MOMENTUM_FEATURE_COLUMNS, attach_momentum_features
from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.auction_behavior.realized_path import (
    EXTEND_HORIZON_BARS,
    EXTEND_HORIZON_POINTS,
    Y_EXTEND_5PTS_25MIN,
    build_extend_horizon_outcomes,
    science_outcome_targets,
)
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import session_date_from_ns
from nq.validation.leakage import assert_availability_not_before_event

_ET = ZoneInfo("America/New_York")
_EPS = 1e-9


def _ns(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    stamp = dt.datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _ohlc_frame(
    *,
    starts: list[int],
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    delta: list[float] | None = None,
    intensity: list[float] | None = None,
    story: list[int] | None = None,
) -> pl.DataFrame:
    n = len(closes)
    highs = highs if highs is not None else [c + 1.0 for c in closes]
    lows = lows if lows is not None else [c - 1.0 for c in closes]
    return pl.DataFrame(
        {
            AVAILABILITY_TS: starts,
            "close": closes,
            "high": highs,
            "low": lows,
            "vp_of_delta": delta if delta is not None else [0.0] * n,
            "lf_arrival_intensity": intensity if intensity is not None else [1.0] * n,
            "_behavior_story_run": story if story is not None else [1] * n,
        }
    )


def test_roc_10_is_lookback_only() -> None:
    closes = [100.0 + i for i in range(12)]
    ts = [_ns(2025, 6, 2, 10, i) for i in range(12)]
    out = attach_momentum_features(_ohlc_frame(starts=ts, closes=closes))
    assert out["roc_10"][9] == pytest.approx(0.0, abs=_EPS)
    # (close_10 - close_0) / close_0
    assert out["roc_10"][10] == pytest.approx((110.0 - 100.0) / 100.0)
    mutated = attach_momentum_features(_ohlc_frame(starts=ts, closes=[*closes[:-1], 999.0]))
    assert out["roc_10"][10] == pytest.approx(float(mutated["roc_10"][10]))
    assert float(mutated["roc_10"][11]) != float(out["roc_10"][11])


def test_cvd_20_is_signed_flow_rolling_sum() -> None:
    n = 5
    ts = [_ns(2025, 6, 2, 10, i) for i in range(n)]
    delta = [0.5, -0.2, 0.1, 0.3, -0.4]
    intensity = [2.0, 2.0, 2.0, 2.0, 2.0]
    out = attach_momentum_features(
        _ohlc_frame(
            starts=ts,
            closes=[100.0] * n,
            delta=delta,
            intensity=intensity,
        )
    )
    expected = []
    acc = 0.0
    for d, w in zip(delta, intensity, strict=True):
        acc += d * w
        expected.append(acc)
    for i, value in enumerate(expected):
        assert out["cvd_20"][i] == pytest.approx(value)


def test_daily_atr_excludes_current_session() -> None:
    day1 = [_ns(2025, 6, 2, 10, i) for i in range(3)]
    day2 = [_ns(2025, 6, 3, 10, i) for i in range(3)]
    # day1 range = 20; day2 range would be 200 if leaked
    frame = _ohlc_frame(
        starts=[*day1, *day2],
        closes=[100.0, 105.0, 110.0, 200.0, 210.0, 300.0],
        highs=[110.0, 115.0, 120.0, 250.0, 260.0, 400.0],
        lows=[100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
        intensity=[1.0] * 6,
        story=[1, 1, 1, 2, 2, 2],
    )
    assert session_date_from_ns(day1[0]) != session_date_from_ns(day2[0])
    out = attach_momentum_features(frame)
    # ATR on day1 is unknown (no prior day) → distance/range use 0
    assert float(out["distance_to_vwap"][0]) == pytest.approx(0.0)
    assert float(out["range_width_ratio"][0]) == pytest.approx(0.0)
    # day2 ATR = day1 true range = 20 (high 120 - low 100); current day 200-wide range unused
    day2_row = out.filter(pl.col(AVAILABILITY_TS) == day2[0]).row(0, named=True)
    assert day2_row["range_width_ratio"] == pytest.approx(50.0 / 20.0)
    mutated = frame.with_columns(
        pl.when(pl.col(AVAILABILITY_TS) == day2[-1])
        .then(10_000.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    mutated_out = attach_momentum_features(mutated)
    assert float(out["distance_to_vwap"][3]) == pytest.approx(
        float(mutated_out["distance_to_vwap"][3])
    )
    assert float(out["range_width_ratio"][3]) == pytest.approx(
        float(mutated_out["range_width_ratio"][3])
    )


def test_vwap_is_causal_session_running_mean() -> None:
    ts = [_ns(2025, 6, 2, 10, i) for i in range(3)]
    closes = [100.0, 110.0, 120.0]
    intensity = [1.0, 1.0, 2.0]
    # force ATR via a prior day so distance is defined
    prior = [_ns(2025, 6, 1, 10, i) for i in range(2)]
    frame = _ohlc_frame(
        starts=[*prior, *ts],
        closes=[100.0, 100.0, *closes],
        highs=[110.0, 110.0, 101.0, 111.0, 121.0],
        lows=[90.0, 90.0, 99.0, 109.0, 119.0],
        intensity=[1.0, 1.0, *intensity],
        story=[0, 0, 1, 1, 1],
    )
    out = attach_momentum_features(frame)
    atr = 20.0  # prior day high-low
    vwap = (100.0 * 1 + 110.0 * 1 + 120.0 * 2) / 4.0
    last = out.tail(1).row(0, named=True)
    assert last["distance_to_vwap"] == pytest.approx((120.0 - vwap) / atr)


def test_range_width_uses_last_ten_bars_and_prior_atr() -> None:
    prior = [_ns(2025, 6, 1, 10, i) for i in range(2)]
    ts = [_ns(2025, 6, 2, 10, i) for i in range(10)]
    closes = [100.0] * 10
    highs = [100.0 + i for i in range(10)]
    lows = [90.0] * 10
    frame = _ohlc_frame(
        starts=[*prior, *ts],
        closes=[100.0, 100.0, *closes],
        highs=[110.0, 110.0, *highs],
        lows=[90.0, 90.0, *lows],
        story=[0, 0, *[1] * 10],
    )
    out = attach_momentum_features(frame)
    last = out.tail(1).row(0, named=True)
    atr = 20.0
    width = 109.0 - 90.0
    assert last["range_width_ratio"] == pytest.approx(width / atr)


def test_family_selection_keeps_momentum_beside_vp() -> None:
    n = 20
    cols: dict[str, object] = {AVAILABILITY_TS: list(range(n))}
    for name in (
        "vp_balance",
        "vp_fsm_break",
        "proj_poc_shift_ticks",
        "path_beyond_asia_ticks",
        "struct_dist_vah_ticks",
        "lf_arrival_intensity",
        "rel_credibility",
        "roc_10",
        "cvd_20",
        "distance_to_vwap",
        "range_width_ratio",
    ):
        cols[name] = [float(i + hash(name) % 5) for i in range(n)]
    names = select_feature_names_by_family(pl.DataFrame(cols), max_features=68)
    for col in MOMENTUM_FEATURE_COLUMNS:
        assert col in names
    assert "vp_balance" in names or "vp_fsm_break" in names
    assert "proj_poc_shift_ticks" in names


def test_extend_horizon_hits_five_points_inside_fifty_bars() -> None:
    n = 12
    close = 100.0
    highs = [close] * n
    highs[4] = close + EXTEND_HORIZON_POINTS
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "close": [close] * n,
            "high": highs,
            "low": [close] * n,
            "path_beyond_asia_ticks": [0.0, 2.0] + [2.0] * (n - 2),
            "vp_fsm_break": [0.0, 1.0] + [0.0] * (n - 2),
            "vp_fsm_retest": [0.0] * n,
            "proj_break_direction": [1.0] * n,
            "_behavior_story_run": [1] * n,
        }
    )
    labels = build_extend_horizon_outcomes(frame, window=8, group_col="_behavior_story_run")
    resolved = labels.filter(pl.col("label_status") == "resolved")
    assert resolved.height == 1
    assert float(resolved["y"][0]) == 1.0
    assert_availability_not_before_event(
        resolved[SETUP_AVAILABILITY_TS].to_numpy(),
        resolved[OUTCOME_AVAILABLE_TS].to_numpy(),
    )


def test_extend_horizon_incomplete_window_is_censored() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [0, 1, 2],
            "close": [100.0, 100.0, 100.0],
            "high": [100.0, 100.1, 100.2],
            "low": [99.0, 99.0, 99.0],
            "path_beyond_asia_ticks": [0.0, 2.0, 2.0],
            "vp_fsm_break": [0.0, 1.0, 0.0],
            "vp_fsm_retest": [0.0, 0.0, 0.0],
            "proj_break_direction": [1.0, 1.0, 1.0],
            "_behavior_story_run": [1, 1, 1],
        }
    )
    labels = build_extend_horizon_outcomes(frame, window=8, group_col="_behavior_story_run")
    assert labels.filter(pl.col("label_status") == "censored").height == 1
    assert labels.filter(pl.col("label_status") == "resolved").height == 0


def test_extend_horizon_uses_fixed_point_prices() -> None:
    tick = round(0.25 / PRICE_SCALE)
    close = round(20_000.0 / PRICE_SCALE)
    five = round(EXTEND_HORIZON_POINTS / PRICE_SCALE)
    n = 6
    highs = [close] * n
    highs[3] = close + five
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: list(range(n)),
            "close": [close] * n,
            "high": highs,
            "low": [close - tick] * n,
            "path_beyond_asia_ticks": [0.0, 2.0] + [2.0] * (n - 2),
            "vp_fsm_break": [0.0, 1.0] + [0.0] * (n - 2),
            "vp_fsm_retest": [0.0] * n,
            "proj_break_direction": [1.0] * n,
            "_behavior_story_run": [1] * n,
        }
    )
    labels = build_extend_horizon_outcomes(frame, window=5, group_col="_behavior_story_run")
    assert float(labels.filter(pl.col("label_status") == "resolved")["y"][0]) == 1.0


def test_science_targets_add_horizon_beside_path() -> None:
    names = science_outcome_targets(include_assumed_scripts=False)
    assert Y_EXTEND_5PTS_25MIN in names
    assert "y_path_further_beyond" in names
    assert names.count(Y_EXTEND_5PTS_25MIN) == 1
    assert EXTEND_HORIZON_BARS == 50
    assert EXTEND_HORIZON_POINTS == 5.0
