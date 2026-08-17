"""طبقة هـ: باسكت هيكل على OOF — وقف من قاع البراميل لا هدف/4."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.structure_basket import (
    PRICE_TICK,
    StructureBasketConfig,
    attach_structure_at_t,
    render_structure_basket_markdown,
    run_structure_basket,
    run_structure_basket_from_period_dir,
    run_structure_lookback_grid,
    write_structure_basket_report,
)

_GROUP = "_behavior_story_run"
_TICK = PRICE_TICK
_CLOSE0 = 20_000_000_000_000.0


def _month_ns(year: int, month: int, day: int = 10) -> int:
    stamp = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def _px(ticks_from_close0: float) -> float:
    return _CLOSE0 + float(ticks_from_close0) * _TICK


def _bars(
    *,
    story: int,
    start: int,
    close_ticks: list[float],
    high_ticks: list[float] | None = None,
    low_ticks: list[float] | None = None,
    asia_vah_ticks: float,
    decision_vah_ticks: list[float] | float,
    decision_poc_ticks: float = 0.0,
    inside: list[float] | None = None,
    break_dir: float = 1.0,
) -> pl.DataFrame:
    n = len(close_ticks)
    vah_dec = (
        [float(decision_vah_ticks)] * n
        if isinstance(decision_vah_ticks, (int, float))
        else list(decision_vah_ticks)
    )
    highs = high_ticks if high_ticks is not None else [c + 2.0 for c in close_ticks]
    lows = low_ticks if low_ticks is not None else [c - 2.0 for c in close_ticks]
    asia_vah = _px(asia_vah_ticks)
    close = [_px(t) for t in close_ticks]
    beyond = [(c - asia_vah) / _TICK for c in close]
    extreme: list[float] = []
    running = 0.0
    for value in beyond:
        running = max(running, value)
        extreme.append(running)
    return pl.DataFrame(
        {
            AVAILABILITY_TS: [start + i for i in range(n)],
            _GROUP: [story] * n,
            "close": close,
            "high": [_px(t) for t in highs],
            "low": [_px(t) for t in lows],
            "asia_vah": [asia_vah] * n,
            "asia_val": [_px(asia_vah_ticks - 40.0)] * n,
            "asia_poc": [_px(asia_vah_ticks - 20.0)] * n,
            "composite_vah": [0.0] * n,
            "composite_val": [0.0] * n,
            "composite_poc": [0.0] * n,
            "decision_vah": [_px(t) for t in vah_dec],
            "decision_val": [0.0] * n,
            "decision_poc": [_px(decision_poc_ticks)] * n,
            "proj_break_direction": [break_dir] * n,
            "path_beyond_asia_ticks": beyond,
            "path_extreme_ticks": extreme,
            "path_inside_asia_va": inside if inside is not None else [0.0] * n,
            "path_depth_follow": [2.0] * n,
            "path_depth_confirm": [2.0] * n,
            "vp_balance": [0.0] * n,
        }
    )


def _setup(
    *,
    story: int,
    ts: int,
    close_ticks: float,
    asia_vah_ticks: float,
    decision_vah_ticks: float,
    p: float = 0.8,
    y: float = 1.0,
    outcome_ts: int | None = None,
    break_dir: float = 1.0,
    decision_poc_ticks: float = 0.0,
    high_ticks: float | None = None,
    low_ticks: float | None = None,
) -> dict[str, float | int | str]:
    close = _px(close_ticks)
    asia_vah = _px(asia_vah_ticks)
    beyond = (close - asia_vah) / _TICK
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: ts + 1 if outcome_ts is None else outcome_ts,
        "outcome_name": "y_path_further_beyond",
        "y": y,
        "label_status": "resolved",
        _GROUP: story,
        "close": close,
        "high": _px(close_ticks + 2.0 if high_ticks is None else high_ticks),
        "low": _px(close_ticks - 2.0 if low_ticks is None else low_ticks),
        "asia_vah": asia_vah,
        "asia_val": _px(asia_vah_ticks - 40.0),
        "asia_poc": _px(asia_vah_ticks - 20.0),
        "decision_vah": _px(decision_vah_ticks),
        "decision_poc": _px(decision_poc_ticks),
        "proj_break_direction": break_dir,
        "path_beyond_asia_ticks": beyond,
        "path_extreme_ticks": beyond,
        "path_inside_asia_va": 0.0,
        "p_y_path_further_beyond": p,
        "path_depth_follow": 2.0,
        "path_depth_confirm": 2.0,
        "vp_balance": 0.0,
        "proj_outside_volume_share": 0.8,
        "wave_frac": 0.95,
        "ticks_remaining_to_peak": 4.0,
    }


def _cfg(**kwargs: float | int | None) -> StructureBasketConfig:
    base: dict[str, float | int | None] = {
        "holdout_months": None,
        "round_trip_cost_pts": 0.75,
        "max_hold_bars": 8,
        "min_ahead_ticks": 16.0,
        "lookback_bars": 8,
        "stop_buffer_ticks": 1.0,
        "swing_radius": 2,
        "min_stop_bars": 3,
    }
    base.update(kwargs)
    return StructureBasketConfig(
        holdout_months=None if base["holdout_months"] is None else int(base["holdout_months"]),
        round_trip_cost_pts=float(base["round_trip_cost_pts"] or 0.0),
        max_hold_bars=int(base["max_hold_bars"] or 8),
        min_ahead_ticks=float(base["min_ahead_ticks"] or 16.0),
        lookback_bars=int(base["lookback_bars"] or 8),
        stop_buffer_ticks=float(base["stop_buffer_ticks"] or 0.0),
        swing_radius=int(base["swing_radius"] or 2),
        min_stop_bars=int(base["min_stop_bars"] or 3),
    )


def _history_then(
    *,
    after_close: list[float],
    after_high: list[float],
    after_low: list[float],
    trough_ticks: float = -24.0,
    decision_vah_ticks: float = 80.0,
    asia_vah_ticks: float = -80.0,
    n_hist: int = 11,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """11 بار تاريخ + الدخول عند آخرها، ثم مسار ما بعد t."""
    hist_close = [0.0] * n_hist
    hist_high = [4.0] * n_hist
    hist_low = [-4.0] * n_hist
    hist_low[4] = float(trough_ticks)
    close_ticks = hist_close + after_close
    high_ticks = hist_high + after_high
    low_ticks = hist_low + after_low
    start = 10
    entry_ts = start + n_hist - 1
    blended = _bars(
        story=1,
        start=start,
        close_ticks=close_ticks,
        high_ticks=high_ticks,
        low_ticks=low_ticks,
        asia_vah_ticks=asia_vah_ticks,
        decision_vah_ticks=decision_vah_ticks,
    )
    labeled = pl.DataFrame(
        [
            _setup(
                story=1,
                ts=entry_ts,
                close_ticks=0.0,
                asia_vah_ticks=asia_vah_ticks,
                decision_vah_ticks=decision_vah_ticks,
                outcome_ts=entry_ts + len(after_close),
            )
        ]
    )
    return labeled, blended


def test_stop_is_lookback_low_not_target_over_four() -> None:
    labeled, blended = _history_then(
        after_close=[-8.0, -26.0],
        after_high=[-6.0, -20.0],
        after_low=[-10.0, -26.0],
        trough_ticks=-24.0,
        decision_vah_ticks=80.0,
    )
    report = run_structure_basket(labeled, blended, config=_cfg())
    assert report.diagnostics["not_fixed_1_to_4"] is True
    assert report.diagnostics["stop_is_structure_low_not_target_over_rr"] is True
    assert report.trades.height == 1
    row = report.trades.row(0, named=True)
    assert float(row["risk_ticks"]) == pytest.approx(25.0)
    assert float(row["target_ticks"]) == pytest.approx(80.0)
    assert float(row["risk_ticks"]) != pytest.approx(float(row["target_ticks"]) / 4.0)
    assert row["exit_reason"] == "stop"
    assert float(row["net_pts"]) == pytest.approx(-(25.0 / 4.0) - 0.75)


def test_take_fills_at_frozen_target() -> None:
    labeled, blended = _history_then(
        after_close=[40.0, 80.0],
        after_high=[42.0, 81.0],
        after_low=[38.0, 70.0],
        trough_ticks=-24.0,
        decision_vah_ticks=80.0,
    )
    report = run_structure_basket(labeled, blended, config=_cfg())
    row = report.trades.row(0, named=True)
    assert row["target_name"] == "decision_vah"
    assert row["exit_reason"] == "take"
    assert int(row["hold_bars"]) == 2
    assert float(row["net_pts"]) == pytest.approx((80.0 / 4.0) - 0.75)


def test_same_bar_stop_before_take() -> None:
    labeled, blended = _history_then(
        after_close=[0.0],
        after_high=[80.0],
        after_low=[-26.0],
        trough_ticks=-24.0,
        decision_vah_ticks=80.0,
    )
    report = run_structure_basket(labeled, blended, config=_cfg())
    assert report.trades["exit_reason"][0] == "stop"


def test_local_high_beats_farther_vah() -> None:
    n_hist = 11
    hist_close = [0.0] * n_hist
    hist_high = [4.0] * n_hist
    hist_low = [-4.0] * n_hist
    hist_low[4] = -24.0
    # Confirmed swing high at i=5 (radius 2): confirm at 7 <= entry 10.
    hist_high[5] = 40.0
    hist_high[3] = 6.0
    hist_high[4] = 8.0
    hist_high[6] = 10.0
    hist_high[7] = 12.0
    after_close = [20.0]
    after_high = [22.0]
    after_low = [18.0]
    blended = _bars(
        story=1,
        start=10,
        close_ticks=hist_close + after_close,
        high_ticks=hist_high + after_high,
        low_ticks=hist_low + after_low,
        asia_vah_ticks=-80.0,
        decision_vah_ticks=80.0,
    )
    labeled = pl.DataFrame(
        [
            _setup(
                story=1,
                ts=10 + n_hist - 1,
                close_ticks=0.0,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=80.0,
            )
        ]
    )
    geo = attach_structure_at_t(
        labeled,
        blended,
        lookback_bars=8,
        min_stop_bars=3,
        stop_buffer_ticks=1.0,
        swing_radius=2,
        min_ahead_ticks=16.0,
    )
    assert geo["target_name"][0] == "local_high"
    assert float(geo["target_ticks"][0]) == pytest.approx(40.0)


def test_later_vah_move_does_not_change_frozen_target() -> None:
    n_hist = 11
    close_ticks = [0.0] * n_hist + [40.0, 80.0]
    high_ticks = [4.0] * n_hist + [42.0, 81.0]
    low_ticks = [-4.0] * n_hist + [38.0, 70.0]
    low_ticks[4] = -24.0
    vah = [80.0] * n_hist + [200.0, 200.0]
    blended = _bars(
        story=1,
        start=10,
        close_ticks=close_ticks,
        high_ticks=high_ticks,
        low_ticks=low_ticks,
        asia_vah_ticks=-80.0,
        decision_vah_ticks=vah,
    )
    labeled = pl.DataFrame(
        [
            _setup(
                story=1,
                ts=10 + n_hist - 1,
                close_ticks=0.0,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=80.0,
            )
        ]
    )
    report = run_structure_basket(labeled, blended, config=_cfg())
    assert report.diagnostics["levels_frozen_at_t"] is True
    assert float(report.trades["target_ticks"][0]) == pytest.approx(80.0)
    assert report.trades["exit_reason"][0] == "take"


def test_skips_when_level_closer_than_min_ahead() -> None:
    labeled, blended = _history_then(
        after_close=[4.0],
        after_high=[6.0],
        after_low=[2.0],
        decision_vah_ticks=8.0,
    )
    report = run_structure_basket(labeled, blended, config=_cfg(min_ahead_ticks=16.0))
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "no_level_ahead"


def test_label_window_does_not_clip_hold() -> None:
    labeled, blended = _history_then(
        after_close=[20.0, 40.0, 80.0],
        after_high=[22.0, 42.0, 81.0],
        after_low=[18.0, 38.0, 70.0],
        decision_vah_ticks=80.0,
    )
    labeled = labeled.with_columns(
        pl.lit(int(labeled[SETUP_AVAILABILITY_TS][0]) + 1).alias(OUTCOME_AVAILABLE_TS)
    )
    report = run_structure_basket(labeled, blended, config=_cfg(max_hold_bars=4))
    assert int(report.trades["hold_bars"][0]) == 3
    assert report.trades["exit_reason"][0] == "take"


def test_y_shuffle_does_not_change_structure() -> None:
    labeled, blended = _history_then(
        after_close=[80.0],
        after_high=[81.0],
        after_low=[70.0],
    )
    a = run_structure_basket(labeled, blended, config=_cfg())
    b = run_structure_basket(labeled.with_columns(pl.col("y") * 0.0), blended, config=_cfg())
    assert a.trades["exit_reason"].to_list() == b.trades["exit_reason"].to_list()
    assert a.trades["net_pts"].to_list() == b.trades["net_pts"].to_list()


def test_peak_columns_are_ignored() -> None:
    labeled, blended = _history_then(
        after_close=[-8.0, -26.0],
        after_high=[-6.0, -20.0],
        after_low=[-10.0, -26.0],
    )
    blended = blended.with_columns(
        pl.lit(0.99).alias("wave_frac"),
        pl.lit(4.0).alias("ticks_remaining_to_peak"),
    )
    report = run_structure_basket(labeled, blended, config=_cfg())
    assert report.diagnostics["completed_wave_peak_not_used"] is True
    assert report.trades["exit_reason"][0] == "stop"


def test_holdout_excluded() -> None:
    rows_b: list[pl.DataFrame] = []
    rows_l: list[dict[str, float | int | str]] = []
    n_hist = 11
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        hist_close = [0.0] * n_hist
        hist_high = [4.0] * n_hist
        hist_low = [-4.0] * n_hist
        hist_low[4] = -24.0
        if month >= 9:
            after_c, after_h, after_l = [80.0], [81.0], [70.0]
        else:
            after_c, after_h, after_l = [-26.0], [-20.0], [-26.0]
        rows_b.append(
            _bars(
                story=month,
                start=ts,
                close_ticks=hist_close + after_c,
                high_ticks=hist_high + after_h,
                low_ticks=hist_low + after_l,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=80.0,
            )
        )
        rows_l.append(
            _setup(
                story=month,
                ts=ts + n_hist - 1,
                close_ticks=0.0,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=80.0,
                outcome_ts=ts + n_hist + 1,
            )
        )
    report = run_structure_basket(
        pl.DataFrame(rows_l),
        pl.concat(rows_b),
        config=_cfg(holdout_months=4, max_hold_bars=3),
    )
    assert report.diagnostics["holdout_scored"] is False
    reasons = report.diagnostics["exit_reasons"]
    assert isinstance(reasons, dict)
    assert "take" not in reasons
    assert reasons.get("stop", 0) >= 1


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame({AVAILABILITY_TS: [1], "order_id": [1], "action": ["A"], _GROUP: [1]})
    labeled = pl.DataFrame(
        [_setup(story=1, ts=1, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=80.0)]
    )
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_structure_basket(labeled, raw, config=_cfg())


def test_short_stop_is_lookback_high() -> None:
    n_hist = 11
    hist_close = [0.0] * n_hist
    hist_high = [4.0] * n_hist
    hist_low = [-4.0] * n_hist
    hist_high[4] = 24.0
    after_close = [8.0, 26.0]
    after_high = [10.0, 26.0]
    after_low = [6.0, 20.0]
    asia_val_ticks = 80.0
    asia_vah_ticks = 120.0
    close0 = _CLOSE0
    n = n_hist + 2
    closes = [_px(t) for t in hist_close + after_close]
    asia_val = close0 + asia_val_ticks * _TICK
    asia_vah = close0 + asia_vah_ticks * _TICK
    target = close0 - 80.0 * _TICK
    beyond = [(asia_val - c) / _TICK for c in closes]
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: [10 + i for i in range(n)],
            _GROUP: [1] * n,
            "close": closes,
            "high": [_px(t) for t in hist_high + after_high],
            "low": [_px(t) for t in hist_low + after_low],
            "asia_vah": [asia_vah] * n,
            "asia_val": [asia_val] * n,
            "asia_poc": [0.0] * n,
            "decision_val": [target] * n,
            "decision_vah": [0.0] * n,
            "decision_poc": [0.0] * n,
            "proj_break_direction": [-1.0] * n,
            "path_beyond_asia_ticks": beyond,
            "path_extreme_ticks": beyond,
            "path_inside_asia_va": [0.0] * n,
        }
    )
    labeled = pl.DataFrame(
        [
            {
                SETUP_AVAILABILITY_TS: 10 + n_hist - 1,
                OUTCOME_AVAILABLE_TS: 10 + n_hist + 1,
                "outcome_name": "y_path_further_beyond",
                "y": 1.0,
                "label_status": "resolved",
                _GROUP: 1,
                "close": close0,
                "asia_vah": asia_vah,
                "asia_val": asia_val,
                "decision_val": target,
                "proj_break_direction": -1.0,
                "path_beyond_asia_ticks": beyond[n_hist - 1],
                "path_extreme_ticks": beyond[n_hist - 1],
                "p_y_path_further_beyond": 0.8,
                "proj_outside_volume_share": 0.8,
            }
        ]
    )
    report = run_structure_basket(labeled, blended, config=_cfg())
    assert report.trades.height == 1
    assert float(report.trades["risk_ticks"][0]) == pytest.approx(25.0)
    assert report.trades["target_name"][0] == "decision_val"
    assert report.trades["exit_reason"][0] == "stop"


def test_lookback_grid_is_not_a_search() -> None:
    labeled, blended = _history_then(
        after_close=[-26.0],
        after_high=[-20.0],
        after_low=[-26.0],
    )
    grid = run_structure_lookback_grid(labeled, blended, lookbacks=(5, 8, 10), config=_cfg())
    assert grid["lookback_bars"].to_list() == [5, 8, 10]
    assert grid.height == 3


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    labeled, blended = _history_then(
        after_close=[80.0],
        after_high=[81.0],
        after_low=[70.0],
    )
    ts = int(labeled[SETUP_AVAILABILITY_TS][0])
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [ts], "p_y_path_further_beyond": [0.81]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    report = run_structure_basket_from_period_dir(period, config=_cfg())
    out = tmp_path / "out"
    write_structure_basket_report(report, out)
    text = (out / "STRUCTURE.md").read_text(encoding="utf-8")
    assert "not live execution" in text.lower()
    assert "1:4" in text or "target / 4" in text
    rendered = render_structure_basket_markdown(report)
    assert "Not** a chart" in rendered or "**Not** a chart" in rendered
    assert report.diagnostics["not_live_execution"] is True
