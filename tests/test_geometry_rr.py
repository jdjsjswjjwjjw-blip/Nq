"""طبقة د: هندسة 1:4 إلى مستوى مجمّد عند t — بلا فريم شارت وبلا Y جديد."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.geometry_rr import (
    PRICE_TICK,
    GeometryRRConfig,
    attach_geometry_at_t,
    render_geometry_rr_markdown,
    run_geometry_rr,
    run_geometry_rr_from_period_dir,
    write_geometry_rr_report,
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
    asia_vah_ticks: float,
    decision_vah_ticks: list[float] | float,
    inside: list[float] | None = None,
) -> pl.DataFrame:
    n = len(close_ticks)
    vah_dec = (
        [float(decision_vah_ticks)] * n
        if isinstance(decision_vah_ticks, (int, float))
        else list(decision_vah_ticks)
    )
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
            "asia_vah": [asia_vah] * n,
            "asia_val": [_px(asia_vah_ticks - 40.0)] * n,
            "asia_poc": [_px(asia_vah_ticks - 20.0)] * n,
            "asia_primary_hvn": [0.0] * n,
            "composite_vah": [0.0] * n,
            "composite_val": [0.0] * n,
            "composite_poc": [0.0] * n,
            "composite_primary_hvn": [0.0] * n,
            "decision_vah": [_px(t) for t in vah_dec],
            "decision_val": [0.0] * n,
            "decision_poc": [0.0] * n,
            "proj_break_direction": [1.0] * n,
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
        "asia_vah": asia_vah,
        "asia_val": _px(asia_vah_ticks - 40.0),
        "asia_poc": _px(asia_vah_ticks - 20.0),
        "decision_vah": _px(decision_vah_ticks),
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


def _cfg(**kwargs: float | int | None) -> GeometryRRConfig:
    base: dict[str, float | int | None] = {
        "holdout_months": None,
        "round_trip_cost_pts": 0.75,
        "max_hold_bars": 8,
        "min_ahead_ticks": 16.0,
        "reward_multiple": 4.0,
    }
    base.update(kwargs)
    return GeometryRRConfig(
        holdout_months=None if base["holdout_months"] is None else int(base["holdout_months"]),
        round_trip_cost_pts=float(base["round_trip_cost_pts"] or 0.0),
        max_hold_bars=int(base["max_hold_bars"] or 8),
        min_ahead_ticks=float(base["min_ahead_ticks"] or 16.0),
        reward_multiple=float(base["reward_multiple"] or 4.0),
    )


def test_nearest_ahead_level_frozen_at_t() -> None:
    """الهدف قرار VAH عند t حتى لو ابتعد المركّب/القرار لاحقًا."""
    # close=0 ticks, asia_vah=-80, decision_vah=+32 at t then +200 later.
    blended = _bars(
        story=1,
        start=10,
        close_ticks=[0.0, 10.0, 32.0, 40.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=[32.0, 80.0, 200.0, 200.0],
    )
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=32.0)]
    )
    report = run_geometry_rr(labeled, blended, config=_cfg())
    assert report.diagnostics["chart_timeframe_unchanged"] is True
    assert report.diagnostics["does_not_modify_science_y"] is True
    assert report.diagnostics["levels_frozen_at_t"] is True
    assert report.trades.height == 1
    row = report.trades.row(0, named=True)
    assert row["target_name"] == "decision_vah"
    assert float(row["target_ticks"]) == pytest.approx(32.0)
    assert float(row["risk_ticks"]) == pytest.approx(8.0)
    assert row["exit_reason"] == "take_4r"
    assert int(row["hold_bars"]) == 2


def test_stop_1r_before_target() -> None:
    blended = _bars(
        story=1,
        start=10,
        close_ticks=[0.0, -8.0, -20.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=32.0,
    )
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=32.0)]
    )
    report = run_geometry_rr(labeled, blended, config=_cfg())
    assert report.trades["exit_reason"][0] == "stop_1r"
    assert int(report.trades["hold_bars"][0]) == 1


def test_asia_return_invalidates() -> None:
    blended = _bars(
        story=1,
        start=10,
        close_ticks=[0.0, -4.0, -4.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=32.0,
        inside=[0.0, 0.0, 1.0],
    )
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=32.0)]
    )
    report = run_geometry_rr(labeled, blended, config=_cfg())
    assert report.trades["exit_reason"][0] == "asia_return"


def test_skips_when_level_closer_than_min_ahead() -> None:
    blended = _bars(
        story=1,
        start=10,
        close_ticks=[0.0, 4.0, 8.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=8.0,
    )
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=8.0)]
    )
    report = run_geometry_rr(labeled, blended, config=_cfg(min_ahead_ticks=16.0))
    assert report.trades.height == 0
    assert report.skipped.height == 1
    assert report.skipped["skip_reason"][0] == "no_level_ahead"


def test_picks_nearest_of_two_ahead_levels() -> None:
    entries = pl.DataFrame(
        [_setup(story=1, ts=10, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=40.0)]
    ).with_columns(pl.lit(_px(24.0)).alias("composite_vah"))
    geo = attach_geometry_at_t(entries, min_ahead_ticks=16.0, reward_multiple=4.0)
    assert geo["target_name"][0] == "composite_vah"
    assert float(geo["target_ticks"][0]) == pytest.approx(24.0)


def test_label_window_does_not_clip_hold() -> None:
    blended = _bars(
        story=1,
        start=10,
        close_ticks=[0.0, 8.0, 16.0, 32.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=32.0,
    )
    labeled = pl.DataFrame(
        [
            _setup(
                story=1,
                ts=10,
                close_ticks=0.0,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=32.0,
                outcome_ts=11,
            )
        ]
    )
    report = run_geometry_rr(labeled, blended, config=_cfg(max_hold_bars=4))
    assert int(report.trades["hold_bars"][0]) == 3
    assert report.trades["exit_reason"][0] == "take_4r"


def test_y_shuffle_does_not_change_geometry() -> None:
    blended = _bars(
        story=1,
        start=10,
        close_ticks=[0.0, 32.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=32.0,
    )
    labeled = pl.DataFrame(
        [
            _setup(
                story=1,
                ts=10,
                close_ticks=0.0,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=32.0,
                y=1.0,
            )
        ]
    )
    a = run_geometry_rr(labeled, blended, config=_cfg())
    b = run_geometry_rr(labeled.with_columns(pl.col("y") * 0.0), blended, config=_cfg())
    assert a.trades["exit_reason"].to_list() == b.trades["exit_reason"].to_list()
    assert a.trades["net_pts"].to_list() == b.trades["net_pts"].to_list()


def test_peak_columns_are_ignored() -> None:
    blended = _bars(
        story=1,
        start=10,
        close_ticks=[0.0, -8.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=32.0,
    ).with_columns(
        pl.lit(0.99).alias("wave_frac"),
        pl.lit(4.0).alias("ticks_remaining_to_peak"),
    )
    labeled = pl.DataFrame(
        [_setup(story=1, ts=10, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=32.0)]
    )
    report = run_geometry_rr(labeled, blended, config=_cfg())
    assert report.diagnostics["completed_wave_peak_not_used"] is True
    assert report.trades["exit_reason"][0] == "stop_1r"


def test_holdout_excluded() -> None:
    rows_b: list[pl.DataFrame] = []
    rows_l: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        ts = _month_ns(2025, month)
        # Holdout would take profit immediately if scored.
        closes = [0.0, 32.0, 32.0] if month >= 9 else [0.0, -8.0, -8.0]
        rows_b.append(
            _bars(
                story=month,
                start=ts,
                close_ticks=closes,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=32.0,
            )
        )
        rows_l.append(
            _setup(
                story=month,
                ts=ts,
                close_ticks=0.0,
                asia_vah_ticks=-80.0,
                decision_vah_ticks=32.0,
                outcome_ts=ts + 2,
            )
        )
    report = run_geometry_rr(
        pl.DataFrame(rows_l),
        pl.concat(rows_b),
        config=_cfg(holdout_months=4, max_hold_bars=3),
    )
    assert report.diagnostics["holdout_scored"] is False
    reasons = report.diagnostics["exit_reasons"]
    assert isinstance(reasons, dict)
    assert "take_4r" not in reasons
    assert reasons.get("stop_1r", 0) >= 1


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame({AVAILABILITY_TS: [1], "order_id": [1], "action": ["A"], _GROUP: [1]})
    labeled = pl.DataFrame(
        [_setup(story=1, ts=1, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=32.0)]
    )
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_geometry_rr(labeled, raw, config=_cfg())


def test_short_take_4r() -> None:
    # close at 0, asia_val at +80 (below-VAL short), decision_val 32 ticks lower.
    close0 = _CLOSE0
    asia_val = close0 + 80.0 * _TICK
    asia_vah = asia_val + 40.0 * _TICK
    target = close0 - 32.0 * _TICK
    n = 3
    closes = [close0, close0 - 16.0 * _TICK, close0 - 32.0 * _TICK]
    beyond = [(asia_val - c) / _TICK for c in closes]
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: [10 + i for i in range(n)],
            _GROUP: [1] * n,
            "close": closes,
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
                SETUP_AVAILABILITY_TS: 10,
                OUTCOME_AVAILABLE_TS: 12,
                "outcome_name": "y_path_further_beyond",
                "y": 1.0,
                "label_status": "resolved",
                _GROUP: 1,
                "close": close0,
                "asia_vah": asia_vah,
                "asia_val": asia_val,
                "decision_val": target,
                "proj_break_direction": -1.0,
                "path_beyond_asia_ticks": beyond[0],
                "path_extreme_ticks": beyond[0],
                "p_y_path_further_beyond": 0.8,
                "proj_outside_volume_share": 0.8,
            }
        ]
    )
    report = run_geometry_rr(labeled, blended, config=_cfg())
    assert report.trades.height == 1
    assert report.trades["target_name"][0] == "decision_val"
    assert report.trades["exit_reason"][0] == "take_4r"


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    ts = _month_ns(2025, 5)
    blended = _bars(
        story=1,
        start=ts,
        close_ticks=[0.0, 32.0],
        asia_vah_ticks=-80.0,
        decision_vah_ticks=32.0,
    )
    labeled = pl.DataFrame(
        [_setup(story=1, ts=ts, close_ticks=0.0, asia_vah_ticks=-80.0, decision_vah_ticks=32.0)]
    )
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [ts], "p_y_path_further_beyond": [0.81]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    report = run_geometry_rr_from_period_dir(period, config=_cfg())
    out = tmp_path / "out"
    write_geometry_rr_report(report, out)
    text = (out / "GEOMETRY.md").read_text(encoding="utf-8")
    assert "timeframe" in text.lower()
    assert "frozen" in text.lower()
    rendered = render_geometry_rr_markdown(report)
    assert "Not** a chart" in rendered or "Not a chart" in rendered or "**Not** a chart" in rendered
    assert report.diagnostics["fvg_not_on_completed_states"] is True
