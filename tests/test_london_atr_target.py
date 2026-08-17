"""طبقة ز: ATR لندن سببي + هاي آسيا — بلا مدى CME الكامل وبلا مدى اليوم الجاري."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.research.london_atr_target import (
    LondonAtrConfig,
    attach_london_atr_at_t,
    causal_london_atr_pts,
    compute_london_ranges,
    pts_to_price,
    render_london_atr_markdown,
    run_london_atr,
    run_london_atr_from_period_dir,
    write_london_atr_report,
)

_GROUP = "_behavior_story_run"
_CLOSE0 = 20_000_000_000_000.0
_ASIA = int(VpLiquiditySession.ASIA)
_LONDON = int(VpLiquiditySession.LONDON)
_NY = int(VpLiquiditySession.NEW_YORK)
_ET = ZoneInfo("America/New_York")


def _px(pts_from_close0: float) -> float:
    return _CLOSE0 + pts_to_price(pts_from_close0)


def _ts_et(day: int, hour: int, minute: int, second: int = 0) -> int:
    base = dt.date(2025, 2, 10)
    d = base + dt.timedelta(days=day)
    stamp = dt.datetime(d.year, d.month, d.day, hour, minute, second, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def _cfg(**kwargs: float | int | bool | None) -> LondonAtrConfig:
    base: dict[str, float | int | bool | None] = {
        "holdout_months": None,
        "round_trip_cost_pts": 0.75,
        "max_hold_bars": 40,
        "atr_days": 3,
        "min_atr_pts": 30.0,
        "target_atr_frac": 0.5,
        "stop_atr_frac": 0.2,
        "min_rr": 2.0,
        "use_asia_extreme": True,
        "london_session_only": True,
    }
    base.update(kwargs)
    return LondonAtrConfig(
        holdout_months=None if base["holdout_months"] is None else int(base["holdout_months"]),
        round_trip_cost_pts=float(base["round_trip_cost_pts"] or 0.0),
        max_hold_bars=int(base["max_hold_bars"] or 40),
        atr_days=int(base["atr_days"] or 3),
        min_atr_pts=float(base["min_atr_pts"] or 30.0),
        target_atr_frac=float(base["target_atr_frac"] or 0.5),
        stop_atr_frac=float(base["stop_atr_frac"] or 0.2),
        min_rr=float(base["min_rr"] or 2.0),
        use_asia_extreme=bool(base["use_asia_extreme"]),
        london_session_only=bool(base["london_session_only"]),
    )


def _bar(
    *,
    ts: int,
    story: int,
    date: str,
    session: int,
    close_pts: float,
    high_pts: float,
    low_pts: float,
    beyond: float = 80.0,
) -> dict[str, float | int | str]:
    return {
        AVAILABILITY_TS: ts,
        _GROUP: story,
        VP_LIQUIDITY_SESSION: session,
        "_session_date": date,
        "close": _px(close_pts),
        "high": _px(high_pts),
        "low": _px(low_pts),
        "asia_vah": _px(-20.0),
        "asia_val": _px(-40.0),
        "proj_break_direction": 1.0,
        "path_beyond_asia_ticks": beyond,
        "path_extreme_ticks": beyond,
        "path_inside_asia_va": 0.0,
    }


def _history_days(*, london_range: float = 60.0, n_hist: int = 3) -> list[pl.DataFrame]:
    half = london_range / 2.0
    out: list[pl.DataFrame] = []
    for i in range(n_hist):
        date = f"2025-02-{10 + i:02d}"
        rows = [
            _bar(
                ts=_ts_et(i, 8, 0),
                story=i + 1,
                date=date,
                session=_LONDON,
                close_pts=0.0,
                high_pts=half,
                low_pts=-half,
            ),
            _bar(
                ts=_ts_et(i, 10, 0),
                story=i + 1,
                date=date,
                session=_NY,
                close_pts=0.0,
                high_pts=400.0,
                low_pts=-400.0,
            ),
        ]
        out.append(pl.DataFrame(rows))
    return out


def _setup(
    *,
    story: int,
    day: int,
    date: str,
    hour: int = 8,
    minute: int = 0,
    p: float = 0.8,
    y: float = 1.0,
) -> dict[str, float | int | str]:
    ts = _ts_et(day, hour, minute)
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: _ts_et(day, hour, minute + 3),
        "outcome_name": "y_path_further_beyond",
        "y": y,
        "label_status": "resolved",
        _GROUP: story,
        "_session_date": date,
        "close": _px(0.0),
        "high": _px(2.0),
        "low": _px(-2.0),
        "asia_vah": _px(-20.0),
        "asia_val": _px(-40.0),
        "proj_break_direction": 1.0,
        "path_beyond_asia_ticks": 80.0,
        "path_extreme_ticks": 80.0,
        "path_inside_asia_va": 0.0,
        "p_y_path_further_beyond": p,
        "proj_outside_volume_share": 0.8,
        "wave_frac": 0.95,
        "ticks_remaining_to_peak": 4.0,
    }


def _world(
    *,
    asia_high_pts: float,
    after: list[tuple[int, int, int, float, float]],
    fire_hour: int = 8,
    fire_minute: int = 0,
    today_early_london_high: float = 400.0,
    today_early_london_low: float = -400.0,
    london_range: float = 60.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    hist = _history_days(london_range=london_range)
    fire_day = 3
    date = "2025-02-13"
    story = 99
    rows = [
        _bar(
            ts=_ts_et(fire_day, 1, 0),
            story=story,
            date=date,
            session=_ASIA,
            close_pts=0.0,
            high_pts=asia_high_pts,
            low_pts=-30.0,
        ),
        _bar(
            ts=_ts_et(fire_day, 3, 15),
            story=story,
            date=date,
            session=_LONDON,
            close_pts=0.0,
            high_pts=today_early_london_high,
            low_pts=today_early_london_low,
        ),
        _bar(
            ts=_ts_et(fire_day, fire_hour, fire_minute),
            story=story,
            date=date,
            session=_LONDON,
            close_pts=0.0,
            high_pts=2.0,
            low_pts=-2.0,
        ),
    ]
    for hour, minute, session, hi, lo in after:
        rows.append(
            _bar(
                ts=_ts_et(fire_day, hour, minute),
                story=story,
                date=date,
                session=session,
                close_pts=(hi + lo) / 2.0,
                high_pts=hi,
                low_pts=lo,
                beyond=80.0 + hi,
            )
        )
    blended = pl.concat([*hist, pl.DataFrame(rows)])
    labeled = pl.DataFrame(
        [_setup(story=story, day=fire_day, date=date, hour=fire_hour, minute=fire_minute)]
    )
    return labeled, blended


def test_london_atr_ignores_ny_and_current_day() -> None:
    _labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 30.0, 20.0)],
        today_early_london_high=500.0,
        today_early_london_low=-500.0,
        london_range=60.0,
    )
    days = compute_london_ranges(blended)
    atr = causal_london_atr_pts(days, window=3)
    assert "2025-02-13" in atr
    assert atr["2025-02-13"] == pytest.approx(60.0)
    assert atr["2025-02-13"] < 100.0


def test_quiet_london_atr_is_skipped() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 30.0, 20.0)],
        london_range=60.0,
    )
    report = run_london_atr(labeled, blended, config=_cfg(min_atr_pts=200.0))
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "atr_below_min"


def test_ny_fire_is_skipped() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(10, 1, _NY, 30.0, 20.0)],
        fire_hour=10,
        fire_minute=0,
    )
    labeled = labeled.with_columns(
        pl.lit(_ts_et(3, 10, 0)).alias(SETUP_AVAILABILITY_TS),
        pl.lit(_px(0.0)).alias("close"),
    )
    report = run_london_atr(labeled, blended, config=_cfg())
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "not_london_session"


def test_asia_far_is_capped_at_half_london_atr() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 30.0, 20.0)],
        london_range=60.0,
    )
    report = run_london_atr(labeled, blended, config=_cfg())
    row = report.trades.row(0, named=True)
    assert row["target_name"] == "atr_extension"
    assert float(row["target_pts"]) == pytest.approx(30.0)
    assert float(row["risk_pts"]) == pytest.approx(12.0)
    assert float(row["rr_multiple"]) == pytest.approx(2.5)
    assert row["exit_reason"] == "take"


def test_asia_near_fails_min_rr_two() -> None:
    labeled, blended = _world(
        asia_high_pts=20.0,
        after=[(8, 1, _LONDON, 20.0, 10.0)],
        london_range=60.0,
    )
    report = run_london_atr(labeled, blended, config=_cfg(min_rr=2.0))
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "rr_below_min"


def test_asia_near_taken_when_min_rr_is_1_5() -> None:
    labeled, blended = _world(
        asia_high_pts=20.0,
        after=[(8, 1, _LONDON, 20.0, 10.0)],
        london_range=60.0,
    )
    report = run_london_atr(labeled, blended, config=_cfg(min_rr=1.5))
    assert report.trades.height == 1
    assert report.trades["target_name"][0] == "asia_session"
    assert float(report.trades["target_pts"][0]) == pytest.approx(20.0)
    assert float(report.trades["rr_multiple"][0]) == pytest.approx(20.0 / 12.0)


def test_stop_is_london_atr_fraction() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 8.0, -12.0)],
        london_range=60.0,
    )
    report = run_london_atr(labeled, blended, config=_cfg())
    row = report.trades.row(0, named=True)
    assert float(row["risk_pts"]) == pytest.approx(12.0)
    assert row["exit_reason"] == "stop"
    assert float(row["net_pts"]) == pytest.approx(-12.0 - 0.75)


def test_same_bar_stop_before_take() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 30.0, -12.0)],
        london_range=60.0,
    )
    report = run_london_atr(labeled, blended, config=_cfg())
    assert report.trades["exit_reason"][0] == "stop"


def test_hold_exits_at_london_end_not_ny_take() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[
            (9, 20, _LONDON, 2.0, -1.0),
            (9, 25, _LONDON, 3.0, -1.0),
            (9, 35, _NY, 40.0, 30.0),
        ],
        fire_hour=9,
        fire_minute=15,
        london_range=60.0,
    )
    report = run_london_atr(labeled, blended, config=_cfg())
    assert report.trades.height == 1
    assert report.trades["exit_reason"][0] == "london_end"
    assert float(report.trades["target_pts"][0]) == pytest.approx(30.0)


def test_holdout_excluded() -> None:
    frames: list[pl.DataFrame] = []
    setups: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        stamp = dt.datetime(2025, month, 10, 8, 0, tzinfo=_ET)
        ts0 = int(stamp.timestamp() * 1_000_000_000)
        date = f"2025-{month:02d}-10"
        n = 6
        hi = [30.0] * n
        lo = [-30.0] * n
        if month >= 9:
            hi[-1] = 30.0
            lo[-1] = 20.0
        else:
            hi[-1] = 8.0
            lo[-1] = -12.0
        ts = [ts0 + i * 60_000_000_000 for i in range(n)]
        frames.append(
            pl.DataFrame(
                {
                    AVAILABILITY_TS: ts,
                    _GROUP: [month] * n,
                    VP_LIQUIDITY_SESSION: [_LONDON] * n,
                    "_session_date": [date] * n,
                    "close": [_px(0.0)] * n,
                    "high": [_px(h) for h in hi],
                    "low": [_px(v) for v in lo],
                    "asia_vah": [_px(-20.0)] * n,
                    "asia_val": [_px(-40.0)] * n,
                    "proj_break_direction": [1.0] * n,
                    "path_beyond_asia_ticks": [80.0] * n,
                    "path_extreme_ticks": [80.0] * n,
                    "path_inside_asia_va": [0.0] * n,
                }
            )
        )
        row = _setup(story=month, day=0, date=date)
        row[SETUP_AVAILABILITY_TS] = ts0
        row[OUTCOME_AVAILABLE_TS] = ts0 + 5
        setups.append(row)
    report = run_london_atr(
        pl.DataFrame(setups),
        pl.concat(frames),
        config=_cfg(holdout_months=4, atr_days=1, min_atr_pts=1.0, use_asia_extreme=False),
    )
    assert report.diagnostics["holdout_scored"] is False
    reasons = report.diagnostics.get("exit_reasons") or {}
    assert "take" not in reasons


def test_peak_columns_ignored() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 8.0, -12.0)],
    )
    blended = blended.with_columns(
        pl.lit(0.99).alias("wave_frac"),
        pl.lit(4.0).alias("ticks_remaining_to_peak"),
    )
    report = run_london_atr(labeled, blended, config=_cfg())
    assert report.diagnostics["completed_wave_peak_not_used"] is True
    assert report.trades["exit_reason"][0] == "stop"


def test_y_shuffle_does_not_change_basket() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 30.0, 10.0)],
    )
    a = run_london_atr(labeled, blended, config=_cfg())
    b = run_london_atr(labeled.with_columns(pl.col("y") * 0.0), blended, config=_cfg())
    assert a.trades["exit_reason"].to_list() == b.trades["exit_reason"].to_list()
    assert a.trades["net_pts"].to_list() == b.trades["net_pts"].to_list()


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame({AVAILABILITY_TS: [1], "order_id": [1], "action": ["A"], _GROUP: [1]})
    labeled = pl.DataFrame([_setup(story=1, day=0, date="2025-02-10")])
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_london_atr(labeled, raw, config=_cfg())


def test_without_asia_always_uses_extension() -> None:
    labeled, blended = _world(
        asia_high_pts=24.0,
        after=[(8, 1, _LONDON, 30.0, 10.0)],
    )
    report = run_london_atr(labeled, blended, config=_cfg(use_asia_extreme=False, min_rr=2.0))
    assert report.trades["target_name"][0] == "atr_extension"
    assert float(report.trades["target_pts"][0]) == pytest.approx(30.0)


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 30.0, 10.0)],
    )
    ts = int(labeled[SETUP_AVAILABILITY_TS][0])
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [ts], "p_y_path_further_beyond": [0.81]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    report = run_london_atr_from_period_dir(period, config=_cfg())
    out = tmp_path / "out"
    write_london_atr_report(report, out)
    text = (out / "LONDON_ATR.md").read_text(encoding="utf-8")
    assert "london-session" in text.lower() or "London" in text
    assert report.diagnostics["current_day_london_range_not_in_atr"] is True
    rendered = render_london_atr_markdown(report)
    assert "**Not** a chart" in rendered


def test_attach_requires_prior_london_days() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after=[(8, 1, _LONDON, 30.0, 10.0)],
    )
    days = compute_london_ranges(blended)
    atr = causal_london_atr_pts(days, window=3)
    geo = attach_london_atr_at_t(
        labeled,
        blended,
        atr_by_date=atr,
        asia_ext=pl.DataFrame(
            {
                "_behavior_story_run": [99],
                "asia_session_high": [_px(80.0)],
                "asia_session_low": [_px(-30.0)],
            }
        ),
        config=_cfg(),
    )
    assert bool(geo["vol_ok"][0]) is True
    assert float(geo["london_atr_pts"][0]) == pytest.approx(60.0)
