"""طبقة و: ATR يومي سببي + هاي آسيا — بلا 1:4 وبلا مدى اليوم الجاري."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.research.volatility_adjusted_target import (
    VolatilityTargetConfig,
    attach_volatility_at_t,
    causal_atr_pts,
    compute_daily_ranges,
    pts_to_price,
    render_volatility_target_markdown,
    run_volatility_target,
    run_volatility_target_from_period_dir,
    write_volatility_target_report,
)

_GROUP = "_behavior_story_run"
_CLOSE0 = 20_000_000_000_000.0
_ASIA = int(VpLiquiditySession.ASIA)
_LONDON = int(VpLiquiditySession.LONDON)


def _px(pts_from_close0: float) -> float:
    return _CLOSE0 + pts_to_price(pts_from_close0)


def _ts(day: int, bar: int) -> int:
    stamp = dt.datetime(2025, 3, 1, 14, 0, tzinfo=dt.UTC) + dt.timedelta(days=day, seconds=bar)
    return int(stamp.timestamp() * 1_000_000_000)


def _cfg(**kwargs: float | int | bool | None) -> VolatilityTargetConfig:
    base: dict[str, float | int | bool | None] = {
        "holdout_months": None,
        "round_trip_cost_pts": 0.75,
        "max_hold_bars": 8,
        "atr_days": 3,
        "min_atr_pts": 60.0,
        "target_atr_frac": 0.4,
        "stop_atr_frac": 0.2,
        "min_rr": 2.0,
        "use_asia_extreme": True,
    }
    base.update(kwargs)
    return VolatilityTargetConfig(
        holdout_months=None if base["holdout_months"] is None else int(base["holdout_months"]),
        round_trip_cost_pts=float(base["round_trip_cost_pts"] or 0.0),
        max_hold_bars=int(base["max_hold_bars"] or 8),
        atr_days=int(base["atr_days"] or 3),
        min_atr_pts=float(base["min_atr_pts"] or 60.0),
        target_atr_frac=float(base["target_atr_frac"] or 0.4),
        stop_atr_frac=float(base["stop_atr_frac"] or 0.2),
        min_rr=float(base["min_rr"] or 2.0),
        use_asia_extreme=bool(base["use_asia_extreme"]),
    )


def _day(
    *,
    story: int,
    day: int,
    date: str,
    high_pts: float,
    low_pts: float,
    close_pts: float = 0.0,
    n: int = 4,
    session: int = _LONDON,
    asia_high_pts: float | None = None,
    asia_low_pts: float | None = None,
    beyond: float = 80.0,
) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if asia_high_pts is not None:
        rows.append(
            {
                AVAILABILITY_TS: _ts(day, 0),
                _GROUP: story,
                VP_LIQUIDITY_SESSION: _ASIA,
                "_session_date": date,
                "close": _px(0.0),
                "high": _px(asia_high_pts),
                "low": _px(asia_low_pts if asia_low_pts is not None else -10.0),
                "asia_vah": _px(-20.0),
                "asia_val": _px(-40.0),
                "proj_break_direction": 1.0,
                "path_beyond_asia_ticks": beyond,
                "path_extreme_ticks": beyond,
                "path_inside_asia_va": 0.0,
            }
        )
    for i in range(n):
        rows.append(
            {
                AVAILABILITY_TS: _ts(day, 10 + i),
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
        )
    return pl.DataFrame(rows)


def _setup(
    *,
    story: int,
    day: int,
    date: str,
    p: float = 0.8,
    y: float = 1.0,
    bar: int = 10,
    outcome_bar: int = 13,
) -> dict[str, float | int | str]:
    ts = _ts(day, bar)
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: _ts(day, outcome_bar),
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


def _history_days(*, atr_pts: float = 80.0, n_hist: int = 3) -> list[pl.DataFrame]:
    half = atr_pts / 2.0
    out: list[pl.DataFrame] = []
    for i in range(n_hist):
        out.append(
            _day(
                story=i + 1,
                day=i,
                date=f"2025-02-{10 + i:02d}",
                high_pts=half,
                low_pts=-half,
            )
        )
    return out


def _world(
    *,
    asia_high_pts: float,
    after_high: list[float],
    after_low: list[float],
    fire_day_high: float = 400.0,
    fire_day_low: float = -400.0,
    atr_pts: float = 80.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    hist = _history_days(atr_pts=atr_pts)
    fire_day = 3
    date = "2025-02-13"
    story = 99
    asia = _day(
        story=story,
        day=fire_day,
        date=date,
        high_pts=fire_day_high,
        low_pts=fire_day_low,
        n=1,
        asia_high_pts=asia_high_pts,
        asia_low_pts=-30.0,
    )
    after_rows: list[dict[str, float | int | str]] = []
    for i, (hi, lo) in enumerate(zip(after_high, after_low, strict=True)):
        after_rows.append(
            {
                AVAILABILITY_TS: _ts(fire_day, 11 + i),
                _GROUP: story,
                VP_LIQUIDITY_SESSION: _LONDON,
                "_session_date": date,
                "close": _px((hi + lo) / 2.0),
                "high": _px(hi),
                "low": _px(lo),
                "asia_vah": _px(-20.0),
                "asia_val": _px(-40.0),
                "proj_break_direction": 1.0,
                "path_beyond_asia_ticks": 80.0 + hi,
                "path_extreme_ticks": 80.0 + hi,
                "path_inside_asia_va": 0.0,
            }
        )
    fire_london = asia.tail(1).with_columns(
        pl.lit(_ts(fire_day, 10)).alias(AVAILABILITY_TS),
        pl.lit(_px(0.0)).alias("close"),
        pl.lit(_px(2.0)).alias("high"),
        pl.lit(_px(-2.0)).alias("low"),
        pl.lit(_LONDON, dtype=pl.Int64).alias(VP_LIQUIDITY_SESSION),
    )
    blended = pl.concat([*hist, asia.head(1), fire_london, pl.DataFrame(after_rows)])
    labeled = pl.DataFrame([_setup(story=story, day=fire_day, date=date)])
    return labeled, blended


def test_atr_ignores_current_day_range() -> None:
    _labeled, blended = _world(
        asia_high_pts=50.0,
        after_high=[32.0],
        after_low=[30.0],
        fire_day_high=500.0,
        fire_day_low=-500.0,
        atr_pts=80.0,
    )
    daily = compute_daily_ranges(blended)
    atr = causal_atr_pts(daily, window=3)
    assert "2025-02-13" in atr
    assert atr["2025-02-13"] == pytest.approx(80.0)
    assert atr["2025-02-13"] < 200.0


def test_quiet_atr_is_skipped() -> None:
    labeled, blended = _world(
        asia_high_pts=50.0,
        after_high=[32.0],
        after_low=[30.0],
        atr_pts=80.0,
    )
    report = run_volatility_target(labeled, blended, config=_cfg(min_atr_pts=200.0))
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "atr_below_min"


def test_asia_far_is_capped_at_atr_extension() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after_high=[32.0],
        after_low=[20.0],
        atr_pts=80.0,
    )
    report = run_volatility_target(labeled, blended, config=_cfg())
    row = report.trades.row(0, named=True)
    assert row["target_name"] == "atr_extension"
    assert float(row["target_pts"]) == pytest.approx(32.0)
    assert float(row["risk_pts"]) == pytest.approx(16.0)
    assert float(row["rr_multiple"]) == pytest.approx(2.0)
    assert row["exit_reason"] == "take"


def test_asia_near_fails_min_rr_two() -> None:
    labeled, blended = _world(
        asia_high_pts=20.0,
        after_high=[20.0],
        after_low=[10.0],
        atr_pts=80.0,
    )
    report = run_volatility_target(labeled, blended, config=_cfg(min_rr=2.0))
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "rr_below_min"


def test_asia_near_taken_when_min_rr_is_1_5() -> None:
    labeled, blended = _world(
        asia_high_pts=24.0,
        after_high=[24.0],
        after_low=[10.0],
        atr_pts=80.0,
    )
    report = run_volatility_target(labeled, blended, config=_cfg(min_rr=1.5))
    assert report.trades.height == 1
    assert report.trades["target_name"][0] == "asia_session"
    assert float(report.trades["target_pts"][0]) == pytest.approx(24.0)
    assert float(report.trades["rr_multiple"][0]) == pytest.approx(1.5)


def test_asia_behind_uses_atr_extension() -> None:
    labeled, blended = _world(
        asia_high_pts=-10.0,
        after_high=[32.0],
        after_low=[10.0],
        atr_pts=80.0,
    )
    report = run_volatility_target(labeled, blended, config=_cfg())
    assert report.trades["target_name"][0] == "atr_extension"
    assert float(report.trades["target_pts"][0]) == pytest.approx(32.0)


def test_stop_is_atr_fraction_not_target_over_four() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after_high=[8.0],
        after_low=[-16.0],
        atr_pts=80.0,
    )
    report = run_volatility_target(labeled, blended, config=_cfg())
    row = report.trades.row(0, named=True)
    assert float(row["risk_pts"]) == pytest.approx(16.0)
    assert float(row["risk_pts"]) != pytest.approx(float(row["target_pts"]) / 4.0)
    assert row["exit_reason"] == "stop"
    assert float(row["net_pts"]) == pytest.approx(-16.0 - 0.75)


def test_same_bar_stop_before_take() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after_high=[32.0],
        after_low=[-16.0],
        atr_pts=80.0,
    )
    report = run_volatility_target(labeled, blended, config=_cfg())
    assert report.trades["exit_reason"][0] == "stop"


def test_holdout_excluded() -> None:
    frames: list[pl.DataFrame] = []
    setups: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        stamp = dt.datetime(2025, month, 10, 14, 0, tzinfo=dt.UTC)
        ts0 = int(stamp.timestamp() * 1_000_000_000)
        date = f"2025-{month:02d}-10"
        n = 6
        hi = [40.0] * n
        lo = [-40.0] * n
        if month >= 9:
            hi[-1] = 32.0
            lo[-1] = 20.0
        else:
            hi[-1] = 8.0
            lo[-1] = -16.0
        frames.append(
            pl.DataFrame(
                {
                    AVAILABILITY_TS: [ts0 + i for i in range(n)],
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
        row = _setup(story=month, day=0, date=date, bar=0, outcome_bar=5)
        row[SETUP_AVAILABILITY_TS] = ts0
        row[OUTCOME_AVAILABLE_TS] = ts0 + 5
        setups.append(row)
    report = run_volatility_target(
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
        after_high=[8.0],
        after_low=[-16.0],
    )
    blended = blended.with_columns(
        pl.lit(0.99).alias("wave_frac"),
        pl.lit(4.0).alias("ticks_remaining_to_peak"),
    )
    report = run_volatility_target(labeled, blended, config=_cfg())
    assert report.diagnostics["completed_wave_peak_not_used"] is True
    assert report.trades["exit_reason"][0] == "stop"


def test_y_shuffle_does_not_change_basket() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after_high=[32.0],
        after_low=[10.0],
    )
    a = run_volatility_target(labeled, blended, config=_cfg())
    b = run_volatility_target(labeled.with_columns(pl.col("y") * 0.0), blended, config=_cfg())
    assert a.trades["exit_reason"].to_list() == b.trades["exit_reason"].to_list()
    assert a.trades["net_pts"].to_list() == b.trades["net_pts"].to_list()


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame({AVAILABILITY_TS: [1], "order_id": [1], "action": ["A"], _GROUP: [1]})
    labeled = pl.DataFrame([_setup(story=1, day=0, date="2025-02-13")])
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_volatility_target(labeled, raw, config=_cfg())


def test_without_asia_always_uses_extension() -> None:
    labeled, blended = _world(
        asia_high_pts=24.0,
        after_high=[32.0],
        after_low=[10.0],
    )
    report = run_volatility_target(
        labeled, blended, config=_cfg(use_asia_extreme=False, min_rr=2.0)
    )
    assert report.trades["target_name"][0] == "atr_extension"
    assert float(report.trades["target_pts"][0]) == pytest.approx(32.0)


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after_high=[32.0],
        after_low=[10.0],
    )
    ts = int(labeled[SETUP_AVAILABILITY_TS][0])
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [ts], "p_y_path_further_beyond": [0.81]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    report = run_volatility_target_from_period_dir(period, config=_cfg())
    out = tmp_path / "out"
    write_volatility_target_report(report, out)
    text = (out / "VOLATILITY.md").read_text(encoding="utf-8")
    assert "high-vol" in text.lower()
    assert report.diagnostics["current_day_range_not_in_atr"] is True
    rendered = render_volatility_target_markdown(report)
    assert "Not** a chart" in rendered or "**Not** a chart" in rendered


def test_attach_requires_prior_atr_days() -> None:
    labeled, blended = _world(
        asia_high_pts=80.0,
        after_high=[32.0],
        after_low=[10.0],
    )
    daily = compute_daily_ranges(blended)
    atr = causal_atr_pts(daily, window=3)
    geo = attach_volatility_at_t(
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
    assert float(geo["daily_atr_pts"][0]) == pytest.approx(80.0)
