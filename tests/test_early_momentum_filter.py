"""طبقة ح: فلتر زخم/حجم/كسر سببي — بلا موقع موجة مكتملة كبوابة."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.research.early_momentum_filter import (
    EarlyMomentumConfig,
    pts_to_price,
    render_early_momentum_markdown,
    run_early_momentum,
    run_early_momentum_from_period_dir,
    write_early_momentum_report,
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


def _cfg(**kwargs: float | int | bool | None) -> EarlyMomentumConfig:
    base: dict[str, float | int | bool | None] = {
        "holdout_months": None,
        "round_trip_cost_pts": 0.75,
        "max_hold_bars": 40,
        "atr_days": 3,
        "momentum_bars": 5,
        "momentum_atr_frac": 0.15,
        "volume_bars": 3,
        "volume_multiple": 1.5,
        "break_pts": 2.0,
        "require_momentum": True,
        "require_volume": True,
        "require_break": True,
    }
    base.update(kwargs)
    return EarlyMomentumConfig(
        holdout_months=None if base["holdout_months"] is None else int(base["holdout_months"]),
        round_trip_cost_pts=float(base["round_trip_cost_pts"] or 0.0),
        max_hold_bars=int(base["max_hold_bars"] or 40),
        atr_days=int(base["atr_days"] or 3),
        momentum_bars=int(base["momentum_bars"] or 5),
        momentum_atr_frac=float(base["momentum_atr_frac"] or 0.15),
        volume_bars=int(base["volume_bars"] or 3),
        volume_multiple=float(base["volume_multiple"] or 1.5),
        break_pts=float(base["break_pts"] or 2.0),
        require_momentum=bool(base["require_momentum"]),
        require_volume=bool(base["require_volume"]),
        require_break=bool(base["require_break"]),
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
    intensity: float = 10.0,
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
        "lf_arrival_intensity": intensity,
        "asia_vah": _px(-20.0),
        "asia_val": _px(-40.0),
        "proj_break_direction": 1.0,
        "path_beyond_asia_ticks": beyond,
        "path_extreme_ticks": beyond,
        "path_inside_asia_va": 0.0,
        "vp_fsm_break": 1.0,
    }


def _history_days(*, london_range: float = 60.0, n_hist: int = 3) -> list[pl.DataFrame]:
    half = london_range / 2.0
    out: list[pl.DataFrame] = []
    for i in range(n_hist):
        date = f"2025-02-{10 + i:02d}"
        out.append(
            pl.DataFrame(
                [
                    _bar(
                        ts=_ts_et(i, 8, 0),
                        story=i + 1,
                        date=date,
                        session=_LONDON,
                        close_pts=0.0,
                        high_pts=half,
                        low_pts=-half,
                    )
                ]
            )
        )
    return out


def _setup(*, story: int, day: int, date: str, minute: int = 8) -> dict[str, float | int | str]:
    ts = _ts_et(day, 8, minute)
    return {
        SETUP_AVAILABILITY_TS: ts,
        OUTCOME_AVAILABLE_TS: _ts_et(day, 8, minute + 3),
        "outcome_name": "y_path_further_beyond",
        "y": 1.0,
        "label_status": "resolved",
        _GROUP: story,
        "_session_date": date,
        "close": _px(0.0),
        "high": _px(3.0),
        "low": _px(-1.0),
        "asia_vah": _px(-20.0),
        "asia_val": _px(-40.0),
        "proj_break_direction": 1.0,
        "path_beyond_asia_ticks": 80.0,
        "path_extreme_ticks": 80.0,
        "path_inside_asia_va": 0.0,
        "p_y_path_further_beyond": 0.8,
        "proj_outside_volume_share": 0.8,
        "wave_frac": 0.95,
        "ticks_remaining_to_peak": 4.0,
        "lf_arrival_intensity": 20.0,
    }


def _world(
    *,
    asia_high_pts: float = -5.0,
    warmup_close: float = -12.0,
    fire_intensity: float = 20.0,
    warmup_intensity: float = 10.0,
    after_high: float = 30.0,
    after_low: float = 10.0,
    n_warmup: int = 8,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    hist = _history_days()
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
            intensity=10.0,
            beyond=0.0,
        )
    ]
    for i in range(n_warmup):
        rows.append(
            _bar(
                ts=_ts_et(fire_day, 8, i),
                story=story,
                date=date,
                session=_LONDON,
                close_pts=warmup_close,
                high_pts=warmup_close + 1.0,
                low_pts=warmup_close - 1.0,
                intensity=warmup_intensity,
                beyond=16.0,
            )
        )
    fire_minute = n_warmup
    rows.append(
        _bar(
            ts=_ts_et(fire_day, 8, fire_minute),
            story=story,
            date=date,
            session=_LONDON,
            close_pts=0.0,
            high_pts=3.0,
            low_pts=-1.0,
            intensity=fire_intensity,
            beyond=80.0,
        )
    )
    rows.append(
        _bar(
            ts=_ts_et(fire_day, 8, fire_minute + 1),
            story=story,
            date=date,
            session=_LONDON,
            close_pts=(after_high + after_low) / 2.0,
            high_pts=after_high,
            low_pts=after_low,
            intensity=12.0,
            beyond=80.0 + after_high * 4.0,
        )
    )
    blended = pl.concat([*hist, pl.DataFrame(rows)])
    labeled = pl.DataFrame([_setup(story=story, day=fire_day, date=date, minute=fire_minute)])
    return labeled, blended


def test_all_three_gates_pass_and_take() -> None:
    labeled, blended = _world()
    report = run_early_momentum(labeled, blended, config=_cfg())
    assert report.trades.height == 1
    row = report.trades.row(0, named=True)
    assert float(row["momentum_pts"]) == pytest.approx(12.0)
    assert float(row["volume_ratio"]) == pytest.approx(2.0)
    assert float(row["break_pts"]) >= 2.0
    assert row["exit_reason"] == "take"
    assert report.diagnostics["wave_frac_not_used_as_entry_filter"] is True


def test_quiet_momentum_is_skipped() -> None:
    labeled, blended = _world(warmup_close=-2.0)
    report = run_early_momentum(labeled, blended, config=_cfg())
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "momentum_below_min"


def test_quiet_volume_is_skipped() -> None:
    labeled, blended = _world(fire_intensity=11.0)
    report = run_early_momentum(labeled, blended, config=_cfg())
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "volume_below_min"


def test_fake_break_is_skipped() -> None:
    labeled, blended = _world(asia_high_pts=10.0)
    report = run_early_momentum(labeled, blended, config=_cfg())
    assert report.trades.height == 0
    assert report.skipped["skip_reason"][0] == "break_below_min"


def test_wave_frac_is_not_an_entry_gate() -> None:
    labeled, blended = _world()
    blended = blended.with_columns(
        pl.lit(0.99).alias("wave_frac"),
        pl.lit(4.0).alias("ticks_remaining_to_peak"),
    )
    report = run_early_momentum(labeled, blended, config=_cfg())
    assert report.trades.height == 1
    assert report.diagnostics["completed_wave_peak_not_used_as_filter"] is True


def test_stop_hits_before_take() -> None:
    labeled, blended = _world(after_high=30.0, after_low=-12.0)
    report = run_early_momentum(labeled, blended, config=_cfg())
    assert report.trades["exit_reason"][0] == "stop"


def test_holdout_excluded() -> None:
    frames: list[pl.DataFrame] = []
    setups: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        stamp = dt.datetime(2025, month, 10, 8, 0, tzinfo=_ET)
        ts0 = int(stamp.timestamp() * 1_000_000_000)
        date = f"2025-{month:02d}-10"
        n = 12
        intensity = [10.0] * n
        intensity[-2] = 20.0
        close = [-12.0] * n
        close[-2] = 0.0
        hi = [-11.0] * n
        hi[-2] = 3.0
        lo = [-13.0] * n
        lo[-2] = -1.0
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
                    "close": [_px(c) for c in close],
                    "high": [_px(h) for h in hi],
                    "low": [_px(v) for v in lo],
                    "lf_arrival_intensity": intensity,
                    "asia_vah": [_px(-20.0)] * n,
                    "asia_val": [_px(-40.0)] * n,
                    "proj_break_direction": [1.0] * n,
                    "path_beyond_asia_ticks": [80.0] * n,
                    "path_extreme_ticks": [80.0] * n,
                    "path_inside_asia_va": [0.0] * n,
                    "vp_fsm_break": [1.0] * n,
                }
            )
        )
        row = _setup(story=month, day=0, date=date, minute=0)
        row[SETUP_AVAILABILITY_TS] = ts[-2]
        row[OUTCOME_AVAILABLE_TS] = ts[-1]
        row["close"] = _px(0.0)
        setups.append(row)
    report = run_early_momentum(
        pl.DataFrame(setups),
        pl.concat(frames),
        config=_cfg(holdout_months=4, atr_days=1, require_break=False),
    )
    assert report.diagnostics["holdout_scored"] is False
    reasons = report.diagnostics.get("exit_reasons") or {}
    assert "take" not in reasons


def test_y_shuffle_does_not_change_filter() -> None:
    labeled, blended = _world()
    a = run_early_momentum(labeled, blended, config=_cfg())
    b = run_early_momentum(labeled.with_columns(pl.col("y") * 0.0), blended, config=_cfg())
    assert a.trades["exit_reason"].to_list() == b.trades["exit_reason"].to_list()
    assert a.trades["net_pts"].to_list() == b.trades["net_pts"].to_list()


def test_refuses_raw_mbo() -> None:
    raw = pl.DataFrame({AVAILABILITY_TS: [1], "order_id": [1], "action": ["A"], _GROUP: [1]})
    labeled = pl.DataFrame([_setup(story=1, day=0, date="2025-02-10")])
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_early_momentum(labeled, raw, config=_cfg())


def test_period_dir_roundtrip(tmp_path: Path) -> None:
    labeled, blended = _world()
    ts = int(labeled[SETUP_AVAILABILITY_TS][0])
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [ts], "p_y_path_further_beyond": [0.81]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    report = run_early_momentum_from_period_dir(period, config=_cfg())
    out = tmp_path / "out"
    write_early_momentum_report(report, out)
    text = (out / "EARLY_MOMENTUM.md").read_text(encoding="utf-8")
    assert "wave_frac" in text
    assert report.diagnostics["wave_frac_not_used_as_entry_filter"] is True
    rendered = render_early_momentum_markdown(report)
    assert "**Not** a new science Y" in rendered
