"""أداة عين: نوافذ 30ث حول إطلاق النموذج — بلا Y جديد وبلا holdout."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from nq.auction_behavior.outcomes import OUTCOME_AVAILABLE_TS, SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.entry_replay import (
    DEFAULT_LOOKAHEAD_NS,
    DEFAULT_LOOKBACK_NS,
    EntryReplayConfig,
    extract_entry_windows,
    limit_replay,
    render_entry_replay_gallery_html,
    render_entry_replay_html,
    run_entry_replay,
    run_entry_replay_from_period_dir,
    to_nq_price,
    write_entry_replay_report,
)

_GROUP = "_behavior_story_run"
_BAR_NS = 30 * 1_000_000_000


def _month_ns(year: int, month: int, day: int = 10) -> int:
    stamp = dt.datetime(year, month, day, 8, 0, tzinfo=dt.UTC)
    return int(stamp.timestamp() * 1_000_000_000)


def _story(*, story: int, t0: int, n_before: int = 20, n_after: int = 30) -> pl.DataFrame:
    starts = t0 - n_before * _BAR_NS
    n = n_before + 1 + n_after
    ts = [starts + i * _BAR_NS for i in range(n)]
    close = [20_000.0 + 0.25 * (i - n_before) for i in range(n)]
    beyond = [80.0 + float(i - n_before) for i in range(n)]
    return pl.DataFrame(
        {
            AVAILABILITY_TS: ts,
            _GROUP: [story] * n,
            "close": close,
            "asia_vah": [19_980.0] * n,
            "asia_val": [19_960.0] * n,
            "decision_vah": [20_010.0] * n,
            "decision_val": [19_990.0] * n,
            "path_beyond_asia_ticks": beyond,
            "path_extreme_ticks": beyond,
            "path_inside_asia_va": [0.0] * n,
            "path_depth_confirm": [0.8] * n,
            "proj_break_direction": [1.0] * n,
        }
    )


def _setup(*, story: int, t0: int, p: float = 0.8, y: float = 1.0) -> dict[str, float | int | str]:
    return {
        SETUP_AVAILABILITY_TS: t0,
        OUTCOME_AVAILABLE_TS: t0 + 1,
        "outcome_name": "y_path_further_beyond",
        "y": y,
        "label_status": "resolved",
        _GROUP: story,
        "close": 20_000.0,
        "asia_vah": 19_980.0,
        "asia_val": 19_960.0,
        "decision_vah": 20_010.0,
        "path_beyond_asia_ticks": 80.0,
        "path_extreme_ticks": 80.0,
        "p_y_path_further_beyond": p,
        "proj_break_direction": 1.0,
        "proj_outside_volume_share": 0.8,
        "path_depth_confirm": 0.8,
    }


def test_window_is_10_minutes_before_and_15_after() -> None:
    t0 = 1_000_000_000_000_000_000
    blended = _story(story=1, t0=t0)
    entries = pl.DataFrame([_setup(story=1, t0=t0)])
    trades, bars = extract_entry_windows(
        entries,
        blended,
        lookback_ns=DEFAULT_LOOKBACK_NS,
        lookahead_ns=DEFAULT_LOOKAHEAD_NS,
    )
    assert trades.height == 1
    assert int(trades["n_bars_before"][0]) == 20
    assert int(trades["n_bars_after"][0]) == 30
    mins = bars["minutes_from_entry"].to_list()
    assert min(mins) == pytest.approx(-10.0)
    assert max(mins) == pytest.approx(15.0)
    assert bars.filter(pl.col("is_entry_bar")).height == 1


def test_label_window_does_not_clip_replay() -> None:
    t0 = 1_000_000_000_000_000_000
    blended = _story(story=1, t0=t0)
    labeled = pl.DataFrame([_setup(story=1, t0=t0)])
    report = run_entry_replay(labeled, blended, config=EntryReplayConfig(holdout_months=None))
    assert report.diagnostics["chart_timeframe_unchanged"] is True
    assert report.diagnostics["post_entry_path_is_inspection_only"] is True
    assert int(report.trades["n_bars_after"][0]) == 30


def test_y_shuffle_does_not_change_window() -> None:
    t0 = 1_000_000_000_000_000_000
    blended = _story(story=1, t0=t0)
    labeled = pl.DataFrame([_setup(story=1, t0=t0, y=1.0)])
    a = run_entry_replay(labeled, blended, config=EntryReplayConfig(holdout_months=None))
    b = run_entry_replay(
        labeled.with_columns(pl.col("y") * 0.0),
        blended,
        config=EntryReplayConfig(holdout_months=None),
    )
    assert a.bars["minutes_from_entry"].to_list() == b.bars["minutes_from_entry"].to_list()
    assert a.bars["close_pts"].to_list() == b.bars["close_pts"].to_list()


def test_holdout_excluded() -> None:
    rows_b: list[pl.DataFrame] = []
    rows_l: list[dict[str, float | int | str]] = []
    for month in range(1, 13):
        t0 = _month_ns(2025, month)
        rows_b.append(_story(story=month, t0=t0, n_before=2, n_after=2))
        rows_l.append(_setup(story=month, t0=t0))
    report = run_entry_replay(
        pl.DataFrame(rows_l),
        pl.concat(rows_b),
        config=EntryReplayConfig(
            holdout_months=4, lookback_ns=2 * _BAR_NS, lookahead_ns=2 * _BAR_NS
        ),
    )
    assert report.diagnostics["holdout_scored"] is False
    months = [
        dt.datetime.fromtimestamp(int(t) / 1e9, tz=dt.UTC).month
        for t in report.trades[SETUP_AVAILABILITY_TS].to_list()
    ]
    assert months
    assert max(months) <= 8


def test_refuses_raw_mbo() -> None:
    t0 = 10
    raw = pl.DataFrame({AVAILABILITY_TS: [t0], "order_id": [1], "action": ["A"], _GROUP: [1]})
    labeled = pl.DataFrame([_setup(story=1, t0=t0)])
    with pytest.raises(ValueError, match="refuses raw MBO"):
        run_entry_replay(labeled, raw, config=EntryReplayConfig(holdout_months=None))


def test_to_nq_price_scaled_and_plain() -> None:
    assert to_nq_price(20_000.0) == pytest.approx(20_000.0)
    assert to_nq_price(20_000_000_000_000.0) == pytest.approx(20_000.0)


def test_html_roundtrip(tmp_path: Path) -> None:
    t0 = _month_ns(2025, 5)
    blended = _story(story=1, t0=t0, n_before=4, n_after=6)
    labeled = pl.DataFrame([_setup(story=1, t0=t0)])
    period = tmp_path / "period"
    period.mkdir()
    labeled.write_parquet(period / "science_labeled.parquet")
    blended.write_parquet(period / "period_blended.parquet")
    pl.DataFrame({AVAILABILITY_TS: [t0], "p_y_path_further_beyond": [0.81]}).write_parquet(
        period / "oof_predictions.parquet"
    )
    report = run_entry_replay_from_period_dir(
        period,
        config=EntryReplayConfig(
            holdout_months=None, lookback_ns=4 * _BAR_NS, lookahead_ns=6 * _BAR_NS
        ),
    )
    out = tmp_path / "out"
    write_entry_replay_report(report, out)
    html = (out / "ENTRY_REPLAY.html").read_text(encoding="utf-8")
    assert "30 ثانية" in html or "30s" in html.lower() or "30" in html
    assert report.trades.height == 1
    rendered = render_entry_replay_html(report)
    assert "entry_utc" in rendered
    assert "inspection" in (out / "ENTRY_REPLAY.md").read_text(encoding="utf-8")
    gallery = (out / "ENTRY_REPLAY_GALLERY.html").read_text(encoding="utf-8")
    assert "<svg" in gallery


def test_limit_replay_keeps_first_n() -> None:
    t0 = 1_000_000_000_000_000_000
    blended = pl.concat(
        [
            _story(story=1, t0=t0, n_before=2, n_after=2),
            _story(story=2, t0=t0 + 3_600_000_000_000, n_before=2, n_after=2),
        ]
    )
    labeled = pl.DataFrame(
        [
            _setup(story=1, t0=t0),
            _setup(story=2, t0=t0 + 3_600_000_000_000),
        ]
    )
    report = run_entry_replay(
        labeled,
        blended,
        config=EntryReplayConfig(
            holdout_months=None, lookback_ns=2 * _BAR_NS, lookahead_ns=2 * _BAR_NS
        ),
    )
    limited = limit_replay(report, 1)
    assert limited.trades.height == 1
    assert int(limited.trades["trade_id"][0]) == int(report.trades["trade_id"][0])
    html = render_entry_replay_gallery_html(limited)
    assert html.count("<svg") == 1
