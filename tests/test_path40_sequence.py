"""مسار 40 برميلًا: اللقطة لا تكفي؛ المستقبل لا يدخل الميزات."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

import nq.research
from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.determinism import make_generator
from nq.core.session import VP_LIQUIDITY_SESSION, VpLiquiditySession
from nq.research.path40_sequence import (
    HOLDOUT_START_DATE,
    HORIZON_BARS,
    LOOKBACK_BARS,
    Y_PATH40,
    build_lookback_tensor,
    build_path40_labels,
    fit_predict_conv,
    fit_predict_logistic,
    flatten_path_matrix,
    last_bar_matrix,
    run_path40_sequence,
    write_path40_report,
)

_ET = ZoneInfo("America/New_York")
_LONDON = int(VpLiquiditySession.LONDON)


def _ns(day: int, minute: int, *, month: int = 6) -> int:
    stamp = dt.datetime(2025, month, day, 4, 0, tzinfo=_ET) + dt.timedelta(minutes=minute)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "run_path40_sequence" not in nq.research.__all__
    assert not hasattr(nq.research, "run_path40_sequence")


def test_lookback_does_not_include_future_bars() -> None:
    n = LOOKBACK_BARS + 3
    ts = [_ns(3, i) for i in range(n)]
    high = [100.0] * n
    high[-1] = 900.0
    setup = ts[-3]
    blended = pl.DataFrame(
        {
            AVAILABILITY_TS: ts,
            "close": [100.0] * n,
            "high": high,
            "low": [100.0] * n,
            "_behavior_story_run": [1] * n,
        }
    )
    x = build_lookback_tensor(blended, np.array([setup], dtype=np.int64))
    assert x.shape == (1, LOOKBACK_BARS, 5)
    assert float(x[0, -1, 4]) == 1.0
    assert float(np.max(x[0, :, 1])) < 2.0


def test_label_horizon_is_forty_and_renamed() -> None:
    prior_n = 3
    n = HORIZON_BARS + 2
    prior = pl.DataFrame(
        {
            AVAILABILITY_TS: [_ns(2, i) for i in range(prior_n)],
            VP_LIQUIDITY_SESSION: [_LONDON] * prior_n,
            "close": [120.0] * prior_n,
            "high": [150.0] * prior_n,
            "low": [100.0] * prior_n,
            "path_beyond_asia_ticks": [0.0] * prior_n,
            "vp_fsm_break": [0.0] * prior_n,
            "vp_fsm_retest": [0.0] * prior_n,
            "proj_break_direction": [1.0] * prior_n,
            "_behavior_story_run": [0] * prior_n,
        }
    )
    today = pl.DataFrame(
        {
            AVAILABILITY_TS: [_ns(3, i) for i in range(n)],
            VP_LIQUIDITY_SESSION: [_LONDON] * n,
            "close": [200.0] * n,
            "high": [200.0] + [230.0] * (n - 1),
            "low": [200.0] + [199.0] * (n - 1),
            "path_beyond_asia_ticks": [0.0] + [2.0] * (n - 1),
            "vp_fsm_break": [0.0, 1.0] + [0.0] * (n - 2),
            "vp_fsm_retest": [0.0] * n,
            "proj_break_direction": [1.0] * n,
            "_behavior_story_run": [1] * n,
        }
    )
    labels = build_path40_labels(pl.concat([prior, today]))
    resolved = labels.filter(pl.col("label_status") == "resolved")
    assert Y_PATH40 in labels["outcome_name"].to_list()
    assert HORIZON_BARS == 40
    assert LOOKBACK_BARS == 40
    assert HOLDOUT_START_DATE == "2025-09-01"
    if resolved.height:
        assert int(resolved["horizon_bars"].to_list()[-1]) >= HORIZON_BARS


def test_last_bar_misses_path_that_linear_sees() -> None:
    rng = make_generator(7)
    n_each = 24
    t_len = LOOKBACK_BARS
    x = np.zeros((n_each * 2, t_len, 5), dtype=np.float64)
    y = np.zeros(n_each * 2, dtype=np.float64)
    for i in range(n_each):
        x[i, :, 4] = 1.0
        x[i, :, 0] = np.linspace(-0.8, 0.0, t_len)
        x[i, -1, :4] = 0.0
        y[i] = 1.0
        j = n_each + i
        x[j, :, 4] = 1.0
        x[j, :, 0] = rng.normal(0.0, 0.3, size=t_len)
        x[j, -1, :4] = 0.0
        y[j] = 0.0
    train = np.arange(0, n_each * 2, 2)
    test = np.arange(1, n_each * 2, 2)
    p_last = fit_predict_logistic(last_bar_matrix(x), y, train)[test]
    p_path = fit_predict_logistic(flatten_path_matrix(x), y, train)[test]
    last_spread = float(np.max(p_last) - np.min(p_last))
    pos = p_path[y[test] > 0.5]
    neg = p_path[y[test] < 0.5]
    assert last_spread < 0.05
    assert float(np.min(pos) - np.max(neg)) > 0.05


def test_conv_trains_deterministically() -> None:
    rng = make_generator(1)
    x = rng.normal(0.0, 0.1, size=(20, LOOKBACK_BARS, 5))
    x[:, :, 4] = 1.0
    y = (x[:, :10, 0].mean(axis=1) > 0).astype(np.float64)
    train = np.arange(20)
    p1 = fit_predict_conv(x, y, train, rng=make_generator(1), epochs=3)
    p2 = fit_predict_conv(x, y, train, rng=make_generator(1), epochs=3)
    assert np.allclose(p1, p2)
    assert np.all((p1 > 0.0) & (p1 < 1.0))


def test_run_writes_report_and_skips_holdout(tmp_path: Path) -> None:
    prior_n = 3
    n = HORIZON_BARS + LOOKBACK_BARS
    rows: list[dict[str, float | int]] = []
    for i in range(prior_n):
        rows.append(
            {
                AVAILABILITY_TS: _ns(2, i, month=4),
                VP_LIQUIDITY_SESSION: _LONDON,
                "close": 120.0,
                "high": 150.0,
                "low": 100.0,
                "path_beyond_asia_ticks": 0.0,
                "vp_fsm_break": 0.0,
                "vp_fsm_retest": 0.0,
                "proj_break_direction": 1.0,
                "_behavior_story_run": 0,
            }
        )
    for story in range(6):
        for bar in range(n):
            onset = bar == LOOKBACK_BARS - 1
            after = bar >= LOOKBACK_BARS - 1
            rows.append(
                {
                    AVAILABILITY_TS: _ns(3 + story, bar, month=5),
                    VP_LIQUIDITY_SESSION: _LONDON,
                    "close": 200.0,
                    "high": 230.0 if after else 200.0,
                    "low": 199.0 if after else 200.0,
                    "path_beyond_asia_ticks": 2.0 if after else 0.0,
                    "vp_fsm_break": 1.0 if onset else 0.0,
                    "vp_fsm_retest": 0.0,
                    "proj_break_direction": 1.0,
                    "_behavior_story_run": story + 1,
                }
            )
    blended = pl.DataFrame(rows)
    scored, diag = run_path40_sequence(blended, conv_epochs=2)
    assert diag["retrained_existing_heads"] is False
    assert diag["holdout_touched"] is False
    assert diag["uses_future_bars_as_features"] is False
    if scored.height:
        assert SETUP_AVAILABILITY_TS in scored.columns
        assert "p_path_conv" in scored.columns
    out = write_path40_report(scored, diag, tmp_path)
    text = (out / "PATH40.md").read_text(encoding="utf-8")
    assert "path_conv" in text
    assert "last_bar" in text
