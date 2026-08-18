"""مسح نمط القمة على شرائح اليوم مقابل ضابط. النتيجة بعد النافذة فقط."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_pattern import (
    FILL_RATIO_MAX,
    T_IMB_MIN,
    T_RATE_MIN,
    scan_peak_pattern,
    write_pattern_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_PX_LOW = 19_800_000_000
_NS = SECOND_NS
_HI = 20_100_000_000


def _t0() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "scan_peak_pattern" not in nq.research.__all__
    assert not hasattr(nq.research, "scan_peak_pattern")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mbo = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _PX, 1, 2)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    with pytest.raises(ValueError, match="multi-day"):
        scan_peak_pattern(mbo)


def test_locked_thresholds_match_pr116() -> None:
    assert T_RATE_MIN == 50.0
    assert T_IMB_MIN == 0.10
    assert FILL_RATIO_MAX == 0.20


def test_pattern_then_drop_counts_reversal_control_does_not_need_pattern() -> None:
    t0 = _t0()
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    seq = 1
    # 30s of buy T + ask cancel/fill (fill ratio 1/11)
    for i in range(6):
        stamp = t0 + i * 5 * _NS
        events.append(("T", "B", _PX, 2, 100 + i))
        ts.append(stamp)
        events.append(("C", "A", _PX, 10, 200 + i))
        ts.append(stamp + 1)
        events.append(("F", "A", _PX, 1, 300 + i))
        ts.append(stamp + 2)
        seq += 3
    # after window: 1% drop
    events.append(("T", "A", _PX_LOW, 1, 999))
    ts.append(t0 + 40 * _NS)
    # later quiet sell window, no drop after it
    late = t0 + 400 * _NS
    events.append(("T", "A", _PX, 2, 400))
    ts.append(late)
    events.append(("C", "A", _PX, 1, 401))
    ts.append(late + 1)
    events.append(("T", "A", _PX, 1, 402))
    ts.append(late + 800 * _NS)
    mbo = make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))
    scored, diag = scan_peak_pattern(
        mbo,
        t_rate_min=0.1,
        t_imb_min=0.10,
        fill_ratio_max=0.20,
        seed=0,
    )
    hits = scored.filter(
        (pl.col("t_rate") > 0.1) & (pl.col("t_imbalance") > 0.10) & (pl.col("fill_ratio") < 0.20)
    )
    assert hits.height >= 1
    assert bool(hits["rev_100bps"].any())
    assert diag["n_pattern_episodes"] >= 1
    assert diag["summary"]["pattern_episodes"]["rate_100bps"] > 0
    ctrl = diag["summary"]["control"]
    assert ctrl["n"] >= 1
    assert "busy_control" in diag["summary"]
    assert diag["n_busy_control"] >= 0
    assert "pattern_hour_utc" in diag


def test_named_peak_is_scored_without_using_future_in_pattern() -> None:
    t0 = _t0()
    mbo = make_stream(
        [
            ("T", "B", _HI, 2, 1),
            ("C", "A", _HI, 8, 2),
            ("F", "A", _HI, 1, 3),
            ("T", "A", _PX_LOW, 1, 9),
        ],
        event_ts=[t0 - _NS, t0 - _NS + 1, t0 - _NS + 2, t0 + 10 * _NS],
        sequence=[1, 2, 3, 4],
    )
    _, diag = scan_peak_pattern(
        mbo,
        price_hi=_HI,
        t_rate_min=0.0,
        t_imb_min=-1.0,
        fill_ratio_max=1.0,
    )
    peak = diag["named_peak"]
    assert peak["high_ts"] == t0 - _NS
    assert peak["drop_frac"] > 0
    assert "matches_pattern" in peak


def test_same_seed_draws_same_random_control(tmp_path: Path) -> None:
    t0 = _t0()
    events = [("T", "B", _PX, 1, i) for i in range(1, 20)]
    ts = [t0 + i * 5 * _NS for i in range(19)]
    mbo = make_stream(events, event_ts=ts, sequence=list(range(1, 20)))
    scored, a = scan_peak_pattern(mbo, t_rate_min=1000.0, seed=3)
    _, b = scan_peak_pattern(mbo, t_rate_min=1000.0, seed=3)
    assert a["summary"]["random_control"]["n"] == b["summary"]["random_control"]["n"]
    written = write_pattern_report(scored, a, tmp_path)
    assert "Locked from PR #116" in (written / "PEAK_PATTERN.md").read_text(encoding="utf-8")
