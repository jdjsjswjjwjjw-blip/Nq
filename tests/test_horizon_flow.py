"""مسار 5 دقائق بعد القمة: الشرائح تبدأ من H لا من +30ث."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

import nq.research
from nq.research.horizon_flow import (
    compare_post_peak_horizon,
    post_peak_windows,
    write_horizon_report,
)
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_pattern import HORIZON_S
from nq.research.triple_tape import scan_triple_pattern
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_HI = 20_001_000_000
_PX_LOW = 19_800_000_000
_NS = SECOND_NS


def _t0() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "compare_post_peak_horizon" not in nq.research.__all__
    assert not hasattr(nq.research, "compare_post_peak_horizon")


def test_horizon_is_300s_and_bins_start_at_h() -> None:
    assert HORIZON_S == 300
    high = _t0()
    wins = post_peak_windows(high)
    names = [w.name for w in wins]
    assert names[0] == "peak"
    assert names[1] == "+0-30s"
    assert names[2] == "+30-60s"
    assert names[-1] == "+0-300s"
    assert wins[0].end_ts == high
    assert wins[1].start_ts == high
    assert wins[2].start_ts == high + 30 * _NS
    assert wins[-1].end_ts == high + 300 * _NS


def test_refuses_concatenated_multi_day() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mnq = make_stream(
        [("T", "B", _HI, 1, 1), ("T", "B", _HI, 1, 2)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    tape = make_stream(
        [("T", "B", _HI, 1, 0)],
        event_ts=[int(day_a.timestamp() * 1_000_000_000)],
        sequence=[1],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mnq)
    with pytest.raises(ValueError, match="multi-day"):
        compare_post_peak_horizon(mnq, tape, tape, price_hi=_HI)


def test_path_captures_nq_fade_after_h_not_only_plus_30() -> None:
    high = _t0()
    mnq = make_stream(
        [
            ("T", "B", _PX, 2, 1),
            ("C", "A", _PX, 8, 2),
            ("F", "A", _PX, 1, 2),
            ("T", "B", _HI, 1, 9),
            ("T", "A", _PX_LOW, 1, 10),
        ],
        event_ts=[high - 10 * _NS, high - 9 * _NS, high - 8 * _NS, high, high + 200 * _NS],
        sequence=[1, 2, 3, 4, 5],
    )
    tape = make_stream(
        [("T", "B", _PX, 2, 0), ("T", "B", _HI, 1, 0), ("T", "A", _PX_LOW, 1, 0)],
        event_ts=[high - 10 * _NS, high, high + 200 * _NS],
        sequence=[1, 4, 5],
    ).drop("order_id")
    nq = make_stream(
        [
            ("T", "B", _PX, 5, 0),
            ("T", "B", _PX, 2, 0),
            ("T", "A", _PX, 2, 0),
            ("T", "A", _PX, 6, 0),
        ],
        event_ts=[high - 10 * _NS, high + 5 * _NS, high + 40 * _NS, high + 100 * _NS],
        sequence=[10, 11, 12, 13],
    ).drop("order_id")
    table, diag = compare_post_peak_horizon(mnq, tape, nq, price_hi=_HI)
    by = {r["name"]: r for r in table.iter_rows(named=True)}
    assert by["+0-30s"]["nq_t_imbalance"] > 0
    assert by["+30-60s"]["nq_t_imbalance"] < 0
    assert by["+0-300s"]["nq_t_imbalance"] < 0
    assert by["peak"]["nq_fill_ratio"] != by["peak"]["nq_fill_ratio"]  # NaN
    assert diag["nq_faded_nonpos"] is True
    assert diag["first_30s_nq_imbalance"] > 0
    assert diag["old_drop_30_60s_nq_imbalance"] < 0
    assert diag["day_scan_drop_already_5m"] is True
    assert by["+180-210s"]["min_px"] == _PX_LOW or by["+0-300s"]["min_px"] == _PX_LOW


def test_fwd_nq_imb_excludes_in_window_prints() -> None:
    t0 = _t0()
    events = [("T", "B", _PX, 2, 1), ("C", "A", _PX, 10, 2), ("F", "A", _PX, 1, 2)]
    ts = [t0, t0 + 1, t0 + 2]
    for i in range(1, 6):
        stamp = t0 + i * 5 * _NS
        events.extend(
            [("T", "B", _PX, 2, 10 + i), ("C", "A", _PX, 10, 20 + i), ("F", "A", _PX, 1, 20 + i)]
        )
        ts.extend([stamp, stamp + 1, stamp + 2])
    events.append(("T", "A", _PX, 1, 99))
    ts.append(t0 + 40 * _NS)
    mbo = make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))
    tape_t = [(a, s, p, sz, 0) for (a, s, p, sz, _oid) in events if a == "T"]
    tape_ts = [t for (a, _s, _p, _sz, _oid), t in zip(events, ts, strict=True) if a == "T"]
    tape = make_stream(tape_t, event_ts=tape_ts, sequence=list(range(1, len(tape_t) + 1))).drop(
        "order_id"
    )
    nq = make_stream(
        [("T", "B", _PX, 4, 0), ("T", "A", _PX, 8, 0)],
        event_ts=[t0 + 10 * _NS, t0 + 50 * _NS],
        sequence=[1, 2],
    ).drop("order_id")
    scored, diag = scan_triple_pattern(mbo, tape, nq, seed=0)
    assert diag["horizon_s"] == 300
    hit = scored.filter(scored["nq_t_imbalance"] > 0.20)
    assert hit.height >= 1
    fwd = hit.select(pl.col("nq_fwd_imbalance").median()).item()
    assert fwd is not None and float(fwd) < 0


def test_horizon_report(tmp_path: Path) -> None:
    high = _t0()
    mnq = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _HI, 1, 2)],
        event_ts=[high - _NS, high],
        sequence=[1, 2],
    )
    tape = make_stream(
        [("T", "B", _PX, 1, 0), ("T", "B", _HI, 1, 0)],
        event_ts=[high - _NS, high],
        sequence=[1, 2],
    ).drop("order_id")
    nq = make_stream(
        [("T", "B", _PX, 1, 0)],
        event_ts=[high - _NS],
        sequence=[9],
    ).drop("order_id")
    table, diag = compare_post_peak_horizon(mnq, tape, nq, price_hi=_HI)
    written = write_horizon_report(table, diag, tmp_path)
    text = (written / "HORIZON_FLOW.md").read_text(encoding="utf-8")
    assert "old drop window was only +30-60s" in text
    assert "+0-30s" in text
