"""دمج ثلاثة مصادر على نوافذ مقفلة ثم مسح فرضية القمة."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import nq.research
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import HOUR_NS
from nq.research.triple_tape import (
    FILL_RATIO_MAX,
    NQ_IMB_MIN,
    NQ_IMB_NEAR_ZERO,
    compare_triple_windows,
    scan_triple_pattern,
    write_triple_report,
)
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
    assert "compare_triple_windows" not in nq.research.__all__
    assert not hasattr(nq.research, "compare_triple_windows")
    assert not hasattr(nq.research, "scan_triple_pattern")


def test_locked_thresholds_match_hypothesis() -> None:
    assert FILL_RATIO_MAX == 0.20
    assert NQ_IMB_MIN == 0.20
    assert NQ_IMB_NEAR_ZERO == 0.05


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
        compare_triple_windows(mnq, tape, tape, price_hi=_HI)


def test_peak_hypothesis_fill_low_nq_still_buying() -> None:
    high = _t0()
    peak = high - 10 * _NS
    drop = high + 40 * _NS
    mnq = make_stream(
        [
            ("A", "A", _PX, 10, 1),
            ("T", "B", _PX, 4, 9),
            ("F", "A", _PX, 1, 1),
            ("C", "A", _PX, 8, 1),
            ("T", "B", _HI, 1, 8),
            ("A", "A", _PX, 5, 2),
            ("T", "B", _PX, 1, 7),
            ("F", "A", _PX, 1, 2),
            ("C", "A", _PX, 5, 2),
        ],
        event_ts=[
            peak,
            peak + 1,
            peak + 2,
            peak + 3,
            high,
            drop,
            drop + 1,
            drop + 2,
            drop + 3,
        ],
        sequence=list(range(1, 10)),
    )
    mnq_tape = make_stream(
        [("T", "B", _PX, 4, 0), ("T", "B", _HI, 1, 0), ("T", "B", _PX, 1, 0)],
        event_ts=[peak + 1, high, drop + 1],
        sequence=[2, 5, 7],
    ).drop("order_id")
    nq_tape = make_stream(
        [
            ("T", "B", _PX, 5, 0),
            ("T", "A", _PX, 1, 0),
            ("T", "B", _PX, 1, 0),
            ("T", "A", _PX, 1, 0),
        ],
        event_ts=[peak + 1, peak + 2, drop + 1, drop + 2],
        sequence=[20, 21, 22, 23],
    ).drop("order_id")
    table, diag = compare_triple_windows(mnq, mnq_tape, nq_tape, price_hi=_HI)
    assert diag["clock"] == "locked_to_mnq_first_T_at_price_hi"
    assert diag["mnq_high_ts"] == high
    assert diag["nq_fill_ratio"] == "unavailable_without_mbo_F_C"
    by = {r["name"]: r for r in table.iter_rows(named=True)}
    assert by["peak"]["mnq_fill_ratio"] == 1 / 9
    assert by["peak"]["mnq_ask_cancel_share"] == 8 / 9
    assert by["peak"]["nq_t_imbalance"] > 0.20
    assert by["peak"]["peak_hypothesis"] is True
    assert by["peak"]["nq_fill_ratio"] != by["peak"]["nq_fill_ratio"]  # NaN
    assert by["peak"]["mbo_tape_n_t_diff"] == 0
    assert by["drop"]["drop_hypothesis"] is True
    assert abs(by["drop"]["nq_t_imbalance"]) < 0.05
    assert diag["peak_hypothesis_holds"] is True
    assert diag["drop_hypothesis_holds"] is True


def test_climb_uses_same_mnq_clock() -> None:
    high = _t0()
    mnq = make_stream(
        [
            ("T", "A", _PX, 2, 1),
            ("T", "B", _PX, 3, 2),
            ("T", "B", _HI, 1, 9),
        ],
        event_ts=[high - HOUR_NS - _NS, high - _NS, high],
        sequence=[1, 2, 3],
    )
    tape = make_stream(
        [("T", "A", _PX, 2, 0), ("T", "B", _PX, 3, 0), ("T", "B", _HI, 1, 0)],
        event_ts=[high - HOUR_NS - _NS, high - _NS, high],
        sequence=[1, 2, 3],
    ).drop("order_id")
    nq = make_stream(
        [("T", "B", _PX, 1, 0)],
        event_ts=[high - _NS],
        sequence=[9],
    ).drop("order_id")
    table, diag = compare_triple_windows(mnq, tape, nq, price_hi=_HI)
    by = {r["name"]: r for r in table.iter_rows(named=True)}
    assert diag["mnq_high_ts"] == high
    assert by["climb"]["mnq_mbo_t_imbalance"] < 0
    assert by["peak"]["mnq_mbo_t_imbalance"] > 0


def test_scan_joint_pattern_then_drop_vs_control() -> None:
    t0 = _t0()
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    nq_events: list[tuple[str, str, int, int, int]] = []
    nq_ts: list[int] = []
    for i in range(6):
        stamp = t0 + i * 5 * _NS
        events.extend(
            [("T", "B", _PX, 2, 100 + i), ("C", "A", _PX, 10, 200 + i), ("F", "A", _PX, 1, 300 + i)]
        )
        ts.extend([stamp, stamp + 1, stamp + 2])
        nq_events.append(("T", "B", _PX, 3, 0))
        nq_ts.append(stamp)
    events.append(("T", "A", _PX_LOW, 1, 999))
    ts.append(t0 + 40 * _NS)
    late = t0 + 400 * _NS
    events.extend([("T", "A", _PX, 2, 400), ("C", "A", _PX, 1, 401), ("T", "A", _PX, 1, 402)])
    ts.extend([late, late + 1, late + 800 * _NS])
    nq_events.append(("T", "A", _PX, 2, 0))
    nq_ts.append(late)
    mbo = make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))
    tape_t = [(a, s, p, sz, 0) for (a, s, p, sz, _oid) in events if a == "T"]
    tape_ts = [t for (a, _s, _p, _sz, _oid), t in zip(events, ts, strict=True) if a == "T"]
    tape = make_stream(tape_t, event_ts=tape_ts, sequence=list(range(1, len(tape_t) + 1))).drop(
        "order_id"
    )
    nq = make_stream(nq_events, event_ts=nq_ts, sequence=list(range(1, len(nq_events) + 1))).drop(
        "order_id"
    )
    scored, diag = scan_triple_pattern(mbo, tape, nq, seed=0)
    assert diag["n_pattern_windows"] >= 1
    assert diag["summary"]["pattern_windows"]["rate_100bps"] > 0
    assert diag["nq_fill_ratio"] == "unavailable_without_mbo_F_C"
    assert "fill_only" in diag["summary"]
    assert "nq_buy_only" in diag["summary"]
    assert "busy_control" in diag["summary"]
    assert "nq_fwd_imbalance" in scored.columns
    assert scored.filter(scored["nq_t_imbalance"] > 0.20).height >= 1


def test_triple_report(tmp_path: Path) -> None:
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
    named, diag = compare_triple_windows(mnq, tape, nq, price_hi=_HI)
    written = write_triple_report(named, diag, tmp_path)
    text = (written / "TRIPLE_TAPE.md").read_text(encoding="utf-8")
    assert "MNQ fill_ratio<0.20" in text
    assert "Not spoofing" in text
