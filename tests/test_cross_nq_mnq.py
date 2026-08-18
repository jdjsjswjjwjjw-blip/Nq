"""NQ trades مقابل MNQ MBO على نوافذ مقفلة. النتيجة من T فقط لـ NQ."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import nq.research
from nq.research.cross_nq_mnq import (
    MNQ_MULT,
    NQ_MULT,
    compare_nq_mnq_windows,
    write_cross_report,
)
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import HOUR_NS
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_HI = 20_001_000_000
_NS = SECOND_NS


def _t0() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "compare_nq_mnq_windows" not in nq.research.__all__
    assert not hasattr(nq.research, "compare_nq_mnq_windows")


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
    nq = make_stream(
        [("T", "B", _HI, 1, 0)],
        event_ts=[int(day_a.timestamp() * 1_000_000_000)],
        sequence=[1],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mnq)
    with pytest.raises(ValueError, match="multi-day"):
        compare_nq_mnq_windows(mnq, nq, price_hi=_HI)


def test_clock_locks_to_mnq_high_nq_fill_is_nan() -> None:
    high = _t0()
    mnq = make_stream(
        [
            ("T", "B", _PX, 4, 1),
            ("F", "A", _PX, 4, 1),
            ("C", "A", _PX, 12, 2),
            ("T", "B", _HI, 1, 9),
        ],
        event_ts=[high - 10 * _NS, high - 10 * _NS, high - 9 * _NS, high],
        sequence=[1, 2, 3, 4],
    )
    nq = make_stream(
        [("T", "B", _PX, 1, 0), ("T", "B", _HI, 1, 0)],
        event_ts=[high - 10 * _NS, high + 2_000_000],
        sequence=[10, 11],
    ).drop("order_id")
    stacked, diffs, diag = compare_nq_mnq_windows(mnq, nq, price_hi=_HI)
    assert diag["clock"] == "locked_to_mnq_first_T_at_price_hi"
    assert diag["mnq_high_ts"] == high
    assert diag["nq_high_ts"] == high + 2_000_000
    assert diag["high_leader"] == "mnq"
    assert diag["high_lag_ns"] == 2_000_000
    assert diag["nq_fill_ratio"] == "unavailable_without_mbo_F_C"
    by = {r["name"]: r for r in diffs.iter_rows(named=True)}
    peak_nq = stacked.filter((stacked["name"] == "peak") & (stacked["contract"] == "NQ")).row(
        0, named=True
    )
    assert peak_nq["n_t"] >= 1
    assert peak_nq["ask_hit_share"] != peak_nq["ask_hit_share"]  # NaN
    peak_mnq = stacked.filter((stacked["name"] == "peak") & (stacked["contract"] == "MNQ")).row(
        0, named=True
    )
    assert peak_mnq["ask_hit_share"] == 4 / 16
    assert by["peak"]["first_print_lag_ns"] == 0.0


def test_diffs_are_nq_minus_mnq_and_notional_uses_multipliers() -> None:
    high = _t0()
    mnq = make_stream(
        [
            ("T", "B", _PX, 10, 1),
            ("T", "B", _HI, 1, 9),
        ],
        event_ts=[high - _NS, high],
        sequence=[1, 2],
    )
    nq = make_stream(
        [("T", "B", _PX, 1, 0), ("T", "A", _PX, 1, 0)],
        event_ts=[high - _NS, high - 500_000_000],
        sequence=[8, 9],
    ).drop("order_id")
    stacked, diffs, diag = compare_nq_mnq_windows(mnq, nq, price_hi=_HI)
    assert diag["not_spoofing"] is True
    peak = {r["name"]: r for r in diffs.iter_rows(named=True)}["peak"]
    mnq_peak = stacked.filter((stacked["name"] == "peak") & (stacked["contract"] == "MNQ")).row(
        0, named=True
    )
    nq_peak = stacked.filter((stacked["name"] == "peak") & (stacked["contract"] == "NQ")).row(
        0, named=True
    )
    assert abs(peak["d_t_per_s"] - (nq_peak["t_per_s"] - mnq_peak["t_per_s"])) < 1e-12
    assert abs(peak["d_t_imbalance"] - (nq_peak["t_imbalance"] - mnq_peak["t_imbalance"])) < 1e-12
    assert mnq_peak["t_notional"] == (mnq_peak["t_buy_size"] + mnq_peak["t_sell_size"]) * MNQ_MULT
    assert nq_peak["t_notional"] == (nq_peak["t_buy_size"] + nq_peak["t_sell_size"]) * NQ_MULT
    assert NQ_MULT / MNQ_MULT == 10.0


def test_climb_and_drop_share_mnq_clock() -> None:
    high = _t0()
    mnq = make_stream(
        [
            ("T", "A", _PX, 2, 1),
            ("T", "B", _PX, 3, 2),
            ("T", "A", _PX, 4, 3),
            ("T", "B", _HI, 1, 9),
        ],
        event_ts=[
            high - HOUR_NS - _NS,
            high - _NS,
            high + 31 * _NS,
            high,
        ],
        sequence=[1, 2, 3, 4],
    )
    nq = make_stream(
        [
            ("T", "B", _PX, 1, 0),
            ("T", "A", _PX, 5, 0),
            ("T", "B", _PX, 2, 0),
        ],
        event_ts=[
            high - HOUR_NS - _NS,
            high - _NS,
            high + 31 * _NS,
        ],
        sequence=[10, 11, 12],
    ).drop("order_id")
    stacked, diffs, diag = compare_nq_mnq_windows(mnq, nq, price_hi=_HI)
    assert diag["mnq_high_ts"] == high
    by = {(r["name"], r["contract"]): r for r in stacked.iter_rows(named=True)}
    assert by[("climb", "MNQ")]["t_sell_size"] == 2
    assert by[("peak", "MNQ")]["t_buy_size"] == 3
    assert by[("drop", "MNQ")]["t_sell_size"] == 4
    assert by[("climb", "NQ")]["t_buy_size"] == 1
    assert by[("peak", "NQ")]["t_sell_size"] == 5
    assert diffs.height == 3


def test_cross_report(tmp_path: Path) -> None:
    high = _t0()
    mnq = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _HI, 1, 2)],
        event_ts=[high - _NS, high],
        sequence=[1, 2],
    )
    nq = make_stream(
        [("T", "B", _PX, 1, 0)],
        event_ts=[high - _NS],
        sequence=[9],
    ).drop("order_id")
    stacked, diffs, diag = compare_nq_mnq_windows(mnq, nq, price_hi=_HI)
    written = write_cross_report(stacked, diffs, diag, tmp_path)
    text = (written / "CROSS_NQ_MNQ.md").read_text(encoding="utf-8")
    assert "Fill_Ratio needs MBO F/C" in text
    assert "Not spoofing" in text
    assert diag["nq_source"] == "trades_tape_T_only"
