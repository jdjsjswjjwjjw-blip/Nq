"""تدفق T/F حول القمة: اختلال العدوان، بلا نموذج."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import nq.research
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import HOUR_NS, NamedWindow
from nq.research.peak_flow import compare_peak_flow, score_flow_window, write_flow_report
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_HI = 20_001_000_000
_NS = SECOND_NS


def _t0() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "compare_peak_flow" not in nq.research.__all__
    assert not hasattr(nq.research, "compare_peak_flow")
    assert not hasattr(nq.research, "score_flow_window")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mbo = make_stream(
        [("T", "B", _HI, 1, 9), ("T", "B", _HI, 1, 10)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    with pytest.raises(ValueError, match="multi-day"):
        compare_peak_flow(mbo, price_hi=_HI)


def test_buy_aggression_and_ask_fills_in_peak_window() -> None:
    high = _t0()
    win = NamedWindow("peak", high - 30 * _NS, high)
    mbo = make_stream(
        [
            ("T", "B", _PX, 4, 11),
            ("F", "A", _PX, 4, 1),
            ("T", "A", _PX, 1, 12),
            ("C", "A", _PX, 2, 2),
        ],
        event_ts=[high - 20 * _NS, high - 20 * _NS, high - 5 * _NS, high - 4 * _NS],
        sequence=[1, 2, 3, 4],
    )
    row = score_flow_window(mbo, win)
    assert row["t_buy_size"] == 4
    assert row["t_sell_size"] == 1
    assert abs(row["t_imbalance"] - 0.6) < 1e-9
    assert row["f_ask_size"] == 4
    assert row["ask_hit_share"] == 4 / 6


def test_late_half_can_flip_from_early() -> None:
    high = _t0()
    start = high - 30 * _NS
    mbo = make_stream(
        [
            ("T", "B", _PX, 5, 1),
            ("T", "A", _PX, 1, 2),
            ("T", "A", _PX, 6, 3),
        ],
        event_ts=[start + _NS, start + 2 * _NS, start + 20 * _NS],
        sequence=[1, 2, 3],
    )
    row = score_flow_window(mbo, NamedWindow("peak", start, high))
    assert row["t_imbalance_early"] > 0
    assert row["t_imbalance_late"] < 0


def test_compare_peak_flow_uses_same_clock_as_controls() -> None:
    high = _t0()
    mbo = make_stream(
        [
            ("T", "B", _PX, 2, 1),
            ("T", "A", _PX, 2, 2),
            ("T", "A", _PX, 3, 3),
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
    table, diag = compare_peak_flow(mbo, price_hi=_HI)
    assert diag["phantom_closed"] is True
    assert diag["not_backtest"] is True
    by = {r["name"]: r for r in table.iter_rows(named=True)}
    assert by["peak"]["end_ts"] == high
    assert by["climb"]["t_buy_size"] == 2
    assert by["peak"]["t_sell_size"] == 2
    assert by["drop"]["t_sell_size"] == 3


def test_flow_report(tmp_path: Path) -> None:
    high = _t0()
    mbo = make_stream(
        [("T", "B", _PX, 1, 1), ("T", "B", _HI, 1, 2)],
        event_ts=[high - _NS, high],
        sequence=[1, 2],
    )
    table, diag = compare_peak_flow(mbo, price_hi=_HI)
    written = write_flow_report(table, diag, tmp_path)
    text = (written / "PEAK_FLOW.md").read_text(encoding="utf-8")
    assert "Phantom/cancel-noise chapter is closed" in text
    assert "Not a model" in text
