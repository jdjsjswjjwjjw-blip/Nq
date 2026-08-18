"""مقارنة MBO مقابل شريط TRADE: تنفيذ فوري أو مرور كأمر معلّق."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import nq.research
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.mbo_trade_overlap import (
    compare_mbo_trades,
    overlap_diagnostics,
    overlap_table,
    prepare_trades_tape,
    write_overlap_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_PX_ASK = 20_000_250_000
_NS = 1_000_000_000


def _t0() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "compare_mbo_trades" not in nq.research.__all__
    assert not hasattr(nq.research, "compare_mbo_trades")
    assert not hasattr(nq.research, "overlap_table")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mbo = make_stream(
        [("A", "B", _PX, 1, 1), ("T", "A", _PX, 1, 9)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    trades = make_stream(
        [("T", "A", _PX, 1, 0)],
        event_ts=[int(day_a.timestamp() * 1_000_000_000)],
        sequence=[2],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    with pytest.raises(ValueError, match="multi-day"):
        compare_mbo_trades(mbo, trades)


def test_resting_fill_vs_immediate_aggressor_print() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 4, 1),
            ("F", "B", _PX, 4, 1),
            ("T", "A", _PX, 4, 99),
        ],
        event_ts=[t - 3 * _NS, t - 2 * _NS, t - 2 * _NS],
        sequence=[1, 2, 2],
    )
    trades = make_stream(
        [("T", "A", _PX, 4, 0)],
        event_ts=[t - 2 * _NS],
        sequence=[2],
    ).drop("order_id")
    result = compare_mbo_trades(mbo, trades)
    assert result.n_add == 1
    assert result.n_mbo_f == 1
    assert result.n_mbo_t == 1
    assert result.n_trades == 1
    assert result.n_mbo_f_rested == 1
    assert result.n_mbo_f_no_prior_add == 0
    assert result.n_mbo_t_rested == 0
    assert result.n_mbo_t_immediate == 1
    assert result.n_oid_added_and_filled == 1
    assert result.n_oid_traded_never_added == 1
    assert result.n_seq_overlap == 1
    assert result.n_trades_unmatched_seq == 0
    assert result.pct_adds_that_fill == 1.0
    assert result.pct_mbo_t_immediate == 1.0
    assert result.pct_mbo_f_rested == 1.0
    assert result.pct_trades_matched_to_mbo_t == 1.0


def test_trade_print_after_add_counts_as_rested() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 2, 7),
            ("T", "B", _PX, 2, 7),
        ],
        event_ts=[t - 2 * _NS, t - _NS],
        sequence=[10, 11],
    )
    trades = make_stream(
        [("T", "B", _PX, 2, 0)],
        event_ts=[t - _NS],
        sequence=[11],
    ).drop("order_id")
    result = compare_mbo_trades(mbo, trades)
    assert result.n_mbo_t_rested == 1
    assert result.n_mbo_t_immediate == 0
    assert result.n_oid_traded_never_added == 0


def test_unmatched_tape_sequence_is_counted() -> None:
    t = _t0()
    mbo = make_stream(
        [("A", "A", _PX_ASK, 1, 3), ("T", "B", _PX_ASK, 1, 8)],
        event_ts=[t - 2 * _NS, t - _NS],
        sequence=[1, 5],
    )
    trades = make_stream(
        [("T", "B", _PX_ASK, 1, 0)],
        event_ts=[t - _NS],
        sequence=[99],
    ).drop("order_id")
    result = compare_mbo_trades(mbo, trades)
    assert result.n_seq_overlap == 0
    assert result.n_trades_unmatched_seq == 1
    assert result.pct_trades_matched_to_mbo_t == 0.0


def test_fill_without_prior_add_is_not_rested() -> None:
    t = _t0()
    mbo = make_stream(
        [("F", "B", _PX, 3, 44), ("T", "A", _PX, 3, 0)],
        event_ts=[t - _NS, t - _NS],
        sequence=[4, 4],
    )
    trades = make_stream(
        [("T", "A", _PX, 3, 0)],
        event_ts=[t - _NS],
        sequence=[4],
    )
    result = compare_mbo_trades(mbo, trades)
    assert result.n_add == 0
    assert result.n_mbo_f_rested == 0
    assert result.n_mbo_f_no_prior_add == 1
    assert result.n_oid_added_and_filled == 0


def test_prepare_trades_tape_fills_missing_order_id() -> None:
    t = _t0()
    trades = make_stream(
        [("T", "A", _PX, 1, 12)],
        event_ts=[t],
        sequence=[1],
    ).drop("order_id")
    prepared = prepare_trades_tape(trades)
    assert "order_id" in prepared.columns
    assert prepared["order_id"].to_list() == [0]


def test_overlap_table_and_report(tmp_path: Path) -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 1, 1),
            ("F", "B", _PX, 1, 1),
            ("T", "A", _PX, 1, 2),
        ],
        event_ts=[t - 2 * _NS, t - _NS, t - _NS],
        sequence=[1, 2, 2],
    )
    trades = make_stream(
        [("T", "A", _PX, 1, 0)],
        event_ts=[t - _NS],
        sequence=[2],
    ).drop("order_id")
    result = compare_mbo_trades(mbo, trades)
    table = overlap_table(result)
    assert "mbo_T_immediate_no_prior_add" in table["metric"].to_list()
    diagnostics = overlap_diagnostics(result, mbo_path="mbo.parquet", trades_path="tr.parquet")
    written = write_overlap_report(table, diagnostics, tmp_path)
    assert (written / "MBO_TRADE_OVERLAP.md").is_file()
    assert (written / "summary.json").is_file()
    assert (written / "mbo_trade_overlap.parquet").is_file()
    assert "immediate" in (written / "MBO_TRADE_OVERLAP.md").read_text(encoding="utf-8")
