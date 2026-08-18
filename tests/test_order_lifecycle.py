"""دورة حياة الأمر: تنفيذ مقابل إلغاء سريع، سببي حتى t."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import nq.research
from nq.research.manual_zone_depth import et_clock_to_ns
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.order_lifecycle import (
    FLEETING_NS,
    close_orders_by_t,
    metrics_frame,
    window_metrics,
    write_lifecycle_report,
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
    assert "window_metrics" not in nq.research.__all__
    assert not hasattr(nq.research, "window_metrics")
    assert not hasattr(nq.research, "close_orders_by_t")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mbo = make_stream(
        [("A", "B", _PX, 1, 1), ("A", "A", _PX_ASK, 1, 2)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    with pytest.raises(ValueError, match="multi-day"):
        window_metrics(mbo, _t0(), windows_s=(10,))


def test_add_trade_cancel_is_genuine_not_fleeting() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 4, 1),
            ("T", "B", _PX, 4, 1),
            ("C", "B", _PX, 4, 1),
        ],
        event_ts=[t - 3 * _NS, t - 2 * _NS, t - _NS],
    )
    closed, _ = close_orders_by_t(mbo, t)
    assert len(closed) == 1
    assert closed[0].kind == "genuine"
    assert closed[0].executed_size == 4


def test_fast_cancel_without_trade_is_fleeting() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 10, 1),
            ("C", "B", _PX, 10, 1),
        ],
        event_ts=[t - _NS, t - _NS // 2],
    )
    closed, _ = close_orders_by_t(mbo, t)
    assert len(closed) == 1
    assert closed[0].kind == "fleeting_unfilled"
    assert closed[0].lifetime_ns < FLEETING_NS
    rows = window_metrics(mbo, t, windows_s=(10,))
    assert rows[0].n_fleeting == 1
    assert rows[0].fleeting_size == 10
    assert rows[0].n_trade == 0


def test_slow_cancel_is_not_fleeting() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 6, 1),
            ("C", "B", _PX, 6, 1),
        ],
        event_ts=[t - 5 * _NS, t - _NS],
    )
    closed, _ = close_orders_by_t(mbo, t)
    assert closed[0].kind == "cancelled_unfilled"
    rows = window_metrics(mbo, t, windows_s=(10,))
    assert rows[0].n_fleeting == 0


def test_future_cancel_not_visible_at_t() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 8, 1),
            ("C", "B", _PX, 8, 1),
        ],
        event_ts=[t - _NS, t + _NS],
    )
    closed, _ = close_orders_by_t(mbo, t)
    assert closed == []
    rows = window_metrics(mbo, t, windows_s=(10,))
    assert rows[0].n_fleeting == 0
    assert rows[0].n_add == 1


def test_short_window_drops_earlier_fleeting() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 5, 1),
            ("C", "B", _PX, 5, 1),
            ("A", "B", _PX, 3, 2),
            ("T", "B", _PX, 3, 2),
            ("C", "B", _PX, 3, 2),
        ],
        event_ts=[t - 25 * _NS, t - 24 * _NS, t - 5 * _NS, t - 4 * _NS, t - 3 * _NS],
    )
    rows = {r.window_s: r for r in window_metrics(mbo, t, windows_s=(10, 30))}
    assert rows[10].n_fleeting == 0
    assert rows[30].n_fleeting == 1
    assert rows[10].n_trade == 1
    assert rows[30].n_trade == 1


def test_refill_after_fill_counts_iceberg() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 5, 1),
            ("T", "B", _PX, 2, 1),
            ("M", "B", _PX, 8, 1),
            ("C", "B", _PX, 8, 1),
        ],
        event_ts=[t - 4 * _NS, t - 3 * _NS, t - 2 * _NS, t - _NS],
    )
    closed, _ = close_orders_by_t(mbo, t)
    assert closed[0].kind == "genuine"
    assert closed[0].had_refill is True
    rows = window_metrics(mbo, t, windows_s=(10,))
    assert rows[0].n_iceberg == 1


def test_report_writes(tmp_path: Path) -> None:
    t = _t0()
    mbo = make_stream(
        [("A", "B", _PX, 2, 1), ("C", "B", _PX, 2, 1)],
        event_ts=[t - _NS, t - _NS // 2],
    )
    frame = metrics_frame(window_metrics(mbo, t, windows_s=(10, 30)))
    written = write_lifecycle_report(
        frame,
        {"reconstructed_order_book": False, "concatenated_raw_mbo": False},
        tmp_path,
    )
    text = (written / "ORDER_LIFECYCLE.md").read_text(encoding="utf-8")
    assert "fleeting_unfilled" in text
    assert "Not a live overlay" in text
    assert et_clock_to_ns("2025-06-03", "10:35:00") == t
