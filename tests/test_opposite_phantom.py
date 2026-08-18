"""ضوضاء عكسية قبل T العدواني: حدس Add/Cancel بلا ملء، ليس سبوفينج قانونيًا."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import nq.research
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.opposite_phantom import (
    SECOND_NS,
    closed_unfilled_orders,
    opposite_phantom,
    write_phantom_report,
)
from tests.mbo_factory import make_stream

_ET = ZoneInfo("America/New_York")
_PX = 20_000_000_000
_PX_ASK = 20_000_250_000
_PX_FAR = _PX_ASK + 8 * 250_000_000
_NS = SECOND_NS


def _t0() -> int:
    stamp = dt.datetime(2025, 6, 3, 10, 35, 0, tzinfo=_ET)
    return int(stamp.timestamp() * 1_000_000_000)


def test_not_exported_from_research_init() -> None:
    assert "opposite_phantom" not in nq.research.__all__
    assert not hasattr(nq.research, "closed_unfilled_orders")
    assert not hasattr(nq.research, "write_phantom_report")


def test_refuses_concatenated_multi_day_mbo() -> None:
    day_a = dt.datetime(2025, 6, 3, 4, 0, tzinfo=_ET)
    day_b = dt.datetime(2025, 6, 5, 4, 0, tzinfo=_ET)
    mbo = make_stream(
        [("A", "A", _PX_ASK, 4, 1), ("C", "A", _PX_ASK, 4, 1), ("T", "B", _PX_ASK, 1, 9)],
        event_ts=[
            int(day_a.timestamp() * 1_000_000_000),
            int(day_a.timestamp() * 1_000_000_000) + 100,
            int(day_b.timestamp() * 1_000_000_000),
        ],
    )
    with pytest.raises(ValueError, match="multi-day"):
        assert_single_day_mbo(mbo)
    with pytest.raises(ValueError, match="multi-day"):
        opposite_phantom(mbo, price_lo=_PX, price_hi=_PX_ASK)


def test_opposite_unfilled_cancel_counts_same_side_does_not() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "A", _PX_ASK, 8, 1),
            ("C", "A", _PX_ASK, 8, 1),
            ("A", "B", _PX, 8, 2),
            ("C", "B", _PX, 8, 2),
            ("T", "B", _PX_ASK, 2, 99),
        ],
        event_ts=[t - 4 * _NS, t - 3 * _NS, t - 4 * _NS, t - 3 * _NS, t],
        sequence=[1, 2, 3, 4, 5],
    )
    windows, per_t, _ = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_ASK,
        windows_s=(5,),
        tick_band=4,
        mean_add_size=1.0,
        size_mult=2.0,
    )
    assert windows.height == 1
    assert windows["unique_opposite_oids"][0] == 1
    assert windows["unique_large_oids"][0] == 1
    assert per_t["opp_size"][0] == 8
    assert per_t["opp_ratio"][0] == 4.0
    assert windows["n_t_ratio_ge_3"][0] == 1


def test_fill_excludes_order_from_phantom() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "A", _PX_ASK, 8, 1),
            ("F", "A", _PX_ASK, 8, 1),
            ("C", "A", _PX_ASK, 8, 1),
            ("T", "B", _PX_ASK, 2, 99),
        ],
        event_ts=[t - 4 * _NS, t - 3 * _NS, t - 2 * _NS, t],
        sequence=[1, 2, 3, 4],
    )
    windows, _, _ = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_ASK,
        windows_s=(5,),
        mean_add_size=1.0,
    )
    assert windows["unique_opposite_oids"][0] == 0
    assert closed_unfilled_orders(mbo).height == 0


def test_events_at_or_after_t_are_ignored() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("T", "B", _PX_ASK, 2, 99),
            ("A", "A", _PX_ASK, 8, 1),
            ("C", "A", _PX_ASK, 8, 1),
        ],
        event_ts=[t, t + _NS, t + 2 * _NS],
        sequence=[1, 2, 3],
    )
    windows, _, _ = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_ASK,
        windows_s=(5,),
        mean_add_size=1.0,
    )
    assert windows["unique_opposite_oids"][0] == 0
    assert windows["mean_phantom_ratio"][0] == 0.0


def test_outside_window_and_far_price_excluded() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "A", _PX_ASK, 8, 1),
            ("C", "A", _PX_ASK, 8, 1),
            ("A", "A", _PX_FAR, 8, 2),
            ("C", "A", _PX_FAR, 8, 2),
            ("T", "B", _PX_ASK, 2, 99),
        ],
        event_ts=[t - 20 * _NS, t - 19 * _NS, t - 2 * _NS, t - _NS, t],
        sequence=[1, 2, 3, 4, 5],
    )
    short, _, _ = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_FAR,
        windows_s=(5,),
        tick_band=4,
        mean_add_size=1.0,
    )
    wide, _, _ = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_FAR,
        windows_s=(30,),
        tick_band=4,
        mean_add_size=1.0,
    )
    assert short["unique_opposite_oids"][0] == 0
    assert wide["unique_opposite_oids"][0] == 1


def test_small_size_is_opposite_but_not_large() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "A", _PX_ASK, 1, 1),
            ("C", "A", _PX_ASK, 1, 1),
            ("T", "B", _PX_ASK, 2, 99),
        ],
        event_ts=[t - 2 * _NS, t - _NS, t],
        sequence=[1, 2, 3],
    )
    windows, per_t, diag = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_ASK,
        windows_s=(5,),
        mean_add_size=1.0,
        size_mult=2.0,
    )
    assert diag["large_min_size"] == 2.0
    assert windows["unique_opposite_oids"][0] == 1
    assert windows["unique_large_oids"][0] == 0
    assert per_t["large_ratio"][0] == 0.0
    assert per_t["opp_ratio"][0] == 0.5


def test_rested_t_is_not_an_anchor() -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "B", _PX, 2, 7),
            ("T", "B", _PX, 2, 7),
        ],
        event_ts=[t - 2 * _NS, t],
        sequence=[1, 2],
    )
    windows, _, diag = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_ASK,
        windows_s=(5,),
        mean_add_size=1.0,
    )
    assert diag["n_t"] == 0
    assert windows.height == 0


def test_instant_cancel_rate_and_report(tmp_path: Path) -> None:
    t = _t0()
    mbo = make_stream(
        [
            ("A", "A", _PX_ASK, 8, 1),
            ("C", "A", _PX_ASK, 8, 1),
            ("T", "B", _PX_ASK, 1, 99),
        ],
        event_ts=[t - 200_000_000, t - 100_000_000, t],
        sequence=[1, 2, 3],
    )
    windows, per_t, diag = opposite_phantom(
        mbo,
        price_lo=_PX,
        price_hi=_PX_ASK,
        windows_s=(1,),
        mean_add_size=1.0,
    )
    assert windows["instant_cancel_rate"][0] == 1.0
    written = write_phantom_report(windows, diag, tmp_path, per_print=per_t)
    text = (written / "OPPOSITE_PHANTOM.md").read_text(encoding="utf-8")
    assert "Not a legal spoofing" in text
    assert "Not an LSTM" in text
    assert (written / "summary.json").is_file()
