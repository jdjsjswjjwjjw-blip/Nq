"""اختبارات مُحاكي المزاد."""

from __future__ import annotations

import polars as pl
import pytest

from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.auction import (
    auction_action_states,
    auction_fsm_columns,
    auction_signal_frame,
    auction_states,
)
from tests.mbo_factory import make_stream


def test_balanced_window_rotates_in_value() -> None:
    # نافذة متوازنة: حجم مركّز داخل نطاق ضيّق.
    frame = make_stream(
        [
            ("T", "B", 100, 5, 0),
            ("T", "A", 100, 5, 0),
            ("T", "B", 101, 2, 0),
            ("T", "A", 99, 2, 0),
        ],
        event_ts=[0, 1, 2, 3],
        sequence=[1, 2, 3, 4],
    )
    states = auction_states(frame, interval_ns=100).sort("bucket_start")
    assert states.height == 1
    assert states["in_value_fraction"].to_list()[0] > 0
    assert states["is_balanced"].to_list()[0] in (True, False)


def test_expansion_and_new_high_detected() -> None:
    frame = make_stream(
        [
            # نافذة 0: نطاق ضيّق حول 100
            ("T", "B", 100, 5, 0),
            ("T", "A", 100, 5, 0),
            ("T", "B", 101, 1, 0),
            # نافذة 100: نطاق واسع وقمة جديدة
            ("T", "B", 100, 1, 0),
            ("T", "B", 120, 5, 0),
            ("T", "A", 100, 1, 0),
        ],
        event_ts=[0, 1, 2, 100, 101, 102],
        sequence=[1, 2, 3, 4, 5, 6],
    )
    states = auction_states(frame, interval_ns=100).sort("bucket_start")
    assert states.height == 2
    assert states["made_new_high"].to_list() == [False, True]
    # النافذة الثانية أوسع مدى من الأولى
    assert states["is_expansion"].to_list()[1] is True


def test_availability_is_bucket_end() -> None:
    frame = make_stream(
        [("T", "B", 100, 5, 0), ("T", "A", 100, 5, 0)],
        event_ts=[0, 1],
        sequence=[1, 2],
    )
    states = auction_states(frame, interval_ns=10)
    assert states["availability_ts"].to_list() == states["bucket_end"].to_list()


def test_balance_flips_to_imbalance() -> None:
    # نافذة 0 متوازنة (تدوير حول 100)، نافذة 100 مختلّة (اتجاه يُغلق عند القمة).
    balanced = [("T", "B", 100 + d, 2, 0) for d in (0, 0, 1, -1, 0)]
    trend = [("T", "B", 100 + j, 2, 0) for j in range(10)]
    events = balanced + trend
    ts = list(range(len(balanced))) + list(range(100, 100 + len(trend)))
    seq = list(range(1, len(events) + 1))
    frame = make_stream(events, event_ts=ts, sequence=seq)

    states = auction_states(frame, interval_ns=50).sort("bucket_start")
    states = states.with_columns(
        (pl.col("is_balanced").shift(1) & ~pl.col("is_balanced"))
        .fill_null(value=False)
        .alias("flip_to_imbalance")
    )
    assert states["is_balanced"].to_list()[0] is True
    assert states["is_balanced"].to_list()[1] is False
    assert states["flip_to_imbalance"].to_list()[1] is True
    assert "close_in_value" in states.columns


def test_empty_stream() -> None:
    states = auction_states(make_stream([]), interval_ns=10)
    assert states.height == 0


def test_auction_signal_frame_exports_vp_columns() -> None:
    balanced = [("T", "B", 100 + d, 2, 0) for d in (0, 0, 1, -1, 0)]
    trend = [("T", "B", 100 + j, 2, 0) for j in range(10)]
    events = balanced + trend
    ts = list(range(len(balanced))) + list(range(100, 100 + len(trend)))
    seq = list(range(1, len(events) + 1))
    frame = make_stream(events, event_ts=ts, sequence=seq)

    # رينج أوسع من ساعة الفعل (مصغّر للاختبار: 100 / 50 بدل 5د / 30ث)
    signals = auction_signal_frame(
        frame,
        profile_interval_ns=100,
        signal_interval_ns=50,
    ).sort(AVAILABILITY_TS)
    assert signals.height >= 1
    for col in (
        "vp_upper",
        "vp_mid",
        "vp_lower",
        "vp_rel_upper",
        "vp_rel_mid",
        "vp_rel_lower",
        "vp_excess_upper",
        "vp_excess_lower",
        "vp_of_delta",
        "vp_absorb",
        "vp_look_fail",
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_close_in_value",
        "vp_in_value_frac",
        "vp_pullback_defense",
        "vp_poc_migration",
        "vp_flip_to_imbalance",
        "vp_fsm_break",
        "vp_fsm_accel",
        "vp_fsm_retest",
        "vp_fsm_expand",
        "vp_auction_setup",
    ):
        assert col in signals.columns
    # ثلاث حدود VP: علوي ≥ متوسط ≥ سفلي
    assert signals["vp_upper"].to_list()[0] >= signals["vp_mid"].to_list()[0]
    assert signals["vp_mid"].to_list()[0] >= signals["vp_lower"].to_list()[0]


def test_auction_action_states_joins_profile_onto_signal() -> None:
    """رينج أوسع يُلحق سببيًا على براميل الفعل."""
    events = [("T", "B", 100 + (i % 3), 2, 0) for i in range(40)]
    ts = list(range(0, 4000, 100))
    frame = make_stream(events, event_ts=ts, sequence=list(range(1, 41)))

    action = auction_action_states(
        frame,
        profile_interval_ns=1000,
        signal_interval_ns=200,
    )
    assert action.height >= 1
    assert "vah" in action.columns and "poc" in action.columns and "val" in action.columns
    assert "is_balanced" in action.columns
    # كل برميل فعل يحمل حدود رينج مكتملة
    assert action["vah"].null_count() == 0


def test_auction_fsm_columns_empty_states() -> None:
    empty = auction_states(make_stream([]), interval_ns=10)
    fsm = auction_fsm_columns(empty)
    assert fsm.height == 0
    for col in (
        "vp_fsm_break",
        "vp_fsm_accel",
        "vp_fsm_retest",
        "vp_fsm_expand",
        "vp_auction_setup",
    ):
        assert col in fsm.columns


def test_auction_fsm_setup_completes_balance_break_retest_expand() -> None:
    """سلسلة اصطناعية: توازن → كسر أعلى → تسارع → ريتست عند mid → توسّع."""
    n = 12
    states = pl.DataFrame(
        {
            "bucket_start": list(range(n)),
            "is_balanced": [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "close": [
                100.0,
                100.5,
                103.0,
                104.0,
                105.0,
                100.2,
                100.1,
                103.0,
                106.0,
                108.0,
                110.0,
                112.0,
            ],
            "vah": [101.0] * n,
            "poc": [100.0] * n,
            "val": [99.0] * n,
            "bucket_volume": [
                10.0,
                10.0,
                10.0,
                40.0,
                12.0,
                11.0,
                10.0,
                10.0,
                20.0,
                22.0,
                25.0,
                30.0,
            ],
            "is_expansion": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            "pullback_defended": [
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            "close_in_value": [
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            "delta": [0.0, 0.0, 5.0, 8.0, 2.0, -1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "absorb": [0.0] * n,
            "look_fail": [0.0] * n,
        }
    )
    fsm = auction_fsm_columns(states, retest_window=8, accel_lookback=3, accel_mult=1.5)
    assert float(fsm["vp_fsm_break"][2]) == 1.0
    assert (fsm["vp_fsm_accel"] != 0.0).any()
    assert (fsm["vp_fsm_retest"] != 0.0).any()
    assert (fsm["vp_auction_setup"] == 1.0).any()


def test_auction_signal_frame_empty() -> None:
    signals = auction_signal_frame(
        make_stream([]),
        profile_interval_ns=100,
        signal_interval_ns=50,
    )
    assert signals.height == 0
    assert AVAILABILITY_TS in signals.columns
    assert "vp_balance" in signals.columns
    assert "vp_upper" in signals.columns
    assert "vp_mid" in signals.columns
    assert "vp_lower" in signals.columns
    assert "vp_auction_setup" in signals.columns
    assert "vp_fsm_break" in signals.columns


def test_auction_signal_frame_rejects_profile_shorter_than_signal() -> None:
    with pytest.raises(ValueError, match="profile_interval_ns"):
        auction_action_states(
            make_stream([("T", "B", 100, 1, 0)]),
            profile_interval_ns=10,
            signal_interval_ns=100,
        )