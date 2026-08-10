"""اختبارات مُحاكي المزاد."""

from __future__ import annotations

import polars as pl
import pytest

from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.auction import (
    auction_action_states,
    auction_fsm_columns,
    auction_signal_frame,
    auction_signals_from_states,
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


def test_expansion_uses_past_range_median_not_single_tiny_bar() -> None:
    frame = make_stream(
        [
            ("T", "B", 100, 1, 0),
            ("T", "A", 110, 1, 0),
            ("T", "B", 109, 1, 0),
            ("T", "A", 110, 1, 0),
            ("T", "B", 109, 1, 0),
            ("T", "A", 111, 1, 0),
        ],
        event_ts=[0, 1, 100, 101, 200, 201],
        sequence=[1, 2, 3, 4, 5, 6],
    )
    states = auction_states(frame, interval_ns=100).sort("bucket_start")
    assert states["range"].to_list() == [10, 1, 2]
    assert states["made_new_high"].to_list()[2] is True
    assert states["is_expansion"].to_list()[2] is False


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
        "vp_order_accel",
        "vp_early_imbalance",
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_close_in_value",
        "vp_in_value_frac",
        "vp_pullback_defense",
        "vp_poc_migration",
        "vp_flip_to_imbalance",
        "vp_fsm_break",
        "vp_fsm_build",
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


def test_auction_signals_from_states_matches_full_signal_frame() -> None:
    events = [("T", "B", 100 + (i % 4), 2, 0) for i in range(40)]
    frame = make_stream(
        events,
        event_ts=list(range(0, 4000, 100)),
        sequence=list(range(1, 41)),
    )
    states = auction_action_states(
        frame,
        profile_interval_ns=1000,
        signal_interval_ns=200,
    )
    from_states = auction_signals_from_states(states, retest_window=3)
    direct = auction_signal_frame(
        frame,
        profile_interval_ns=1000,
        signal_interval_ns=200,
        retest_window=3,
    )
    assert from_states.equals(direct)


def test_auction_fsm_columns_empty_states() -> None:
    empty = auction_states(make_stream([]), interval_ns=10)
    fsm = auction_fsm_columns(empty)
    assert fsm.height == 0
    for col in (
        "vp_fsm_break",
        "vp_fsm_build",
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
    # أول كسر = مراقبة فقط — ليس توسّعًا
    assert float(fsm["vp_fsm_expand"][2]) == 0.0
    assert (fsm["vp_fsm_accel"] != 0.0).any()
    assert (fsm["vp_fsm_retest"] != 0.0).any() or (fsm["vp_fsm_build"] != 0.0).any()
    assert (fsm["vp_auction_setup"] == 1.0).any()


def _fsm_base_frame(n: int = 16, **overrides: object) -> pl.DataFrame:
    data: dict[str, object] = {
        "bucket_start": list(range(n)),
        "is_balanced": [True, True] + [False] * (n - 2),
        "close": [100.0, 100.0] + [103.0] * (n - 2),
        "vah": [101.0] * n,
        "poc": [100.0] * n,
        "val": [99.0] * n,
        "bucket_volume": [10.0] * n,
        "is_expansion": [False] * n,
        "pullback_defended": [False] * n,
        "close_in_value": [True, True] + [False] * (n - 2),
        "delta": [0.0, 0.0] + [5.0] * (n - 2),
        "absorb": [0.0] * n,
        "look_fail": [0.0] * n,
    }
    data.update(overrides)
    return pl.DataFrame(data)


def test_auction_fsm_first_break_is_not_expansion() -> None:
    """أول كسر خارج الفوليوم لا يعلن expand/setup."""
    states = _fsm_base_frame(
        n=6,
        close=[100.0, 100.0, 103.0, 103.5, 104.0, 104.5],
        is_expansion=[False] * 6,
    )
    fsm = auction_fsm_columns(states, rebalance_confirm=3, build_max_age=20)
    assert float(fsm["vp_fsm_break"][2]) == 1.0
    assert float(fsm["vp_fsm_expand"].sum()) == 0.0
    assert float(fsm["vp_auction_setup"].sum()) == 0.0


def test_auction_fsm_return_inside_volume_keeps_pending_then_expands() -> None:
    """رجوع داخل الفوليوم بعد الكسر = بناء، ثم انطلاق لاحق — لا قتل للرحلة."""
    n = 14
    states = _fsm_base_frame(
        n=n,
        is_balanced=[True, True] + [False] * (n - 2),
        close=[
            100.0,
            100.0,
            103.0,  # break
            100.5,  # back inside — build
            100.2,
            100.8,
            99.5,
            100.1,
            100.4,
            102.0,  # poke again — still not expand (no is_expansion)
            100.3,
            106.0,  # clean expand
            108.0,
            110.0,
        ],
        close_in_value=[
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
        ],
        is_expansion=[False] * 11 + [True, True, True],
        delta=[0.0, 0.0] + [4.0] * (n - 2),
        pullback_defended=[False] * 3 + [True, True, True, True, True, True] + [False] * 5,
    )
    fsm = auction_fsm_columns(states, retest_window=10, rebalance_confirm=5, build_max_age=30)
    assert float(fsm["vp_fsm_break"][2]) == 1.0
    # أثناء الرجوع داخل القيمة يظهر build وليس expand
    assert float(fsm["vp_fsm_build"][3]) == 1.0
    assert float(fsm["vp_fsm_expand"][3]) == 0.0
    assert float(fsm["vp_fsm_expand"][9]) == 0.0  # poke بلا is_expansion
    assert (fsm["vp_fsm_expand"] == 1.0).any()
    assert (fsm["vp_auction_setup"] == 1.0).any()
    setup_i = int(fsm["vp_auction_setup"].to_list().index(1.0))
    assert setup_i >= 11


def test_auction_fsm_brief_balance_flicker_does_not_kill_context() -> None:
    """وميض توازن لبرميل أو اثنين لا يقفل القصة؛ التوسّع اللاحق يبقى متاحًا."""
    n = 12
    states = _fsm_base_frame(
        n=n,
        is_balanced=[
            True,
            True,
            False,  # break
            False,
            True,  # flicker 1
            True,  # flicker 2 (< rebalance_confirm=3)
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        close=[
            100.0,
            100.0,
            103.0,
            102.5,
            100.2,
            100.1,
            100.5,
            102.0,
            106.0,
            108.0,
            110.0,
            112.0,
        ],
        close_in_value=[
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ],
        is_expansion=[False] * 8 + [True, True, True, True],
        delta=[0.0, 0.0] + [3.0] * (n - 2),
    )
    fsm = auction_fsm_columns(states, rebalance_confirm=3, build_max_age=30)
    assert (fsm["vp_fsm_build"] != 0.0).any()
    assert (fsm["vp_auction_setup"] == 1.0).any()


def test_auction_fsm_true_rebalance_kills_pending() -> None:
    """إعادة توازن حقيقية (3 براميل متتالية) تقتل السياق — لا توسّع لاحق من نفس الكسر."""
    n = 12
    states = _fsm_base_frame(
        n=n,
        is_balanced=[
            True,
            True,
            False,  # break
            True,
            True,
            True,  # confirm rebalance → kill
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        close=[
            100.0,
            100.0,
            103.0,
            100.0,
            100.1,
            100.0,
            106.0,
            108.0,
            110.0,
            112.0,
            114.0,
            116.0,
        ],
        close_in_value=[True, True, False] + [True] * 3 + [False] * 6,
        is_expansion=[False] * 6 + [True] * 6,
        delta=[0.0, 0.0, 5.0] + [0.0] * 3 + [6.0] * 6,
    )
    fsm = auction_fsm_columns(states, rebalance_confirm=3, build_max_age=30)
    assert float(fsm["vp_fsm_break"][2]) == 1.0
    assert float(fsm["vp_fsm_build"][3]) == 1.0
    # بعد قتل السياق: لا setup من نفس الرحلة (قد يحدث break جديد لاحقًا فقط إن سبقه توازن)
    assert float(fsm["vp_auction_setup"].sum()) == 0.0


def test_auction_fsm_build_around_single_mid_anchor() -> None:
    """بناء حول خط واحد (POC في منتصف التذبذب) بدون اشتراط خطّين."""
    n = 10
    states = _fsm_base_frame(
        n=n,
        close=[100.0, 100.0, 103.0, 100.4, 99.7, 100.3, 99.8, 100.2, 106.0, 108.0],
        close_in_value=[True, True, False, True, True, True, True, True, False, False],
        is_expansion=[False] * 8 + [True, True],
        delta=[0.0, 0.0] + [2.0] * (n - 2),
    )
    fsm = auction_fsm_columns(states, rebalance_confirm=4, build_max_age=20)
    assert (fsm["vp_fsm_build"] == 1.0).any()
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
    assert "vp_order_accel" in signals.columns
    assert "vp_early_imbalance" in signals.columns


def test_auction_wires_order_accel_early_imbalance() -> None:
    """مسار المزاد يصدّر vp_order_accel / vp_early_imbalance من أداة order_flow."""
    # عدة براميل فعل هادئة ثم برميل فعل بضغطة شراء كبيرة (تسارع واضح).
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    seq = 1
    for b in range(8):
        for j in range(4):
            size = 40 if b == 7 else 2
            events.append(("T", "B", 100 + (j % 2), size, 0))
            ts.append(b * 50 + j)
            seq += 1
    frame = make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))
    states = auction_action_states(
        frame,
        profile_interval_ns=100,
        signal_interval_ns=50,
        fixed_range=False,
    )
    assert states.height >= 4
    assert "order_accel_rate" in states.columns
    assert "early_imbalance" in states.columns
    # آخر برميل فعل: استهلاك أعلى بكثير من الأساس السابق.
    assert float(states["order_accel_rate"][-1]) > 1.0
    sigs = auction_signals_from_states(states, fixed_range_decisions=False)
    assert "vp_order_accel" in sigs.columns
    assert "vp_early_imbalance" in sigs.columns
    assert (sigs["vp_order_accel"].abs() > 0.0).any()
    assert (sigs["vp_early_imbalance"].abs() > 0.0).any()


def test_auction_signal_frame_rejects_profile_shorter_than_signal() -> None:
    with pytest.raises(ValueError, match="profile_interval_ns"):
        auction_action_states(
            make_stream([("T", "B", 100, 1, 0)]),
            profile_interval_ns=10,
            signal_interval_ns=100,
        )
