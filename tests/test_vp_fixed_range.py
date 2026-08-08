"""اختبارات رينج الفوليوم الثابت (Fixed Range) — قبول توسّع / عرضي / خروج."""

from __future__ import annotations

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import VP_LIQUIDITY_SESSION
from nq.simulation.auction import (
    auction_action_states,
    auction_fsm_columns,
    auction_signals_from_states,
)
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.vp_fixed_range import (
    VP_FIXED_RANGE_COLUMNS,
    VpFixedRangeConfig,
    attach_vp_fixed_range,
)
from tests.mbo_factory import make_stream

_IV = 100


def _bucket_trades(
    *,
    bucket: int,
    prices: list[int],
    size: int = 5,
) -> tuple[list[tuple[str, str, int, int, int]], list[int]]:
    """صفقات داخل برميل واحد."""
    events = [("T", "B", px, size, 0) for px in prices]
    ts = [bucket * _IV + i for i in range(len(prices))]
    return events, ts


def _profile_from_specs(
    specs: list[tuple[int, int, bool, bool, bool, float, int]],
    price_sets: list[list[int]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: dict[str, list[object]] = {
        BUCKET_START: [],
        BUCKET_END: [],
        AVAILABILITY_TS: [],
        "close": [],
        "range": [],
        "is_expansion": [],
        "is_balanced": [],
        "close_in_value": [],
        "in_value_fraction": [],
        VP_LIQUIDITY_SESSION: [],
    }
    all_events: list[tuple[str, str, int, int, int]] = []
    all_ts: list[int] = []
    for i, (close, rng, exp, bal, civ, ivf, sess) in enumerate(specs):
        rows[BUCKET_START].append(i * _IV)
        rows[BUCKET_END].append((i + 1) * _IV)
        rows[AVAILABILITY_TS].append((i + 1) * _IV)
        rows["close"].append(close)
        rows["range"].append(rng)
        rows["is_expansion"].append(exp)
        rows["is_balanced"].append(bal)
        rows["close_in_value"].append(civ)
        rows["in_value_fraction"].append(ivf)
        rows[VP_LIQUIDITY_SESSION].append(sess)
        ev, ts = _bucket_trades(bucket=i, prices=price_sets[i])
        all_events.extend(ev)
        all_ts.extend(ts)
    profile = pl.DataFrame(rows)
    mbo = make_stream(all_events, event_ts=all_ts, sequence=list(range(1, len(all_events) + 1)))
    return profile, mbo


def _synthetic_profile_and_mbo() -> tuple[pl.DataFrame, pl.DataFrame]:
    """سيناريو: توسّع → قبول عرضي → بناء → وخزة → خروج صريح."""
    specs = [
        (100, 2, False, True, True, 0.9, 2),  # 0 seed balance
        (108, 12, True, False, False, 0.2, 2),  # 1 expansion
        (104, 3, False, True, True, 0.85, 2),  # 2 accept
        (103, 2, False, True, True, 0.9, 2),  # 3 extend
        (105, 3, False, True, True, 0.8, 2),  # 4 extend
        (112, 4, False, False, False, 0.1, 2),  # 5 probe
        (118, 14, True, False, False, 0.05, 2),  # 6 clear exit
    ]
    price_sets = [
        [99, 100, 101, 100],
        [100, 105, 110, 108],
        [103, 104, 105, 104],
        [102, 103, 104, 103],
        [104, 105, 106, 105],
        [110, 111, 112, 112],
        [112, 115, 120, 118],
    ]
    return _profile_from_specs(specs, price_sets)


def test_attach_opens_on_accepted_expansion_and_extends_only_in_balance() -> None:
    profile, mbo = _synthetic_profile_and_mbo()
    out = attach_vp_fixed_range(
        profile,
        mbo,
        interval_ns=_IV,
        config=VpFixedRangeConfig(accept_window=3, exit_expansion_ratio=1.5),
    )
    for c in VP_FIXED_RANGE_COLUMNS:
        assert c in out.columns

    # العلم السببي على برميل القبول (2) لا برميل الاندفاع (1)
    assert out["vp_fr_accepted_expansion"].to_list()[1] == 0.0
    assert out["vp_fr_accepted_expansion"].to_list()[2] == 1.0
    assert out["vp_fr_start_ts"].to_list()[2] == 1 * _IV  # البداية = برميل التوسّع
    assert out["vp_fr_active"].to_list()[0] == 0.0
    assert out["vp_fr_active"].to_list()[2] == 1.0
    assert out["vp_fr_active"].to_list()[3] == 1.0
    assert out["vp_fr_active"].to_list()[4] == 1.0

    end_ts = out["vp_fr_end_ts"].to_list()
    assert end_ts[3] == 4 * _IV
    assert end_ts[4] == 5 * _IV
    assert end_ts[5] == 5 * _IV  # وخزة: لا مدّ

    # خروج: إشارة مع active=0 حتى لا يلصق use_fr عبر asof
    assert out["vp_fr_exit"].to_list()[6] == 1.0
    assert out["vp_fr_active"].to_list()[6] == 0.0


def test_accepted_flag_not_leaked_before_accept_availability() -> None:
    profile, mbo = _synthetic_profile_and_mbo()
    out = attach_vp_fixed_range(profile, mbo, interval_ns=_IV)
    # عند availability برميل الاندفاع لم يُعرف القبول بعد
    exp_row = out.filter(pl.col(BUCKET_START) == 1 * _IV)
    assert float(exp_row["vp_fr_accepted_expansion"][0]) == 0.0
    accept_row = out.filter(pl.col(BUCKET_START) == 2 * _IV)
    assert float(accept_row["vp_fr_accepted_expansion"][0]) == 1.0


def test_last_expansion_wins_before_accept() -> None:
    """البداية = آخر expansion قبل القبول، لا الأول."""
    specs = [
        (100, 2, False, True, True, 0.9, 2),
        (108, 12, True, False, False, 0.2, 2),  # early expansion
        (112, 14, True, False, False, 0.1, 2),  # later expansion (should win)
        (105, 3, False, True, True, 0.85, 2),  # accept
    ]
    price_sets = [
        [99, 100, 101, 100],
        [100, 105, 110, 108],
        [108, 110, 114, 112],
        [104, 105, 106, 105],
    ]
    profile, mbo = _profile_from_specs(specs, price_sets)
    out = attach_vp_fixed_range(profile, mbo, interval_ns=_IV)
    assert out["vp_fr_accepted_expansion"].to_list()[3] == 1.0
    assert out["vp_fr_start_ts"].to_list()[3] == 2 * _IV


def test_session_change_closes_range_without_exit_signal() -> None:
    profile, mbo = _synthetic_profile_and_mbo()
    profile = profile.with_columns(
        pl.when(pl.col(BUCKET_START) >= 4 * _IV)
        .then(pl.lit(0, dtype=pl.Int64))
        .otherwise(pl.col(VP_LIQUIDITY_SESSION))
        .alias(VP_LIQUIDITY_SESSION)
    )
    out = attach_vp_fixed_range(
        profile,
        mbo,
        interval_ns=_IV,
        config=VpFixedRangeConfig(accept_window=3),
    )
    assert out["vp_fr_active"].to_list()[4] == 0.0
    assert out["vp_fr_exit"].to_list()[4] == 0.0
    assert out["vp_fr_exit"].sum() == 0.0


def test_fsm_fires_setup_once_on_fr_exit_edge() -> None:
    n = 5
    states = pl.DataFrame(
        {
            BUCKET_START: list(range(n)),
            "close": [100.0] * n,
            "vah": [105.0] * n,
            "poc": [100.0] * n,
            "val": [95.0] * n,
            "bucket_volume": [10.0] * n,
            "is_balanced": [False] * n,
            "is_expansion": [False] * n,
            "pullback_defended": [False] * n,
            "close_in_value": [False] * n,
            "delta": [1.0] * n,
            "absorb": [0.0] * n,
            "look_fail": [0.0] * n,
            VP_LIQUIDITY_SESSION: [2] * n,
            # بعد الخروج: active=0 مع تكرار exit عبر asof
            "vp_fr_active": [1.0, 0.0, 0.0, 0.0, 0.0],
            "vp_fr_upper": [105.0, 105.0, 105.0, None, None],
            "vp_fr_mid": [100.0, 100.0, 100.0, None, None],
            "vp_fr_lower": [95.0, 95.0, 95.0, None, None],
            "vp_fr_exit": [0.0, 1.0, 1.0, 0.0, 0.0],
            "vp_fr_in_balance": [1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    fsm = auction_fsm_columns(states, fixed_range_decisions=True)
    setups = fsm["vp_auction_setup"].to_list()
    assert setups == [0.0, 1.0, 0.0, 0.0, 0.0]
    # بعد الحافة: لا use_fr sticky ولا setup كلاسيكي
    assert float(fsm["vp_fsm_build"].to_list()[2]) == 0.0


def test_fsm_fixed_range_decisions_suppress_classic_setup() -> None:
    """بين دورات FR: لا vp_auction_setup من المسار الكلاسيكي."""
    n = 8
    states = pl.DataFrame(
        {
            BUCKET_START: list(range(n)),
            "close": [100.0, 100.0, 110.0, 111.0, 112.0, 108.0, 115.0, 116.0],
            "vah": [105.0] * n,
            "poc": [100.0] * n,
            "val": [95.0] * n,
            "bucket_volume": [10.0] * n,
            "is_balanced": [True, True, False, False, False, False, False, False],
            "is_expansion": [False, False, True, False, True, False, True, True],
            "pullback_defended": [False] * n,
            "close_in_value": [True, True, False, False, False, True, False, False],
            "delta": [0.0, 0.0, 5.0, 5.0, 5.0, 1.0, 5.0, 5.0],
            "absorb": [0.0] * n,
            "look_fail": [0.0] * n,
            VP_LIQUIDITY_SESSION: [2] * n,
            "vp_fr_active": [0.0] * n,
            "vp_fr_upper": [None] * n,
            "vp_fr_mid": [None] * n,
            "vp_fr_lower": [None] * n,
            "vp_fr_exit": [0.0] * n,
            "vp_fr_in_balance": [0.0] * n,
        }
    )
    classic = auction_fsm_columns(states, fixed_range_decisions=False)
    fr_only = auction_fsm_columns(states, fixed_range_decisions=True)
    assert float(classic["vp_auction_setup"].abs().sum()) > 0.0
    assert float(fr_only["vp_auction_setup"].abs().sum()) == 0.0


def test_auction_action_states_exposes_fr_and_signals() -> None:
    events: list[tuple[str, str, int, int, int]] = []
    ts: list[int] = []
    for b in range(8):
        base = 100 + (b if b < 3 else (0 if b < 6 else b))
        for j, d in enumerate((0, 1, -1, 0)):
            events.append(("T", "B", base + d, 3, 0))
            ts.append(b * _IV + j)
    events[8:12] = [("T", "B", 100 + k * 3, 4, 0) for k in range(4)]
    frame = make_stream(events, event_ts=ts, sequence=list(range(1, len(events) + 1)))
    states = auction_action_states(
        frame,
        profile_interval_ns=_IV,
        signal_interval_ns=_IV,
        fixed_range=True,
    )
    assert states.height >= 1
    for c in VP_FIXED_RANGE_COLUMNS:
        assert c in states.columns
    sigs = auction_signals_from_states(states, fixed_range_decisions=True)
    for c in (
        "vp_fr_active",
        "vp_fr_accepted_expansion",
        "vp_fr_in_balance",
        "vp_fr_exit",
        "vp_fr_upper",
        "vp_fr_mid",
        "vp_fr_lower",
        "vp_fr_start_ts",
        "vp_fr_end_ts",
    ):
        assert c in sigs.columns
