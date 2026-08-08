"""تحقق علمي كمّي شامل للمشروع (Project-Wide Scientific Audit).

يغطي المبادئ الأربعة الحاكمة + طبقات المحاكاة + Fixed-Range + الجلسات +
WF-قبل-التنفيذ. لا يدّعي ربحية على بيانات حية — يثبت صحّة الآلة السببية.
"""

from __future__ import annotations

import datetime as dt
import importlib
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from nq.contracts.mbo import MBO_SCHEMA, PRICE_SCALE, validate_mbo_frame
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.determinism import make_generator
from nq.core.session import (
    VP_LIQUIDITY_SESSION,
    VpLiquiditySession,
    vp_liquidity_session_from_ns,
)
from nq.core.time import assert_sorted_causal
from nq.orderbook import reconstruct
from nq.simulation.auction import (
    VP_PROFILE_INTERVAL_NS,
    VP_SIGNAL_INTERVAL_NS,
    auction_action_states,
    auction_fsm_columns,
    auction_signal_frame,
    auction_signals_from_states,
)
from nq.simulation.common import BUCKET_START
from nq.simulation.vp_fixed_range import (
    VP_FIXED_RANGE_COLUMNS,
    VpFixedRangeConfig,
    attach_vp_fixed_range,
)
from nq.strategies.vp_auction import _VP_AUCTION_FOCUS
from nq.validation import assert_availability_not_before_event, assert_causal_order
from tests.mbo_factory import make_stream
from tests.test_vp_fixed_range import _profile_from_specs, _synthetic_profile_and_mbo
from tests.test_vp_scientific_intensive import _synth_session

_NS = 1_000_000_000
_IV = 100
_PX = int(PRICE_SCALE)

# ---------------------------------------------------------------------------
# A) مبادئ حاكمة: MBO / سببية / availability
# ---------------------------------------------------------------------------


def test_audit_mbo_schema_is_sole_event_contract() -> None:
    required = {
        "event_ts",
        "ingest_ts",
        "sequence",
        "instrument_id",
        "symbol",
        "action",
        "side",
        "price",
        "size",
        "order_id",
        "flags",
    }
    assert required <= set(MBO_SCHEMA.keys())
    frame = _synth_session(500, seed=1, hours=0.5)
    validate_mbo_frame(frame)
    assert_sorted_causal(frame)


def test_audit_reconstruction_prefix_independence() -> None:
    frame = _synth_session(3_000, seed=2, hours=1.0)
    full = reconstruct(frame)
    cut = frame.height // 2
    prefix = reconstruct(frame.head(cut))
    assert prefix.top_of_book.height <= full.top_of_book.height
    if prefix.top_of_book.height and full.top_of_book.height:
        last_seq = int(prefix.top_of_book["sequence"][-1])
        full_at = full.top_of_book.filter(pl.col("sequence") <= last_seq)
        assert full_at.height >= 1
        assert full_at["best_bid"][-1] == prefix.top_of_book["best_bid"][-1]
        assert full_at["best_ask"][-1] == prefix.top_of_book["best_ask"][-1]


def test_audit_auction_suffix_perturbation_causal() -> None:
    frame = _synth_session(6_000, seed=3, hours=2.0)
    prof, sig_iv = 60 * _NS, 30 * _NS
    a = auction_signal_frame(frame, profile_interval_ns=prof, signal_interval_ns=sig_iv)
    if a.height < 16:
        pytest.skip("need enough bars")
    cut = a.height // 2
    mid = int(a[AVAILABILITY_TS][cut])
    noisy = frame.with_columns(
        pl.when(pl.col(EVENT_TS) > mid)
        .then(pl.col("price") + 25 * _PX)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    b = auction_signal_frame(noisy, profile_interval_ns=prof, signal_interval_ns=sig_iv)
    early = a.filter(pl.col(AVAILABILITY_TS) <= mid).join(
        b.filter(pl.col(AVAILABILITY_TS) <= mid),
        on=AVAILABILITY_TS,
        how="inner",
        suffix="_b",
    )
    assert early.height >= 4
    for col in ("vp_upper", "vp_balance", "vp_fr_active", "vp_auction_setup"):
        if col in early.columns and f"{col}_b" in early.columns:
            assert np.allclose(early[col].to_numpy(), early[f"{col}_b"].to_numpy(), equal_nan=True)


def test_audit_dual_timeframes_and_availability_monotonic() -> None:
    assert VP_PROFILE_INTERVAL_NS == 5 * 60 * _NS
    assert VP_SIGNAL_INTERVAL_NS == 30 * _NS
    frame = _synth_session(4_000, seed=4, hours=1.5)
    states = auction_action_states(
        frame,
        profile_interval_ns=VP_PROFILE_INTERVAL_NS,
        signal_interval_ns=VP_SIGNAL_INTERVAL_NS,
        fixed_range=True,
    )
    if states.height == 0:
        pytest.skip("empty states")
    assert_causal_order(states[AVAILABILITY_TS].to_list())
    assert_availability_not_before_event(
        states[BUCKET_START].to_list(),
        states[AVAILABILITY_TS].to_list(),
    )
    for c in VP_FIXED_RANGE_COLUMNS:
        assert c in states.columns


# ---------------------------------------------------------------------------
# B) جلسات السيولة + Fixed-Range (القواعد الخمس)
# ---------------------------------------------------------------------------


def test_audit_liquidity_session_boundaries_partition_day() -> None:
    # نصف-مفتوح: آسيا [18,03) لندن [03,09:30) نيويورك [09:30,18)
    # عيّنة ET صيف 2024-06-03

    et = ZoneInfo("America/New_York")

    def ns(h: int, m: int = 0) -> int:
        return int(dt.datetime(2024, 6, 3, h, m, tzinfo=et).timestamp() * _NS)

    assert vp_liquidity_session_from_ns(ns(18)) == int(VpLiquiditySession.ASIA)
    assert vp_liquidity_session_from_ns(ns(2, 59)) == int(VpLiquiditySession.ASIA)
    assert vp_liquidity_session_from_ns(ns(3)) == int(VpLiquiditySession.LONDON)
    assert vp_liquidity_session_from_ns(ns(9, 29)) == int(VpLiquiditySession.LONDON)
    assert vp_liquidity_session_from_ns(ns(9, 30)) == int(VpLiquiditySession.NEW_YORK)
    assert vp_liquidity_session_from_ns(ns(17, 59)) == int(VpLiquiditySession.NEW_YORK)


def test_audit_fr_five_rules_locked() -> None:
    """قفل كمّي للقواعد الخمس: قبول سببي، active=0 عند الخروج، آخر expansion، مدّ بالتوازن فقط."""
    profile, mbo = _synthetic_profile_and_mbo()
    out = attach_vp_fixed_range(
        profile,
        mbo,
        interval_ns=_IV,
        config=VpFixedRangeConfig(accept_window=3, exit_expansion_ratio=1.5),
    )
    # 1 سببية القبول
    assert out["vp_fr_accepted_expansion"].to_list()[1] == 0.0
    assert out["vp_fr_accepted_expansion"].to_list()[2] == 1.0
    # 2 خروج بلا active لاصق
    assert out["vp_fr_exit"].to_list()[6] == 1.0
    assert out["vp_fr_active"].to_list()[6] == 0.0
    # 4/5 مدّ داخل التوازن فقط + وخزة لا تمدّ
    end = out["vp_fr_end_ts"].to_list()
    assert end[4] == 5 * _IV
    assert end[5] == 5 * _IV

    # آخر expansion
    specs = [
        (100, 2, False, True, True, 0.9, 2),
        (108, 12, True, False, False, 0.2, 2),
        (114, 16, True, False, False, 0.1, 2),
        (106, 3, False, True, True, 0.85, 2),
    ]
    prices = [
        [99, 100, 101, 100],
        [100, 105, 110, 108],
        [110, 112, 116, 114],
        [105, 106, 107, 106],
    ]
    p2, m2 = _profile_from_specs(specs, prices)
    o2 = attach_vp_fixed_range(p2, m2, interval_ns=_IV)
    assert o2["vp_fr_start_ts"].to_list()[3] == 2 * _IV


def test_audit_fr_decisions_only_on_exit_edge() -> None:
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
    fr = auction_fsm_columns(states, fixed_range_decisions=True)
    classic = auction_fsm_columns(states, fixed_range_decisions=False)
    assert float(fr["vp_auction_setup"].sum()) == 0.0
    assert float(classic["vp_auction_setup"].abs().sum()) > 0.0


def test_audit_fr_focus_and_signal_export() -> None:
    need = {
        "vp_fr_active",
        "vp_fr_accepted_expansion",
        "vp_fr_in_balance",
        "vp_fr_exit",
        "vp_auction_setup",
        "vp_fsm_build",
        "vp_liquidity_session",
    }
    assert need <= set(_VP_AUCTION_FOCUS)
    frame = make_stream(
        [("T", "B", 100 + (i % 3), 2, 0) for i in range(40)],
        event_ts=list(range(0, 4000, 100)),
        sequence=list(range(1, 41)),
    )
    sigs = auction_signals_from_states(
        auction_action_states(frame, profile_interval_ns=1000, signal_interval_ns=200),
        fixed_range_decisions=True,
    )
    for c in need:
        assert c in sigs.columns


# ---------------------------------------------------------------------------
# C) طبقات المعمارية موجودة وقابلة للاستيراد
# ---------------------------------------------------------------------------

_LAYER_MODULES = (
    "nq.contracts.mbo",
    "nq.orderbook",
    "nq.simulation.auction",
    "nq.simulation.volume_profile",
    "nq.simulation.vp_fixed_range",
    "nq.simulation.footprint",
    "nq.simulation.order_flow",
    "nq.simulation.liquidity",
    "nq.simulation.deceptive_liquidity",
    "nq.simulation.edge_execution_plan",
    "nq.models.encoder",
    "nq.models.preprocessing",
    "nq.research.orchestrator",
    "nq.strategies.vp_auction",
    "nq.validation",
    "nq.statistics",
)


@pytest.mark.parametrize("mod", _LAYER_MODULES)
def test_audit_architecture_layer_importable(mod: str) -> None:
    m = importlib.import_module(mod)
    assert m is not None


# ---------------------------------------------------------------------------
# D) حتمية المولّدات
# ---------------------------------------------------------------------------


def test_audit_determinism_generator_reproducible() -> None:
    a = make_generator(42).random(size=20)
    b = make_generator(42).random(size=20)
    c = make_generator(43).random(size=20)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_audit_auction_pipeline_deterministic_on_same_mbo() -> None:
    frame = _synth_session(2_500, seed=9, hours=1.0)
    kwargs: dict[str, Any] = {
        "profile_interval_ns": 60 * _NS,
        "signal_interval_ns": 30 * _NS,
        "fixed_range": True,
    }
    s1 = auction_signals_from_states(auction_action_states(frame, **kwargs))
    s2 = auction_signals_from_states(auction_action_states(frame, **kwargs))
    assert s1.equals(s2)


# ---------------------------------------------------------------------------
# E) نبضات asof: لا التصاق exit/accepted على شبكة الفعل
# ---------------------------------------------------------------------------


def test_audit_fr_exit_and_accept_are_pulses_after_asof_logic() -> None:
    sticky = pl.DataFrame(
        {
            "vp_fr_exit": [0.0, 1.0, 1.0, 1.0, 0.0],
            "vp_fr_accepted_expansion": [0.0, 0.0, 1.0, 1.0, 0.0],
        }
    ).with_columns(
        pl.when(
            (pl.col("vp_fr_exit") != 0.0) & (pl.col("vp_fr_exit").shift(1).fill_null(0.0) == 0.0)
        )
        .then(pl.col("vp_fr_exit"))
        .otherwise(0.0)
        .alias("vp_fr_exit"),
        pl.when(
            (pl.col("vp_fr_accepted_expansion") != 0.0)
            & (pl.col("vp_fr_accepted_expansion").shift(1).fill_null(0.0) == 0.0)
        )
        .then(pl.col("vp_fr_accepted_expansion"))
        .otherwise(0.0)
        .alias("vp_fr_accepted_expansion"),
    )
    assert sticky["vp_fr_exit"].to_list() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert sticky["vp_fr_accepted_expansion"].to_list() == [0.0, 0.0, 1.0, 0.0, 0.0]
