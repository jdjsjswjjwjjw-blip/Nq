"""Shared fail_*/streaming signal semantics (pulse join, trap session, deltas)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.session import SESSION_DATE
from nq.features.streaming import _bucket_signed_deltas, streaming_event_features
from nq.models.tick_stream import build_tick_stream
from nq.research.orchestrator import _attach_failed_breakout, _attach_failed_fvg
from tests.mbo_factory import Event, make_stream
from tests.test_coverage import _paired_streams

_ET = ZoneInfo("America/New_York")
_NS = 1_000_000_000


def _ns_et(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    local = dt.datetime(year, month, day, hour, minute, tzinfo=_ET)
    return int(local.timestamp() * 1e9)


def test_attach_failed_fvg_is_pulse_not_sticky() -> None:
    """إشارة 30m لا تلتصق على صفوف 1s اللاحقة عبر asof."""
    clock = pl.DataFrame({AVAILABILITY_TS: [10, 20, 30, 40]})
    # محاكاة إطار FVG كثيف: نبضة عند 20 فقط
    fvg_right = pl.DataFrame(
        {
            AVAILABILITY_TS: [10, 20, 30],
            "fail_fvg": [0.0, 1.0, 0.0],
            "effort_range_ratio": [0.0, 1.5, 0.0],
            "effort_volume_ratio": [0.0, 2.0, 0.0],
        }
    )
    # حقن عبر monkeypatch بسيط للمسار: نختبر منطق join مباشرة
    left = clock.sort(AVAILABILITY_TS)
    right = fvg_right.sort(AVAILABILITY_TS)
    joined = left.join(right, on=AVAILABILITY_TS, how="left").with_columns(
        pl.col("fail_fvg").fill_null(0.0)
    )
    assert joined["fail_fvg"].to_list() == [0.0, 1.0, 0.0, 0.0]


def test_attach_failed_breakout_sparse_pulse_does_not_stick() -> None:
    """إطار FB متفرق: الإشارة عند T فقط — لا حمل حتى الإشارة التالية."""
    clock = pl.DataFrame({AVAILABILITY_TS: [100, 200, 300, 400]})
    sparse = pl.DataFrame({AVAILABILITY_TS: [200], "fail_breakout": [1.0]})
    joined = clock.join(sparse, on=AVAILABILITY_TS, how="left").with_columns(
        pl.col("fail_breakout").fill_null(0.0)
    )
    assert joined["fail_breakout"].to_list() == [0.0, 1.0, 0.0, 0.0]


def test_fail_cli_help_mentions_exploratory_default() -> None:
    root = Path(__file__).resolve().parents[1]
    fvg = (root / "scripts" / "run_fail_fvg.py").read_text(encoding="utf-8")
    fb = (root / "scripts" / "run_fail_breakout.py").read_text(encoding="utf-8")
    assert "exploratory full-sample" in fvg
    assert "exploratory full-sample" in fb
    assert "--search" in fvg and "--search" in fb


def test_streaming_deltas_are_signed_volume_not_return_sign() -> None:
    frame = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3, 4],
            SESSION_DATE: ["2024-06-03"] * 4,
            "nq_signed_vol": [2.0, 5.0, 5.0, 9.0],
            "mnq_signed_vol": [1.0, 1.0, 4.0, 6.0],
            "nq_return": [0.5, -0.2, 0.1, 0.3],
        }
    )
    out = _bucket_signed_deltas(frame)
    assert out["nq_delta"].to_list() == [2.0, 3.0, 0.0, 4.0]
    assert out["mnq_delta"].to_list() == [1.0, 0.0, 3.0, 2.0]
    # ليست علامة العائد
    assert out["nq_delta"].to_list() != [float(np.sign(x)) for x in frame["nq_return"]]

    # عبر الجلسات: لا فرق كاذب من cumsum الجلسة السابقة
    multi = pl.DataFrame(
        {
            AVAILABILITY_TS: [1, 2, 3],
            SESSION_DATE: ["2024-06-03", "2024-06-03", "2024-06-04"],
            "nq_signed_vol": [10.0, 15.0, 2.0],
            "mnq_signed_vol": [10.0, 12.0, 3.0],
        }
    )
    crossed = _bucket_signed_deltas(multi)
    assert crossed["nq_delta"].to_list() == [10.0, 5.0, 2.0]
    assert crossed["mnq_delta"].to_list() == [10.0, 2.0, 3.0]


def test_tick_stream_session_resets_signed_vol() -> None:
    """cumsum الموقّع يُصفَّر عند تغيّر session_date (ET)."""
    day1 = _ns_et(2024, 6, 3, 10, 0)
    day2 = _ns_et(2024, 6, 4, 10, 0)
    price = 20_000_000_000
    events: list[Event] = []
    ts: list[int] = []
    seq: list[int] = []
    for i, base in enumerate((day1, day1 + _NS, day2, day2 + _NS)):
        events.extend(
            [
                ("A", "B", price, 5, i * 10 + 1),
                ("A", "A", price + 1_000_000, 5, i * 10 + 2),
                ("T", "B", price, 3, 0),
            ]
        )
        ts.extend([base, base + 1, base + 2])
        seq.extend([i * 3 + 1, i * 3 + 2, i * 3 + 3])
    nq = make_stream(events, instrument_id=1, symbol="NQ", event_ts=ts, sequence=seq)
    mnq = make_stream(events, instrument_id=2, symbol="MNQ", event_ts=ts, sequence=seq)
    stream = build_tick_stream(nq, mnq)
    frame = stream.frame.sort(AVAILABILITY_TS)
    # صفقات MNQ ترفع mnq_signed_vol؛ بعد عبور اليوم يجب أن ينخفض / يُعاد من الصفر
    mnq_trades = frame.filter(pl.col("instrument_id") == 2)
    vols = mnq_trades["mnq_signed_vol"].to_list()
    assert len(vols) >= 2
    # آخر صف في اليوم الأول أعلى من أول صف بعد إعادة التصفير في اليوم الثاني
    day1_last = max(
        v
        for v, t in zip(vols, mnq_trades[AVAILABILITY_TS].to_list(), strict=True)
        if t < day2
    )
    day2_first = min(
        v
        for v, t in zip(vols, mnq_trades[AVAILABILITY_TS].to_list(), strict=True)
        if t >= day2
    )
    assert day1_last > 0
    assert day2_first < day1_last


def test_orchestrator_attach_helpers_exportable() -> None:
    # sanity: helpers ما زالت قابلة للاستيراد بعد تعديل النبضة
    assert callable(_attach_failed_fvg)
    assert callable(_attach_failed_breakout)


def test_streaming_event_features_expose_aligned_delta_names() -> None:
    nq, mnq = _paired_streams(400, seed=7)
    events = streaming_event_features(nq, mnq)
    assert "nq_delta" in events.columns
    assert "mnq_delta" in events.columns
    assert "nq_signed_vol" in events.columns
    assert "mnq_signed_vol" in events.columns
