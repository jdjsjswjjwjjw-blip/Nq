"""محرّك الميزات اللحظية (Streaming / State Machine) من MBO.

يحدّث الحالة من **كل حدث** بنفس ترتيب السوق الحي (دفتر + VP + أنظمة + trap)،
ثم يُنتج إطار بحث مع ``availability_ts = event_ts`` (point-in-time).

الفرق عن الـ batch: لا انتظار لاكتمال نافذة زمنية قبل تحديث الإشارة؛ العيّنة
البحثية تأخذ **آخر حالة معروفة** داخل كل ``interval_ns`` (سببي).
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.session import add_session_columns
from nq.models.tick_stream import TICK_FEATURE_NAMES, build_tick_stream

_REF_PRICE: Final = 20_000_000_000.0

STREAMING_SIGNAL_COLUMNS: Final[tuple[str, ...]] = (
    "trap_setup",
    "mnq_delta",
    "nq_return",
    "mnq_return",
    "phase_balance",
    "phase_expansion",
    "in_value_area",
    "near_vah",
    "near_val",
    "poc_dist_norm",
    "nq_spread_norm",
    "stream_vah_bid_liq",
    "stream_vah_ask_liq",
    "stream_val_bid_liq",
    "stream_val_ask_liq",
    "stream_trail_bid_liq",
    "stream_trail_ask_liq",
    "depth_cum_bid",
    "depth_cum_ask",
    "depth_imbalance",
    "depth_trail_bid",
    "depth_trail_ask",
)


def streaming_event_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
    progress: object | None = None,
) -> pl.DataFrame:
    """إطار حدث-بحدث من آلة الحالة (متاح عند ``event_ts``)."""
    if progress is not None:
        progress.op("streaming: استدعاء build_tick_stream")  # type: ignore[union-attr]
    tick = build_tick_stream(
        nq,
        mnq,
        nq_instrument_id=nq_instrument_id,
        mnq_instrument_id=mnq_instrument_id,
        progress=progress,
    )
    raw = tick.frame
    if raw.height == 0:
        return raw

    if progress is not None:
        progress.op(f"streaming: تحويل أسعار/عوائد من {raw.height:,} حدث")  # type: ignore[union-attr]
    ref = _REF_PRICE
    return raw.with_columns(
        (pl.col("nq_mid_norm") * ref).alias("nq_close"),
        (pl.col("mnq_mid_norm") * ref).alias("mnq_close"),
        (pl.col("nq_best_bid_norm") * ref).alias("nq_bid"),
        (pl.col("nq_best_ask_norm") * ref).alias("nq_ask"),
        (pl.col("mnq_best_bid_norm") * ref).alias("mnq_bid"),
        (pl.col("mnq_best_ask_norm") * ref).alias("mnq_ask"),
        pl.col("mnq_signed_vol").alias("mnq_delta"),
        pl.col("nq_vah_bid_liq_log").alias("stream_vah_bid_liq"),
        pl.col("nq_vah_ask_liq_log").alias("stream_vah_ask_liq"),
        pl.col("nq_val_bid_liq_log").alias("stream_val_bid_liq"),
        pl.col("nq_val_ask_liq_log").alias("stream_val_ask_liq"),
        pl.col("nq_trail_bid_liq_log").alias("stream_trail_bid_liq"),
        pl.col("nq_trail_ask_liq_log").alias("stream_trail_ask_liq"),
    ).with_columns(
        pl.col("nq_close").diff().fill_null(0.0).alias("nq_return"),
        pl.col("mnq_close").diff().fill_null(0.0).alias("mnq_return"),
        pl.col("nq_close").diff().fill_null(0.0).sign().alias("nq_delta"),
    )


def sample_streaming_to_interval(
    events: pl.DataFrame,
    *,
    interval_ns: int,
) -> pl.DataFrame:
    """آخر حالة لحظية داخل كل فاصل.

    الحالة تُحدَّث حدثًا بحدث؛ عند أخذ عيّنة بحثية تُثبَّت
    ``availability_ts = bucket_end`` حتى تُحاذى مع M9/الألفا على ساعة موحّدة،
    بينما المحتوى نفسه هو آخر حالة سببية داخل الفاصل.
    """
    if interval_ns < 1:
        raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
    if events.height == 0:
        return events
    if AVAILABILITY_TS not in events.columns:
        raise ValueError(f"events require {AVAILABILITY_TS}")

    work = events.sort(AVAILABILITY_TS).with_columns(
        (pl.col(AVAILABILITY_TS) // interval_ns * interval_ns).alias("_stream_bucket")
    )
    sampled = work.group_by("_stream_bucket", maintain_order=True).agg(pl.all().last())
    return (
        sampled.with_columns((pl.col("_stream_bucket") + interval_ns).alias(AVAILABILITY_TS))
        .drop("_stream_bucket")
        .sort(AVAILABILITY_TS)
    )


def build_streaming_research_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
    progress: object | None = None,
) -> pl.DataFrame:
    """يبني إطار البحث من آلة حالة MBO لحظية (بديل الـ batch العريض)."""
    events = streaming_event_features(
        nq,
        mnq,
        nq_instrument_id=nq_instrument_id,
        mnq_instrument_id=mnq_instrument_id,
        progress=progress,
    )
    if events.height == 0:
        return events

    if progress is not None:
        progress.op(  # type: ignore[union-attr]
            f"عيّنة بحثية على interval_ns={interval_ns} من {events.height:,} حدث"
        )
    sampled = sample_streaming_to_interval(events, interval_ns=interval_ns)
    preferred = (
        AVAILABILITY_TS,
        EVENT_TS,
        "nq_close",
        "mnq_close",
        "nq_bid",
        "nq_ask",
        "mnq_bid",
        "mnq_ask",
        "nq_return",
        "mnq_return",
        "nq_delta",
        "mnq_delta",
        "trap_setup",
        "phase_balance",
        "phase_expansion",
        "in_value_area",
        "near_vah",
        "near_val",
        "poc_dist_norm",
        "vah_dist_norm",
        "val_dist_norm",
        "nq_spread_norm",
        "stream_vah_bid_liq",
        "stream_vah_ask_liq",
        "stream_val_bid_liq",
        "stream_val_ask_liq",
        "stream_trail_bid_liq",
        "stream_trail_ask_liq",
        "depth_cum_bid",
        "depth_cum_ask",
        "depth_imbalance",
        "depth_trail_bid",
        "depth_trail_ask",
        "market_phase",
        *TICK_FEATURE_NAMES,
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for col in preferred:
        if col in sampled.columns and col not in seen:
            seen.add(col)
            ordered.append(col)
    frame = sampled.select(ordered)
    if progress is not None:
        progress.op(f"إضافة أعمدة الجلسة — عيّنة={frame.height:,} صف")  # type: ignore[union-attr]
    return add_session_columns(frame, time_col=AVAILABILITY_TS)


__all__ = [
    "STREAMING_SIGNAL_COLUMNS",
    "build_streaming_research_features",
    "sample_streaming_to_interval",
    "streaming_event_features",
]
