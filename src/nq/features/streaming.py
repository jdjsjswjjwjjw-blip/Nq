"""محرّك الميزات اللحظية (Streaming / State Machine) من MBO.

يحدّث الحالة من **كل حدث** بنفس ترتيب السوق الحي (دفتر + VP + أنظمة + trap)،
ثم يُنتج إطار بحث مع ``availability_ts = event_ts`` (point-in-time).

الفرق عن الـ batch: لا انتظار لاكتمال نافذة زمنية قبل تحديث الإشارة؛ العيّنة
البحثية تأخذ **آخر حالة معروفة** داخل كل ``interval_ns`` (سببي).
"""

from __future__ import annotations

from typing import Final, Literal, overload

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.session import SESSION_DATE, add_session_columns
from nq.models.tick_stream import TICK_FEATURE_NAMES, TickStream, build_tick_stream
from nq.research.progress import ProgressLike

_REF_PRICE: Final = 20_000_000_000.0

STREAMING_SIGNAL_COLUMNS: Final[tuple[str, ...]] = (
    "trap_setup",
    "mnq_delta",
    "nq_delta",
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


def _bucket_signed_deltas(frame: pl.DataFrame) -> pl.DataFrame:
    """``nq_delta`` / ``mnq_delta`` = حجم موقّع للفاصل (فرق cumsum الجلسة) — يطابق batch."""
    if frame.height == 0 or SESSION_DATE not in frame.columns:
        return frame
    exprs: list[pl.Expr] = []
    if "mnq_signed_vol" in frame.columns:
        exprs.append(
            pl.col("mnq_signed_vol")
            .diff()
            .over(SESSION_DATE)
            .fill_null(pl.col("mnq_signed_vol"))
            .alias("mnq_delta")
        )
    if "nq_signed_vol" in frame.columns:
        exprs.append(
            pl.col("nq_signed_vol")
            .diff()
            .over(SESSION_DATE)
            .fill_null(pl.col("nq_signed_vol"))
            .alias("nq_delta")
        )
    return frame.with_columns(exprs) if exprs else frame


def _events_from_tick(tick: TickStream, *, progress: ProgressLike | None) -> pl.DataFrame:
    raw = tick.frame
    if raw.height == 0:
        return raw
    if progress is not None:
        progress.op(f"streaming: تحويل أسعار/عوائد من {raw.height:,} حدث")
    ref = _REF_PRICE
    events = raw.with_columns(
        (pl.col("nq_mid_norm") * ref).alias("nq_close"),
        (pl.col("mnq_mid_norm") * ref).alias("mnq_close"),
        (pl.col("nq_best_bid_norm") * ref).alias("nq_bid"),
        (pl.col("nq_best_ask_norm") * ref).alias("nq_ask"),
        (pl.col("mnq_best_bid_norm") * ref).alias("mnq_bid"),
        (pl.col("mnq_best_ask_norm") * ref).alias("mnq_ask"),
        pl.col("nq_vah_bid_liq_log").alias("stream_vah_bid_liq"),
        pl.col("nq_vah_ask_liq_log").alias("stream_vah_ask_liq"),
        pl.col("nq_val_bid_liq_log").alias("stream_val_bid_liq"),
        pl.col("nq_val_ask_liq_log").alias("stream_val_ask_liq"),
        pl.col("nq_trail_bid_liq_log").alias("stream_trail_bid_liq"),
        pl.col("nq_trail_ask_liq_log").alias("stream_trail_ask_liq"),
    ).with_columns(
        pl.col("nq_close").diff().fill_null(0.0).alias("nq_return"),
        pl.col("mnq_close").diff().fill_null(0.0).alias("mnq_return"),
    )
    # على مستوى الحدث: دلتا = مساهمة الحدث (فرق cumsum الجلسة) — ليست sign(return)
    events = add_session_columns(events, time_col=AVAILABILITY_TS)
    return _bucket_signed_deltas(events)


def streaming_event_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """إطار حدث-بحدث من آلة الحالة (متاح عند ``event_ts``)."""
    if progress is not None:
        progress.op("streaming: استدعاء build_tick_stream")
    tick = build_tick_stream(
        nq,
        mnq,
        nq_instrument_id=nq_instrument_id,
        mnq_instrument_id=mnq_instrument_id,
        progress=progress,
    )
    return _events_from_tick(tick, progress=progress)


def sample_streaming_to_interval(
    events: pl.DataFrame,
    *,
    interval_ns: int,
) -> pl.DataFrame:
    """آخر حالة لحظية داخل كل فاصل (``sample_mode=last_state``).

    الحالة تُحدَّث حدثًا بحدث؛ عند أخذ عيّنة بحثية تُثبَّت
    ``availability_ts = bucket_end`` حتى تُحاذى مع M9/الألفا على ساعة موحّدة،
    بينما المحتوى نفسه هو آخر حالة سببية داخل الفاصل.

    تحذير دقة: فاصل كبير (مثل 1s) يطمس تطوّر الميكرو-هيكل داخل الشمعة.
    للبحوث الدقيقة اختر ``interval_ns`` أصغر (انظر ``[streaming]`` في
    ``configs/default.toml``) أو استخدم مسار أحداث العمق / bottom-book
    التي تحتفظ بدقة الحدث داخل الشمعة قبل النشر عند ``bucket_end``.
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


def _assemble_streaming_frame(
    events: pl.DataFrame,
    *,
    interval_ns: int,
    progress: ProgressLike | None,
) -> pl.DataFrame:
    if events.height == 0:
        return events
    if progress is not None:
        progress.op(f"عيّنة بحثية على interval_ns={interval_ns} من {events.height:,} حدث")
    sampled = sample_streaming_to_interval(events, interval_ns=interval_ns)
    # أعد حساب دلتا الفاصل من cumsum الجلسة عند نهاية كل bucket (وليس last(event_delta))
    if SESSION_DATE not in sampled.columns:
        sampled = add_session_columns(sampled, time_col=AVAILABILITY_TS)
    sampled = _bucket_signed_deltas(sampled)
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
        progress.op(f"إضافة أعمدة الجلسة — عيّنة={frame.height:,} صف")
    if SESSION_DATE not in frame.columns:
        return add_session_columns(frame, time_col=AVAILABILITY_TS)
    return frame


@overload
def build_streaming_research_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
    progress: ProgressLike | None = None,
    return_tick: Literal[False] = False,
) -> pl.DataFrame: ...


@overload
def build_streaming_research_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
    progress: ProgressLike | None = None,
    return_tick: Literal[True],
) -> tuple[pl.DataFrame, TickStream]: ...


def build_streaming_research_features(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
    progress: ProgressLike | None = None,
    return_tick: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, TickStream]:
    """يبني إطار البحث من آلة حالة MBO لحظية (بديل الـ batch العريض).

    ``return_tick=True`` يعيد ``TickStream`` لإعادة استخدامه في SSL-tick.
    """
    if progress is not None:
        progress.op("streaming: استدعاء build_tick_stream")
    tick = build_tick_stream(
        nq,
        mnq,
        nq_instrument_id=nq_instrument_id,
        mnq_instrument_id=mnq_instrument_id,
        progress=progress,
    )
    events = _events_from_tick(tick, progress=progress)
    frame = _assemble_streaming_frame(events, interval_ns=interval_ns, progress=progress)
    if return_tick:
        return frame, tick
    return frame


__all__ = [
    "STREAMING_SIGNAL_COLUMNS",
    "build_streaming_research_features",
    "sample_streaming_to_interval",
    "streaming_event_features",
]
