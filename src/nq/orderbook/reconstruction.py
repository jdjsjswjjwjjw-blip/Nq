"""إعادة بناء دفتر الأوامر من MBO (Order Book Reconstruction).

تُعالَج الأحداث بالترتيب السببي الصارم ``(event_ts, sequence)``، فتكون حالة
الدفتر عند أي زمن ``t`` دالةً في الأحداث حتى ``t`` فقط (سببية تامة، بلا تسريب).

المخرج الأساسي للطبقات اللاحقة هو سلسلة **top-of-book** الزمنية: أفضل طلب/عرض
وحجمهما بعد كل حدث.

ملاحظة أداء/ذاكرة: الأعمدة النصية لا تُمادَّ كلها دفعة واحدة إلى ``list[str]``
(كان ذلك يضاعف الذاكرة على أيام ~30–60M حدث ويساهم في قتل workers تحت التوازي).
المعالجة تتم على شرائح؛ الأعمدة الرقمية تُقرأ كـ NumPy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS, SEQUENCE
from nq.core.time import assert_sorted_causal
from nq.orderbook.book import OrderBook
from nq.orderbook.integrity import IntegrityReport, check_integrity
from nq.research.progress import ProgressLike

_TOB_SCHEMA: dict[str, pl.DataType] = {
    EVENT_TS: pl.Int64(),
    SEQUENCE: pl.UInt64(),
    "best_bid": pl.Int64(),
    "bid_size": pl.Int64(),
    "best_ask": pl.Int64(),
    "ask_size": pl.Int64(),
}

# شريحة صغيرة بما يكفي لإبقاء قوائم action/side في حدود معقولة تحت التوازي.
_RECONSTRUCT_CHUNK = 250_000
# لا يُستخدم كسعر حقيقي في MBO (الأسعار بنقطة ثابتة موجبة كبيرة).
_MISSING_INT = int(np.iinfo(np.int64).min)
_MISSING = np.int64(_MISSING_INT)


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """نتيجة إعادة البناء: سلسلة top-of-book، الحالة النهائية، وتقرير السلامة."""

    top_of_book: pl.DataFrame
    book: OrderBook
    integrity: IntegrityReport


def _empty_tob() -> pl.DataFrame:
    return pl.DataFrame(schema=_TOB_SCHEMA)


def _top_of_book_frame(
    frame: pl.DataFrame,
    bb_price: npt.NDArray[np.int64],
    bb_size: npt.NDArray[np.int64],
    ba_price: npt.NDArray[np.int64],
    ba_size: npt.NDArray[np.int64],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            EVENT_TS: frame[EVENT_TS],
            SEQUENCE: frame[SEQUENCE],
            "best_bid": pl.Series("best_bid", bb_price).replace(_MISSING_INT, None),
            "bid_size": pl.Series("bid_size", bb_size).replace(_MISSING_INT, None),
            "best_ask": pl.Series("best_ask", ba_price).replace(_MISSING_INT, None),
            "ask_size": pl.Series("ask_size", ba_size).replace(_MISSING_INT, None),
        }
    )


def _crossed_book_events(tob: pl.DataFrame) -> int:
    return tob.filter(
        pl.col("best_bid").is_not_null()
        & pl.col("best_ask").is_not_null()
        & (pl.col("best_bid") >= pl.col("best_ask"))
    ).height


def _apply_chunk(
    book: OrderBook,
    chunk: pl.DataFrame,
    *,
    bb_price: npt.NDArray[np.int64] | None,
    bb_size: npt.NDArray[np.int64] | None,
    ba_price: npt.NDArray[np.int64] | None,
    ba_size: npt.NDArray[np.int64] | None,
    offset: int,
    progress: ProgressLike | None,
    progress_label: str,
    total: int,
) -> None:
    """يعالج شريحة واحدة — يمادّ النصوص لهذه الشريحة فقط."""
    actions: list[str] = chunk["action"].cast(pl.Utf8).to_list()
    sides: list[str] = chunk["side"].cast(pl.Utf8).to_list()
    prices = chunk["price"].to_numpy()
    sizes = chunk["size"].to_numpy()
    order_ids = chunk["order_id"].to_numpy()
    apply = book.apply
    best_bid = book.best_bid
    best_ask = book.best_ask
    n_chunk = len(actions)
    record = bb_price is not None
    for i in range(n_chunk):
        apply(
            actions[i],
            sides[i],
            int(prices[i]),
            int(sizes[i]),
            int(order_ids[i]),
        )
        if record:
            idx = offset + i
            bid = best_bid()
            ask = best_ask()
            if bid is None:
                bb_price[idx] = _MISSING  # type: ignore[index]
                bb_size[idx] = _MISSING  # type: ignore[index]
            else:
                bb_price[idx] = bid[0]  # type: ignore[index]
                bb_size[idx] = bid[1]  # type: ignore[index]
            if ask is None:
                ba_price[idx] = _MISSING  # type: ignore[index]
                ba_size[idx] = _MISSING  # type: ignore[index]
            else:
                ba_price[idx] = ask[0]  # type: ignore[index]
                ba_size[idx] = ask[1]  # type: ignore[index]
        done = offset + i + 1
        if progress is not None and (done % 500 == 0 or done == total):
            progress.heartbeat(done, total, label=progress_label)


def reconstruct(
    frame: pl.DataFrame,
    *,
    record_top_of_book: bool = True,
    progress: ProgressLike | None = None,
    progress_label: str = "reconstruct",
) -> ReconstructionResult:
    """يُعيد بناء دفتر أوامر أداة واحدة من أحداث MBO.

    يفترض أن الإطار لأداة واحدة (``instrument_id`` وحيد)؛ استخدم
    ``reconstruct_by_instrument`` لتعدّد الأدوات. يتحقق من الترتيب السببي أولًا.
    """
    n_instruments = frame["instrument_id"].n_unique() if frame.height else 0
    if n_instruments > 1:
        raise ValueError("reconstruct expects a single instrument; use reconstruct_by_instrument.")
    assert_sorted_causal(frame)

    book = OrderBook()
    base_integrity = check_integrity(frame)

    if frame.height == 0:
        return ReconstructionResult(_empty_tob(), book, base_integrity)

    n = frame.height
    if progress is not None:
        progress.op(f"{progress_label}: إعادة بناء دفتر · أحداث={n:,}")

    bb_price: npt.NDArray[np.int64] | None = None
    bb_size: npt.NDArray[np.int64] | None = None
    ba_price: npt.NDArray[np.int64] | None = None
    ba_size: npt.NDArray[np.int64] | None = None
    if record_top_of_book:
        bb_price = np.full(n, _MISSING, dtype=np.int64)
        bb_size = np.full(n, _MISSING, dtype=np.int64)
        ba_price = np.full(n, _MISSING, dtype=np.int64)
        ba_size = np.full(n, _MISSING, dtype=np.int64)

    for start in range(0, n, _RECONSTRUCT_CHUNK):
        chunk = frame.slice(start, min(_RECONSTRUCT_CHUNK, n - start))
        _apply_chunk(
            book,
            chunk,
            bb_price=bb_price,
            bb_size=bb_size,
            ba_price=ba_price,
            ba_size=ba_size,
            offset=start,
            progress=progress,
            progress_label=progress_label,
            total=n,
        )

    crossed = 0
    if record_top_of_book:
        assert bb_price is not None and bb_size is not None
        assert ba_price is not None and ba_size is not None
        tob = _top_of_book_frame(frame, bb_price, bb_size, ba_price, ba_size)
        crossed = _crossed_book_events(tob)
    else:
        tob = _empty_tob()

    if progress is not None:
        progress.op(f"{progress_label}: انتهى · tob={tob.height:,}")

    integrity = replace(
        base_integrity,
        unknown_order_refs=book.unknown_order_refs,
        crossed_book_events=crossed,
    )
    return ReconstructionResult(tob, book, integrity)


def scan_book_tob_and_depth(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    interval_ns_list: Sequence[int],
    n_levels: int = 5,
    record_top_of_book: bool = True,
    progress: ProgressLike | None = None,
    progress_label: str = "book_scan",
) -> tuple[ReconstructionResult, dict[int, pl.DataFrame]]:
    """يبني TOB ولقطات إغلاق العمق في مرور سببي واحد على دفتر واحد."""
    from nq.simulation.common import BUCKET_END, BUCKET_START  # noqa: PLC0415
    from nq.simulation.depth_lifecycle import (  # noqa: PLC0415
        _depth_bar_empty_schema,
        snapshot_to_row,
    )

    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    intervals = tuple(dict.fromkeys(int(value) for value in interval_ns_list))
    if not intervals:
        raise ValueError("interval_ns_list must be non-empty")
    for interval_ns in intervals:
        if interval_ns < 1:
            raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")

    n_instruments = frame["instrument_id"].n_unique() if frame.height else 0
    if n_instruments > 1:
        raise ValueError(
            "scan_book_tob_and_depth expects a single instrument; "
            "split the frame by instrument first."
        )
    assert_sorted_causal(frame)

    book = OrderBook()
    base_integrity = check_integrity(frame)
    empty_schema = _depth_bar_empty_schema(n_levels=n_levels)
    if frame.height == 0:
        result = ReconstructionResult(_empty_tob(), book, base_integrity)
        return result, {interval: pl.DataFrame(schema=empty_schema) for interval in intervals}

    n = frame.height
    if progress is not None:
        iv_text = ",".join(str(interval) for interval in intervals)
        progress.op(f"{progress_label}: أحداث={n:,} · فواصل=[{iv_text}] · L1–L{n_levels}")

    bb_price: npt.NDArray[np.int64] | None = None
    bb_size: npt.NDArray[np.int64] | None = None
    ba_price: npt.NDArray[np.int64] | None = None
    ba_size: npt.NDArray[np.int64] | None = None
    if record_top_of_book:
        bb_price = np.full(n, _MISSING, dtype=np.int64)
        bb_size = np.full(n, _MISSING, dtype=np.int64)
        ba_price = np.full(n, _MISSING, dtype=np.int64)
        ba_size = np.full(n, _MISSING, dtype=np.int64)

    rows_by_interval: dict[int, list[dict[str, float | int | None]]] = {
        interval: [] for interval in intervals
    }
    current_bucket: dict[int, int | None] = dict.fromkeys(intervals)
    last_event_in_bucket: dict[int, int] = dict.fromkeys(intervals, -1)

    def _emit(interval_ns: int, bucket_start: int) -> None:
        bucket_end = bucket_start + interval_ns
        snap = book.snapshot(n_levels, availability_ts=bucket_end)
        row = snapshot_to_row(snap, event_ts=last_event_in_bucket[interval_ns])
        row[AVAILABILITY_TS] = bucket_end
        row[BUCKET_START] = bucket_start
        row[BUCKET_END] = bucket_end
        rows_by_interval[interval_ns].append(row)

    apply = book.apply
    best_bid = book.best_bid
    best_ask = book.best_ask
    for start in range(0, n, _RECONSTRUCT_CHUNK):
        chunk = frame.slice(start, min(_RECONSTRUCT_CHUNK, n - start))
        actions: list[str] = chunk["action"].cast(pl.Utf8).to_list()
        sides: list[str] = chunk["side"].cast(pl.Utf8).to_list()
        prices = chunk["price"].to_numpy()
        sizes = chunk["size"].to_numpy()
        order_ids = chunk["order_id"].to_numpy()
        event_ts = chunk[EVENT_TS].to_numpy()

        for local_i, action in enumerate(actions):
            idx = start + local_i
            ts = int(event_ts[local_i])
            for interval_ns in intervals:
                bucket = (ts // interval_ns) * interval_ns
                current = current_bucket[interval_ns]
                if current is None:
                    current_bucket[interval_ns] = bucket
                elif bucket != current:
                    _emit(interval_ns, current)
                    current_bucket[interval_ns] = bucket

            apply(
                action,
                sides[local_i],
                int(prices[local_i]),
                int(sizes[local_i]),
                int(order_ids[local_i]),
            )
            for interval_ns in intervals:
                last_event_in_bucket[interval_ns] = ts

            if record_top_of_book:
                bid = best_bid()
                ask = best_ask()
                if bid is not None:
                    bb_price[idx], bb_size[idx] = bid  # type: ignore[index]
                if ask is not None:
                    ba_price[idx], ba_size[idx] = ask  # type: ignore[index]

            done = idx + 1
            if progress is not None and (done % 500 == 0 or done == n):
                progress.heartbeat(done, n, label=progress_label)

    for interval_ns in intervals:
        current = current_bucket[interval_ns]
        if current is not None:
            _emit(interval_ns, current)

    crossed = 0
    if record_top_of_book:
        assert bb_price is not None and bb_size is not None
        assert ba_price is not None and ba_size is not None
        tob = _top_of_book_frame(frame, bb_price, bb_size, ba_price, ba_size)
        crossed = _crossed_book_events(tob)
    else:
        tob = _empty_tob()
    integrity = replace(
        base_integrity,
        unknown_order_refs=book.unknown_order_refs,
        crossed_book_events=crossed,
    )
    result = ReconstructionResult(tob, book, integrity)

    depth_by_interval: dict[int, pl.DataFrame] = {}
    for interval_ns in intervals:
        rows = rows_by_interval[interval_ns]
        depth_by_interval[interval_ns] = (
            pl.DataFrame(rows).sort(AVAILABILITY_TS) if rows else pl.DataFrame(schema=empty_schema)
        )
    if progress is not None:
        progress.op(f"{progress_label}: انتهى · tob={tob.height:,}")
    return result, depth_by_interval


def reconstruct_by_instrument(
    frame: pl.DataFrame,
    *,
    record_top_of_book: bool = True,
    progress: ProgressLike | None = None,
) -> dict[int, ReconstructionResult]:
    """يُعيد البناء لكل أداة على حدة ويُعيد قاموسًا ``instrument_id -> نتيجة``."""
    results: dict[int, ReconstructionResult] = {}
    if frame.height == 0:
        return results
    groups = list(frame.group_by(["instrument_id"], maintain_order=True))
    n = len(groups)
    for i, ((instrument_id,), group) in enumerate(groups, start=1):
        if progress is not None:
            progress.op(f"reconstruct_by_instrument [{i}/{n}] id={instrument_id}")
        results[int(instrument_id)] = reconstruct(
            group,
            record_top_of_book=record_top_of_book,
            progress=progress,
            progress_label=f"reconstruct:{instrument_id}",
        )
    return results
