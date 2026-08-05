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

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.temporal import EVENT_TS, SEQUENCE
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
_MISSING = np.int64(np.iinfo(np.int64).min)


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """نتيجة إعادة البناء: سلسلة top-of-book، الحالة النهائية، وتقرير السلامة."""

    top_of_book: pl.DataFrame
    book: OrderBook
    integrity: IntegrityReport


def _empty_tob() -> pl.DataFrame:
    return pl.DataFrame(schema=_TOB_SCHEMA)


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
        tob = pl.DataFrame(
            {
                EVENT_TS: frame[EVENT_TS],
                SEQUENCE: frame[SEQUENCE],
                "best_bid": pl.Series("best_bid", bb_price).replace(_MISSING, None),
                "bid_size": pl.Series("bid_size", bb_size).replace(_MISSING, None),
                "best_ask": pl.Series("best_ask", ba_price).replace(_MISSING, None),
                "ask_size": pl.Series("ask_size", ba_size).replace(_MISSING, None),
            }
        )
        crossed = tob.filter(
            pl.col("best_bid").is_not_null()
            & pl.col("best_ask").is_not_null()
            & (pl.col("best_bid") >= pl.col("best_ask"))
        ).height
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
