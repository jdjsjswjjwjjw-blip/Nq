"""إعادة بناء دفتر الأوامر من MBO (Order Book Reconstruction).

تُعالَج الأحداث بالترتيب السببي الصارم ``(event_ts, sequence)``، فتكون حالة
الدفتر عند أي زمن ``t`` دالةً في الأحداث حتى ``t`` فقط (سببية تامة، بلا تسريب).

المخرج الأساسي للطبقات اللاحقة هو سلسلة **top-of-book** الزمنية: أفضل طلب/عرض
وحجمهما بعد كل حدث.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import polars as pl

from nq.contracts.temporal import EVENT_TS, SEQUENCE
from nq.core.time import assert_sorted_causal
from nq.orderbook.book import OrderBook
from nq.orderbook.integrity import IntegrityReport, check_integrity

_TOB_SCHEMA: dict[str, pl.DataType] = {
    EVENT_TS: pl.Int64(),
    SEQUENCE: pl.UInt64(),
    "best_bid": pl.Int64(),
    "bid_size": pl.Int64(),
    "best_ask": pl.Int64(),
    "ask_size": pl.Int64(),
}


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """نتيجة إعادة البناء: سلسلة top-of-book، الحالة النهائية، وتقرير السلامة."""

    top_of_book: pl.DataFrame
    book: OrderBook
    integrity: IntegrityReport


def _empty_tob() -> pl.DataFrame:
    return pl.DataFrame(schema=_TOB_SCHEMA)


def _record_loop(
    book: OrderBook,
    actions: list[str],
    sides: list[str],
    prices: list[int],
    sizes: list[int],
    order_ids: list[int],
) -> tuple[list[int | None], list[int | None], list[int | None], list[int | None]]:
    """يعالج الأحداث ويسجّل top-of-book بعد كل حدث."""
    apply = book.apply
    bids = book.bids
    asks = book.asks
    bb_price: list[int | None] = []
    bb_size: list[int | None] = []
    ba_price: list[int | None] = []
    ba_size: list[int | None] = []
    for action, side, price, size, order_id in zip(
        actions, sides, prices, sizes, order_ids, strict=True
    ):
        apply(action, side, price, size, order_id)
        if bids:
            p = max(bids)
            bb_price.append(p)
            bb_size.append(bids[p])
        else:
            bb_price.append(None)
            bb_size.append(None)
        if asks:
            p = min(asks)
            ba_price.append(p)
            ba_size.append(asks[p])
        else:
            ba_price.append(None)
            ba_size.append(None)
    return bb_price, bb_size, ba_price, ba_size


def _plain_loop(
    book: OrderBook,
    actions: list[str],
    sides: list[str],
    prices: list[int],
    sizes: list[int],
    order_ids: list[int],
) -> None:
    apply = book.apply
    for action, side, price, size, order_id in zip(
        actions, sides, prices, sizes, order_ids, strict=True
    ):
        apply(action, side, price, size, order_id)


def reconstruct(
    frame: pl.DataFrame,
    *,
    record_top_of_book: bool = True,
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

    actions: list[str] = frame["action"].cast(pl.Utf8).to_list()
    sides: list[str] = frame["side"].cast(pl.Utf8).to_list()
    prices: list[int] = frame["price"].to_list()
    sizes: list[int] = frame["size"].to_list()
    order_ids: list[int] = frame["order_id"].to_list()

    crossed = 0
    if record_top_of_book:
        bb_price, bb_size, ba_price, ba_size = _record_loop(
            book, actions, sides, prices, sizes, order_ids
        )
        tob = pl.DataFrame(
            {
                EVENT_TS: frame[EVENT_TS],
                SEQUENCE: frame[SEQUENCE],
                "best_bid": pl.Series("best_bid", bb_price, dtype=pl.Int64),
                "bid_size": pl.Series("bid_size", bb_size, dtype=pl.Int64),
                "best_ask": pl.Series("best_ask", ba_price, dtype=pl.Int64),
                "ask_size": pl.Series("ask_size", ba_size, dtype=pl.Int64),
            }
        )
        crossed = tob.filter(
            pl.col("best_bid").is_not_null()
            & pl.col("best_ask").is_not_null()
            & (pl.col("best_bid") >= pl.col("best_ask"))
        ).height
    else:
        _plain_loop(book, actions, sides, prices, sizes, order_ids)
        tob = _empty_tob()

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
) -> dict[int, ReconstructionResult]:
    """يُعيد البناء لكل أداة على حدة ويُعيد قاموسًا ``instrument_id -> نتيجة``."""
    results: dict[int, ReconstructionResult] = {}
    if frame.height == 0:
        return results
    for (instrument_id,), group in frame.group_by(["instrument_id"], maintain_order=True):
        results[int(instrument_id)] = reconstruct(
            group, record_top_of_book=record_top_of_book
        )
    return results
