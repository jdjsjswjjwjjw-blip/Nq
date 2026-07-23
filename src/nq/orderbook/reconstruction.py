"""إعادة بناء دفتر الأوامر من MBO (Order Book Reconstruction).

تُعالَج الأحداث بالترتيب السببي الصارم ``(event_ts, sequence)``، فتكون حالة
الدفتر عند أي زمن ``t`` دالةً في الأحداث حتى ``t`` فقط (سببية تامة، بلا تسريب).

المخرج الأساسي للطبقات اللاحقة هو سلسلة **top-of-book** الزمنية: أفضل طلب/عرض
وحجمهما بعد كل حدث.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import polars as pl

from nq.contracts.instruments import require_single_contract_identity
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


class StrictReconstructionError(ValueError):
    """Raised when strict research/backtest reconstruction rejects an MBO stream."""


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
) -> tuple[
    list[int | None],
    list[int | None],
    list[int | None],
    list[int | None],
    int,
]:
    """يعالج الأحداث ويسجّل top-of-book بعد كل حدث."""
    apply = book.apply
    bids = book.bids
    asks = book.asks
    bb_price: list[int | None] = []
    bb_size: list[int | None] = []
    ba_price: list[int | None] = []
    ba_size: list[int | None] = []
    crossed = 0
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
        if bb_price[-1] is not None and ba_price[-1] is not None and bb_price[-1] >= ba_price[-1]:
            crossed += 1
    return bb_price, bb_size, ba_price, ba_size, crossed


def _plain_loop(
    book: OrderBook,
    actions: list[str],
    sides: list[str],
    prices: list[int],
    sizes: list[int],
    order_ids: list[int],
) -> int:
    apply = book.apply
    crossed = 0
    for action, side, price, size, order_id in zip(
        actions, sides, prices, sizes, order_ids, strict=True
    ):
        apply(action, side, price, size, order_id)
        bid = book.best_bid()
        ask = book.best_ask()
        if bid is not None and ask is not None and bid[0] >= ask[0]:
            crossed += 1
    return crossed


def _raise_if_strict_rejected(integrity: IntegrityReport) -> None:
    if integrity.strict_ok:
        return
    failures = integrity.strict_failures()
    detail = ", ".join(f"{name}={count}" for name, count in failures.items())
    raise StrictReconstructionError(f"strict MBO reconstruction rejected stream: {detail}")


def _raise_if_contract_identity_changes(frame: pl.DataFrame) -> None:
    try:
        require_single_contract_identity(frame, context="strict reconstruction")
    except ValueError as exc:
        raise StrictReconstructionError(str(exc)) from exc


def reconstruct(
    frame: pl.DataFrame,
    *,
    record_top_of_book: bool = True,
    strict: bool = False,
) -> ReconstructionResult:
    """يُعيد بناء دفتر أوامر أداة واحدة من أحداث MBO.

    يفترض أن الإطار لأداة واحدة (``instrument_id`` وحيد)؛ استخدم
    ``reconstruct_by_instrument`` لتعدّد الأدوات. يتحقق من الترتيب السببي أولًا.
    """
    n_instruments = frame["instrument_id"].n_unique() if frame.height else 0
    if n_instruments > 1:
        raise ValueError("reconstruct expects a single instrument; use reconstruct_by_instrument.")
    assert_sorted_causal(frame)
    if strict:
        _raise_if_contract_identity_changes(frame)

    book = OrderBook()
    base_integrity = check_integrity(frame)

    if frame.height == 0:
        if strict:
            _raise_if_strict_rejected(base_integrity)
        return ReconstructionResult(_empty_tob(), book, base_integrity)

    actions: list[str] = frame["action"].cast(pl.Utf8).to_list()
    sides: list[str] = frame["side"].cast(pl.Utf8).to_list()
    prices: list[int] = frame["price"].to_list()
    sizes: list[int] = frame["size"].to_list()
    order_ids: list[int] = frame["order_id"].to_list()

    crossed = 0
    if record_top_of_book:
        bb_price, bb_size, ba_price, ba_size, crossed = _record_loop(
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
    else:
        crossed = _plain_loop(book, actions, sides, prices, sizes, order_ids)
        tob = _empty_tob()

    integrity = replace(
        base_integrity,
        unknown_order_refs=book.unknown_order_refs,
        crossed_book_events=crossed,
    )
    if strict:
        _raise_if_strict_rejected(integrity)
    return ReconstructionResult(tob, book, integrity)


def reconstruct_by_instrument(
    frame: pl.DataFrame,
    *,
    record_top_of_book: bool = True,
    strict: bool = False,
) -> dict[int, ReconstructionResult]:
    """يُعيد البناء لكل أداة على حدة ويُعيد قاموسًا ``instrument_id -> نتيجة``."""
    results: dict[int, ReconstructionResult] = {}
    if frame.height == 0:
        return results
    for (instrument_id,), group in frame.group_by(["instrument_id"], maintain_order=True):
        results[int(instrument_id)] = reconstruct(
            group,
            record_top_of_book=record_top_of_book,
            strict=strict,
        )
    return results
