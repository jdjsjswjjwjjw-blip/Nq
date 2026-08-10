"""اختبارات حالة دفتر الأوامر (OrderBook) — دلالات Databento MBO."""

from __future__ import annotations

from nq.orderbook import OrderBook


def _book() -> OrderBook:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("A", "B", 99, 3, 2)
    book.apply("A", "A", 102, 4, 3)
    return book


def test_best_bid_ask_and_spread() -> None:
    book = _book()
    assert book.best_bid() == (100, 5)
    assert book.best_ask() == (102, 4)
    assert book.spread() == 2


def test_add_aggregates_same_level() -> None:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("A", "B", 100, 2, 2)
    assert book.best_bid() == (100, 7)


def test_cancel_full_removes_level() -> None:
    book = _book()
    book.apply("C", "N", 0, 5, 1)  # cancel full size of order 1
    assert book.best_bid() == (99, 3)


def test_cancel_partial_reduces_size() -> None:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("C", "N", 0, 2, 1)  # partial cancel
    assert book.best_bid() == (100, 3)
    assert 1 in book.orders


def test_fill_does_not_change_book() -> None:
    """Databento: Fill لا يغيّر الدفتر — الإلغاء المصاحب هو من يحدّث."""
    book = OrderBook()
    book.apply("A", "A", 102, 4, 1)
    book.apply("F", "N", 0, 1, 1)
    assert book.best_ask() == (102, 4)
    book.apply("C", "N", 0, 1, 1)  # cancel accompanying fill
    assert book.best_ask() == (102, 3)


def test_modify_moves_price_level() -> None:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("M", "B", 101, 5, 1)
    assert book.best_bid() == (101, 5)
    assert 100 not in book.bids


def test_clear_empties_book() -> None:
    book = _book()
    book.apply("R", "N", 0, 0, 0)
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.spread() is None


def test_unknown_reference_counted() -> None:
    book = OrderBook()
    book.apply("C", "N", 0, 0, 999)
    book.apply("F", "N", 0, 1, 888)
    assert book.unknown_order_refs == 2


def test_modify_unknown_does_not_create_ghost() -> None:
    book = OrderBook()
    book.apply("M", "A", 105, 4, 777)
    assert book.unknown_order_refs == 1
    assert book.best_ask() is None
    assert 777 not in book.orders


def test_duplicate_add_replaces_not_doubles() -> None:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("A", "B", 101, 3, 1)  # same order_id
    assert book.duplicate_add_refs == 1
    assert book.best_bid() == (101, 3)
    assert 100 not in book.bids


def test_trade_and_none_are_noops() -> None:
    book = _book()
    before = (book.best_bid(), book.best_ask())
    book.apply("T", "B", 100, 2, 0)
    book.apply("N", "N", 0, 0, 0)
    assert (book.best_bid(), book.best_ask()) == before


def test_same_sequence_fill_cancel_trade_safe() -> None:
    """Fill+Cancel+Trade بنفس sequence لا تُفسد الدفتر."""
    book = OrderBook()
    book.apply("A", "B", 100, 10, 1)
    book.apply("F", "N", 0, 4, 1)  # no-op on book
    book.apply("C", "N", 0, 4, 1)  # reduce by 4
    book.apply("T", "A", 100, 4, 0)
    assert book.best_bid() == (100, 6)
