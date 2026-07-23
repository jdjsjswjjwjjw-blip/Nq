"""اختبارات حالة دفتر الأوامر (OrderBook)."""

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


def test_cancel_reduces_and_removes_level() -> None:
    book = _book()
    book.apply("C", "N", 0, 0, 1)  # cancel order 1 (bid 100 x5)
    assert book.best_bid() == (99, 3)


def test_cancel_can_partially_reduce_resting_order() -> None:
    book = OrderBook()
    book.apply("A", "B", 100, 5, 1)
    book.apply("C", "N", 0, 2, 1)
    assert book.best_bid() == (100, 3)
    assert book.orders[1] == (True, 100, 3)


def test_fill_does_not_mutate_databento_resting_book_state() -> None:
    book = OrderBook()
    book.apply("A", "A", 102, 4, 1)
    book.apply("F", "N", 0, 1, 1)
    assert book.best_ask() == (102, 4)


def test_cancel_partial_then_full() -> None:
    book = OrderBook()
    book.apply("A", "A", 102, 4, 1)
    book.apply("C", "N", 0, 1, 1)  # partial cancel 1
    assert book.best_ask() == (102, 3)
    book.apply("C", "N", 0, 3, 1)  # remaining canceled
    assert book.best_ask() is None


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
    assert book.unknown_order_refs == 1


def test_modify_unknown_order_counted_without_inventing_liquidity() -> None:
    book = OrderBook()
    book.apply("M", "A", 105, 4, 777)  # modify an order we never saw
    assert book.unknown_order_refs == 1
    assert book.best_ask() is None


def test_trade_and_none_are_noops() -> None:
    book = _book()
    before = (book.best_bid(), book.best_ask())
    book.apply("T", "B", 100, 2, 0)
    book.apply("N", "N", 0, 0, 0)
    assert (book.best_bid(), book.best_ask()) == before
