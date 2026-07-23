"""حالة دفتر الأوامر (Order Book State).

يتتبّع الدفتر لكل جانب (طلب/عرض) الحجم المُجمّع عند كل مستوى سعري، إضافةً إلى
تتبّع كل أمر مفرد عبر ``order_id`` لمعالجة الإلغاء/التعديل/التنفيذ بدقّة.

الأسعار أعداد صحيحة بنقطة ثابتة (fixed-point) وفق عقد MBO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nq.contracts.mbo import MboAction, MboSide

if TYPE_CHECKING:
    from nq.orderbook.depth import DepthSnapshot

_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_MODIFY = MboAction.MODIFY.value
_CLEAR = MboAction.CLEAR.value
_FILL = MboAction.FILL.value
_BID = MboSide.BID.value


class OrderBook:
    """دفتر أوامر قابل للتحديث حدثًا بحدث بترتيب سببي صارم.

    الحالة:

    * ``bids`` / ``asks``: ``dict[price -> aggregated_size]`` لكل جانب.
    * ``orders``: ``dict[order_id -> (is_bid, price, size)]`` لتتبّع الأوامر.
    """

    __slots__ = ("asks", "bids", "orders", "unknown_order_refs")

    def __init__(self) -> None:
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.orders: dict[int, tuple[bool, int, int]] = {}
        self.unknown_order_refs: int = 0

    def clear(self) -> None:
        """يمسح الدفتر بالكامل (book reset)."""
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()

    @staticmethod
    def _reduce(level: dict[int, int], price: int, size: int) -> None:
        remaining = level.get(price, 0) - size
        if remaining > 0:
            level[price] = remaining
        else:
            level.pop(price, None)

    def apply(self, action: str, side: str, price: int, size: int, order_id: int) -> None:
        """يطبّق حدث MBO مفردًا على الحالة.

        ``TRADE`` و ``NONE`` لا يعدّلان الأوامر القائمة (التنفيذ يجري عبر ``FILL``).
        كل مرجع لأمر غير معروف يزيد ``unknown_order_refs``.
        """
        if action == _ADD:
            if order_id in self.orders:
                self.unknown_order_refs += 1
            else:
                is_bid = side == _BID
                self.orders[order_id] = (is_bid, price, size)
                level = self.bids if is_bid else self.asks
                level[price] = level.get(price, 0) + size
            return

        if action == _CANCEL:
            rec = self.orders.get(order_id)
            if rec is None:
                self.unknown_order_refs += 1
                return
            is_bid, p, s = rec
            cancel_size = s if size <= 0 else size
            if cancel_size > s:
                raise ValueError(
                    f"cancel size exceeds resting order size for order_id={order_id}: "
                    f"{cancel_size} > {s}"
                )
            self._reduce(self.bids if is_bid else self.asks, p, cancel_size)
            remaining = s - cancel_size
            if remaining > 0:
                self.orders[order_id] = (is_bid, p, remaining)
            else:
                self.orders.pop(order_id, None)
            return

        if action == _FILL:
            # Databento MBO fill records do not mutate resting book state; the
            # paired cancel/modify record carries the book-size update.
            return

        if action == _MODIFY:
            rec = self.orders.get(order_id)
            if rec is None:
                self.unknown_order_refs += 1
                return
            is_bid, old_price, old_size = rec
            level = self.bids if is_bid else self.asks
            self._reduce(level, old_price, old_size)
            level[price] = level.get(price, 0) + size
            self.orders[order_id] = (is_bid, price, size)
            return

        if action == _CLEAR:
            self.clear()
        # TRADE / NONE: لا تغيير في الأوامر القائمة.

    def best_bid(self) -> tuple[int, int] | None:
        """أفضل طلب ``(price, size)`` أو ``None`` إن كان الجانب فارغًا."""
        if not self.bids:
            return None
        price = max(self.bids)
        return price, self.bids[price]

    def best_ask(self) -> tuple[int, int] | None:
        """أفضل عرض ``(price, size)`` أو ``None`` إن كان الجانب فارغًا."""
        if not self.asks:
            return None
        price = min(self.asks)
        return price, self.asks[price]

    def spread(self) -> int | None:
        """الفارق السعري (best_ask - best_bid) بالنقطة الثابتة، أو ``None``."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    def size_at(self, side: str, price: int) -> int:
        """الحجم المعلّق عند سعر محدد (0 إن لم يوجد مستوى)."""
        book = self.bids if side == _BID else self.asks
        return int(book.get(price, 0))

    def top_n(self, side: str, n: int) -> list[tuple[int, int]]:
        """أفضل ``n`` مستويات ``(price, size)`` مرتّبة من الأفضل للأسوأ."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if side == _BID:
            prices = sorted(self.bids.keys(), reverse=True)[:n]
            return [(p, self.bids[p]) for p in prices]
        prices = sorted(self.asks.keys())[:n]
        return [(p, self.asks[p]) for p in prices]

    def cum_depth(self, side: str, n: int) -> int:
        """مجموع الحجم على أفضل ``n`` مستويات."""
        return int(sum(sz for _, sz in self.top_n(side, n)))

    def depth_imbalance(self, n: int) -> float:
        """اختلال عمق ``(bid_n - ask_n) / (bid_n + ask_n)`` ∈ [-1, 1]."""
        bid_n = self.cum_depth(_BID, n)
        ask_n = self.cum_depth("A", n)
        total = bid_n + ask_n
        if total <= 0:
            return 0.0
        return (bid_n - ask_n) / total

    def trail_liquidity(self) -> tuple[int, int]:
        """سيولة خلف أفضل طلب/عرض ``(trail_bid, trail_ask)``."""
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        trail_bid = (
            sum(sz for p, sz in self.bids.items() if best_bid is not None and p < best_bid[0])
            if best_bid is not None
            else 0
        )
        trail_ask = (
            sum(sz for p, sz in self.asks.items() if best_ask is not None and p > best_ask[0])
            if best_ask is not None
            else 0
        )
        return int(trail_bid), int(trail_ask)

    def snapshot(self, n: int = 5, *, availability_ts: int = 0) -> DepthSnapshot:
        """لقطة عمق سببية من الحالة الحالية (بدون آثار جانبية)."""
        from nq.orderbook.depth import DepthSnapshot  # noqa: PLC0415

        bid = self.best_bid()
        ask = self.best_ask()
        bids = tuple(self.top_n(_BID, n))
        asks = tuple(self.top_n("A", n))
        trail_bid, trail_ask = self.trail_liquidity()
        return DepthSnapshot(
            availability_ts=int(availability_ts),
            best_bid=None if bid is None else bid[0],
            bid_size=0 if bid is None else bid[1],
            best_ask=None if ask is None else ask[0],
            ask_size=0 if ask is None else ask[1],
            bid_levels=bids,
            ask_levels=asks,
            cum_bid=int(sum(sz for _, sz in bids)),
            cum_ask=int(sum(sz for _, sz in asks)),
            imbalance=self.depth_imbalance(n),
            trail_bid=trail_bid,
            trail_ask=trail_ask,
            n_levels=n,
        )
