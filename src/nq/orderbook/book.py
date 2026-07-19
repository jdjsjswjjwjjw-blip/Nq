"""حالة دفتر الأوامر (Order Book State).

يتتبّع الدفتر لكل جانب (طلب/عرض) الحجم المُجمّع عند كل مستوى سعري، إضافةً إلى
تتبّع كل أمر مفرد عبر ``order_id`` لمعالجة الإلغاء/التعديل/التنفيذ بدقّة.

الأسعار أعداد صحيحة بنقطة ثابتة (fixed-point) وفق عقد MBO.
"""

from __future__ import annotations

from nq.contracts.mbo import MboAction, MboSide

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

    def apply(  # noqa: PLR0911 -- dispatch على نوع الحدث؛ العودة المبكرة أوضح
        self, action: str, side: str, price: int, size: int, order_id: int
    ) -> None:
        """يطبّق حدث MBO مفردًا على الحالة.

        ``TRADE`` و ``NONE`` لا يعدّلان الأوامر القائمة (التنفيذ يجري عبر ``FILL``).
        كل مرجع لأمر غير معروف يزيد ``unknown_order_refs``.
        """
        if action == _ADD:
            is_bid = side == _BID
            self.orders[order_id] = (is_bid, price, size)
            level = self.bids if is_bid else self.asks
            level[price] = level.get(price, 0) + size
            return

        if action == _CANCEL:
            rec = self.orders.pop(order_id, None)
            if rec is None:
                self.unknown_order_refs += 1
                return
            is_bid, p, s = rec
            self._reduce(self.bids if is_bid else self.asks, p, s)
            return

        if action == _FILL:
            rec = self.orders.get(order_id)
            if rec is None:
                self.unknown_order_refs += 1
                return
            is_bid, p, s = rec
            self._reduce(self.bids if is_bid else self.asks, p, size)
            remaining = s - size
            if remaining > 0:
                self.orders[order_id] = (is_bid, p, remaining)
            else:
                self.orders.pop(order_id, None)
            return

        if action == _MODIFY:
            rec = self.orders.get(order_id)
            if rec is None:
                self.unknown_order_refs += 1
                is_bid = side == _BID
                self.orders[order_id] = (is_bid, price, size)
                level = self.bids if is_bid else self.asks
                level[price] = level.get(price, 0) + size
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
