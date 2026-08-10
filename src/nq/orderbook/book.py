"""حالة دفتر الأوامر (Order Book State).

يتتبّع الدفتر لكل جانب (طلب/عرض) الحجم المُجمّع عند كل مستوى سعري، إضافةً إلى
تتبّع كل أمر مفرد عبر ``order_id``.

آلة حالات Databento MBO:

* ``A`` Add — إدراج؛ تكرار ``order_id`` = استبدال + عدّاد duplicate.
* ``M`` Modify — أمر معروف فقط (مجهول = سلامة بلا شبح).
* ``C`` Cancel — إلغاء جزئي/كامل: ``size`` كمية مُلغاة.
* ``R`` Clear — تفريغ.
* ``T``/``F``/``N`` — لا تعدّل الدفتر (التغيير مع Cancel).

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
_TRADE = MboAction.TRADE.value
_NONE = MboAction.NONE.value
_BID = MboSide.BID.value


class OrderBook:
    """دفتر أوامر قابل للتحديث حدثًا بحدث بترتيب سببي صارم.

    الحالة:

    * ``bids`` / ``asks``: ``dict[price -> aggregated_size]`` لكل جانب.
    * ``orders``: ``dict[order_id -> (is_bid, price, size)]`` لتتبّع الأوامر.
    * ``_bid_vol`` / ``_ask_vol``: مجموع الحجم لكل جانب (للـ trail بنفس الأرقام).
    * ``_best_bid`` / ``_best_ask``: كاش O(1) لأفضل سعر (يُعاد مسحه عند الحاجة).
    """

    __slots__ = (
        "_ask_vol",
        "_best_ask",
        "_best_bid",
        "_bid_vol",
        "asks",
        "bids",
        "duplicate_add_refs",
        "orders",
        "unknown_order_refs",
    )

    def __init__(self) -> None:
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.orders: dict[int, tuple[bool, int, int]] = {}
        self.unknown_order_refs: int = 0
        self.duplicate_add_refs: int = 0
        self._bid_vol: int = 0
        self._ask_vol: int = 0
        self._best_bid: int | None = None
        self._best_ask: int | None = None

    def clear(self) -> None:
        """يمسح الدفتر بالكامل (book reset)."""
        self.bids.clear()
        self.asks.clear()
        self.orders.clear()
        self._bid_vol = 0
        self._ask_vol = 0
        self._best_bid = None
        self._best_ask = None

    def _add_level(self, level: dict[int, int], price: int, size: int, *, is_bid: bool) -> None:
        level[price] = level.get(price, 0) + size
        if is_bid:
            self._bid_vol += size
            if self._best_bid is None or price > self._best_bid:
                self._best_bid = price
        else:
            self._ask_vol += size
            if self._best_ask is None or price < self._best_ask:
                self._best_ask = price

    def _reduce(self, level: dict[int, int], price: int, size: int, *, is_bid: bool) -> None:
        old = level.get(price, 0)
        remaining = old - size
        if remaining > 0:
            level[price] = remaining
            removed = size
        else:
            level.pop(price, None)
            removed = old
            # زال المستوى — إن كان الأفضل نُعيد الحساب عند الطلب التالي
            if is_bid and self._best_bid == price:
                self._best_bid = None
            if not is_bid and self._best_ask == price:
                self._best_ask = None
        if is_bid:
            self._bid_vol -= removed
        else:
            self._ask_vol -= removed

    def _ensure_best_bid(self) -> int | None:
        if self._best_bid is None and self.bids:
            self._best_bid = max(self.bids)
        return self._best_bid

    def _ensure_best_ask(self) -> int | None:
        if self._best_ask is None and self.asks:
            self._best_ask = min(self.asks)
        return self._best_ask

    def apply(  # noqa: PLR0911 -- dispatch على نوع الحدث؛ العودة المبكرة أوضح
        self, action: str, side: str, price: int, size: int, order_id: int
    ) -> None:
        """يطبّق حدث MBO مفردًا وفق دلالات Databento الرسمية."""
        orders = self.orders
        bids = self.bids
        asks = self.asks
        add_level = self._add_level
        reduce_level = self._reduce

        # Trade / Fill / None: لا تغيير في الأوامر القائمة.
        if action in (_TRADE, _FILL, _NONE):
            if action == _FILL and order_id not in orders and order_id != 0:
                self.unknown_order_refs += 1
            return

        if action == _CLEAR:
            self.clear()
            return

        if action == _ADD:
            is_bid = side == _BID
            existing = orders.get(order_id)
            if existing is not None:
                self.duplicate_add_refs += 1
                old_bid, old_price, old_size = existing
                reduce_level(
                    bids if old_bid else asks, old_price, old_size, is_bid=old_bid
                )
            orders[order_id] = (is_bid, price, size)
            add_level(bids if is_bid else asks, price, size, is_bid=is_bid)
            return

        if action == _CANCEL:
            rec = orders.get(order_id)
            if rec is None:
                self.unknown_order_refs += 1
                return
            is_bid, p, s = rec
            # size>0: إلغاء جزئي/كامل بالكمية؛ size<=0: إلغاء كامل للمتبقي (توافق اصطناعي).
            cancel_qty = int(size) if int(size) > 0 else int(s)
            if cancel_qty <= 0:
                return
            removed = min(s, cancel_qty)
            reduce_level(bids if is_bid else asks, p, removed, is_bid=is_bid)
            remaining = s - removed
            if remaining > 0:
                orders[order_id] = (is_bid, p, remaining)
            else:
                orders.pop(order_id, None)
            return

        if action == _MODIFY:
            rec = orders.get(order_id)
            if rec is None:
                self.unknown_order_refs += 1
                return
            is_bid, old_price, old_size = rec
            level = bids if is_bid else asks
            reduce_level(level, old_price, old_size, is_bid=is_bid)
            add_level(level, price, size, is_bid=is_bid)
            orders[order_id] = (is_bid, price, size)
            return

    def best_bid(self) -> tuple[int, int] | None:
        """أفضل طلب ``(price, size)`` أو ``None`` إن كان الجانب فارغًا."""
        price = self._ensure_best_bid()
        if price is None:
            return None
        return price, self.bids[price]

    def best_ask(self) -> tuple[int, int] | None:
        """أفضل عرض ``(price, size)`` أو ``None`` إن كان الجانب فارغًا."""
        price = self._ensure_best_ask()
        if price is None:
            return None
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

    def path_liquidity(self, n: int = 5) -> tuple[float, float, float]:
        """سيولة المسار ``(cum_bid, cum_ask, imbalance)`` بدون لقطة L1–L5 كاملة."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        bid_n = self.cum_depth(_BID, n)
        ask_n = self.cum_depth("A", n)
        total = bid_n + ask_n
        imb = 0.0 if total <= 0 else (bid_n - ask_n) / float(total)
        return float(bid_n), float(ask_n), float(imb)

    def trail_liquidity(self) -> tuple[int, int]:
        """سيولة خلف أفضل طلب/عرض ``(trail_bid, trail_ask)``.

        مكافئ رياضيًا لـ ``sum(levels) - best_size`` عبر مجاميع الجانب المخزّنة.
        """
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        trail_bid = self._bid_vol - best_bid[1] if best_bid is not None else 0
        trail_ask = self._ask_vol - best_ask[1] if best_ask is not None else 0
        return int(trail_bid), int(trail_ask)

    def snapshot(self, n: int = 5, *, availability_ts: int = 0) -> DepthSnapshot:
        """لقطة عمق سببية من الحالة الحالية (بدون آثار جانبية)."""
        from nq.orderbook.depth import DepthSnapshot  # noqa: PLC0415

        bid = self.best_bid()
        ask = self.best_ask()
        bids = tuple(self.top_n(_BID, n))
        asks = tuple(self.top_n("A", n))
        trail_bid, trail_ask = self.trail_liquidity()
        cum_bid = int(sum(sz for _, sz in bids))
        cum_ask = int(sum(sz for _, sz in asks))
        total = cum_bid + cum_ask
        imbalance = 0.0 if total <= 0 else (cum_bid - cum_ask) / float(total)
        return DepthSnapshot(
            availability_ts=int(availability_ts),
            best_bid=None if bid is None else bid[0],
            bid_size=0 if bid is None else bid[1],
            best_ask=None if ask is None else ask[0],
            ask_size=0 if ask is None else ask[1],
            bid_levels=bids,
            ask_levels=asks,
            cum_bid=cum_bid,
            cum_ask=cum_ask,
            imbalance=imbalance,
            trail_bid=trail_bid,
            trail_ask=trail_ask,
            n_levels=n,
        )
