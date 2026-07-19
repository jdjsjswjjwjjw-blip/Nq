"""إعادة بناء دفتر الأوامر من MBO (Order Book Reconstruction).

المكوّنات:

* ``check_integrity`` — فحوص سلامة تدفّق MBO (out-of-order، فجوات التسلسل).
* ``OrderBook`` — حالة دفتر الأوامر (مستويات الأسعار + تتبّع الأوامر).
* ``reconstruct`` — إعادة بناء الحالة حدثًا بحدث وإخراج سلسلة top-of-book زمنية.
"""

from __future__ import annotations

from nq.orderbook.book import OrderBook
from nq.orderbook.integrity import IntegrityReport, check_integrity
from nq.orderbook.reconstruction import (
    ReconstructionResult,
    reconstruct,
    reconstruct_by_instrument,
)

__all__ = [
    "IntegrityReport",
    "OrderBook",
    "ReconstructionResult",
    "check_integrity",
    "reconstruct",
    "reconstruct_by_instrument",
]
