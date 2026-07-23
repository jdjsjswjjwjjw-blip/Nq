"""إعادة بناء دفتر الأوامر من MBO (Order Book Reconstruction).

المكوّنات:

* ``check_integrity`` — فحوص سلامة تدفّق MBO (out-of-order، فجوات التسلسل).
* ``OrderBook`` — حالة دفتر الأوامر (مستويات الأسعار + تتبّع الأوامر).
* ``reconstruct`` — إعادة بناء الحالة حدثًا بحدث وإخراج سلسلة top-of-book زمنية.
"""

from __future__ import annotations

from nq.orderbook.book import OrderBook
from nq.orderbook.depth import DepthSnapshot, walk_buy_vwap, walk_sell_vwap
from nq.orderbook.integrity import IntegrityReport, check_integrity
from nq.orderbook.reconstruction import (
    ReconstructionResult,
    StrictReconstructionError,
    reconstruct,
    reconstruct_by_instrument,
)

__all__ = [
    "DepthSnapshot",
    "IntegrityReport",
    "OrderBook",
    "ReconstructionResult",
    "StrictReconstructionError",
    "check_integrity",
    "reconstruct",
    "reconstruct_by_instrument",
    "walk_buy_vwap",
    "walk_sell_vwap",
]
