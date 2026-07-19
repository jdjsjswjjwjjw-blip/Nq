"""أساس مشترك لطبقة المحاكاة (Simulation Common Foundation).

يوفّر:

* اصطلاح المُعتدي (aggressor) في أحداث الصفقات.
* استخراج الصفقات (tape) مع أحجام الشراء/البيع العدوانية.
* تقطيع زمني (time bucketing) سببي مع طابع إتاحة (``availability_ts``) عند
  اكتمال النافذة، حتى لا تُستخدم أي ميزة مُجمّعة قبل إغلاق نافذتها.

اصطلاح المُعتدي (aggressor convention) لأحداث ``TRADE``:

* ``side == "B"`` → مشترٍ عدواني يرفع العرض (aggressive buy, lifts ask).
* ``side == "A"`` → بائع عدواني يضرب الطلب (aggressive sell, hits bid).
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nq.contracts.mbo import MboAction, MboSide
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS

#: جانب المُعتدي: شراء عدواني (يرفع العرض) وبيع عدواني (يضرب الطلب).
AGGRESSOR_BUY: Final = MboSide.BID.value
AGGRESSOR_SELL: Final = MboSide.ASK.value

#: أسماء أعمدة التقطيع الزمني.
BUCKET_START: Final = "bucket_start"
BUCKET_END: Final = "bucket_end"


def add_time_bucket(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يضيف أعمدة نافذة زمنية سببية: ``bucket_start``, ``bucket_end``, ``availability_ts``.

    ``bucket_start = floor(event_ts / interval) * interval`` و
    ``bucket_end = bucket_start + interval`` (زمن اكتمال النافذة = زمن الإتاحة).
    أي ميزة مُجمّعة على نافذة تصبح متاحة فقط عند ``bucket_end`` (point-in-time).
    """
    if interval_ns < 1:
        raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
    start = (pl.col(EVENT_TS) // interval_ns) * interval_ns
    return frame.with_columns(
        start.alias(BUCKET_START),
        (start + interval_ns).alias(BUCKET_END),
        (start + interval_ns).alias(AVAILABILITY_TS),
    )


def extract_trades(frame: pl.DataFrame) -> pl.DataFrame:
    """يستخرج الصفقات (``action == TRADE``) مع أحجام الشراء/البيع العدوانية.

    يضيف الأعمدة:

    * ``buy_volume``  — الحجم عندما يكون المُعتدي مشتريًا (side == B).
    * ``sell_volume`` — الحجم عندما يكون المُعتدي بائعًا (side == A).
    * ``signed_volume`` — ``buy_volume - sell_volume`` (تدفّق موقّع).
    """
    trades = frame.filter(pl.col("action") == MboAction.TRADE.value)
    size = pl.col("size").cast(pl.Int64)
    is_buy = pl.col("side") == AGGRESSOR_BUY
    is_sell = pl.col("side") == AGGRESSOR_SELL
    return trades.with_columns(
        pl.when(is_buy).then(size).otherwise(0).alias("buy_volume"),
        pl.when(is_sell).then(size).otherwise(0).alias("sell_volume"),
        pl.when(is_buy).then(size).when(is_sell).then(-size).otherwise(0).alias("signed_volume"),
    )
