"""عقد بيانات MBO (Market By Order) — المصدر الوحيد للحقيقة في النظام.

يُصمّم هذا العقد على غرار مخطط MBO القياسي لعقود CME الآجلة (NQ / MNQ):
كل حدث يمثّل تعديلًا ذريًّا على دفتر الأوامر (إضافة/إلغاء/تعديل/مسح/صفقة/تنفيذ).
الأسعار مُخزّنة كأعداد صحيحة بنقطة ثابتة (fixed-point) بمقياس ``PRICE_SCALE``
لتفادي أخطاء الفاصلة العائمة وضمان الدقّة الكمية.

كل الطبقات الأعلى تُشتق **حصريًا** من هذا العقد (مبدأ MBO-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import polars as pl

from nq.contracts.temporal import EVENT_TS, INGEST_TS, SEQUENCE

#: مقياس النقطة الثابتة للأسعار: السعر الحقيقي = price * PRICE_SCALE.
PRICE_SCALE: Final = 1e-9


class MboAction(StrEnum):
    """أنواع أحداث MBO الذرية على دفتر الأوامر."""

    ADD = "A"  # إدراج أمر جديد (add resting order)
    CANCEL = "C"  # إلغاء أمر قائم
    MODIFY = "M"  # تعديل أمر قائم (سعر/حجم)
    CLEAR = "R"  # مسح جانب/دفتر بالكامل (book reset)
    TRADE = "T"  # صفقة تنفيذية
    FILL = "F"  # تنفيذ (fill) مرتبط بأمر
    NONE = "N"  # لا فعل (heartbeat/administrative)


class MboSide(StrEnum):
    """جانب دفتر الأوامر الذي يتعلق به الحدث."""

    BID = "B"  # جانب الطلب (buy side)
    ASK = "A"  # جانب العرض (sell side)
    NONE = "N"  # غير محدد (مثل بعض الصفقات)


@dataclass(frozen=True, slots=True)
class MboEvent:
    """تمثيل مُكتمل التنميط (fully-typed) لحدث MBO مفرد.

    يُستخدم أساسًا في الاختبارات والتحقق والتوثيق؛ المعالجة الجماهيرية تجري
    على أُطر Polars عمودية عالية الأداء وفق ``MBO_SCHEMA``.
    """

    event_ts: int  # زمن الحدث في السوق (ns since epoch)
    ingest_ts: int  # زمن الاستلام لدينا (ns since epoch)
    sequence: int  # تسلسل رتيب لفضّ التعادل
    instrument_id: int  # معرّف الأداة
    symbol: str  # الرمز (NQ / MNQ)
    action: MboAction
    side: MboSide
    price: int  # سعر بنقطة ثابتة (fixed-point)
    size: int  # الحجم
    order_id: int  # معرّف الأمر
    flags: int = 0  # أعلام السوق (bit flags)

    def __post_init__(self) -> None:
        if self.ingest_ts < self.event_ts:
            msg = (
                "point-in-time violation: ingest_ts < event_ts "
                f"({self.ingest_ts} < {self.event_ts}) — المعلومة لا تُستلم قبل وقوعها."
            )
            raise ValueError(msg)
        if self.size < 0:
            raise ValueError(f"size must be non-negative, got {self.size}")


#: المخطط العمودي القانوني لأُطر MBO (Polars schema).
MBO_SCHEMA: Final[dict[str, pl.DataType]] = {
    EVENT_TS: pl.Int64(),
    INGEST_TS: pl.Int64(),
    SEQUENCE: pl.UInt64(),
    "instrument_id": pl.UInt32(),
    "symbol": pl.Utf8(),
    "action": pl.Enum([a.value for a in MboAction]),
    "side": pl.Enum([s.value for s in MboSide]),
    "price": pl.Int64(),
    "size": pl.UInt32(),
    "order_id": pl.UInt64(),
    "flags": pl.UInt8(),
}


def validate_mbo_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """يتحقق من مطابقة إطار Polars لعقد MBO بنيويًا ونقطيًا-زمنيًا.

    الفحوص:

    * وجود كل الأعمدة المطلوبة (لا نقص ولا أعمدة غير معرّفة تُكسر العقد).
    * تطابق أنواع الأعمدة مع ``MBO_SCHEMA``.
    * سلامة النقطة الزمنية: ``ingest_ts >= event_ts`` لكل صف.

    يُعيد نفس الإطار عند النجاح ليُستخدم انسيابيًا (fluent)، ويرفع ``ValueError``
    عند أي خرق للعقد.
    """
    actual = set(frame.columns)
    expected = set(MBO_SCHEMA)

    missing = expected - actual
    if missing:
        raise ValueError(f"MBO contract violation: missing columns {sorted(missing)}")

    extra = actual - expected
    if extra:
        raise ValueError(f"MBO contract violation: unexpected columns {sorted(extra)}")

    mismatched: dict[str, tuple[str, str]] = {}
    for name, expected_dtype in MBO_SCHEMA.items():
        got = frame.schema[name]
        if got != expected_dtype:
            mismatched[name] = (str(expected_dtype), str(got))
    if mismatched:
        raise ValueError(f"MBO contract violation: dtype mismatch {mismatched}")

    if frame.height > 0:
        violations = frame.filter(pl.col(INGEST_TS) < pl.col(EVENT_TS)).height
        if violations:
            raise ValueError(
                f"MBO contract violation: {violations} rows with ingest_ts < event_ts "
                "(point-in-time / temporal-leakage guard)."
            )

    return frame
