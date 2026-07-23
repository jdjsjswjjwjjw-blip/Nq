"""الترتيب الزمني السببي (Causal Time Ordering).

الترتيب القانوني لأي تدفّق أحداث يبدأ بـ ``(event_ts, sequence)``، ثم يستخدم
حقول المصدر المتاحة كفواصل تعادل حتمية عند تكرار الطابع/التسلسل.
"""

from __future__ import annotations

import polars as pl

from nq.contracts.temporal import INGEST_TS, TemporalFields

_OPTIONAL_CAUSAL_TIE_BREAKERS = (
    INGEST_TS,
    "publisher_id",
    "instrument_id",
    "action",
    "side",
    "order_id",
    "price",
    "size",
    "flags",
)


def causal_sort_columns(frame: pl.DataFrame) -> list[str]:
    """أقوى أعمدة ترتيب سببي متاحة في الإطار."""
    keys = list(TemporalFields.CAUSAL_ORDER)
    keys.extend(col for col in _OPTIONAL_CAUSAL_TIE_BREAKERS if col in frame.columns)
    return list(dict.fromkeys(keys))


def sort_causal(frame: pl.DataFrame) -> pl.DataFrame:
    """يرتّب الإطار بالترتيب السببي القانوني وفواصل التعادل المتاحة بثبات."""
    return frame.sort(causal_sort_columns(frame), maintain_order=True)


def is_sorted_causal(frame: pl.DataFrame) -> bool:
    """يُعيد ما إذا كان الإطار مرتّبًا بالفعل بالترتيب السببي القانوني."""
    if frame.height <= 1:
        return True
    keys = causal_sort_columns(frame)
    ordered = frame.select(keys).sort(keys, maintain_order=True)
    return frame.select(keys).equals(ordered)


def assert_sorted_causal(frame: pl.DataFrame) -> pl.DataFrame:
    """يتحقق من الترتيب السببي ويرفع ``ValueError`` إن اختلّ. يُعيد الإطار انسيابيًا."""
    if not is_sorted_causal(frame):
        raise ValueError(
            "causal-order violation: frame is not sorted by "
            f"{TemporalFields.CAUSAL_ORDER} — required to prevent temporal leakage."
        )
    return frame
