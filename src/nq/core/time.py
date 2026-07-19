"""الترتيب الزمني السببي (Causal Time Ordering).

الترتيب القانوني لأي تدفّق أحداث هو ``(event_ts, sequence)``. هذا الترتيب
حتمي ويضمن أن كل حساب لاحق يرى الأحداث بالترتيب الذي وقعت به فعلًا، وهو
شرط ضروري (لكنه غير كافٍ وحده) لمنع التسريب الزمني.
"""

from __future__ import annotations

import polars as pl

from nq.contracts.temporal import TemporalFields


def sort_causal(frame: pl.DataFrame) -> pl.DataFrame:
    """يرتّب الإطار بالترتيب السببي القانوني ``(event_ts, sequence)`` بثبات."""
    return frame.sort(list(TemporalFields.CAUSAL_ORDER), maintain_order=True)


def is_sorted_causal(frame: pl.DataFrame) -> bool:
    """يُعيد ما إذا كان الإطار مرتّبًا بالفعل بالترتيب السببي القانوني."""
    if frame.height <= 1:
        return True
    keys = list(TemporalFields.CAUSAL_ORDER)
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
