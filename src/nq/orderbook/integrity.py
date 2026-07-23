"""فحوص سلامة تدفّق MBO (Stream Integrity Checks).

تُقاس هذه الفحوص على البيانات كما وردت (per-instrument) لكشف مشكلات التسليم
والمصدر قبل الاعتماد عليها علميًا:

* ``out_of_order``            — أحداث وردت خارج الترتيب السببي.
* ``sequence_non_monotonic``  — أرقام تسلسل غير متزايدة تمامًا (تكرار/تراجع).
* ``sequence_skips``          — قفزات في التسلسل (فجوات: diff > 1).
* ``unknown_order_refs``      — إلغاء/تعديل/تنفيذ لأمر غير معروف (يُحسب أثناء البناء).
* ``crossed_book_events``     — لحظات تقاطع الدفتر (best_bid >= best_ask).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nq.contracts.temporal import EVENT_TS, SEQUENCE

#: أدنى فرق تسلسل سليم بين حدثين متتاليين لنفس الأداة.
_EXPECTED_SEQUENCE_STEP = 1


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """تقرير سلامة تدفّق MBO."""

    n_events: int
    out_of_order: int
    sequence_non_monotonic: int
    sequence_skips: int
    unknown_order_refs: int
    crossed_book_events: int

    @property
    def ok(self) -> bool:
        """سليم عندما لا يوجد اختلال ترتيب ولا تسلسل غير رتيب ولا مراجع مجهولة."""
        return (
            self.out_of_order == 0
            and self.sequence_non_monotonic == 0
            and self.unknown_order_refs == 0
        )

    @property
    def strict_ok(self) -> bool:
        """سليم لبحث/باكتيست صارم: بلا فجوات تسلسل ولا دفاتر متقاطعة أيضًا."""
        return self.ok and self.sequence_skips == 0 and self.crossed_book_events == 0

    def strict_failures(self) -> dict[str, int]:
        """عدادات الفشل غير الصفرية التي تمنع استخدام التدفق في الوضع الصارم."""
        values = {
            "out_of_order": self.out_of_order,
            "sequence_non_monotonic": self.sequence_non_monotonic,
            "sequence_skips": self.sequence_skips,
            "unknown_order_refs": self.unknown_order_refs,
            "crossed_book_events": self.crossed_book_events,
        }
        return {name: count for name, count in values.items() if count}


def check_integrity(frame: pl.DataFrame) -> IntegrityReport:
    """يحسب فحوص السلامة المعتمدة على الإطار (per-instrument).

    لا يشمل ``unknown_order_refs`` و ``crossed_book_events`` لأنهما يُشتقّان أثناء
    إعادة البناء؛ يُعيدهما هذا التقرير أصفارًا ويملؤهما ``reconstruct``.
    """
    n = frame.height
    if n == 0:
        return IntegrityReport(0, 0, 0, 0, 0, 0)

    prev_ts = pl.col(EVENT_TS).shift(1).over("instrument_id")
    prev_seq = pl.col(SEQUENCE).cast(pl.Int64).shift(1).over("instrument_id")
    seq = pl.col(SEQUENCE).cast(pl.Int64)

    out_of_order = (pl.col(EVENT_TS) < prev_ts) | ((pl.col(EVENT_TS) == prev_ts) & (seq < prev_seq))
    non_monotonic = seq <= prev_seq
    skips = (seq - prev_seq) > _EXPECTED_SEQUENCE_STEP

    stats = frame.select(
        out_of_order.fill_null(value=False).sum().alias("ooo"),
        non_monotonic.fill_null(value=False).sum().alias("nonmono"),
        skips.fill_null(value=False).sum().alias("skips"),
    ).row(0)

    return IntegrityReport(
        n_events=n,
        out_of_order=int(stats[0]),
        sequence_non_monotonic=int(stats[1]),
        sequence_skips=int(stats[2]),
        unknown_order_refs=0,
        crossed_book_events=0,
    )
