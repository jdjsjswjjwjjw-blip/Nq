"""الحقول الزمنية القانونية (Canonical Temporal Fields).

كل صفّ بيانات في النظام — من MBO الخام حتى الميزات النهائية — يجب أن يحمل
طابعًا زمنيًا صريحًا يحدّد **متى وقع الحدث** و**متى أصبحت المعلومة متاحة**.
هذا هو حجر الأساس لمنع التسريب الزمني (temporal leakage).

الثوابت:

* ``EVENT_TS``        — الطابع الزمني لوقوع الحدث في السوق (exchange event time), ns.
* ``INGEST_TS``       — الطابع الزمني لاستلام/معالجة الحدث لدينا (receive time), ns.
* ``SEQUENCE``        — رقم تسلسلي رتيب لفضّ التعادل ضمن نفس الطابع الزمني.
* ``AVAILABILITY_TS`` — الطابع الزمني الذي تصبح عنده الميزة/المخرَج متاحًا للاستخدام.

القاعدة الحاكمة (point-in-time): ``AVAILABILITY_TS >= EVENT_TS`` دائمًا؛ لا يمكن
لأي معلومة أن تصبح متاحة قبل وقوع الحدث الذي اشتُقّت منه.
"""

from __future__ import annotations

from typing import Final

EVENT_TS: Final = "event_ts"
INGEST_TS: Final = "ingest_ts"
SEQUENCE: Final = "sequence"
AVAILABILITY_TS: Final = "availability_ts"


class TemporalFields:
    """أسماء الحقول الزمنية القانونية مُجمّعة للوصول البرمجي المتّسق."""

    EVENT_TS: Final = EVENT_TS
    INGEST_TS: Final = INGEST_TS
    SEQUENCE: Final = SEQUENCE
    AVAILABILITY_TS: Final = AVAILABILITY_TS

    #: الترتيب السببي القانوني: (زمن الحدث، ثم التسلسل) — يضمن ترتيبًا حتميًا.
    CAUSAL_ORDER: Final = (EVENT_TS, SEQUENCE)

    __slots__ = ()
