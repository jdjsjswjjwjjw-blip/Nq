"""أدلّة كمية قابلة للتتبّع (Traceable Quantitative Evidence).

كل ``Evidence`` وحدة حقيقة كمية مُشتقّة من مكوّن محسوب (اختبار إحصائي، مقياس،
حالة سوقية...) مع مصدرها وقيمتها ودلالتها وإصدارها — بحيث يمكن تتبّع أي ادعاء
إلى مصدره الرقمي.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Evidence:
    """دليل كمي مفرد قابل للتتبّع."""

    id: str
    source: str  # المكوّن المُنتِج، مثل "regime_difference_test"
    metric: str  # اسم المقياس، مثل "F" أو "sharpe"
    value: float
    pvalue: float | None = None
    sample_size: int | None = None
    version: str | None = None
    detail: str = ""

    def is_significant(self, alpha: float) -> bool:
        """هل الدليل دالّ إحصائيًا عند مستوى ``alpha``؟ (يتطلّب ``pvalue``)."""
        return self.pvalue is not None and self.pvalue <= alpha


class EvidenceStore:
    """سجلّ أدلّة مفهرس بالمعرّف لضمان التتبّع وعدم التكرار."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def add(self, evidence: Evidence) -> str:
        """يسجّل دليلًا ويُعيد معرّفه؛ يرفض المعرّفات المكرّرة."""
        if evidence.id in self._items:
            raise ValueError(f"duplicate evidence id: {evidence.id!r}")
        self._items[evidence.id] = evidence
        return evidence.id

    def get(self, evidence_id: str) -> Evidence:
        if evidence_id not in self._items:
            raise KeyError(f"unknown evidence id: {evidence_id!r}")
        return self._items[evidence_id]

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[Evidence]:
        """كل الأدلّة المسجّلة (بترتيب الإدراج)."""
        return list(self._items.values())
