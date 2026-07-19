"""تصحيح التعدّد (Multiple-Testing Correction).

عند اختبار فرضيات كثيرة يرتفع معدّل الإيجابيات الكاذبة؛ تُصحَّح القيم الاحتمالية
بواحدة من:

* ``benjamini_hochberg`` — يضبط معدّل الاكتشاف الخاطئ (FDR).
* ``holm``               — إجراء نزولي يضبط معدّل الخطأ العائلي (FWER).
* ``bonferroni``         — أبسط ضبط لـ FWER (الأكثر تحفّظًا).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """نتيجة تصحيح التعدّد: قرار الرفض والقيم الاحتمالية المُعدّلة."""

    reject: BoolArray
    adjusted: FloatArray
    alpha: float
    method: str


def _as_pvalues(pvalues: npt.NDArray[np.floating] | list[float]) -> FloatArray:
    arr = np.asarray(pvalues, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"pvalues must be 1-D, got shape {arr.shape}")
    if bool(np.any((arr < 0) | (arr > 1))):
        raise ValueError("pvalues must lie in [0, 1].")
    return arr


def bonferroni(
    pvalues: npt.NDArray[np.floating] | list[float], *, alpha: float = 0.05
) -> CorrectionResult:
    """ضبط Bonferroni لمعدّل الخطأ العائلي."""
    p = _as_pvalues(pvalues)
    m = p.shape[0]
    adjusted = np.minimum(p * m, 1.0)
    return CorrectionResult(
        reject=adjusted <= alpha, adjusted=adjusted, alpha=alpha, method="bonferroni"
    )


def holm(
    pvalues: npt.NDArray[np.floating] | list[float], *, alpha: float = 0.05
) -> CorrectionResult:
    """إجراء Holm النزولي (step-down) لضبط FWER."""
    p = _as_pvalues(pvalues)
    m = p.shape[0]
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adjusted_sorted[i] = min(running, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return CorrectionResult(reject=adjusted <= alpha, adjusted=adjusted, alpha=alpha, method="holm")


def benjamini_hochberg(
    pvalues: npt.NDArray[np.floating] | list[float], *, alpha: float = 0.05
) -> CorrectionResult:
    """إجراء Benjamini-Hochberg لضبط معدّل الاكتشاف الخاطئ (FDR)."""
    p = _as_pvalues(pvalues)
    m = p.shape[0]
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    scaled = p[order] * m / ranks
    # فرض الرتابة من الأعلى للأسفل (monotone q-values)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = q_sorted
    return CorrectionResult(
        reject=adjusted <= alpha, adjusted=adjusted, alpha=alpha, method="benjamini_hochberg"
    )
