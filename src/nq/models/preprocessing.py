"""تطبيع سببي (Causal Standardization).

يُلائَم المقياس (mean/std) على بيانات التدريب **فقط** ثم يُطبّق للأمام على
الاختبار. هذا يمنع تسرّب إحصاءات المستقبل إلى الماضي (fit-on-past).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class CausalStandardScaler:
    """مطبّع قياسي يُلائَم على الماضي (train) ويُطبّق للأمام.

    يعمل على المحور الأخير (الميزات)؛ يقبل مصفوفات ثنائية ``(n, d)`` أو
    ثلاثية ``(n, window, d)``.
    """

    __slots__ = ("_fitted", "mean_", "std_")

    def __init__(self) -> None:
        self.mean_: FloatArray | None = None
        self.std_: FloatArray | None = None
        self._fitted = False

    def fit(self, x: FloatArray) -> CausalStandardScaler:
        """يحسب المتوسّط والانحراف المعياري على بيانات التدريب فقط."""
        arr = np.asarray(x, dtype=np.float64)
        axes = tuple(range(arr.ndim - 1))
        self.mean_ = arr.mean(axis=axes)
        std = arr.std(axis=axes)
        scale = np.maximum(np.abs(self.mean_), 1.0)
        self.std_ = np.where(std > 1e-12 * scale, std, 1.0)
        self._fitted = True
        return self

    def transform(self, x: FloatArray) -> FloatArray:
        """يطبّق التطبيع باستخدام إحصاءات التدريب الملائَمة مسبقًا."""
        if not self._fitted or self.mean_ is None or self.std_ is None:
            raise RuntimeError("CausalStandardScaler must be fitted before transform().")
        arr = np.asarray(x, dtype=np.float64)
        return (arr - self.mean_) / self.std_

    def fit_transform(self, x: FloatArray) -> FloatArray:
        """يلائم على التدريب ثم يطبّق (يُستخدم على طيّة التدريب حصرًا)."""
        return self.fit(x).transform(x)
