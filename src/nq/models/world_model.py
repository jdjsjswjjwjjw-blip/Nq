"""نموذج العالم التنبّئي (Predictive World Model).

يتعلّم خريطة من التمثيل الكامن الحالي إلى الحالة التالية (next-state prediction).
يُلائَم بانحدار ريدج (Ridge) بصيغة مغلقة على التدريب فقط، ويُقيَّم خارج العيّنة.
الهدف مستقبلي بطبيعته (label)، لذا يجب أن يُبنى دومًا ضمن تقسيم زمني صارم
(walk-forward) حتى لا يتسرّب المستقبل إلى تقييم الماضي.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def r2_score(y_true: FloatArray, y_pred: FloatArray) -> float:
    """معامل التحديد R² (نسخة متعدّدة المخرجات، مجمّعة)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean(axis=0)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


class NextStatePredictor:
    """متنبّئ الحالة التالية بانحدار ريدج مغلق الصيغة."""

    __slots__ = ("_fitted", "alpha", "coef_")

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha
        self.coef_: FloatArray | None = None
        self._fitted = False

    def fit(self, x: FloatArray, y: FloatArray) -> NextStatePredictor:
        """يلائم على التدريب فقط: ``(XᵀX + αI)⁻¹ Xᵀy`` مع عمود تحيّز."""
        xb = self._with_bias(np.asarray(x, dtype=np.float64))
        yb = np.asarray(y, dtype=np.float64)
        d = xb.shape[1]
        reg = self.alpha * np.eye(d)
        reg[-1, -1] = 0.0  # لا نُعاقب التحيّز
        self.coef_ = np.asarray(np.linalg.solve(xb.T @ xb + reg, xb.T @ yb), dtype=np.float64)
        self._fitted = True
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        """يتنبّأ بالحالة التالية للتمثيلات المُدخلة."""
        if not self._fitted or self.coef_ is None:
            raise RuntimeError("NextStatePredictor must be fitted before predict().")
        xb = self._with_bias(np.asarray(x, dtype=np.float64))
        return xb @ self.coef_

    @staticmethod
    def _with_bias(x: FloatArray) -> FloatArray:
        return np.hstack([x, np.ones((x.shape[0], 1))])
