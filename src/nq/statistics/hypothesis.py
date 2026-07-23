"""تحقّق الفرضيات مع تصحيح التعدّد (Hypothesis Verification).

يجمع نتائج بطارية اختبارات (اسم الفرضية → p-value) ويطبّق تصحيح التعدّد، ثم
يُصدر تقريرًا موحّدًا مرتّبًا بالدلالة. هذا يفرض ألّا تُعتمد أي فرضية دون المرور
ببروتوكول إحصائي مصحّح (شرط قبول المحطة).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import polars as pl

from nq.statistics.multiple_testing import benjamini_hochberg, bonferroni, holm

_METHODS = {
    "benjamini_hochberg": benjamini_hochberg,
    "holm": holm,
    "bonferroni": bonferroni,
}

CorrectionMethod = Literal["benjamini_hochberg", "holm", "bonferroni"]


def verify_hypotheses(
    pvalues: Mapping[str, float],
    *,
    alpha: float = 0.05,
    method: CorrectionMethod = "benjamini_hochberg",
) -> pl.DataFrame:
    """يبني تقرير تحقّق فرضيات موحّدًا مع تصحيح التعدّد.

    يُعيد إطارًا بالأعمدة: ``hypothesis``, ``pvalue``, ``adjusted_pvalue``,
    ``reject`` (هل نرفض فرضية العدم عند ``alpha``؟)، مرتّبًا تصاعديًا بالقيمة
    المُعدّلة. يرفع ``ValueError`` لطريقة غير معروفة أو ``alpha`` غير صالح.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    if not pvalues:
        return pl.DataFrame(
            schema={
                "hypothesis": pl.Utf8(),
                "pvalue": pl.Float64(),
                "adjusted_pvalue": pl.Float64(),
                "reject": pl.Boolean(),
            }
        )

    names = list(pvalues.keys())
    raw = np.asarray([pvalues[n] for n in names], dtype=np.float64)
    result = _METHODS[method](raw, alpha=alpha)
    return pl.DataFrame(
        {
            "hypothesis": names,
            "pvalue": raw,
            "adjusted_pvalue": result.adjusted,
            "reject": result.reject,
        }
    ).sort("adjusted_pvalue")
