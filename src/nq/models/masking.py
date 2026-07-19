"""النمذجة المُقنّعة (Masked Modeling).

تُقنّع نسبة من عناصر المدخلات (masked event/state)، ويُدرَّب المشفّر على إعادة
بنائها. القناع محلّي داخل العيّنة (لا يستخدم أي معلومة مستقبلية)، وحتمي عبر بذرة
مولّد عشوائي، ما يضمن قابلية إعادة الإنتاج ومنع التسريب.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MaskedMatrix:
    """مصفوفة مُقنّعة مع قناعها والقيم الأصلية (الأهداف)."""

    masked: FloatArray
    mask: BoolArray
    targets: FloatArray


def mask_matrix(
    x: FloatArray,
    *,
    mask_ratio: float,
    rng: np.random.Generator,
    fill_value: float = 0.0,
) -> MaskedMatrix:
    """يُقنّع نسبة ``mask_ratio`` من خلايا المصفوفة حتميًا.

    يُعيد المصفوفة المُقنّعة (بقيمة ``fill_value`` عند المواضع المُقنّعة)، وقناع
    منطقي بالمواضع المُقنّعة، والقيم الأصلية كأهداف لإعادة البناء.
    """
    if not 0 < mask_ratio < 1:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    arr = np.asarray(x, dtype=np.float64)
    mask = rng.random(arr.shape) < mask_ratio
    masked = arr.copy()
    masked[mask] = fill_value
    return MaskedMatrix(masked=masked, mask=mask, targets=arr)


def masked_reconstruction_error(
    reconstruction: FloatArray,
    target: MaskedMatrix,
) -> float:
    """متوسّط مربّع الخطأ على المواضع المُقنّعة فقط (masked MSE)."""
    if not bool(np.any(target.mask)):
        return 0.0
    diff = reconstruction[target.mask] - target.targets[target.mask]
    return float(np.mean(diff**2))
