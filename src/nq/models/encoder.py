"""مشفّر تمثيلي ذاتي الإشراف (Self-Supervised Representation Encoder).

``Encoder`` بروتوكول موحّد (fit/transform/reconstruct) يسمح باستبدال المشفّر
الأساسي بمشفّر عصبي (Transformer) لاحقًا دون تغيير بقية المسار.

``PCAEncoder`` مشفّر أساسي (baseline) يتعلّم تمثيلًا كامنًا بلا إشراف عبر تحليل
القيم المفردة (SVD)؛ يُلائَم على التدريب فقط (سببي)، ويُنتج متجهات كامنة (market
embeddings) قابلة لإعادة البناء (أساس النمذجة المُقنّعة).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_MATRIX_NDIM = 2


@runtime_checkable
class Encoder(Protocol):
    """واجهة المشفّر: يُلائَم على التدريب، ويشفّر ويعيد البناء."""

    def fit(self, x: FloatArray) -> Encoder: ...

    def transform(self, x: FloatArray) -> FloatArray: ...

    def reconstruct(self, x: FloatArray) -> FloatArray: ...


class PCAEncoder:
    """مشفّر تمثيلي أساسي عبر PCA (SVD)، يُلائَم على الماضي فقط."""

    __slots__ = ("_fitted", "components_", "mean_", "n_components")

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.mean_: FloatArray | None = None
        self.components_: FloatArray | None = None
        self._fitted = False

    def fit(self, x: FloatArray) -> PCAEncoder:
        """يتعلّم المحاور الرئيسية من بيانات التدريب (2-D: عيّنات × ميزات)."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != _MATRIX_NDIM:
            raise ValueError(f"PCAEncoder expects a 2-D matrix, got shape {arr.shape}")
        self.mean_ = arr.mean(axis=0)
        centered = arr - self.mean_
        k = min(self.n_components, *centered.shape)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = vt[:k]
        self._fitted = True
        return self

    def _check_fitted(self) -> tuple[FloatArray, FloatArray]:
        if not self._fitted or self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAEncoder must be fitted before use.")
        return self.mean_, self.components_

    def transform(self, x: FloatArray) -> FloatArray:
        """يشفّر المدخلات إلى الفضاء الكامن (market embeddings)."""
        mean, components = self._check_fitted()
        arr = np.asarray(x, dtype=np.float64)
        return (arr - mean) @ components.T

    def fit_transform(self, x: FloatArray) -> FloatArray:
        return self.fit(x).transform(x)

    def reconstruct(self, x: FloatArray) -> FloatArray:
        """يعيد بناء المدخلات من تمثيلها الكامن (للنمذجة المُقنّعة/التقييم)."""
        mean, components = self._check_fitted()
        embedding = self.transform(x)
        return embedding @ components + mean
