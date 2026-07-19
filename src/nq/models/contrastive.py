"""التعلّم التبايني ذاتي الإشراف (Contrastive Self-Supervised Learning).

يولّد "مناظير" (views) من العيّنة نفسها عبر تحسينات (augmentations) حتمية، ثم
يقيس هدف InfoNCE الذي يقرّب المناظير الإيجابية (من العيّنة ذاتها) ويباعد السلبية
(من عيّنات أخرى ضمن نفس الطيّة الزمنية — بلا تسريب مستقبلي).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_MIN_BATCH_FOR_NEGATIVES = 2


def augment_windows(
    x: FloatArray,
    *,
    rng: np.random.Generator,
    noise_std: float = 0.05,
    mask_ratio: float = 0.1,
) -> FloatArray:
    """يُنتج منظورًا مُحسَّنًا: تشويش غاوسي + إخفاء عشوائي لبعض الخلايا.

    التحسين محلّي لكل عيّنة (لا يخلط عيّنات مختلفة)، وحتمي عبر ``rng``.
    """
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if not 0 <= mask_ratio < 1:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
    arr = np.asarray(x, dtype=np.float64)
    view = arr + rng.standard_normal(arr.shape) * noise_std
    if mask_ratio > 0:
        drop = rng.random(arr.shape) < mask_ratio
        view = np.where(drop, 0.0, view)
    return view


def _l2_normalize(z: FloatArray) -> FloatArray:
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.where(norms > 0, norms, 1.0)


def info_nce_loss(
    z_anchor: FloatArray,
    z_positive: FloatArray,
    *,
    temperature: float = 0.1,
) -> float:
    """هدف InfoNCE (متماثل) لدفعة من المناظير الإيجابية المتناظرة.

    لكل عيّنة ``i``، المنظور الإيجابي هو ``z_positive[i]``، والسلبيات هي بقية
    الدفعة. قيمة أقل تعني تمثيلات أكثر تمييزًا للعيّنات. حتمي وقابل لإعادة الإنتاج.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    a = _l2_normalize(np.asarray(z_anchor, dtype=np.float64))
    p = _l2_normalize(np.asarray(z_positive, dtype=np.float64))
    n = a.shape[0]
    if n < _MIN_BATCH_FOR_NEGATIVES:
        raise ValueError("info_nce_loss needs at least 2 samples for negatives.")

    logits = (a @ p.T) / temperature
    log_denom = _logsumexp_rows(logits)
    positives = np.diagonal(logits)
    return float(np.mean(log_denom - positives))


def _logsumexp_rows(logits: FloatArray) -> FloatArray:
    row_max = np.max(logits, axis=1, keepdims=True)
    stable = logits - row_max
    result = row_max.squeeze(axis=1) + np.log(np.sum(np.exp(stable), axis=1))
    return np.asarray(result, dtype=np.float64)
