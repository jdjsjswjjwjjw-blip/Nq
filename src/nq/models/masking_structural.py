"""إخفاء هيكلي موجّه (Structural Masking) — البُعد 6.

مساران لا يتداخلان:

* **Standalone** (NQ/MNQ منفرد): balance (إخفاء سيولة عند VAH/VAL) أو
  expansion (إخفاء trailing liquidity خلف السعر).
* **Cross-trap**: MNQ مكشوف، إخفاء أوامر NQ التجميعية (bid/ask size).

يُكمّل ``mask_matrix`` العشوائي في ``masking.py`` دون تعديله.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from nq.models.masking import MaskedMatrix
from nq.models.tick_stream import (
    TICK_FEATURE_NAMES,
    MarketPhase,
    MaskPath,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# فهارس الميزات للإخفاء السريع
_NAME_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(TICK_FEATURE_NAMES)}

_NQ_LIQUIDITY_IDX: tuple[int, ...] = (
    _NAME_TO_IDX["nq_bid_size_log"],
    _NAME_TO_IDX["nq_ask_size_log"],
)
_MNQ_FLOW_IDX: tuple[int, ...] = (
    _NAME_TO_IDX["mnq_signed_vol"],
    _NAME_TO_IDX["trap_setup"],
)
_VP_BOUNDARY_IDX: tuple[int, ...] = (
    _NAME_TO_IDX["near_vah"],
    _NAME_TO_IDX["near_val"],
    _NAME_TO_IDX["in_value_area"],
)
_TRAILING_IDX: tuple[int, ...] = (
    _NAME_TO_IDX["nq_bid_size_log"],
    _NAME_TO_IDX["nq_ask_size_log"],
    _NAME_TO_IDX["nq_spread_norm"],
)


def _mask_feature_indices(
    x: FloatArray,
    indices: Sequence[int],
    *,
    time_slice: slice | None = None,
) -> BoolArray:
    """قناع على أبعاد الميزات (الأخير) لكل النافذة أو شريحة زمنية."""
    mask = np.zeros_like(x, dtype=bool)
    t_slice = time_slice if time_slice is not None else slice(None)
    for idx in indices:
        mask[t_slice, idx] = True
    return mask


def structural_mask_sample(
    x: FloatArray,
    *,
    mask_path: int,
    market_phase: int,
    feature_names: Sequence[str] | None = None,
    fill_value: float = 0.0,
) -> MaskedMatrix:
    """يُقنّع عيّنة tick واحدة ``(window, n_features)`` حسب المسار والمرحلة."""
    _ = feature_names  # أسماء ثابتة من TICK_FEATURE_NAMES؛ للتوافق المستقبلي
    arr = np.asarray(x, dtype=np.float64)
    mask = np.zeros_like(arr, dtype=bool)

    if mask_path == int(MaskPath.CROSS_TRAP):
        # مسار 2: إخفاء سيولة NQ التجميعية، إبقاء تدفّق MNQ
        mask |= _mask_feature_indices(arr, _NQ_LIQUIDITY_IDX)
    elif market_phase == int(MarketPhase.BALANCE):
        near_vah = bool(np.any(arr[:, _NAME_TO_IDX["near_vah"]] > 0))
        near_val = bool(np.any(arr[:, _NAME_TO_IDX["near_val"]] > 0))
        in_va = bool(np.any(arr[:, _NAME_TO_IDX["in_value_area"]] > 0))
        if near_vah or near_val or in_va:
            mask |= _mask_feature_indices(arr, _NQ_LIQUIDITY_IDX)
            mask |= _mask_feature_indices(arr, _VP_BOUNDARY_IDX)
    elif market_phase == int(MarketPhase.EXPANSION):
        # مسار 1 — expansion: إخفاء trailing liquidity (آخر tick في النافذة)
        mask |= _mask_feature_indices(arr, _TRAILING_IDX, time_slice=slice(-1, None))
    else:
        # neutral: إخفاء خفيف على spread فقط
        spread_idx = _NAME_TO_IDX["nq_spread_norm"]
        mask |= _mask_feature_indices(arr, (spread_idx,), time_slice=slice(-1, None))

    masked = arr.copy()
    masked[mask] = fill_value
    return MaskedMatrix(masked=masked, mask=mask, targets=arr)


def structural_mask_batch(
    x: FloatArray,
    *,
    mask_paths: npt.NDArray[np.integer],
    market_phases: npt.NDArray[np.integer],
    fill_value: float = 0.0,
) -> list[MaskedMatrix]:
    """يُقنّع دفعة من نوافذ tick ``(n_samples, window, n_features)``."""
    results: list[MaskedMatrix] = []
    for i in range(x.shape[0]):
        results.append(
            structural_mask_sample(
                x[i],
                mask_path=int(mask_paths[i]),
                market_phase=int(market_phases[i]),
                fill_value=fill_value,
            )
        )
    return results


def batch_masked_mse(
    reconstruction: FloatArray,
    masked_batch: list[MaskedMatrix],
) -> float:
    """متوسط MSE على المواضع المُقنّعة عبر الدفعة."""
    if not masked_batch:
        return 0.0
    errors: list[float] = []
    for i, target in enumerate(masked_batch):
        if not bool(np.any(target.mask)):
            continue
        diff = reconstruction[i][target.mask] - target.targets[target.mask]
        errors.append(float(np.mean(diff**2)))
    return float(np.mean(errors)) if errors else 0.0


__all__ = [
    "batch_masked_mse",
    "structural_mask_batch",
    "structural_mask_sample",
]
