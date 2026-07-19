"""أدوات التحقق العامة — وعلى رأسها اختبار التسريب الزمني (Leakage Test)."""

from __future__ import annotations

from nq.validation.leakage import (
    LeakageError,
    LeakageReport,
    assert_availability_not_before_event,
    assert_causal_order,
    assert_temporal_split,
    detect_leakage_by_perturbation,
)

__all__ = [
    "LeakageError",
    "LeakageReport",
    "assert_availability_not_before_event",
    "assert_causal_order",
    "assert_temporal_split",
    "detect_leakage_by_perturbation",
]
