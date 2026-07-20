"""سياسة زمنية مركزية لمنع التسريب (Temporal Policy).

توحّد إعدادات walk-forward: embargo بالنانوثانية، purge لنوافذ SSL
المتداخلة، وأفق التقييم. تُقرأ من ``configs/default.toml`` عند الطلب.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

_NS_SCALE_THRESHOLD: Final = 10**14  # طوابع فوق هذا المقياس تُعامل كـ nanoseconds


def _times_in_nanoseconds(times: npt.NDArray[np.integer]) -> bool:
    if times.size == 0:
        return True
    return int(np.max(times)) >= _NS_SCALE_THRESHOLD


@dataclass(frozen=True, slots=True)
class TemporalPolicy:
    """إعدادات زمنية ملزمة لمسارات SSL والمراقب والألفا."""

    embargo_ns: int = 1_000_000_000
    window: int = 5
    stride: int = 1
    horizon: int = 1

    @classmethod
    def default(cls) -> TemporalPolicy:
        return cls()

    @classmethod
    def from_config(cls, path: Path | None = None) -> TemporalPolicy:
        """يقرأ ``[temporal]`` من ملف TOML (افتراضي: ``configs/default.toml``)."""
        config_path = path if path is not None else Path("configs/default.toml")
        if not config_path.is_file():
            return cls.default()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        temporal = raw.get("temporal", {})
        embargo = int(temporal.get("embargo_ns", 1_000_000_000))
        return cls(embargo_ns=embargo)

    @classmethod
    def for_run(
        cls,
        *,
        interval_ns: int,
        window: int = 5,
        stride: int = 1,
        horizon: int = 1,
        config_path: Path | None = None,
    ) -> TemporalPolicy:
        """يبني سياسة لجلسة تشغيل مع نافذة SSL و``interval_ns`` معروفين."""
        base = cls.from_config(config_path)
        return cls(
            embargo_ns=base.embargo_ns,
            window=window,
            stride=stride,
            horizon=horizon,
        )

    def purge_samples(self) -> int:
        """عدد عيّنات التدريب المُزالة قبل الاختبار بسبب تداخل النوافذ."""
        if self.window <= 1:
            return 0
        return (self.window - 1 + self.stride - 1) // self.stride

    def embargo_time_units(
        self,
        *,
        interval_ns: int,
        times: npt.NDArray[np.integer] | None = None,
    ) -> int:
        """فترة الحظر بنفس وحدات ``times`` (عادة nanoseconds).

        * بيانات إنتاج (ns): ``max(embargo_ns, فجوة النافذة + bucket)``.
        * بيانات اختبار صغيرة: فجوة النافذة + bucket واحد فقط (لا ``embargo_ns`` الضخم).
        """
        if interval_ns < 1:
            raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
        window_gap = (self.window - 1) * self.stride * interval_ns
        minimum = window_gap + interval_ns
        if times is not None and not _times_in_nanoseconds(times):
            return minimum
        return max(self.embargo_ns, minimum)


__all__ = ["TemporalPolicy"]
