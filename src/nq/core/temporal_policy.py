"""سياسة زمنية مركزية لمنع التسريب (Temporal Policy).

توحّد إعدادات walk-forward: embargo بالنانوثانية، purge لنوافذ SSL
المتداخلة، وأفق التقييم. تُقرأ من ``configs/default.toml`` عند الطلب.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
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
        horizon = int(temporal.get("horizon", 1))
        unknown = set(temporal) - {"split_strategy", "embargo_ns", "horizon", "interval_ns", "profile_interval_ns"}
        if unknown:
            raise ValueError(f"unknown [temporal] keys: {sorted(unknown)}")
        return cls(embargo_ns=embargo, horizon=horizon)

    @classmethod
    def for_run(
        cls,
        *,
        interval_ns: int,
        window: int = 5,
        stride: int = 1,
        horizon: int = 1,
        config_path: Path | None = None,
        embargo_ns: int | None = None,
    ) -> TemporalPolicy:
        """يبني سياسة لجلسة تشغيل مع نافذة SSL و``interval_ns`` معروفين.

        ``embargo_ns`` إن وُجد يتجاوز قيمة الملف؛ وإلا تُقرأ من ``config_path``.
        ``interval_ns`` يُمرَّر للواجهة فقط (يُستخدم لاحقًا في ``embargo_time_units``).
        """
        del interval_ns  # يُستهلك عبر embargo_time_units عند الاستدعاء
        base = cls.from_config(config_path)
        return cls(
            embargo_ns=int(embargo_ns) if embargo_ns is not None else base.embargo_ns,
            window=window,
            stride=stride,
            horizon=horizon,
        )

    def purge_samples(self) -> int:
        """عدد عيّنات التدريب المُزالة قبل الاختبار.

        يشمل تداخل نوافذ SSL **وأفق التقييم** (``horizon``) حتى لا تتسرّب
        تسمية العائد الأمامي إلى كتلة الاختبار — نفس فلسفة ``symbolic_gp``.
        """
        window_purge = 0 if self.window <= 1 else (self.window - 1 + self.stride - 1) // self.stride
        return max(window_purge, int(self.horizon), 0)

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


def resolve_grid_context_interval(
    signal_intervals: Sequence[int],
    *,
    default_ns: int,
) -> tuple[int, bool]:
    """أطول إطار إشارة في الشبكة + هل الشبكة مختلطة TF.

    عند الاختلاط يُستخدم ``max`` لسياق العمق/الفوليوم/محاذاة الأفق — مرشّحات
    TF أقصر تُقيَّم تحت أفق أطول (توثيق صريح، ليس خطأ صامت).
    """
    vals = [int(x) for x in signal_intervals if int(x) > 0]
    if not vals:
        return int(default_ns), False
    ctx = max(vals)
    return ctx, min(vals) != ctx


def align_horizon_to_context(
    horizon: int,
    *,
    research_interval_ns: int,
    context_interval_ns: int,
) -> int:
    """إن ``horizon<=1`` حوّله لعدد فواصل البحث داخل شمعة السياق."""
    h = max(1, int(horizon))
    if h > 1 or int(research_interval_ns) < 1:
        return h
    return max(1, int(context_interval_ns) // int(research_interval_ns))


__all__ = [
    "TemporalPolicy",
    "align_horizon_to_context",
    "resolve_grid_context_interval",
]
