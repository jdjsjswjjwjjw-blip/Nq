"""تقطيع التسلسلات الزمنية السببية (Causal Sequence Windowing).

يبني عيّنات تسلسلية منزلقة من مصفوفة ميزات مرتّبة زمنيًا: العيّنة عند المؤشّر
``i`` تضمّ النافذة ``[i-window+1 .. i]`` (الماضي والحاضر فقط)، وطابعها الزمني هو
``availability_ts[i]``. اختلاف طول النافذة يمثّل مقاييس زمنية متعدّدة
(multi-scale): من microseconds إلى minutes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    """مجموعة تسلسلات سببية.

    * ``x``: مصفوفة ``(n_samples, window, n_features)``.
    * ``times``: ``(n_samples,)`` الطابع الزمني لإتاحة كل عيّنة (نهاية النافذة).
    * ``feature_names``: أسماء الميزات بترتيب البُعد الأخير.
    """

    x: FloatArray
    times: IntArray
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def flatten(self) -> FloatArray:
        """يفرد النوافذ إلى مصفوفة ثنائية ``(n_samples, window * n_features)``."""
        n = self.x.shape[0]
        return self.x.reshape(n, -1)


def build_sequences(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    window: int,
    time_col: str = AVAILABILITY_TS,
    stride: int = 1,
    progress: ProgressLike | None = None,
) -> SequenceDataset:
    """يبني ``SequenceDataset`` سببيًا من إطار ميزات مرتّب زمنيًا.

    يُفترض أن الإطار مرتّب تصاعديًا بـ ``time_col`` (سببي). كل نافذة تستخدم
    الماضي والحاضر فقط، فلا تسريب مستقبليًا.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")

    times_all = frame[time_col].to_numpy().astype(np.int64)
    if times_all.shape[0] and bool(np.any(np.diff(times_all) < 0)):
        raise ValueError(f"{time_col} must be non-decreasing (causal order).")

    values = frame.select(feature_columns).to_numpy().astype(np.float64, copy=False)
    n_rows = values.shape[0]
    n_features = len(feature_columns)
    ends = np.arange(window - 1, n_rows, stride, dtype=np.int64)
    n_ends = int(ends.shape[0])
    if progress is not None:
        progress.op(f"build_sequences: {n_ends:,} نافذة · window={window} · rows={n_rows:,}")

    if n_ends == 0:
        x = np.empty((0, window, n_features), dtype=np.float64)
        times = np.empty(0, dtype=np.int64)
    else:
        # sliding_window_view(..., axis=0) → (n-w+1, n_features, window)
        windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window, axis=0)
        x = np.ascontiguousarray(np.transpose(windows[::stride], (0, 2, 1)))
        times = times_all[ends]
        if progress is not None:
            progress.heartbeat(n_ends, n_ends, label="sequences", force=True)

    return SequenceDataset(
        x=x,
        times=times,
        feature_names=tuple(feature_columns),
    )


@dataclass(frozen=True, slots=True)
class TickSequenceDataset(SequenceDataset):
    """تسلسل tick مع بيانات وصفية لمسارات الإخفاء الهيكلي.

    * ``mask_paths``: ``(n_samples,)`` مسار الإخفاء عند نهاية النافذة.
    * ``market_phases``: ``(n_samples,)`` مرحلة السوق (balance/expansion/neutral).
    """

    mask_paths: IntArray
    market_phases: IntArray


def build_tick_sequences(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    window: int,
    time_col: str = AVAILABILITY_TS,
    stride: int = 1,
    progress: ProgressLike | None = None,
) -> TickSequenceDataset:
    """يبني تسلسلات tick/event مع ``mask_path`` و ``market_phase`` لكل عيّنة."""
    base = build_sequences(
        frame,
        feature_columns=feature_columns,
        window=window,
        time_col=time_col,
        stride=stride,
        progress=progress,
    )
    if base.x.shape[0] == 0:
        return TickSequenceDataset(
            x=base.x,
            times=base.times,
            feature_names=base.feature_names,
            mask_paths=np.empty(0, dtype=np.int64),
            market_phases=np.empty(0, dtype=np.int64),
        )

    ends = np.arange(window - 1, frame.height, stride, dtype=np.int64)
    mask_paths = frame["mask_path"].to_numpy().astype(np.int64)[ends]
    market_phases = frame["market_phase"].to_numpy().astype(np.int64)[ends]
    return TickSequenceDataset(
        x=base.x,
        times=base.times,
        feature_names=base.feature_names,
        mask_paths=mask_paths,
        market_phases=market_phases,
    )
