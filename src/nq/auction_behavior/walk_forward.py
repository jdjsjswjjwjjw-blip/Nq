"""Walk-forward متعدد الشرائح — التقسيم على setup فريد ثم توسيع للنتائج."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.models.splitting import WalkForwardFold, purged_walk_forward_split
from nq.validation.leakage import assert_causal_order, assert_temporal_split


@dataclass(frozen=True, slots=True)
class ScienceFold:
    """طيّة علم: حدود زمنية صريحة + مؤشرات الصفوف في جدول الإعدادات."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end_ts: int
    test_start_ts: int
    test_end_ts: int
    segment: str = "time"


def _month_key_from_ns(ts_ns: int) -> str:
    d = dt.datetime.fromtimestamp(ts_ns / 1e9, tz=dt.UTC)
    return f"{d.year:04d}-{d.month:02d}"


def labeled_fold_order(frame: pl.DataFrame, *, ts_col: str = SETUP_AVAILABILITY_TS) -> pl.DataFrame:
    """ترتيب ثابت قبل بناء المؤشرات: الطابع ثم اسم الهدف حتى لا تنقلب الصفوف المتزامنة."""
    if frame.height == 0:
        return frame
    keys = [c for c in (ts_col, "outcome_name") if c in frame.columns]
    if not keys:
        return frame
    return frame.sort(keys, maintain_order=True)


def attach_segment_keys(
    frame: pl.DataFrame,
    *,
    ts_col: str = SETUP_AVAILABILITY_TS,
    instrument_col: str | None = "instrument_id",
) -> pl.DataFrame:
    """يضيف ``segment_month`` و ``segment_contract`` للتقسيم المتعدد."""
    if frame.height == 0 or ts_col not in frame.columns:
        return frame
    ts = [int(x) for x in frame[ts_col].to_list()]
    months = [_month_key_from_ns(t) for t in ts]
    out = frame.with_columns(pl.Series("segment_month", months, dtype=pl.Utf8))
    if instrument_col and instrument_col in out.columns:
        out = out.with_columns(pl.col(instrument_col).cast(pl.Utf8).alias("segment_contract"))
    else:
        out = out.with_columns(pl.lit("SINGLE").alias("segment_contract"))
    return out


def _expand_setup_indices_to_rows(
    frame: pl.DataFrame,
    *,
    ts_col: str,
    setup_times: np.ndarray,
    train_setup_idx: np.ndarray,
    test_setup_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """يوسّع مؤشرات setup الفريدة إلى كل صفوف outcomes التابعة لنفس الطابع."""
    times = frame[ts_col].to_numpy().astype(np.int64)
    train_ts = set(int(x) for x in setup_times[train_setup_idx].tolist())
    test_ts = set(int(x) for x in setup_times[test_setup_idx].tolist())
    # لا تداخل زمني بين القطار والاختبار على مستوى setup
    overlap = train_ts & test_ts
    if overlap:
        raise AssertionError(
            f"setup timestamp assigned to both train and test: {sorted(overlap)[:5]}"
        )
    train_rows = np.flatnonzero(np.isin(times, list(train_ts)))
    test_rows = np.flatnonzero(np.isin(times, list(test_ts)))
    return train_rows.astype(np.intp), test_rows.astype(np.intp)


def unique_setup_timestamps(
    frame: pl.DataFrame, *, ts_col: str = SETUP_AVAILABILITY_TS
) -> np.ndarray:
    """طوابع إعداد فريدة مرتبة (وحدة التقسيم الصحيحة)."""
    if frame.height == 0 or ts_col not in frame.columns:
        return np.zeros(0, dtype=np.int64)
    return (
        frame.select(ts_col)
        .unique(maintain_order=True)
        .sort(ts_col)[ts_col]
        .to_numpy()
        .astype(np.int64)
    )


def build_time_folds(
    times: np.ndarray,
    *,
    n_splits: int,
    embargo: int = 0,
    purge_samples: int = 1,
    min_train_size: int = 12,
) -> list[ScienceFold]:
    """طيّات زمنية purged على طوابع إعداد فريدة (وليس صفوف outcomes مكررة)."""
    times = np.asarray(times, dtype=np.int64)
    # فريد + مرتب
    setup_times = np.unique(times)
    if setup_times.size == 0:
        return []
    assert_causal_order(setup_times)
    try:
        raw = purged_walk_forward_split(
            setup_times,
            n_splits=n_splits,
            embargo=embargo,
            purge_samples=purge_samples,
            min_train_size=min_train_size,
        )
    except ValueError:
        return []
    folds: list[ScienceFold] = []
    for i, wf in enumerate(raw):
        assert_temporal_split(
            setup_times[wf.train_idx], setup_times[wf.test_idx], embargo=float(embargo)
        )
        # هنا المؤشرات على setup_times؛ المُستدعي يوسّعها لصفوف الإطار إن لزم
        folds.append(
            ScienceFold(
                fold=i,
                train_idx=wf.train_idx,
                test_idx=wf.test_idx,
                train_end_ts=int(setup_times[wf.train_idx].max()),
                test_start_ts=int(setup_times[wf.test_idx].min()),
                test_end_ts=int(setup_times[wf.test_idx].max()),
                segment="time",
            )
        )
    return folds


def build_time_folds_for_frame(
    frame: pl.DataFrame,
    *,
    ts_col: str = SETUP_AVAILABILITY_TS,
    n_splits: int = 3,
    embargo: int = 0,
    purge_samples: int = 1,
    min_train_size: int = 12,
) -> list[ScienceFold]:
    """طيّات على setup فريد ثم توسيع لكل صفوف النتائج التابعة."""
    if frame.height == 0 or ts_col not in frame.columns:
        return []
    work = labeled_fold_order(frame, ts_col=ts_col)
    setup_times = unique_setup_timestamps(work, ts_col=ts_col)
    setup_folds = build_time_folds(
        setup_times,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=purge_samples,
        min_train_size=min_train_size,
    )
    out: list[ScienceFold] = []
    for sf in setup_folds:
        train_idx, test_idx = _expand_setup_indices_to_rows(
            work,
            ts_col=ts_col,
            setup_times=setup_times,
            train_setup_idx=sf.train_idx,
            test_setup_idx=sf.test_idx,
        )
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        # ضمان: لا setup_ts مشترك بين train و test
        train_ts = set(work[ts_col].to_numpy()[train_idx].tolist())
        test_ts = set(work[ts_col].to_numpy()[test_idx].tolist())
        if train_ts & test_ts:
            raise AssertionError("duplicate setup_ts across train/test after expansion")
        out.append(
            ScienceFold(
                fold=sf.fold,
                train_idx=train_idx,
                test_idx=test_idx,
                train_end_ts=sf.train_end_ts,
                test_start_ts=sf.test_start_ts,
                test_end_ts=sf.test_end_ts,
                segment=sf.segment,
            )
        )
    return out


def build_expanding_month_folds(
    frame: pl.DataFrame,
    *,
    ts_col: str = SETUP_AVAILABILITY_TS,
    min_train_months: int = 1,
    embargo_ns: int = 0,
    purge_samples: int = 1,
) -> list[ScienceFold]:
    """تدريب متوسّع شهرًا فشهرًا على setup فريد؛ الاختبار = الشهر التالي."""
    if frame.height == 0 or ts_col not in frame.columns:
        return []
    if purge_samples < 0:
        raise ValueError("purge_samples must be non-negative")
    work = attach_segment_keys(labeled_fold_order(frame, ts_col=ts_col), ts_col=ts_col)
    # جدول setup فريد لحساب الشهور
    setups = (
        work.select(ts_col, "segment_month")
        .unique(subset=[ts_col], maintain_order=True)
        .sort(ts_col)
    )
    months = setups["segment_month"].unique(maintain_order=True).to_list()
    if len(months) < min_train_months + 1:
        return []
    setup_times = setups[ts_col].to_numpy().astype(np.int64)
    folds: list[ScienceFold] = []
    fold_i = 0
    for k in range(min_train_months, len(months)):
        train_months = set(months[:k])
        test_month = months[k]
        train_setup_idx = np.flatnonzero(
            setups["segment_month"].is_in(list(train_months)).to_numpy()
        )
        test_setup_idx = np.flatnonzero((setups["segment_month"] == test_month).to_numpy())
        if train_setup_idx.size == 0 or test_setup_idx.size == 0:
            continue
        train_end = int(setup_times[train_setup_idx].max())
        test_start = int(setup_times[test_setup_idx].min())
        if test_start < train_end + int(embargo_ns):
            cutoff = test_start - int(embargo_ns)
            train_setup_idx = train_setup_idx[setup_times[train_setup_idx] <= cutoff]
            if train_setup_idx.size == 0:
                continue
            train_end = int(setup_times[train_setup_idx].max())
        if purge_samples > 0:
            # احذف آخر setups من التدريب على مستوى الطابع الفريد، لا outcome rows.
            train_setup_idx = train_setup_idx[: max(0, train_setup_idx.size - purge_samples)]
            if train_setup_idx.size == 0:
                continue
            train_end = int(setup_times[train_setup_idx].max())
        assert_temporal_split(
            setup_times[train_setup_idx], setup_times[test_setup_idx], embargo=float(embargo_ns)
        )
        train_idx, test_idx = _expand_setup_indices_to_rows(
            work,
            ts_col=ts_col,
            setup_times=setup_times,
            train_setup_idx=train_setup_idx,
            test_setup_idx=test_setup_idx,
        )
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        folds.append(
            ScienceFold(
                fold=fold_i,
                train_idx=train_idx,
                test_idx=test_idx,
                train_end_ts=train_end,
                test_start_ts=int(setup_times[test_setup_idx].min()),
                test_end_ts=int(setup_times[test_setup_idx].max()),
                segment=f"month->{test_month}",
            )
        )
        fold_i += 1
    return folds


def build_contract_aware_folds(
    frame: pl.DataFrame,
    *,
    ts_col: str = SETUP_AVAILABILITY_TS,
    n_splits: int = 3,
    embargo: int = 0,
    purge_samples: int = 1,
    min_train_size: int = 12,
) -> list[ScienceFold]:
    """طيّات زمنية على setup فريد مع تشخيص تعدد الأدوات/العقود.

    وجود أكثر من ``instrument_id`` لا يجعل الاختبار leave-one-contract-out؛
    الزمن يبقى وحدة الفصل لمنع ادعاء تعميم عقدي غير مقاس.
    """
    if frame.height == 0:
        return []
    work = attach_segment_keys(labeled_fold_order(frame, ts_col=ts_col), ts_col=ts_col)
    folds = build_time_folds_for_frame(
        work,
        ts_col=ts_col,
        n_splits=n_splits,
        embargo=embargo,
        purge_samples=purge_samples,
        min_train_size=min_train_size,
    )
    n_contracts = work["segment_contract"].n_unique()
    if n_contracts <= 1:
        return folds
    return [
        ScienceFold(
            fold=f.fold,
            train_idx=f.train_idx,
            test_idx=f.test_idx,
            train_end_ts=f.train_end_ts,
            test_start_ts=f.test_start_ts,
            test_end_ts=f.test_end_ts,
            segment=f"time_multi_instrument({n_contracts})",
        )
        for f in folds
    ]


def folds_to_frame(folds: list[ScienceFold]) -> pl.DataFrame:
    rows = [
        {
            "fold": f.fold,
            "segment": f.segment,
            "train_n": int(f.train_idx.size),
            "test_n": int(f.test_idx.size),
            "train_end_ts": f.train_end_ts,
            "test_start_ts": f.test_start_ts,
            "test_end_ts": f.test_end_ts,
        }
        for f in folds
    ]
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "fold": pl.Int64(),
                "segment": pl.Utf8(),
                "train_n": pl.Int64(),
                "test_n": pl.Int64(),
                "train_end_ts": pl.Int64(),
                "test_start_ts": pl.Int64(),
                "test_end_ts": pl.Int64(),
            }
        )
    )


__all__ = [
    "ScienceFold",
    "WalkForwardFold",
    "attach_segment_keys",
    "build_contract_aware_folds",
    "build_expanding_month_folds",
    "build_time_folds",
    "build_time_folds_for_frame",
    "folds_to_frame",
    "labeled_fold_order",
    "unique_setup_timestamps",
]
