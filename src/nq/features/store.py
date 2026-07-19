"""مخزن الميزات point-in-time (Point-in-Time Feature Store).

المخطط القانوني الطويل (long-form) لكل رصدة ميزة:

* ``feature``         — اسم الميزة (يُنصح بترميز الأداة في الاسم عند اللزوم).
* ``instrument_id``   — معرّف الأداة (كيان)؛ 0 يعني ميزة عبر-سوقية/عامّة.
* ``value``           — القيمة العددية (Float64؛ الأعلام تُحوّل إلى 0/1).
* ``event_ts``        — زمن الحدث الأساس الذي اشتُقّت منه الميزة.
* ``availability_ts`` — زمن إتاحة الميزة (متى أصبحت معلومة) — أساس point-in-time.
* ``version``         — وسم إصدار مجموعة الميزات.

قاعدة منع التسريب: ``availability_ts >= event_ts`` لكل رصدة، ولا يُرجَع في أي
استعلام إلا ما تحقّق شرط ``availability_ts <= t``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.simulation.common import BUCKET_START

FEATURE: Final = "feature"
INSTRUMENT_ID: Final = "instrument_id"
VALUE: Final = "value"
VERSION: Final = "version"

FEATURE_STORE_SCHEMA: Final[dict[str, pl.DataType]] = {
    FEATURE: pl.Utf8(),
    INSTRUMENT_ID: pl.UInt32(),
    VALUE: pl.Float64(),
    EVENT_TS: pl.Int64(),
    AVAILABILITY_TS: pl.Int64(),
    VERSION: pl.Utf8(),
}


def wide_to_features(
    frame: pl.DataFrame,
    *,
    value_columns: Sequence[str],
    version: str,
    instrument_id: int = 0,
    event_col: str = BUCKET_START,
    availability_col: str = AVAILABILITY_TS,
) -> pl.DataFrame:
    """يحوّل مخرَج مُحاكٍ عريضًا (wide) إلى المخطط القانوني الطويل (long).

    تُحوَّل أعمدة القيم إلى ``Float64`` (الأعلام المنطقية إلى 0/1) قبل التذويب
    (unpivot)، ويُنسخ الطابعان الزمنيان ``event_ts`` و ``availability_ts``.
    """
    if not value_columns:
        raise ValueError("value_columns must be non-empty")
    missing = [c for c in (*value_columns, event_col, availability_col) if c not in frame.columns]
    if missing:
        raise ValueError(f"columns not found in frame: {missing}")

    casted = frame.with_columns(pl.col(c).cast(pl.Float64) for c in value_columns)
    long = casted.unpivot(
        index=[event_col, availability_col],
        on=list(value_columns),
        variable_name=FEATURE,
        value_name=VALUE,
    )
    return long.select(
        pl.col(FEATURE),
        pl.lit(instrument_id).cast(pl.UInt32).alias(INSTRUMENT_ID),
        pl.col(VALUE).cast(pl.Float64),
        pl.col(event_col).cast(pl.Int64).alias(EVENT_TS),
        pl.col(availability_col).cast(pl.Int64).alias(AVAILABILITY_TS),
        pl.lit(version).alias(VERSION),
    )


class FeatureStore:
    """مخزن ميزات point-in-time مبني على Polars."""

    __slots__ = ("_data",)

    def __init__(self, data: pl.DataFrame | None = None) -> None:
        self._data = (
            self._validate(data) if data is not None else pl.DataFrame(schema=FEATURE_STORE_SCHEMA)
        )

    @staticmethod
    def _validate(frame: pl.DataFrame) -> pl.DataFrame:
        missing = set(FEATURE_STORE_SCHEMA) - set(frame.columns)
        if missing:
            raise ValueError(f"feature frame missing columns {sorted(missing)}")
        frame = frame.select(list(FEATURE_STORE_SCHEMA)).cast(FEATURE_STORE_SCHEMA)  # type: ignore[arg-type]
        if frame.height:
            bad = frame.filter(pl.col(AVAILABILITY_TS) < pl.col(EVENT_TS)).height
            if bad:
                raise ValueError(
                    f"point-in-time violation: {bad} rows with availability_ts < event_ts."
                )
        return frame

    @property
    def data(self) -> pl.DataFrame:
        """نسخة من كامل بيانات المخزن بالمخطط القانوني."""
        return self._data

    def __len__(self) -> int:
        return self._data.height

    def ingest(self, features: pl.DataFrame) -> FeatureStore:
        """يضيف رصدات ميزات (بالمخطط القانوني) إلى المخزن بعد التحقق."""
        validated = self._validate(features)
        self._data = validated if self._data.height == 0 else pl.concat([self._data, validated])
        return self

    def ingest_wide(
        self,
        frame: pl.DataFrame,
        *,
        value_columns: Sequence[str],
        version: str,
        instrument_id: int = 0,
        event_col: str = BUCKET_START,
        availability_col: str = AVAILABILITY_TS,
    ) -> FeatureStore:
        """يحوّل مخرَج مُحاكٍ عريضًا إلى المخطط القانوني ثم يضيفه."""
        return self.ingest(
            wide_to_features(
                frame,
                value_columns=value_columns,
                version=version,
                instrument_id=instrument_id,
                event_col=event_col,
                availability_col=availability_col,
            )
        )

    def versions(self) -> list[str]:
        """قائمة الإصدارات المتوفّرة في المخزن."""
        return sorted(self._data[VERSION].unique().to_list())

    def _scope(self, *, version: str | None, instrument_id: int | None) -> pl.DataFrame:
        scoped = self._data
        if version is not None:
            scoped = scoped.filter(pl.col(VERSION) == version)
        if instrument_id is not None:
            scoped = scoped.filter(pl.col(INSTRUMENT_ID) == instrument_id)
        return scoped

    def as_of(
        self,
        timestamp: int,
        *,
        version: str | None = None,
        instrument_id: int | None = None,
    ) -> pl.DataFrame:
        """يُعيد أحدث قيمة لكل ميزة كانت متاحة عند ``timestamp`` (point-in-time).

        يُرشّح ``availability_ts <= timestamp`` ثم يأخذ الرصدة الأحدث لكل
        ``(feature, instrument_id)``. لا يُرجِع أي قيمة مستقبلية إطلاقًا.
        """
        scoped = self._scope(version=version, instrument_id=instrument_id).filter(
            pl.col(AVAILABILITY_TS) <= timestamp
        )
        if scoped.height == 0:
            return scoped
        latest = pl.col(AVAILABILITY_TS).max().over([FEATURE, INSTRUMENT_ID])
        return (
            scoped.filter(pl.col(AVAILABILITY_TS) == latest)
            .unique(subset=[FEATURE, INSTRUMENT_ID], keep="last")
            .sort([INSTRUMENT_ID, FEATURE])
        )

    def snapshot_series(
        self,
        *,
        version: str | None = None,
        instrument_id: int | None = None,
    ) -> pl.DataFrame:
        """يبني سلسلة لقطات point-in-time عريضة (feature-per-column) مملوءة أماميًا.

        كل صف عند ``availability_ts`` يحمل آخر قيمة معروفة لكل ميزة حتى تلك
        اللحظة (forward-fill سببي)، وهو الأساس لدمج point-in-time.
        """
        scoped = self._scope(version=version, instrument_id=instrument_id)
        if scoped.height == 0:
            return pl.DataFrame(schema={AVAILABILITY_TS: pl.Int64()})
        wide = scoped.pivot(
            on=FEATURE,
            index=AVAILABILITY_TS,
            values=VALUE,
            aggregate_function="last",
        ).sort(AVAILABILITY_TS)
        feature_cols = [c for c in wide.columns if c != AVAILABILITY_TS]
        return wide.with_columns(pl.col(feature_cols).forward_fill())

    def point_in_time_join(
        self,
        query: pl.DataFrame,
        *,
        ts_col: str,
        version: str | None = None,
        instrument_id: int | None = None,
    ) -> pl.DataFrame:
        """يدمج قيم الميزات المتاحة point-in-time على طوابع زمنية استعلامية.

        لكل صف في ``query`` عند زمن ``t``، يُلحَق آخر قيمة معروفة لكل ميزة حيث
        ``availability_ts <= t`` عبر ``join_asof`` رجعي (backward) — بلا تسريب.
        """
        snapshots = self.snapshot_series(version=version, instrument_id=instrument_id)
        if snapshots.height == 0 or snapshots.width == 1:
            return query
        query_sorted = query.sort(ts_col)
        return query_sorted.join_asof(
            snapshots,
            left_on=ts_col,
            right_on=AVAILABILITY_TS,
            strategy="backward",
        )

    def to_parquet(self, path: str | Path) -> None:
        """يحفظ المخزن عموديًا (Parquet)."""
        self._data.write_parquet(Path(path))

    @classmethod
    def read_parquet(cls, path: str | Path) -> FeatureStore:
        """يقرأ مخزنًا من ملف Parquet ويتحقق من مخططه."""
        return cls(pl.read_parquet(Path(path)))
