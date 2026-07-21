"""قارئ MBO التدفّقي (Streaming MBO Reader).

المصدر الوحيد للحقيقة هو تدفّق MBO الخام. يقرأ هذا القارئ البيانات من:

* إطار Polars جاهز (``pl.DataFrame``)، أو
* ملف عمودي على القرص (``.parquet`` / ``.arrow`` / ``.ipc`` / ``.csv`` / ``.zst``).

ثم يُخضِع البيانات لعقد ``MBO_SCHEMA`` (بنية + نقطة زمنية)، ويرتّبها سببيًا
``(event_ts, sequence)``، ويسلّمها اختياريًا على دفعات ثابتة الحجم.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from nq.contracts.mbo import MBO_SCHEMA, MboAction, validate_mbo_frame
from nq.core.time import sort_causal
from nq.ingestion.databento import is_databento_frame, normalize_databento_frame

_CLEAR = MboAction.CLEAR.value
_NONE = MboAction.NONE.value


def _read_zst_bytes(path: Path) -> bytes:
    try:
        import zstandard as zstd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dep path
        raise ValueError(
            f"cannot read {path.name!r}: install optional dependency `zstandard` for .zst files"
        ) from exc
    with path.open("rb") as handle:
        return zstd.ZstdDecompressor().decompress(handle.read())


def _read_columnar(path: Path, *, max_rows: int | None = None) -> pl.DataFrame:
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".parquet" or name.endswith(".parquet.zst"):
        if max_rows is not None:
            return pl.read_parquet(path, n_rows=max_rows)
        return pl.read_parquet(path)

    if suffix in {".arrow", ".ipc", ".feather"}:
        frame = pl.read_ipc(path)
        return frame.head(max_rows) if max_rows is not None else frame

    if suffix == ".csv":
        if max_rows is not None:
            return pl.read_csv(path, n_rows=max_rows)
        return pl.read_csv(path)

    if suffix == ".zst":
        raw = _read_zst_bytes(path)
        if raw[:4] == b"PAR1":
            frame = pl.read_parquet(io.BytesIO(raw))
        else:
            frame = pl.read_csv(io.BytesIO(raw))
        return frame.head(max_rows) if max_rows is not None else frame

    raise ValueError(
        f"unsupported MBO file format {suffix!r}; "
        "expected .parquet/.arrow/.ipc/.csv/.zst"
    )


def sanitize_mbo_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """يُعالج أسعار null قبل إعادة بناء الدفتر (Clear/None → 0)."""
    if "price" not in frame.columns:
        return frame
    action_col = pl.col("action").cast(pl.Utf8).str.to_uppercase()
    return frame.with_columns(
        pl.when(pl.col("price").is_null() & action_col.is_in([_CLEAR, _NONE]))
        .then(0)
        .otherwise(pl.col("price"))
        .alias("price")
    ).filter(pl.col("price").is_not_null() | action_col.is_in([_CLEAR, _NONE]))


def _prepare_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if is_databento_frame(frame):
        frame = normalize_databento_frame(frame)
    frame = frame.select([name for name in MBO_SCHEMA if name in frame.columns])
    frame = sanitize_mbo_frame(frame)
    validate_mbo_frame(frame)
    return sort_causal(frame)


def load_mbo_frame(
    source: pl.DataFrame | str | Path,
    *,
    max_rows: int | None = None,
    progress: object | None = None,
) -> pl.DataFrame:
    """يُحمّل بيانات MBO ويتحقق من العقد ويرتّبها سبقيًا.

    يقبل إطار Polars مباشرةً أو مسار ملف عمودي. ``max_rows`` يحدّ الحجم للتجارب
    أو الأجهزة محدودة الذاكرة (يُطبَّق بعد الترتيب السببي).
    """
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")

    log = progress
    if isinstance(source, pl.DataFrame):
        if log is not None:
            log.op(f"MBO من DataFrame جاهز: {source.height:,} صف")  # type: ignore[union-attr]
        frame = source
        if max_rows is not None:
            if log is not None:
                log.op(f"قص DataFrame إلى max_rows={max_rows:,}")  # type: ignore[union-attr]
            frame = frame.head(max_rows)
    else:
        path = Path(source)
        if log is not None:
            detail = f" (n_rows={max_rows:,})" if max_rows is not None else ""
            log.op(f"قراءة ملف MBO: {path.resolve()}{detail}")  # type: ignore[union-attr]
        frame = _read_columnar(path, max_rows=max_rows)
        if log is not None:
            log.op(f"قُرئ الخام: {frame.height:,} صف × {frame.width} عمود")  # type: ignore[union-attr]

    if log is not None:
        log.op("تطبيع Databento / التحقق من MBO_SCHEMA / ترتيب سببي")  # type: ignore[union-attr]
    frame = _prepare_frame(frame)
    if log is not None:
        log.op(f"جاهز بعد التحضير: {frame.height:,} صف")  # type: ignore[union-attr]
    return frame


def iter_mbo_batches(
    source: pl.DataFrame | str | Path,
    *,
    batch_size: int = 5_000_000,
    max_rows: int | None = None,
) -> Iterator[pl.DataFrame]:
    """يسلّم بيانات MBO على دفعات سبقية متتابعة بذاكرة ثابتة."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    frame = load_mbo_frame(source, max_rows=max_rows)
    total = frame.height
    for start in range(0, total, batch_size):
        yield frame.slice(start, batch_size)
