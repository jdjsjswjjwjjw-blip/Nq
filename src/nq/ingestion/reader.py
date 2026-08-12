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
from typing import cast

import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as pa_csv  # type: ignore[import-untyped]
import pyarrow.parquet as pa_parquet  # type: ignore[import-untyped]

from nq.contracts.mbo import MBO_SCHEMA, MboAction, validate_mbo_frame
from nq.core.time import sort_causal
from nq.ingestion.databento import is_databento_frame, normalize_databento_frame
from nq.research.progress import ProgressLike

_CLEAR = MboAction.CLEAR.value
_NONE = MboAction.NONE.value
_ARROW_FILE_MAGIC = b"ARROW1"


def _read_zst_bytes(path: Path) -> bytes:
    try:
        import zstandard as zstd  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dep path
        raise ValueError(
            f"cannot read {path.name!r}: install optional dependency `zstandard` for .zst files"
        ) from exc
    with path.open("rb") as handle:
        raw = zstd.ZstdDecompressor().decompress(handle.read())
    return bytes(raw)


def _read_columnar(path: Path, *, max_rows: int | None = None) -> pl.DataFrame:
    """يقرأ الملف كاملًا؛ القصّ السببي يتم بعد الترتيب في ``load_mbo_frame``.

    ``max_rows`` هنا مُتجاهل عند القراءة الخام حتى لا نأخذ رأس الملف غير المرتّب.
    """
    del max_rows  # القص بعد sort_causal فقط
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".parquet" or name.endswith(".parquet.zst"):
        return pl.read_parquet(path)

    if suffix in {".arrow", ".ipc", ".feather"}:
        return pl.read_ipc(path)

    if suffix == ".csv":
        return pl.read_csv(path)

    if suffix == ".zst":
        raw = _read_zst_bytes(path)
        if raw[:4] == b"PAR1":
            return pl.read_parquet(io.BytesIO(raw))
        return pl.read_csv(io.BytesIO(raw))

    raise ValueError(
        f"unsupported MBO file format {suffix!r}; expected .parquet/.arrow/.ipc/.csv/.zst"
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
    # CSV/Arrow قد يعيدان الأعداد كـInt64 والنصوص كـString؛ طبّع إلى العقد
    # القانوني بقصّ صارم (القيم السالبة/الفئات المجهولة تظل أخطاء).
    frame = frame.select(
        [pl.col(name).cast(dtype) for name, dtype in MBO_SCHEMA.items() if name in frame.columns]
    )
    validate_mbo_frame(frame)
    return sort_causal(frame)


def _slice_arrow_batch(batch: pa.RecordBatch, batch_size: int) -> Iterator[pl.DataFrame]:
    """حوّل دفعة Arrow إلى شرائح Polars لا تتجاوز ``batch_size``."""
    for start in range(0, batch.num_rows, batch_size):
        yield cast(pl.DataFrame, pl.from_arrow(batch.slice(start, batch_size)))


def _iter_columnar_batches(path: Path, batch_size: int) -> Iterator[pl.DataFrame]:
    """اقرأ ملفًا عموديًا بدفعات فعلية من القرص، من دون تجسيد الملف كاملًا."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        parquet = pa_parquet.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield cast(pl.DataFrame, pl.from_arrow(batch))
        return

    if suffix in {".arrow", ".ipc", ".feather"}:
        with pa.memory_map(str(path), "r") as source:
            # ملفات IPC لها magic ثابت؛ stream IPC لا يملكه.
            magic_size = len(_ARROW_FILE_MAGIC)
            is_file = (
                source.size() >= magic_size and source.read_at(magic_size, 0) == _ARROW_FILE_MAGIC
            )
            if is_file:
                reader = pa.ipc.open_file(source)
                for index in range(reader.num_record_batches):
                    yield from _slice_arrow_batch(reader.get_batch(index), batch_size)
            else:
                reader = pa.ipc.open_stream(source)
                for batch in reader:
                    yield from _slice_arrow_batch(batch, batch_size)
        return

    if suffix == ".csv":
        # block_size يحد ذاكرة parser؛ عدد الصفوف النهائي يُضبط بالـ slicing.
        read_options = pa_csv.ReadOptions(block_size=max(1 << 20, batch_size * 128))
        with pa_csv.open_csv(path, read_options=read_options) as reader:
            for batch in reader:
                yield from _slice_arrow_batch(batch, batch_size)
        return

    if suffix == ".zst" or path.name.lower().endswith(".parquet.zst"):
        raise ValueError(
            "true streaming does not support whole-file .zst containers; "
            "decompress once to .parquet/.arrow/.csv, then stream that file"
        )

    raise ValueError(f"unsupported MBO file format {suffix!r}; expected .parquet/.arrow/.ipc/.csv")


def _causal_key(frame: pl.DataFrame, row: int) -> tuple[int, int]:
    return int(frame["event_ts"][row]), int(frame["sequence"][row])


def load_mbo_frame(
    source: pl.DataFrame | str | Path,
    *,
    max_rows: int | None = None,
    progress: ProgressLike | None = None,
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
            log.op(f"MBO من DataFrame جاهز: {source.height:,} صف")
        frame = source
    else:
        path = Path(source)
        if log is not None:
            log.op(f"قراءة ملف MBO: {path.resolve()}")
        frame = _read_columnar(path)
        if log is not None:
            log.op(f"قُرئ الخام: {frame.height:,} صف × {frame.width} عمود")

    if log is not None:
        log.op("تطبيع Databento / التحقق من MBO_SCHEMA / ترتيب سببي")
    frame = _prepare_frame(frame)
    if max_rows is not None and frame.height > max_rows:
        if log is not None:
            log.op(f"قص سببي بعد الترتيب إلى max_rows={max_rows:,} (أقدم {max_rows:,})")
        frame = frame.head(max_rows)
    if log is not None:
        log.op(f"جاهز بعد التحضير: {frame.height:,} صف")
    return frame


def iter_mbo_batches(
    source: pl.DataFrame | str | Path,
    *,
    batch_size: int = 5_000_000,
    max_rows: int | None = None,
) -> Iterator[pl.DataFrame]:
    """يسلّم بيانات MBO على دفعات سببية متتابعة بذاكرة محدودة.

    الملفات تُقرأ مباشرة بدفعات Arrow. تُرتّب كل دفعة داخليًا، ثم يُتحقق من
    الحد الفاصل مع الدفعة السابقة. لا يمكن للتدفّق إصلاح ملف غير مرتّب عالميًا
    من دون تحميله كاملًا؛ لذلك يُرفض الحد المخالف بدل إصدار ترتيب فاسد صامت.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")

    raw_batches: Iterator[pl.DataFrame]
    if isinstance(source, pl.DataFrame):
        # إطار الذاكرة موجود أصلًا؛ رتّبه عالميًا مرة واحدة للحفاظ على العقد القديم.
        prepared = _prepare_frame(source)
        raw_batches = (
            prepared.slice(start, batch_size) for start in range(0, prepared.height, batch_size)
        )
    else:
        raw_batches = _iter_columnar_batches(Path(source), batch_size)

    emitted = 0
    generated_sequence_offset = 0
    previous_last: tuple[int, int] | None = None
    for raw in raw_batches:
        if raw.is_empty():
            continue
        generated_sequence = is_databento_frame(raw) and "sequence" not in raw.columns
        frame = raw if isinstance(source, pl.DataFrame) else _prepare_frame(raw)
        if generated_sequence:
            # normalize_databento_frame يولّد sequence داخل كل دفعة؛ أضف offset
            # عالميًا حتى لا يعاد 0 عند حد Arrow ويبطل ترتيب أحداث ذات ts واحد.
            frame = frame.with_columns(
                (pl.col("sequence") + generated_sequence_offset).alias("sequence")
            )
            generated_sequence_offset += raw.height
        if frame.is_empty():
            continue

        first = _causal_key(frame, 0)
        if previous_last is not None and first < previous_last:
            raise ValueError(
                "causal-order violation across streaming batches: "
                f"current first key {first} precedes previous last key {previous_last}; "
                "sort the source once before streaming"
            )

        remaining = frame.height if max_rows is None else max_rows - emitted
        if remaining <= 0:
            break
        if frame.height > remaining:
            frame = frame.head(remaining)

        previous_last = _causal_key(frame, -1)
        emitted += frame.height
        yield frame
        if max_rows is not None and emitted >= max_rows:
            break
