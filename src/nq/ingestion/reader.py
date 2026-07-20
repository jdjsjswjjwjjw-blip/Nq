"""قارئ MBO التدفّقي (Streaming MBO Reader).

المصدر الوحيد للحقيقة هو تدفّق MBO الخام. يقرأ هذا القارئ البيانات من:

* إطار Polars جاهز (``pl.DataFrame``)، أو
* ملف عمودي على القرص (``.parquet`` / ``.arrow`` / ``.ipc``).

ثم يُخضِع البيانات لعقد ``MBO_SCHEMA`` (بنية + نقطة زمنية)، ويرتّبها سببيًا
``(event_ts, sequence)``، ويسلّمها اختياريًا على دفعات ثابتة الحجم.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from nq.contracts.mbo import MBO_SCHEMA, validate_mbo_frame
from nq.core.time import sort_causal
from nq.ingestion.databento import is_databento_frame, normalize_databento_frame


def _read_columnar(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".arrow", ".ipc", ".feather"}:
        return pl.read_ipc(path)
    raise ValueError(f"unsupported MBO file format {suffix!r}; expected .parquet/.arrow/.ipc")


def load_mbo_frame(source: pl.DataFrame | str | Path) -> pl.DataFrame:
    """يُحمّل بيانات MBO ويتحقق من العقد ويرتّبها سببيًا.

    يقبل إطار Polars مباشرةً أو مسار ملف عمودي. يرفع ``ValueError`` عند أي
    خرق للعقد (نقص أعمدة، أنواع خاطئة، أو ``ingest_ts < event_ts``).
    """
    frame = source if isinstance(source, pl.DataFrame) else _read_columnar(Path(source))

    if is_databento_frame(frame):
        frame = normalize_databento_frame(frame)

    # فرض ترتيب الأعمدة القانوني قبل التحقق لتفادي التباسات المخطط.
    frame = frame.select([name for name in MBO_SCHEMA if name in frame.columns])
    validate_mbo_frame(frame)
    return sort_causal(frame)


def iter_mbo_batches(
    source: pl.DataFrame | str | Path,
    *,
    batch_size: int = 5_000_000,
) -> Iterator[pl.DataFrame]:
    """يسلّم بيانات MBO على دفعات سببية متتابعة بذاكرة ثابتة.

    يحافظ على الترتيب السببي العام عبر الدفعات (الدفعة i تسبق i+1 زمنيًا)،
    ما يجعله ملائمًا للمعالجة التدفّقية للبيانات الضخمة دون تحميلها كاملة.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    frame = load_mbo_frame(source)
    total = frame.height
    for start in range(0, total, batch_size):
        yield frame.slice(start, batch_size)
