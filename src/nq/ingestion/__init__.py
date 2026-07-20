"""استيعاب بيانات MBO (Ingestion) — قراءة تدفّقية تُخضِع كل صفّ للعقد.

يقرأ هذا المكوّن بيانات MBO من ملفات عمودية (Parquet/Arrow) أو من إطار Polars
جاهز، ويتحقق من مطابقتها لعقد ``MBO_SCHEMA``، ثم يرتّبها سببيًا ويسلّمها على
دفعات (batches) بذاكرة ثابتة لدعم البيانات الضخمة.
"""

from __future__ import annotations

from nq.ingestion.databento import is_databento_frame, normalize_databento_frame
from nq.ingestion.reader import iter_mbo_batches, load_mbo_frame

__all__ = [
    "is_databento_frame",
    "iter_mbo_batches",
    "load_mbo_frame",
    "normalize_databento_frame",
]
