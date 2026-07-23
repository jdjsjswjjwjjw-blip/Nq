"""تحويل مخرجات Databento MBO إلى عقد ``MBO_SCHEMA`` القانوني.

يُستدعى تلقائيًا من ``load_mbo_frame`` عند اكتشاف أعمدة Databento.
لا يتطلّب حزمة ``databento`` — يعمل على Parquet/Arrow المُصدَّرة.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nq.contracts.mbo import MBO_SCHEMA, PRICE_SCALE, MboAction, MboSide
from nq.contracts.temporal import EVENT_TS, INGEST_TS, SEQUENCE

# إعادة تسمية بدون تصادم (لا نُعيد ts_recv و ts_in_delta إلى نفس العمود)
_DATABENTO_RENAMES: Final[dict[str, str]] = {
    "ts_event": EVENT_TS,
    "ts_recv": INGEST_TS,
}

_ACTION_MAP: Final[dict[str, str]] = {
    "add": MboAction.ADD.value,
    "cancel": MboAction.CANCEL.value,
    "modify": MboAction.MODIFY.value,
    "clear": MboAction.CLEAR.value,
    "trade": MboAction.TRADE.value,
    "fill": MboAction.FILL.value,
    "a": MboAction.ADD.value,
    "c": MboAction.CANCEL.value,
    "m": MboAction.MODIFY.value,
    "r": MboAction.CLEAR.value,
    "t": MboAction.TRADE.value,
    "f": MboAction.FILL.value,
}

_SIDE_MAP: Final[dict[str, str]] = {
    "bid": MboSide.BID.value,
    "ask": MboSide.ASK.value,
    "none": MboSide.NONE.value,
    "b": MboSide.BID.value,
    "a": MboSide.ASK.value,
    "n": MboSide.NONE.value,
}

_CLEAR_ACTION = MboAction.CLEAR.value
_NONE_ACTION = MboAction.NONE.value
_SOURCE_ROW = "__databento_source_row"


def is_databento_frame(frame: pl.DataFrame) -> bool:
    """هل الإطار يحمل أعمدة Databento (وليس العقد القانوني بعد)؟"""
    cols = set(frame.columns)
    if EVENT_TS in cols:
        return False
    return "ts_event" in cols or "ts_recv" in cols


def _rename_databento_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """يُعيد تسمية أعمدة Databento دون إنشاء أعمدة مكرّرة."""
    renamed = frame
    for src, dst in _DATABENTO_RENAMES.items():
        if src in renamed.columns and dst not in renamed.columns:
            renamed = renamed.rename({src: dst})
    if INGEST_TS not in renamed.columns and "ts_in_delta" in renamed.columns:
        renamed = renamed.rename({"ts_in_delta": INGEST_TS})
    return renamed


def _ensure_flags_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "flags" in frame.columns:
        return frame
    if "rtype" in frame.columns:
        return frame.with_columns(pl.col("rtype").cast(pl.UInt8).alias("flags"))
    return frame.with_columns(pl.lit(0, dtype=pl.UInt8).alias("flags"))


def _scale_price_column(frame: pl.DataFrame) -> pl.DataFrame:
    """يحوّل أسعار float (دولار) إلى fixed-point Int64 وفق ``PRICE_SCALE``."""
    dtype = frame.schema["price"]
    if dtype.is_float():
        scaled = (pl.col("price") / PRICE_SCALE).round().cast(pl.Int64)
    else:
        scaled = pl.col("price").cast(pl.Int64)
    return frame.with_columns(
        pl.when(pl.col("price").is_null())
        .then(0)
        .otherwise(scaled)
        .alias("price")
    )


def _sanitize_prices(frame: pl.DataFrame) -> pl.DataFrame:
    """يُسقط صفوف السعر الفارغ (ما عدا Clear/None) ثم يُحوّل السعر إلى fixed-point."""
    action = pl.col("action").cast(pl.Utf8).str.to_uppercase()
    keep = pl.col("price").is_not_null() | action.is_in([_CLEAR_ACTION, _NONE_ACTION])
    cleaned = frame.filter(keep)
    return _scale_price_column(cleaned)


def _source_order_columns(frame: pl.DataFrame) -> list[str]:
    preferred = [
        EVENT_TS,
        INGEST_TS,
        "publisher_id",
        "instrument_id",
        "order_id",
        "action",
        "side",
        "price",
        "size",
        "flags",
        _SOURCE_ROW,
    ]
    return [col for col in preferred if col in frame.columns]


def normalize_databento_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """يحوّل إطار Databento MBO إلى مخطط ``MBO_SCHEMA``."""
    renamed = _rename_databento_columns(frame).with_columns(
        pl.arange(0, frame.height, dtype=pl.UInt64).alias(_SOURCE_ROW)
    )
    renamed = _ensure_flags_column(renamed)

    if EVENT_TS not in renamed.columns:
        raise ValueError("Databento frame must contain ts_event or event_ts")

    if INGEST_TS not in renamed.columns:
        renamed = renamed.with_columns(pl.col(EVENT_TS).alias(INGEST_TS))

    action_expr = pl.col("action").cast(pl.Utf8).str.to_lowercase()
    for raw, canonical in _ACTION_MAP.items():
        action_expr = pl.when(action_expr == raw).then(pl.lit(canonical)).otherwise(action_expr)
    renamed = renamed.with_columns(action_expr.alias("action"))

    side_expr = pl.col("side").cast(pl.Utf8).str.to_lowercase()
    for raw, canonical in _SIDE_MAP.items():
        side_expr = pl.when(side_expr == raw).then(pl.lit(canonical)).otherwise(side_expr)
    renamed = renamed.with_columns(side_expr.alias("side"))

    renamed = _sanitize_prices(renamed)

    if SEQUENCE not in renamed.columns:
        renamed = renamed.sort(_source_order_columns(renamed)).with_columns(
            pl.arange(1, renamed.height + 1, dtype=pl.UInt64).alias(SEQUENCE)
        )

    missing = [name for name in MBO_SCHEMA if name not in renamed.columns]
    if missing:
        raise ValueError(f"Databento frame missing required fields after mapping: {missing}")

    drop_extra = [c for c in renamed.columns if c not in MBO_SCHEMA]
    if drop_extra:
        renamed = renamed.drop(drop_extra)

    return renamed.select([pl.col(name).cast(dtype) for name, dtype in MBO_SCHEMA.items()])


__all__ = ["is_databento_frame", "normalize_databento_frame"]
