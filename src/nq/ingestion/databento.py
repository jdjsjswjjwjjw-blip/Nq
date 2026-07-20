"""تحويل مخرجات Databento MBO إلى عقد ``MBO_SCHEMA`` القانوني.

يُستدعى تلقائيًا من ``load_mbo_frame`` عند اكتشاف أعمدة Databento.
لا يتطلّب حزمة ``databento`` — يعمل على Parquet/Arrow المُصدَّرة.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nq.contracts.mbo import MBO_SCHEMA, MboAction, MboSide
from nq.contracts.temporal import EVENT_TS, INGEST_TS, SEQUENCE

# أعمدة Databento الشائعة → العقد القانوني
_DATABENTO_COLUMN_MAP: Final[dict[str, str]] = {
    "ts_event": EVENT_TS,
    "ts_recv": INGEST_TS,
    "ts_in_delta": INGEST_TS,
    "rtype": "flags",
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


def is_databento_frame(frame: pl.DataFrame) -> bool:
    """هل الإطار يحمل أعمدة Databento (وليس العقد القانوني بعد)؟"""
    cols = set(frame.columns)
    if EVENT_TS in cols:
        return False
    return "ts_event" in cols or "ts_recv" in cols


def normalize_databento_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """يحوّل إطار Databento MBO إلى مخطط ``MBO_SCHEMA``."""
    renamed = frame.rename({k: v for k, v in _DATABENTO_COLUMN_MAP.items() if k in frame.columns})

    if EVENT_TS not in renamed.columns:
        raise ValueError("Databento frame must contain ts_event or event_ts")

    if INGEST_TS not in renamed.columns:
        renamed = renamed.with_columns(pl.col(EVENT_TS).alias(INGEST_TS))

    if SEQUENCE not in renamed.columns:
        renamed = renamed.with_columns(
            pl.arange(0, renamed.height, dtype=pl.UInt64).alias(SEQUENCE)
        )

    if "flags" not in renamed.columns:
        renamed = renamed.with_columns(pl.lit(0, dtype=pl.UInt8).alias("flags"))

    action_expr = pl.col("action").cast(pl.Utf8).str.to_lowercase()
    for raw, canonical in _ACTION_MAP.items():
        action_expr = pl.when(action_expr == raw).then(pl.lit(canonical)).otherwise(action_expr)
    renamed = renamed.with_columns(action_expr.alias("action"))

    side_expr = pl.col("side").cast(pl.Utf8).str.to_lowercase()
    for raw, canonical in _SIDE_MAP.items():
        side_expr = pl.when(side_expr == raw).then(pl.lit(canonical)).otherwise(side_expr)
    renamed = renamed.with_columns(side_expr.alias("side"))

    missing = [name for name in MBO_SCHEMA if name not in renamed.columns]
    if missing:
        raise ValueError(f"Databento frame missing required fields after mapping: {missing}")

    return renamed.select([pl.col(name).cast(dtype) for name, dtype in MBO_SCHEMA.items()])


__all__ = ["is_databento_frame", "normalize_databento_frame"]
