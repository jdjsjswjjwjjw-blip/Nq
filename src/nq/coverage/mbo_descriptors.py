"""إحصاءات MBO كافية لكل نافذة زمنية (MBO Sufficient Statistics).

تُشتق سببيًا من تدفّق MBO الخام وإعادة بناء الدفتر — لا OHLC خارجي.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import AVAILABILITY_TS
from nq.orderbook import reconstruct
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades

_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_FILL = MboAction.FILL.value
_TRADE = MboAction.TRADE.value

_DESCRIPTOR_COLUMNS = (
    "add_count",
    "cancel_count",
    "fill_count",
    "trade_count",
    "trade_volume",
    "cancel_ratio",
    "depth_change",
    "spread",
    "mid",
)


def mbo_window_descriptors(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يبني واصفات MBO لكل نافذة زمنية سبقية (متاحة عند ``bucket_end``)."""
    if frame.height == 0:
        return pl.DataFrame(
            schema={
                BUCKET_START: pl.Int64(),
                BUCKET_END: pl.Int64(),
                AVAILABILITY_TS: pl.Int64(),
                **_descriptor_schema(),
            }
        )

    bucketed = add_time_bucket(frame, interval_ns=interval_ns)
    event_stats = bucketed.group_by(BUCKET_START).agg(
        pl.col(BUCKET_END).first().alias(BUCKET_END),
        (pl.col("action") == _ADD).sum().alias("add_count"),
        (pl.col("action") == _CANCEL).sum().alias("cancel_count"),
        (pl.col("action") == _FILL).sum().alias("fill_count"),
        (pl.col("action") == _TRADE).sum().alias("trade_count"),
    )

    trades = extract_trades(frame)
    if trades.height > 0:
        trade_stats = (
            add_time_bucket(trades, interval_ns=interval_ns)
            .group_by(BUCKET_START)
            .agg(pl.col("size").sum().alias("trade_volume"))
        )
        event_stats = event_stats.join(trade_stats, on=BUCKET_START, how="left")
    else:
        event_stats = event_stats.with_columns(pl.lit(0).alias("trade_volume"))

    tob = reconstruct(frame).top_of_book
    tob_bucketed = add_time_bucket(tob, interval_ns=interval_ns)
    both = pl.col("best_bid").is_not_null() & pl.col("best_ask").is_not_null()
    tob_bucketed = tob_bucketed.with_columns(
        pl.when(both)
        .then((pl.col("best_bid") + pl.col("best_ask")) / 2.0)
        .otherwise(None)
        .alias("mid"),
        pl.when(both).then(pl.col("best_ask") - pl.col("best_bid")).otherwise(None).alias("spread"),
        (pl.col("bid_size").fill_null(0) + pl.col("ask_size").fill_null(0)).alias("total_tob_size"),
    )
    book_stats = tob_bucketed.group_by(BUCKET_START).agg(
        pl.col("mid").last().alias("mid"),
        pl.col("spread").last().alias("spread"),
        (pl.col("total_tob_size").last() - pl.col("total_tob_size").first())
        .abs()
        .alias("depth_change"),
    )

    out = (
        event_stats.join(book_stats, on=BUCKET_START, how="left")
        .with_columns(
            pl.col(BUCKET_END).alias(AVAILABILITY_TS),
            (pl.col("cancel_count") / (pl.col("add_count") + 1)).alias("cancel_ratio"),
            pl.col("trade_volume").fill_null(0),
            pl.col("depth_change").fill_null(0),
            pl.col("spread").fill_null(0),
        )
        .sort(BUCKET_START)
    )
    return out.select(BUCKET_START, BUCKET_END, AVAILABILITY_TS, *_DESCRIPTOR_COLUMNS)


def descriptor_matrix(descriptors: pl.DataFrame) -> tuple[list[str], npt.NDArray[np.float64]]:
    """يحوّل إطار الواصفات إلى مصفوفة رقمية (للاستخدام في المقاييس)."""
    cols = [c for c in _DESCRIPTOR_COLUMNS if c in descriptors.columns]
    matrix = descriptors.select(cols).fill_null(0).to_numpy().astype(np.float64)
    return cols, matrix


def _descriptor_schema() -> dict[str, pl.DataType]:
    return {name: pl.Float64() for name in _DESCRIPTOR_COLUMNS}
