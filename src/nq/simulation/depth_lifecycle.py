"""دورة حياة العمق السببية — دخول / مراقبة / تنفيذ / خروج.

يبني سلسلة لقطات عمق من MBO دون طمس السلم:

* عند كل حدث: لقطة مراقبة (``availability_ts = event_ts``).
* عند إغلاق كل شمعة: لقطة قرار (``availability_ts = bucket_end``)
  من الدفتر بعد آخر حدث داخل الشمعة.

التنفيذ والخروج يستخدمان مسح المستويات الظاهرة فقط (بلا اختلاق عمق).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from nq.contracts.instruments import require_single_contract_identity
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.time import sort_causal
from nq.orderbook.book import OrderBook
from nq.orderbook.depth import DepthSnapshot
from nq.simulation.common import BUCKET_END, BUCKET_START

_DEFAULT_LEVELS: Final = 5

DEPTH_MONITOR_COLUMNS: Final[tuple[str, ...]] = (
    "depth_cum_bid",
    "depth_cum_ask",
    "depth_imbalance",
    "depth_trail_bid",
    "depth_trail_ask",
    "depth_bid_sz_1",
    "depth_ask_sz_1",
    "depth_l1_spread",
)


def _empty_depth_schema(*, n_levels: int) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        AVAILABILITY_TS: pl.Int64(),
        EVENT_TS: pl.Int64(),
        "depth_cum_bid": pl.Float64(),
        "depth_cum_ask": pl.Float64(),
        "depth_imbalance": pl.Float64(),
        "depth_trail_bid": pl.Float64(),
        "depth_trail_ask": pl.Float64(),
        "depth_l1_spread": pl.Float64(),
        "nq_bid": pl.Float64(),
        "nq_ask": pl.Float64(),
    }
    for k in range(1, n_levels + 1):
        schema[f"depth_bid_px_{k}"] = pl.Float64()
        schema[f"depth_bid_sz_{k}"] = pl.Float64()
        schema[f"depth_ask_px_{k}"] = pl.Float64()
        schema[f"depth_ask_sz_{k}"] = pl.Float64()
    return schema


def snapshot_to_row(snap: DepthSnapshot, *, event_ts: int) -> dict[str, float | int | None]:
    """يحوّل لقطة عمق إلى صف أعمدة مسطّحة (أسعار حقيقية)."""
    row: dict[str, float | int | None] = {
        AVAILABILITY_TS: snap.availability_ts,
        EVENT_TS: event_ts,
        "depth_cum_bid": float(snap.cum_bid),
        "depth_cum_ask": float(snap.cum_ask),
        "depth_imbalance": float(snap.imbalance),
        "depth_trail_bid": float(snap.trail_bid),
        "depth_trail_ask": float(snap.trail_ask),
        "depth_l1_spread": (
            float((snap.best_ask - snap.best_bid) * PRICE_SCALE)
            if snap.best_bid is not None and snap.best_ask is not None
            else None
        ),
        "nq_bid": None if snap.best_bid is None else float(snap.best_bid) * PRICE_SCALE,
        "nq_ask": None if snap.best_ask is None else float(snap.best_ask) * PRICE_SCALE,
    }
    for k in range(1, snap.n_levels + 1):
        if k <= len(snap.bid_levels):
            px, sz = snap.bid_levels[k - 1]
            row[f"depth_bid_px_{k}"] = float(px) * PRICE_SCALE
            row[f"depth_bid_sz_{k}"] = float(sz)
        else:
            row[f"depth_bid_px_{k}"] = None
            row[f"depth_bid_sz_{k}"] = 0.0
        if k <= len(snap.ask_levels):
            px, sz = snap.ask_levels[k - 1]
            row[f"depth_ask_px_{k}"] = float(px) * PRICE_SCALE
            row[f"depth_ask_sz_{k}"] = float(sz)
        else:
            row[f"depth_ask_px_{k}"] = None
            row[f"depth_ask_sz_{k}"] = 0.0
    return row


def depth_event_series(
    frame: pl.DataFrame,
    *,
    n_levels: int = _DEFAULT_LEVELS,
) -> pl.DataFrame:
    """لقطات عمق بعد كل حدث — للمراقبة (``availability_ts = event_ts``)."""
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    require_single_contract_identity(frame, context="depth_event_series")
    if frame.height == 0:
        return pl.DataFrame(schema=_empty_depth_schema(n_levels=n_levels))

    work = sort_causal(frame)
    book = OrderBook()
    actions = work["action"].to_list()
    sides = work["side"].to_list()
    prices = work["price"].to_list()
    sizes = work["size"].to_list()
    order_ids = work["order_id"].to_list()
    event_ts = work[EVENT_TS].to_list()

    rows: list[dict[str, float | int | None]] = []
    for i in range(len(actions)):
        book.apply(str(actions[i]), str(sides[i]), int(prices[i]), int(sizes[i]), int(order_ids[i]))
        ts = int(event_ts[i])
        snap = book.snapshot(n_levels, availability_ts=ts)
        rows.append(snapshot_to_row(snap, event_ts=ts))
    return pl.DataFrame(rows).sort(AVAILABILITY_TS)


def depth_at_bar_close(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    n_levels: int = _DEFAULT_LEVELS,
) -> pl.DataFrame:
    """لقطة عمق عند إغلاق كل شمعة — للدخول/القرار.

    الدفتر يُحدَّث بكل أحداث الشمعة ثم تُثبَّت
    ``availability_ts = bucket_end`` (point-in-time عند الإغلاق).
    """
    if interval_ns < 1:
        raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    require_single_contract_identity(frame, context="depth_at_bar_close")
    empty_schema = _empty_depth_schema(n_levels=n_levels)
    empty_schema[BUCKET_START] = pl.Int64()
    empty_schema[BUCKET_END] = pl.Int64()
    if frame.height == 0:
        return pl.DataFrame(schema=empty_schema)

    work = sort_causal(frame)
    book = OrderBook()
    actions = work["action"].to_list()
    sides = work["side"].to_list()
    prices = work["price"].to_list()
    sizes = work["size"].to_list()
    order_ids = work["order_id"].to_list()
    event_ts = work[EVENT_TS].to_list()

    rows: list[dict[str, float | int | None]] = []
    current_bucket: int | None = None
    last_event_in_bucket = -1

    def _emit(bucket_start: int) -> None:
        bucket_end = bucket_start + interval_ns
        snap = book.snapshot(n_levels, availability_ts=bucket_end)
        row = snapshot_to_row(snap, event_ts=last_event_in_bucket)
        row[AVAILABILITY_TS] = bucket_end
        row[BUCKET_START] = bucket_start
        row[BUCKET_END] = bucket_end
        rows.append(row)

    for i in range(len(actions)):
        ts = int(event_ts[i])
        bucket = (ts // interval_ns) * interval_ns
        if current_bucket is None:
            current_bucket = bucket
        elif bucket != current_bucket:
            _emit(current_bucket)
            current_bucket = bucket
        book.apply(str(actions[i]), str(sides[i]), int(prices[i]), int(sizes[i]), int(order_ids[i]))
        last_event_in_bucket = ts

    if current_bucket is not None:
        _emit(current_bucket)

    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.DataFrame(rows).sort(AVAILABILITY_TS)


def attach_depth_asof(
    features: pl.DataFrame,
    depth: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """يلحق أعمدة عمق بـ asof خلفي على ``availability_ts``."""
    if features.height == 0 or depth.height == 0:
        return features
    if AVAILABILITY_TS not in features.columns or AVAILABILITY_TS not in depth.columns:
        raise ValueError(f"both frames require {AVAILABILITY_TS}")

    skip = {AVAILABILITY_TS, EVENT_TS, BUCKET_START, BUCKET_END}
    if columns is None:
        cols = [c for c in depth.columns if c not in skip]
    else:
        cols = [c for c in columns if c in depth.columns and c not in skip]
    if not cols:
        return features

    keep = [AVAILABILITY_TS, *cols]
    right = depth.select(keep).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop = [c for c in cols if c in left.columns]
    if drop:
        left = left.drop(drop)
    joined = left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")
    fill_cols = []
    for c in cols:
        if c not in joined.columns:
            continue
        dtype = joined.schema[c]
        if dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64):
            fill_cols.append(pl.col(c).fill_null(0.0))
    return joined.with_columns(fill_cols) if fill_cols else joined


__all__ = [
    "DEPTH_MONITOR_COLUMNS",
    "attach_depth_asof",
    "depth_at_bar_close",
    "depth_event_series",
    "snapshot_to_row",
]
