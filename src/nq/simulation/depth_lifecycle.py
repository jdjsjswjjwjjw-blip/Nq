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

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.time import sort_causal
from nq.orderbook.book import OrderBook
from nq.orderbook.depth import DepthSnapshot
from nq.research.progress import ProgressLike
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

# مقاييس مسار أحداث العمق داخل الشمعة (تُنشر عند bucket_end فقط)
DEPTH_PATH_COLUMNS: Final[tuple[str, ...]] = (
    "depth_path_imbalance",
    "depth_path_imbalance_delta",
    "depth_path_bid_drain",
    "depth_path_ask_drain",
    "depth_path_pressure",
    "depth_path_n_events",
    # مسار أغنى من open/close فقط — حدّ أقصى/أدنى اختلال داخل الشمعة
    "depth_path_imbalance_max",
    "depth_path_imbalance_min",
    "depth_path_l2_l5_bid_drain",
    "depth_path_l2_l5_ask_drain",
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
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """لقطات عمق بعد كل حدث — للمراقبة (``availability_ts = event_ts``)."""
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
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
    n = len(actions)
    log = progress
    if log is not None:
        log.op(f"depth_event_series: مسح {n:,} حدث MBO → لقطات L1–L{n_levels}")

    rows: list[dict[str, float | int | None]] = []
    hb_every = 5_000 if n else 1
    next_hb = hb_every
    for i in range(n):
        book.apply(str(actions[i]), str(sides[i]), int(prices[i]), int(sizes[i]), int(order_ids[i]))
        ts = int(event_ts[i])
        snap = book.snapshot(n_levels, availability_ts=ts)
        rows.append(snapshot_to_row(snap, event_ts=ts))
        done = i + 1
        if log is not None and (done >= next_hb or done == n):
            log.heartbeat(done, n, label="depth_events", force=True, every=hb_every)
            next_hb = done + hb_every
    if log is not None:
        log.op(f"depth_event_series انتهى: {len(rows):,} لقطة — بناء DataFrame…")
    return pl.DataFrame(rows).sort(AVAILABILITY_TS)


def _depth_bar_empty_schema(*, n_levels: int) -> dict[str, pl.DataType]:
    empty_schema = _empty_depth_schema(n_levels=n_levels)
    empty_schema[BUCKET_START] = pl.Int64()
    empty_schema[BUCKET_END] = pl.Int64()
    return empty_schema


def depth_at_bar_close_multi(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    interval_ns_list: Sequence[int],
    n_levels: int = _DEFAULT_LEVELS,
    progress: ProgressLike | None = None,
) -> dict[int, pl.DataFrame]:
    """لقطات عمق عند إغلاق الشموع لعدة فواصل في **مرور دفتر واحد**.

    كل فاصل يستقلّ بـ buckets خاصّة به؛ النتائج مطابقة لاستدعاء
    ``depth_at_bar_close`` منفصل لكل فاصل (نفس الأرقام السببية).
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    intervals = tuple(dict.fromkeys(int(x) for x in interval_ns_list))
    if not intervals:
        raise ValueError("interval_ns_list must be non-empty")
    for interval_ns in intervals:
        if interval_ns < 1:
            raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")

    empty_schema = _depth_bar_empty_schema(n_levels=n_levels)
    if frame.height == 0:
        return {iv: pl.DataFrame(schema=empty_schema) for iv in intervals}

    work = sort_causal(frame)
    book = OrderBook()
    actions = work["action"].to_list()
    sides = work["side"].to_list()
    prices = work["price"].to_list()
    sizes = work["size"].to_list()
    order_ids = work["order_id"].to_list()
    event_ts = work[EVENT_TS].to_list()
    n = len(actions)
    log = progress
    if log is not None:
        iv_txt = ",".join(str(iv) for iv in intervals)
        log.op(f"depth_at_bar_close_multi: {n:,} حدث → فواصل [{iv_txt}] · L1–L{n_levels}")

    rows_by_iv: dict[int, list[dict[str, float | int | None]]] = {iv: [] for iv in intervals}
    current_bucket: dict[int, int | None] = dict.fromkeys(intervals)
    last_event_in_bucket: dict[int, int] = dict.fromkeys(intervals, -1)
    hb_every = 5_000 if n else 1
    next_hb = hb_every

    def _emit(interval_ns: int, bucket_start: int) -> None:
        bucket_end = bucket_start + interval_ns
        snap = book.snapshot(n_levels, availability_ts=bucket_end)
        row = snapshot_to_row(snap, event_ts=last_event_in_bucket[interval_ns])
        row[AVAILABILITY_TS] = bucket_end
        row[BUCKET_START] = bucket_start
        row[BUCKET_END] = bucket_end
        rows_by_iv[interval_ns].append(row)

    for i in range(n):
        ts = int(event_ts[i])
        for interval_ns in intervals:
            bucket = (ts // interval_ns) * interval_ns
            cur = current_bucket[interval_ns]
            if cur is None:
                current_bucket[interval_ns] = bucket
            elif bucket != cur:
                _emit(interval_ns, cur)
                current_bucket[interval_ns] = bucket
        book.apply(str(actions[i]), str(sides[i]), int(prices[i]), int(sizes[i]), int(order_ids[i]))
        for interval_ns in intervals:
            last_event_in_bucket[interval_ns] = ts
        done = i + 1
        if log is not None and (done >= next_hb or done == n):
            log.heartbeat(done, n, label="depth_bars", force=True, every=hb_every)
            next_hb = done + hb_every

    for interval_ns in intervals:
        cur = current_bucket[interval_ns]
        if cur is not None:
            _emit(interval_ns, cur)

    out: dict[int, pl.DataFrame] = {}
    for interval_ns in intervals:
        rows = rows_by_iv[interval_ns]
        if log is not None:
            log.op(
                f"depth_at_bar_close[{interval_ns}]: {len(rows):,} شمعة بعمق — بناء DataFrame…"
            )
        out[interval_ns] = (
            pl.DataFrame(rows).sort(AVAILABILITY_TS) if rows else pl.DataFrame(schema=empty_schema)
        )
    return out


def depth_at_bar_close(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    n_levels: int = _DEFAULT_LEVELS,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """لقطة عمق عند إغلاق كل شمعة — للدخول/القرار.

    الدفتر يُحدَّث بكل أحداث الشمعة ثم تُثبَّت
    ``availability_ts = bucket_end`` (point-in-time عند الإغلاق).
    """
    return depth_at_bar_close_multi(
        frame,
        interval_ns_list=(interval_ns,),
        n_levels=n_levels,
        progress=progress,
    )[interval_ns]


def depth_event_path_at_bar_close(  # noqa: PLR0912, PLR0915    frame: pl.DataFrame,
    *,
    interval_ns: int,
    n_levels: int = _DEFAULT_LEVELS,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يتتبّع مسار أحداث العمق **داخل** كل شمعة وينشر المقاييس عند ``bucket_end``.

    لا يستخدم أحداثًا بعد إغلاق الشمعة. المقاييس:
    * ``depth_path_imbalance`` / ``_delta`` — اختلال الإغلاق وتغيّره منذ أول حدث
    * ``depth_path_bid/ask_drain`` — سحب سيولة نسبي من افتتاح المسار إلى الإغلاق
    * ``depth_path_pressure`` — ``ask_drain - bid_drain`` (موجب ≈ ضغط صاعد سببيًا)
    * ``depth_path_n_events`` — عدد أحداث MBO داخل الشمعة
    * ``depth_path_imbalance_max/min`` — أقصى/أدنى اختلال لُوحظ داخل الشمعة
    * ``depth_path_l2_l5_*_drain`` — استنزاف سيولة خلف L1 (حافة أسفل الدفتر)

    تسريع آمن (نفس الأرقام لـ open/close الأساسي):
    * كل حدث يحدّث الدفتر (``apply``) — سببية كاملة.
    * قياس السيولة الكلية عند **أول** حدث وعند **الإغلاق**؛ اختلال المسار يُحدَّث
      كل حدث عبر ``path_liquidity`` الخفيف (بدون لقطة L1–L5 كاملة).
    """
    if interval_ns < 1:
        raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")

    schema: dict[str, pl.DataType] = {
        AVAILABILITY_TS: pl.Int64(),
        EVENT_TS: pl.Int64(),
        BUCKET_START: pl.Int64(),
        BUCKET_END: pl.Int64(),
        **{c: pl.Float64() for c in DEPTH_PATH_COLUMNS},
    }
    if frame.height == 0:
        return pl.DataFrame(schema=schema)

    work = sort_causal(frame)
    book = OrderBook()
    actions = work["action"].to_list()
    sides = work["side"].to_list()
    prices = work["price"].to_list()
    sizes = work["size"].to_list()
    order_ids = work["order_id"].to_list()
    event_ts = work[EVENT_TS].to_list()
    n = len(actions)
    log = progress
    if log is not None:
        log.op(
            f"depth_event_path: {n:,} حدث → مسار داخل الشمعة · "
            f"interval_ns={interval_ns} · sample=path+close"
        )

    rows: list[dict[str, float | int]] = []
    current_bucket: int | None = None
    last_event = -1
    n_events = 0
    open_imb = 0.0
    open_bid = 0.0
    open_ask = 0.0
    open_l25_bid = 0.0
    open_l25_ask = 0.0
    imb_max = 0.0
    imb_min = 0.0
    opened = False
    hb_every = 5_000 if n else 1
    next_hb = hb_every

    def _l2_l5(side: str) -> float:
        levels = book.top_n(side, n_levels)
        if len(levels) <= 1:
            return 0.0
        return float(sum(sz for _, sz in levels[1:]))

    def _emit(bucket_start: int) -> None:
        nonlocal opened
        if not opened:
            return
        close_bid, close_ask, close_imb = book.path_liquidity(n_levels)
        close_l25_bid = _l2_l5("B")
        close_l25_ask = _l2_l5("A")
        bucket_end = bucket_start + interval_ns
        bid_base = max(open_bid, 1.0)
        ask_base = max(open_ask, 1.0)
        bid_drain = max(0.0, (open_bid - close_bid) / bid_base)
        ask_drain = max(0.0, (open_ask - close_ask) / ask_base)
        l25_bid_base = max(open_l25_bid, 1.0)
        l25_ask_base = max(open_l25_ask, 1.0)
        l25_bid_drain = max(0.0, (open_l25_bid - close_l25_bid) / l25_bid_base)
        l25_ask_drain = max(0.0, (open_l25_ask - close_l25_ask) / l25_ask_base)
        rows.append(
            {
                AVAILABILITY_TS: bucket_end,
                EVENT_TS: last_event,
                BUCKET_START: bucket_start,
                BUCKET_END: bucket_end,
                "depth_path_imbalance": float(close_imb),
                "depth_path_imbalance_delta": float(close_imb - open_imb),
                "depth_path_bid_drain": float(bid_drain),
                "depth_path_ask_drain": float(ask_drain),
                "depth_path_pressure": float(ask_drain - bid_drain),
                "depth_path_n_events": float(n_events),
                "depth_path_imbalance_max": float(imb_max),
                "depth_path_imbalance_min": float(imb_min),
                "depth_path_l2_l5_bid_drain": float(l25_bid_drain),
                "depth_path_l2_l5_ask_drain": float(l25_ask_drain),
            }
        )
        opened = False

    for i in range(n):
        ts = int(event_ts[i])
        bucket = (ts // interval_ns) * interval_ns
        if current_bucket is None:
            current_bucket = bucket
        elif bucket != current_bucket:
            _emit(current_bucket)
            current_bucket = bucket
            n_events = 0
            opened = False

        book.apply(str(actions[i]), str(sides[i]), int(prices[i]), int(sizes[i]), int(order_ids[i]))
        last_event = ts
        n_events += 1
        _bid, _ask, cur_imb = book.path_liquidity(n_levels)
        if not opened:
            open_bid, open_ask, open_imb = _bid, _ask, cur_imb
            open_l25_bid = _l2_l5("B")
            open_l25_ask = _l2_l5("A")
            imb_max = cur_imb
            imb_min = cur_imb
            opened = True
        else:
            imb_max = max(imb_max, cur_imb)
            imb_min = min(imb_min, cur_imb)
        done = i + 1
        if log is not None and (done >= next_hb or done == n):
            log.heartbeat(done, n, label="depth_path", force=True, every=hb_every)
            next_hb = done + hb_every

    if current_bucket is not None:
        _emit(current_bucket)

    if log is not None:
        log.op(f"depth_event_path انتهى: {len(rows):,} شمعة — بناء DataFrame…")
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).sort(AVAILABILITY_TS)


def attach_depth_asof(
    features: pl.DataFrame,
    depth: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    fill_missing: bool = False,
) -> pl.DataFrame:
    """يلحق أعمدة عمق بـ asof خلفي على ``availability_ts``.

    افتراضيًا **لا** يملأ القيم الناقصة بأصفار — ``null`` يعني «لا لقطة عمق
    سابقة» وليس دفترًا فارغًا. مرّر ``fill_missing=True`` فقط إن احتجت توافقًا
    قديمًا صريحًا.
    """
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
    if not fill_missing:
        return joined
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
    "DEPTH_PATH_COLUMNS",
    "attach_depth_asof",
    "depth_at_bar_close",
    "depth_at_bar_close_multi",
    "depth_event_path_at_bar_close",
    "depth_event_series",
    "snapshot_to_row",
]
