"""حافة أسفل الدفتر (L2–L5) — امتصاص / استنزاف طابور / آيسبرغ حيّ.

مقاييس سببية تُحسب داخل الشمعة من مسار أحداث MBO وتُنشر عند ``bucket_end`` فقط.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.core.time import sort_causal
from nq.orderbook.book import OrderBook
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.depth_noise import DepthNoiseConfig, filter_depth_noise
from nq.simulation.liquidity import detect_icebergs

_TRADE = "T"
_FILL = "F"

_DEFAULT_LEVELS: Final = 5
_MIN_BOTTOM_LEVELS: Final = 2

#: أعمدة حافة أسفل الدفتر (تُلحق asof خلفي)
BOTTOM_BOOK_COLUMNS: Final[tuple[str, ...]] = (
    "bb_l2_l5_bid",
    "bb_l2_l5_ask",
    "bb_l2_l5_imbalance",
    "bb_bid_depletion",
    "bb_ask_depletion",
    "bb_absorption_bid",
    "bb_absorption_ask",
    "bb_queue_pressure",
    "bb_iceberg_hit",
    "bb_path_n_events",
)


def _l2_l5_size(book: OrderBook, side: str, n_levels: int) -> float:
    levels = book.top_n(side, n_levels)
    if len(levels) <= 1:
        return 0.0
    return float(sum(sz for _, sz in levels[1:]))


def _iceberg_hit(slice_frame: pl.DataFrame) -> float:
    if slice_frame.height == 0:
        return 0.0
    ice = detect_icebergs(slice_frame)
    if ice.height == 0:
        return 0.0
    return 1.0 if bool(ice["is_iceberg"].any()) else 0.0


def bottom_book_features_at_bar_close(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    n_levels: int = _DEFAULT_LEVELS,
    filter_noise: bool = True,
    noise_config: DepthNoiseConfig | None = None,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يستخرج حافة L2–L5 داخل كل شمعة وينشرها عند ``bucket_end``.

    * ``bb_l2_l5_*`` — سيولة خلف L1 عند الإغلاق
    * ``bb_*_depletion`` — استنزاف نسبي لـ L2–L5 من افتتاح المسار → الإغلاق
    * ``bb_absorption_*`` — تنفيذ مقابل بقاء السيولة الظاهرة (امتصاص)
    * ``bb_queue_pressure`` — ``ask_depletion - bid_depletion``
    * ``bb_iceberg_hit`` — آيسبرغ سببي داخل أحداث الشمعة
    """
    if interval_ns < 1:
        raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
    if n_levels < _MIN_BOTTOM_LEVELS:
        raise ValueError(
            f"n_levels must be >= {_MIN_BOTTOM_LEVELS} for bottom-book, got {n_levels}"
        )

    schema: dict[str, pl.DataType] = {
        AVAILABILITY_TS: pl.Int64(),
        EVENT_TS: pl.Int64(),
        BUCKET_START: pl.Int64(),
        BUCKET_END: pl.Int64(),
        **{c: pl.Float64() for c in BOTTOM_BOOK_COLUMNS},
    }
    if frame.height == 0:
        return pl.DataFrame(schema=schema)

    raw = filter_depth_noise(frame, config=noise_config) if filter_noise else frame
    work = sort_causal(raw)
    book = OrderBook()
    actions = work["action"].cast(pl.Utf8).to_list()
    sides = work["side"].cast(pl.Utf8).to_list()
    prices = work["price"].to_list()
    sizes = work["size"].to_list()
    order_ids = work["order_id"].to_list()
    event_ts = work[EVENT_TS].to_list()
    n = len(actions)
    log = progress
    if log is not None:
        log.op(
            f"bottom_book: {n:,} حدث → L2–L{n_levels} · interval_ns={interval_ns} · "
            f"noise_filter={filter_noise}"
        )

    rows: list[dict[str, float | int]] = []
    current_bucket: int | None = None
    last_event = -1
    n_events = 0
    open_bid = 0.0
    open_ask = 0.0
    opened = False
    trade_bid = 0.0
    trade_ask = 0.0
    bucket_start_idx = 0
    hb_every = 5_000 if n else 1
    next_hb = hb_every

    def _emit(bucket_start: int, end_idx: int) -> None:
        nonlocal opened, trade_bid, trade_ask
        if not opened:
            return
        close_bid = _l2_l5_size(book, "B", n_levels)
        close_ask = _l2_l5_size(book, "A", n_levels)
        total = close_bid + close_ask
        imb = 0.0 if total <= 0 else (close_bid - close_ask) / total
        bid_base = max(open_bid, 1.0)
        ask_base = max(open_ask, 1.0)
        bid_dep = max(0.0, (open_bid - close_bid) / bid_base)
        ask_dep = max(0.0, (open_ask - close_ask) / ask_base)
        abs_bid = trade_bid / bid_base if close_bid >= open_bid * 0.9 else 0.0
        abs_ask = trade_ask / ask_base if close_ask >= open_ask * 0.9 else 0.0
        sl = work.slice(bucket_start_idx, max(0, end_idx - bucket_start_idx))
        iceberg = _iceberg_hit(sl)
        bucket_end = bucket_start + interval_ns
        rows.append(
            {
                AVAILABILITY_TS: bucket_end,
                EVENT_TS: last_event,
                BUCKET_START: bucket_start,
                BUCKET_END: bucket_end,
                "bb_l2_l5_bid": float(close_bid),
                "bb_l2_l5_ask": float(close_ask),
                "bb_l2_l5_imbalance": float(imb),
                "bb_bid_depletion": float(bid_dep),
                "bb_ask_depletion": float(ask_dep),
                "bb_absorption_bid": float(abs_bid),
                "bb_absorption_ask": float(abs_ask),
                "bb_queue_pressure": float(ask_dep - bid_dep),
                "bb_iceberg_hit": float(iceberg),
                "bb_path_n_events": float(n_events),
            }
        )
        opened = False
        trade_bid = 0.0
        trade_ask = 0.0

    for i in range(n):
        ts = int(event_ts[i])
        bucket = (ts // interval_ns) * interval_ns
        if current_bucket is None:
            current_bucket = bucket
            bucket_start_idx = i
        elif bucket != current_bucket:
            _emit(current_bucket, i)
            current_bucket = bucket
            bucket_start_idx = i
            n_events = 0
            opened = False

        book.apply(str(actions[i]), str(sides[i]), int(prices[i]), int(sizes[i]), int(order_ids[i]))
        last_event = ts
        n_events += 1

        action = str(actions[i])
        side = str(sides[i])
        size = float(sizes[i])
        if action in (_TRADE, _FILL):
            if side == "B":
                trade_ask += size
            elif side == "A":
                trade_bid += size

        if not opened:
            open_bid = _l2_l5_size(book, "B", n_levels)
            open_ask = _l2_l5_size(book, "A", n_levels)
            opened = True

        done = i + 1
        if log is not None and (done >= next_hb or done == n):
            log.heartbeat(done, n, label="bottom_book", force=True, every=hb_every)
            next_hb = done + hb_every

    if current_bucket is not None:
        _emit(current_bucket, n)

    if log is not None:
        log.op(f"bottom_book انتهى: {len(rows):,} شمعة")
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).sort(AVAILABILITY_TS)


def attach_bottom_book_asof(
    features: pl.DataFrame,
    bottom: pl.DataFrame,
) -> pl.DataFrame:
    """يلحق أعمدة أسفل الدفتر بـ asof خلفي دون ``fill_null(0)`` زائف."""
    from nq.simulation.depth_lifecycle import attach_depth_asof  # noqa: PLC0415

    return attach_depth_asof(features, bottom, columns=BOTTOM_BOOK_COLUMNS, fill_missing=False)


__all__ = [
    "BOTTOM_BOOK_COLUMNS",
    "attach_bottom_book_asof",
    "bottom_book_features_at_bar_close",
]
