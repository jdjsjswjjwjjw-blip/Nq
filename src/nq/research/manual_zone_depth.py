"""عمق الدفتر عند مناطق يحددها العين — بلا overlay حيّة.

البروتوكول: أنت تسمّي اليوم والساعة (ET) ومستويات بالنقطة (NQ). الطبقة تعيد
بناء دفتر ذلك اليوم فقط حتى ``t`` وتُظهر الحجم عند كل مستوى + السياق ±N تيك.
رأس المسار يبقى أداة التيك؛ الخروج يدوي. ليست إشارة تلقائية.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import polars as pl

from nq.contracts.mbo import PRICE_SCALE, MboSide
from nq.core.time import sort_causal
from nq.orderbook.book import OrderBook
from nq.research.mbo_sequence_mlp import (
    assert_no_future_events,
    assert_single_day_mbo,
)

LAYER_ID = "manual_zone_depth"
TICK_POINTS: Final = 0.25
DEFAULT_BAND_TICKS: Final = 4
_ET: Final = ZoneInfo("America/New_York")
_BID = MboSide.BID.value
_ASK = MboSide.ASK.value
_TICK_FIXED: Final[int] = round(TICK_POINTS / PRICE_SCALE)


@dataclass(frozen=True, slots=True)
class ManualZone:
    """منطقة يحدّدها المراقب: يوم جلسة + زمن إتاحة + مستويات بالنقطة."""

    day: str
    availability_ts: int
    levels_points: tuple[float, ...]
    label: str = ""
    band_ticks: int = DEFAULT_BAND_TICKS


def points_to_fixed(points: float) -> int:
    """نقطة NQ → سعر النقطة الثابتة، مُثبَّت على تيك 0.25."""
    snapped = round(float(points) / TICK_POINTS) * TICK_POINTS
    return round(snapped / PRICE_SCALE)


def fixed_to_points(price: int) -> float:
    """سعر النقطة الثابتة → نقطة NQ."""
    return float(int(price)) * PRICE_SCALE


def et_clock_to_ns(day: str, clock: str) -> int:
    """``2025-05-01`` + ``10:35:00`` (America/New_York) → نانوثانية."""
    hour, minute, *rest = clock.split(":")
    second = int(rest[0]) if rest else 0
    yyyy, mm, dd = (int(p) for p in day.split("-"))
    stamp = dt.datetime(
        yyyy,
        mm,
        dd,
        int(hour),
        int(minute),
        second,
        tzinfo=_ET,
    )
    return int(stamp.timestamp() * 1_000_000_000)


def parse_zone(
    *,
    day: str,
    clock: str,
    levels_points: Sequence[float],
    label: str = "",
    band_ticks: int = DEFAULT_BAND_TICKS,
) -> ManualZone:
    if not levels_points:
        raise ValueError("levels_points must not be empty")
    if int(band_ticks) < 0:
        raise ValueError(f"band_ticks must be >= 0, got {band_ticks}")
    return ManualZone(
        day=day,
        availability_ts=et_clock_to_ns(day, clock),
        levels_points=tuple(float(p) for p in levels_points),
        label=label,
        band_ticks=int(band_ticks),
    )


def book_at(mbo: pl.DataFrame, availability_ts: int) -> OrderBook:
    """دفتر يوم واحد بعد أحداث ``event_ts`` و ``ingest_ts`` ≤ ``t`` فقط."""
    assert_single_day_mbo(mbo)
    known = assert_no_future_events(mbo, availability_ts)
    book = OrderBook()
    if known.height == 0:
        return book
    work = sort_causal(known)
    actions = work["action"].cast(pl.Utf8).fill_null("").to_list()
    sides = (
        work["side"].cast(pl.Utf8).fill_null("N").to_list()
        if "side" in work.columns
        else ["N"] * work.height
    )
    prices = work["price"].cast(pl.Int64).fill_null(0).to_list()
    sizes = work["size"].cast(pl.Int64).fill_null(0).to_list()
    oids = (
        work["order_id"].cast(pl.Int64).fill_null(0).to_list()
        if "order_id" in work.columns
        else [0] * work.height
    )
    for i, action in enumerate(actions):
        book.apply(str(action), str(sides[i]), int(prices[i]), int(sizes[i]), int(oids[i]))
    return book


def _band_prices(center: int, band_ticks: int) -> list[int]:
    span = int(band_ticks)
    return [center + k * _TICK_FIXED for k in range(-span, span + 1)]


def _zone_stack(book: OrderBook, lo: int, hi: int) -> tuple[int, int]:
    left, right = (lo, hi) if lo <= hi else (hi, lo)
    bid = 0
    ask = 0
    for price, size in book.bids.items():
        if left <= price <= right:
            bid += int(size)
    for price, size in book.asks.items():
        if left <= price <= right:
            ask += int(size)
    return bid, ask


def _quote_points(book: OrderBook) -> tuple[float | None, float | None, float | None]:
    bid = book.best_bid()
    ask = book.best_ask()
    best_bid = None if bid is None else fixed_to_points(bid[0])
    best_ask = None if ask is None else fixed_to_points(ask[0])
    mid = None if best_bid is None or best_ask is None else 0.5 * (best_bid + best_ask)
    return best_bid, best_ask, mid


def zone_depth(mbo: pl.DataFrame, zone: ManualZone) -> pl.DataFrame:
    """حجم معلّق عند كل مستوى عيّنه العين، مع سياق ±``band_ticks``."""
    book = book_at(mbo, zone.availability_ts)
    best_bid, best_ask, mid = _quote_points(book)
    bid_quote = book.best_bid()
    ask_quote = book.best_ask()
    rows: list[dict[str, object]] = []
    centers = [points_to_fixed(p) for p in zone.levels_points]
    lo = min(centers)
    hi = max(centers)
    if zone.band_ticks > 0 and lo == hi:
        lo -= zone.band_ticks * _TICK_FIXED
        hi += zone.band_ticks * _TICK_FIXED
    stack_bid, stack_ask = _zone_stack(book, lo, hi)
    seen: set[int] = set()
    for raw_points, center in zip(zone.levels_points, centers, strict=True):
        for price in _band_prices(center, zone.band_ticks):
            if price in seen:
                continue
            seen.add(price)
            bid_sz = book.size_at(_BID, price)
            ask_sz = book.size_at(_ASK, price)
            ticks_from_bid = None if bid_quote is None else (price - bid_quote[0]) / _TICK_FIXED
            ticks_from_ask = None if ask_quote is None else (price - ask_quote[0]) / _TICK_FIXED
            rows.append(
                {
                    "day": zone.day,
                    "label": zone.label,
                    "availability_ts": zone.availability_ts,
                    "eye_level_points": float(raw_points),
                    "level_points": fixed_to_points(price),
                    "is_eye_level": price == center,
                    "bid_size": bid_sz,
                    "ask_size": ask_sz,
                    "ticks_from_best_bid": ticks_from_bid,
                    "ticks_from_best_ask": ticks_from_ask,
                    "best_bid_points": best_bid,
                    "best_ask_points": best_ask,
                    "mid_points": mid,
                    "zone_bid_stack": stack_bid,
                    "zone_ask_stack": stack_ask,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "day": pl.Utf8(),
                "label": pl.Utf8(),
                "availability_ts": pl.Int64(),
                "eye_level_points": pl.Float64(),
                "level_points": pl.Float64(),
                "is_eye_level": pl.Boolean(),
                "bid_size": pl.Int64(),
                "ask_size": pl.Int64(),
                "ticks_from_best_bid": pl.Float64(),
                "ticks_from_best_ask": pl.Float64(),
                "best_bid_points": pl.Float64(),
                "best_ask_points": pl.Float64(),
                "mid_points": pl.Float64(),
                "zone_bid_stack": pl.Int64(),
                "zone_ask_stack": pl.Int64(),
            }
        )
    return pl.DataFrame(rows).sort(["eye_level_points", "level_points"])


def write_zone_depth_report(
    depth: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if depth.height:
        depth.write_parquet(out / "manual_zone_depth.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Manual zone depth",
        "",
        "Eye-specified NQ-point levels. Per-day book at t. Not a live overlay.",
        "Path/tick head stays the working model; exits stay manual.",
        "",
        f"n_rows={depth.height} reconstructed_order_book="
        f"{diagnostics.get('reconstructed_order_book')} "
        f"concatenated_raw_mbo={diagnostics.get('concatenated_raw_mbo')}",
        "",
    ]
    if depth.height:
        eye = depth.filter(pl.col("is_eye_level"))
        lines.append("| level | bid | ask | ticks vs bid | ticks vs ask |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in eye.iter_rows(named=True):
            lines.append(
                f"| {float(row['level_points']):.2f} | {row['bid_size']} | "
                f"{row['ask_size']} | {row['ticks_from_best_bid']} | "
                f"{row['ticks_from_best_ask']} |"
            )
        lines.append("")
    (out / "MANUAL_ZONE_DEPTH.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "DEFAULT_BAND_TICKS",
    "LAYER_ID",
    "TICK_POINTS",
    "ManualZone",
    "book_at",
    "et_clock_to_ns",
    "fixed_to_points",
    "parse_zone",
    "points_to_fixed",
    "write_zone_depth_report",
    "zone_depth",
]
