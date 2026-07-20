"""تدفّق tick/event موحّد — دفتر حي + ميزات inline (الأبعاد 1–4).

يبني tensor سببي لكل حدث MBO:

* **الدفتر الحي** (top-of-book NQ/MNQ) — جزء من المدخل.
* **volume profile متطوّر** (POC/VAH/VAL) — يُحدَّث مع كل صفقة.
* **مرحلة السوق** (balance / expansion) — لمسار الإخفاء standalone.
* **إشارات cross** (delta MNQ، trap setup) — لمسار الإخفاء cross.

كل صف متاح عند ``event_ts`` للحدث (point-in-time).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS, SEQUENCE
from nq.core.time import sort_causal
from nq.orderbook.book import OrderBook
from nq.simulation.volume_profile import value_area

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_TRADE = MboAction.TRADE.value
_BID = "B"
_ASK = "A"
_NEAR_TICKS = 2  # قرب VAH/VAL بوحدات السعر الثابتة (fixed-point steps)
_PRICE_SCALE: Final = 1_000_000  # خطوة ~1$ لعقود NQ تقريبًا
_IN_VALUE_ACTIVE = 0.5


class MarketPhase(IntEnum):
    """مرحلة السوق لمسار الإخفاء standalone."""

    BALANCE = 0
    EXPANSION = 1
    NEUTRAL = 2


class MaskPath(IntEnum):
    """مسار الإخفاء — standalone أو cross-trap (لا يتداخلان في نفس العيّنة)."""

    STANDALONE = 0
    CROSS_TRAP = 1


# أسماء القنوات بالترتيب الثابت
BOOK_NQ_NAMES: Final = (
    "nq_best_bid_norm",
    "nq_best_ask_norm",
    "nq_bid_size_log",
    "nq_ask_size_log",
    "nq_spread_norm",
    "nq_mid_norm",
)
BOOK_MNQ_NAMES: Final = (
    "mnq_best_bid_norm",
    "mnq_best_ask_norm",
    "mnq_bid_size_log",
    "mnq_ask_size_log",
    "mnq_spread_norm",
    "mnq_mid_norm",
)
VP_NAMES: Final = (
    "poc_dist_norm",
    "vah_dist_norm",
    "val_dist_norm",
    "near_vah",
    "near_val",
    "in_value_area",
)
PHASE_NAMES: Final = (
    "phase_balance",
    "phase_expansion",
)
CROSS_NAMES: Final = (
    "mnq_signed_vol",
    "trap_setup",
)
TICK_FEATURE_NAMES: Final = (
    *BOOK_NQ_NAMES,
    *BOOK_MNQ_NAMES,
    *VP_NAMES,
    *PHASE_NAMES,
    *CROSS_NAMES,
)

_TICK_SCHEMA: Final[dict[str, pl.DataType]] = {
    EVENT_TS: pl.Int64(),
    SEQUENCE: pl.UInt64(),
    "instrument_id": pl.UInt32(),
    "mask_path": pl.Int8(),
    "market_phase": pl.Int8(),
    AVAILABILITY_TS: pl.Int64(),
    **{name: pl.Float64() for name in TICK_FEATURE_NAMES},
}


@dataclass(frozen=True, slots=True)
class TickStream:
    """تسلسل tick موحّد جاهز لـ ``build_tick_sequences``."""

    frame: pl.DataFrame
    feature_names: tuple[str, ...] = TICK_FEATURE_NAMES

    @property
    def height(self) -> int:
        return self.frame.height


def _log_size(size: int | None) -> float:
    if size is None or size <= 0:
        return 0.0
    return float(np.log1p(size))


def _norm_price(price: int | None, ref: float) -> float:
    if price is None or ref <= 0:
        return 0.0
    return float(price) / ref


def _book_row(
    book: OrderBook,
    *,
    ref_price: float,
) -> tuple[float, float, float, float, float, float]:
    bb = max(book.bids) if book.bids else None
    ba = min(book.asks) if book.asks else None
    bb_sz = book.bids.get(bb, 0) if bb is not None else None
    ba_sz = book.asks.get(ba, 0) if ba is not None else None
    spread = float(ba - bb) if bb is not None and ba is not None else 0.0
    mid = (bb + ba) / 2.0 if bb is not None and ba is not None else (bb or ba or 0)
    return (
        _norm_price(bb, ref_price),
        _norm_price(ba, ref_price),
        _log_size(bb_sz),
        _log_size(ba_sz),
        spread / ref_price if ref_price > 0 else 0.0,
        float(mid) / ref_price if ref_price > 0 else 0.0,
    )


def _update_profile(profile: dict[int, int], price: int, size: int) -> None:
    profile[price] = profile.get(price, 0) + size


def _profile_features(
    profile: dict[int, int],
    mid: float,
    *,
    ref_price: float,
) -> tuple[float, float, float, float, float, float]:
    if not profile or mid <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = pl.DataFrame({"price": list(profile.keys()), "volume": list(profile.values())}).sort(
        "price"
    )
    va = value_area(rows)
    if va is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    poc_d = (mid - va.poc) / ref_price
    vah_d = (mid - va.vah) / ref_price
    val_d = (mid - va.val) / ref_price
    near_vah = 1.0 if abs(mid - va.vah) <= _NEAR_TICKS * _PRICE_SCALE else 0.0
    near_val = 1.0 if abs(mid - va.val) <= _NEAR_TICKS * _PRICE_SCALE else 0.0
    in_va = 1.0 if va.val <= mid <= va.vah else 0.0
    return (poc_d, vah_d, val_d, near_vah, near_val, in_va)


def _detect_phase(
    in_value_area: float,
    near_vah: float,
    near_val: float,
    price: float,
    prev_high: float | None,
    prev_low: float | None,
) -> MarketPhase:
    if near_vah > 0 or near_val > 0 or in_value_area > _IN_VALUE_ACTIVE:
        return MarketPhase.BALANCE
    if (
        prev_high is not None
        and prev_low is not None
        and (price > prev_high or price < prev_low)
    ):
        return MarketPhase.EXPANSION
    return MarketPhase.NEUTRAL


def _phase_one_hot(phase: MarketPhase) -> tuple[float, float]:
    return (
        1.0 if phase == MarketPhase.BALANCE else 0.0,
        1.0 if phase == MarketPhase.EXPANSION else 0.0,
    )


def _trap_setup(
    mnq_delta: int,
    nq_mid: float,
    mnq_mid: float,
    nq_high: float,
    mnq_high: float,
    *,
    min_delta: int = 1,
) -> float:
    """إعداد مصيدة سببي: MNQ يتحرك عدوانيًا دون قمة NQ جديدة."""
    if mnq_delta >= min_delta and mnq_mid >= mnq_high and nq_mid < nq_high:
        return 1.0
    if mnq_delta <= -min_delta and mnq_mid <= mnq_high and nq_mid > nq_high:
        return -1.0
    return 0.0


def _tick_row(
    *,
    action: str,
    side: str,
    price: int,
    size: int,
    order_id: int,
    ts: int,
    seq: int,
    inst: int,
    nq_instrument_id: int,
    nq_book: OrderBook,
    mnq_book: OrderBook,
    nq_profile: dict[int, int],
    mnq_signed: int,
    nq_high: float,
    mnq_high: float,
    mnq_low: float,
    ref_price: float,
    prev_nq_mid: float | None,
    prev_low: float | None,
    prev_high: float | None,
) -> tuple[
    dict[str, float | int],
    int,
    float,
    float,
    float,
    float | None,
    float | None,
    float | None,
]:
    """يُحدّث الدفاتر ويُرجع صف tick واحد مع حالة السوق المحدّثة."""
    is_nq = int(inst) == nq_instrument_id
    book = nq_book if is_nq else mnq_book
    book.apply(str(action), str(side), int(price), int(size), int(order_id))

    if is_nq and str(action) == _TRADE:
        _update_profile(nq_profile, int(price), int(size))

    if not is_nq and str(action) == _TRADE:
        trade_size = int(size)
        signed = trade_size if str(side) == _BID else -trade_size
        mnq_signed += signed

    nq_row = _book_row(nq_book, ref_price=ref_price)
    mnq_row = _book_row(mnq_book, ref_price=ref_price)
    nq_mid = nq_row[5] * ref_price
    mnq_mid = mnq_row[5] * ref_price

    if nq_mid > 0:
        nq_high = max(nq_high, nq_mid)
        if prev_nq_mid is not None:
            prev_low = nq_mid if prev_low is None else min(prev_low, nq_mid)
            prev_high = nq_mid if prev_high is None else max(prev_high, nq_mid)

    if mnq_mid > 0:
        mnq_high = max(mnq_high, mnq_mid)
        mnq_low = mnq_mid if mnq_low == 0.0 else min(mnq_low, mnq_mid)

    vp_feats = _profile_features(nq_profile, nq_mid, ref_price=ref_price)
    phase = _detect_phase(vp_feats[5], vp_feats[3], vp_feats[4], nq_mid, prev_high, prev_low)
    phase_oh = _phase_one_hot(phase)
    trap = _trap_setup(mnq_signed, nq_mid, mnq_mid, nq_high, mnq_high)
    mask_path = MaskPath.CROSS_TRAP if abs(trap) > 0 else MaskPath.STANDALONE

    row: dict[str, float | int] = {
        EVENT_TS: int(ts),
        SEQUENCE: int(seq),
        "instrument_id": int(inst),
        "mask_path": int(mask_path),
        "market_phase": int(phase),
        AVAILABILITY_TS: int(ts),
        **dict(zip(BOOK_NQ_NAMES, nq_row, strict=True)),
        **dict(zip(BOOK_MNQ_NAMES, mnq_row, strict=True)),
        **dict(zip(VP_NAMES, vp_feats, strict=True)),
        **dict(zip(PHASE_NAMES, phase_oh, strict=True)),
        "mnq_signed_vol": float(mnq_signed),
        "trap_setup": trap,
    }
    next_prev_nq_mid = nq_mid if nq_mid > 0 else prev_nq_mid
    return row, mnq_signed, nq_high, mnq_high, mnq_low, next_prev_nq_mid, prev_low, prev_high


def build_tick_stream(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
) -> TickStream:
    """يبني تسلسل tick موحّد من MBO خام (NQ + MNQ) مع دفتر حي وميزات inline."""
    nq_sorted = sort_causal(nq.with_columns(pl.lit(nq_instrument_id).alias("instrument_id")))
    mnq_sorted = sort_causal(mnq.with_columns(pl.lit(mnq_instrument_id).alias("instrument_id")))
    combined = pl.concat([nq_sorted, mnq_sorted], how="vertical").sort([EVENT_TS, SEQUENCE])

    nq_book = OrderBook()
    mnq_book = OrderBook()
    nq_profile: dict[int, int] = {}
    mnq_signed = 0
    nq_high = 0.0
    mnq_high = 0.0
    mnq_low = 0.0
    ref_price = 20_000_000_000.0

    rows: list[dict[str, float | int]] = []
    actions = combined["action"].to_list()
    sides = combined["side"].to_list()
    prices = combined["price"].to_list()
    sizes = combined["size"].to_list()
    order_ids = combined["order_id"].to_list()
    event_times = combined[EVENT_TS].to_list()
    sequences = combined["sequence"].to_list()
    instruments = combined["instrument_id"].to_list()

    prev_nq_mid: float | None = None
    prev_low: float | None = None
    prev_high: float | None = None

    for action, side, price, size, order_id, ts, seq, inst in zip(
        actions,
        sides,
        prices,
        sizes,
        order_ids,
        event_times,
        sequences,
        instruments,
        strict=True,
    ):
        row, mnq_signed, nq_high, mnq_high, mnq_low, prev_nq_mid, prev_low, prev_high = _tick_row(
            action=str(action),
            side=str(side),
            price=int(price),
            size=int(size),
            order_id=int(order_id),
            ts=int(ts),
            seq=int(seq),
            inst=int(inst),
            nq_instrument_id=nq_instrument_id,
            nq_book=nq_book,
            mnq_book=mnq_book,
            nq_profile=nq_profile,
            mnq_signed=mnq_signed,
            nq_high=nq_high,
            mnq_high=mnq_high,
            mnq_low=mnq_low,
            ref_price=ref_price,
            prev_nq_mid=prev_nq_mid,
            prev_low=prev_low,
            prev_high=prev_high,
        )
        rows.append(row)

    if not rows:
        return TickStream(frame=pl.DataFrame(schema=_TICK_SCHEMA))

    return TickStream(frame=pl.DataFrame(rows))


__all__ = [
    "TICK_FEATURE_NAMES",
    "MarketPhase",
    "MaskPath",
    "TickStream",
    "build_tick_stream",
]
