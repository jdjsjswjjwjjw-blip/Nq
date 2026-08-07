"""تدفّق tick/event موحّد — دفتر حي + ميزات inline (الأبعاد 1–4).

يبني tensor سببي لكل حدث MBO:

* **الدفتر الحي** (top-of-book NQ/MNQ) — جزء من المدخل.
* **سيولة الدفتر عند VAH/VAL** — مستويات فعلية من ``OrderBook``.
* **volume profile متطوّر** (POC/VAH/VAL) — عبر ``DevelopingVolumeProfile``.
* **مرحلة السوق** (balance / expansion) — عبر ``CausalRegimeTracker`` (KMeansRegimes).
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
from nq.core.session import session_date_from_ns
from nq.core.time import sort_causal
from nq.orderbook.book import OrderBook
from nq.research.progress import ProgressLike
from nq.simulation.volume_profile import DevelopingVolumeProfile, ValueArea
from nq.states.regimes import CausalRegimeTracker

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_TRADE = MboAction.TRADE.value
_BID = "B"
_NEAR_TICKS = 2  # قرب VAH/VAL بعدد تيكات NQ (كل تيك = 0.25$)
_REF_PRICE: Final = 20_000_000_000.0
_HEARTBEAT_LARGE_EVENT_THRESHOLD: Final = 100_000


class MarketPhase(IntEnum):
    """مرحلة السوق لمسار الإخفاء standalone."""

    BALANCE = 0
    EXPANSION = 1
    NEUTRAL = 2


class MaskPath(IntEnum):
    """مسار الإخفاء — standalone أو cross-trap (لا يتداخلان في نفس العيّنة)."""

    STANDALONE = 0
    CROSS_TRAP = 1


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
BOOK_DEPTH_NAMES: Final = (
    "nq_vah_bid_liq_log",
    "nq_vah_ask_liq_log",
    "nq_val_bid_liq_log",
    "nq_val_ask_liq_log",
    "nq_trail_bid_liq_log",
    "nq_trail_ask_liq_log",
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
    "nq_signed_vol",
    "mnq_signed_vol",
    "trap_setup",
)
TICK_FEATURE_NAMES: Final = (
    *BOOK_NQ_NAMES,
    *BOOK_MNQ_NAMES,
    *BOOK_DEPTH_NAMES,
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
    bb_t = book.best_bid()
    ba_t = book.best_ask()
    bb = bb_t[0] if bb_t is not None else None
    ba = ba_t[0] if ba_t is not None else None
    bb_sz = bb_t[1] if bb_t is not None else None
    ba_sz = ba_t[1] if ba_t is not None else None
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


def _book_depth_features(
    book: OrderBook,
    va: ValueArea | None,
) -> tuple[float, float, float, float, float, float]:
    """سيولة فعلية عند VAH/VAL و trailing liquidity خلف أفضل سعر."""
    if va is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vah_bid = book.bids.get(va.vah, 0)
    vah_ask = book.asks.get(va.vah, 0)
    val_bid = book.bids.get(va.val, 0)
    val_ask = book.asks.get(va.val, 0)
    trail_bid, trail_ask = book.trail_liquidity()
    return (
        _log_size(vah_bid),
        _log_size(vah_ask),
        _log_size(val_bid),
        _log_size(val_ask),
        _log_size(trail_bid),
        _log_size(trail_ask),
    )


def _phase_one_hot(phase: MarketPhase) -> tuple[float, float]:
    return (
        1.0 if phase == MarketPhase.BALANCE else 0.0,
        1.0 if phase == MarketPhase.EXPANSION else 0.0,
    )


def _trap_setup(
    mnq_event_delta: int,
    *,
    mnq_new_high: bool,
    mnq_new_low: bool,
    nq_new_high: bool,
    nq_new_low: bool,
    min_delta: int = 1,
) -> float:
    """إعداد مصيدة سببي: MNQ قمة/قاع جلسة جديدة بعدوانية دون تأكيد NQ.

    يطابق فلسفة ``cross_market_features`` (جلسة + دلتا الحدث/الفاصل)، لا قمم عمرية.
    """
    if mnq_event_delta >= min_delta and mnq_new_high and not nq_new_high:
        return 1.0
    if mnq_event_delta <= -min_delta and mnq_new_low and not nq_new_low:
        return -1.0
    return 0.0


def _regime_features(
    vp_feats: tuple[float, float, float, float, float, float],
    depth_feats: tuple[float, float, float, float, float, float],
) -> list[float]:
    """متجه ميزات KMeansRegimes (يطابق ``MARKET_REGIME_FEATURE_NAMES``)."""
    return [
        vp_feats[3],
        vp_feats[4],
        vp_feats[5],
        vp_feats[0],
        vp_feats[1],
        vp_feats[2],
        depth_feats[4],
        depth_feats[5],
    ]


def _mid_from_book(book: OrderBook, *, ref_price: float) -> float:
    bb = book.best_bid()
    ba = book.best_ask()
    if bb is None and ba is None:
        return 0.0
    if bb is None:
        assert ba is not None
        return float(ba[0])
    if ba is None:
        return float(bb[0])
    return (bb[0] + ba[0]) / 2.0


def _apply_event_state(
    *,
    action: str,
    side: str,
    price: int,
    size: int,
    order_id: int,
    inst: int,
    nq_instrument_id: int,
    nq_book: OrderBook,
    mnq_book: OrderBook,
    nq_profile: DevelopingVolumeProfile,
    nq_signed: int,
    mnq_signed: int,
    nq_high: float,
    nq_low: float,
    mnq_high: float,
    mnq_low: float,
) -> tuple[int, int, float, float, float, float, int, int, bool, bool, bool, bool]:
    """يحدّث الدفتر/VP/الجلسة فقط — بلا تعبئة ميزات (سريع لكل حدث)."""
    is_nq = int(inst) == nq_instrument_id
    book = nq_book if is_nq else mnq_book
    book.apply(str(action), str(side), int(price), int(size), int(order_id))

    event_nq_delta = 0
    event_mnq_delta = 0
    if is_nq and str(action) == _TRADE:
        nq_profile.add_trade(int(price), int(size))
        trade_size = int(size)
        event_nq_delta = trade_size if str(side) == _BID else -trade_size
        nq_signed += event_nq_delta
    if not is_nq and str(action) == _TRADE:
        trade_size = int(size)
        event_mnq_delta = trade_size if str(side) == _BID else -trade_size
        mnq_signed += event_mnq_delta

    nq_mid = _mid_from_book(nq_book, ref_price=_REF_PRICE)
    mnq_mid = _mid_from_book(mnq_book, ref_price=_REF_PRICE)
    prev_nq_high, prev_nq_low = nq_high, nq_low
    prev_mnq_high, prev_mnq_low = mnq_high, mnq_low
    if nq_mid > 0:
        nq_high = max(nq_high, nq_mid) if nq_high > 0 else nq_mid
        nq_low = min(nq_low, nq_mid) if nq_low > 0 else nq_mid
    if mnq_mid > 0:
        mnq_high = max(mnq_high, mnq_mid) if mnq_high > 0 else mnq_mid
        mnq_low = min(mnq_low, mnq_mid) if mnq_low > 0 else mnq_mid

    nq_new_high = prev_nq_high > 0 and nq_mid > prev_nq_high
    nq_new_low = prev_nq_low > 0 and nq_mid < prev_nq_low
    mnq_new_high = prev_mnq_high > 0 and mnq_mid > prev_mnq_high
    mnq_new_low = prev_mnq_low > 0 and mnq_mid < prev_mnq_low
    return (
        nq_signed,
        mnq_signed,
        nq_high,
        nq_low,
        mnq_high,
        mnq_low,
        event_nq_delta,
        event_mnq_delta,
        nq_new_high,
        nq_new_low,
        mnq_new_high,
        mnq_new_low,
    )


def _pack_tick_row(
    *,
    ts: int,
    seq: int,
    inst: int,
    availability_ts: int,
    nq_book: OrderBook,
    mnq_book: OrderBook,
    nq_profile: DevelopingVolumeProfile,
    regime_tracker: CausalRegimeTracker,
    nq_signed: int,
    mnq_signed: int,
    event_mnq_delta: int,
    nq_new_high: bool,
    nq_new_low: bool,
    mnq_new_high: bool,
    mnq_new_low: bool,
    ref_price: float,
) -> dict[str, float | int]:
    """يُعبّئ صف ميزات من الحالة الحالية (عند الـ snapshot فقط)."""
    nq_row = _book_row(nq_book, ref_price=ref_price)
    mnq_row = _book_row(mnq_book, ref_price=ref_price)
    nq_mid = nq_row[5] * ref_price
    va = nq_profile.value_area()
    vp_feats = nq_profile.features_at_mid(
        nq_mid, ref_price=ref_price, near_ticks=_NEAR_TICKS, va=va
    )
    depth_feats = _book_depth_features(nq_book, va)
    phase = MarketPhase(regime_tracker.update(_regime_features(vp_feats, depth_feats)))
    phase_oh = _phase_one_hot(phase)
    trap = _trap_setup(
        event_mnq_delta,
        mnq_new_high=mnq_new_high,
        mnq_new_low=mnq_new_low,
        nq_new_high=nq_new_high,
        nq_new_low=nq_new_low,
    )
    mask_path = MaskPath.CROSS_TRAP if abs(trap) > 0 else MaskPath.STANDALONE
    return {
        EVENT_TS: int(ts),
        SEQUENCE: int(seq),
        "instrument_id": int(inst),
        "mask_path": int(mask_path),
        "market_phase": int(phase),
        AVAILABILITY_TS: int(availability_ts),
        **dict(zip(BOOK_NQ_NAMES, nq_row, strict=True)),
        **dict(zip(BOOK_MNQ_NAMES, mnq_row, strict=True)),
        **dict(zip(BOOK_DEPTH_NAMES, depth_feats, strict=True)),
        **dict(zip(VP_NAMES, vp_feats, strict=True)),
        **dict(zip(PHASE_NAMES, phase_oh, strict=True)),
        "nq_signed_vol": float(nq_signed),
        "mnq_signed_vol": float(mnq_signed),
        "trap_setup": trap,
    }


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
    nq_profile: DevelopingVolumeProfile,
    regime_tracker: CausalRegimeTracker,
    nq_signed: int,
    mnq_signed: int,
    nq_high: float,
    nq_low: float,
    mnq_high: float,
    mnq_low: float,
    ref_price: float,
    prev_nq_mid: float | None,
) -> tuple[
    dict[str, float | int],
    int,
    int,
    float,
    float,
    float,
    float,
    float | None,
]:
    """توافق: يحدّث الحالة ويعبّئ صفًا كاملًا (وضع كل-حدث)."""
    del prev_nq_mid
    (
        nq_signed,
        mnq_signed,
        nq_high,
        nq_low,
        mnq_high,
        mnq_low,
        _event_nq_delta,
        event_mnq_delta,
        nq_new_high,
        nq_new_low,
        mnq_new_high,
        mnq_new_low,
    ) = _apply_event_state(
        action=action,
        side=side,
        price=price,
        size=size,
        order_id=order_id,
        inst=inst,
        nq_instrument_id=nq_instrument_id,
        nq_book=nq_book,
        mnq_book=mnq_book,
        nq_profile=nq_profile,
        nq_signed=nq_signed,
        mnq_signed=mnq_signed,
        nq_high=nq_high,
        nq_low=nq_low,
        mnq_high=mnq_high,
        mnq_low=mnq_low,
    )
    row = _pack_tick_row(
        ts=ts,
        seq=seq,
        inst=inst,
        availability_ts=ts,
        nq_book=nq_book,
        mnq_book=mnq_book,
        nq_profile=nq_profile,
        regime_tracker=regime_tracker,
        nq_signed=nq_signed,
        mnq_signed=mnq_signed,
        event_mnq_delta=event_mnq_delta,
        nq_new_high=nq_new_high,
        nq_new_low=nq_new_low,
        mnq_new_high=mnq_new_high,
        mnq_new_low=mnq_new_low,
        ref_price=ref_price,
    )
    nq_mid = _mid_from_book(nq_book, ref_price=ref_price)
    return (
        row,
        nq_signed,
        mnq_signed,
        nq_high,
        nq_low,
        mnq_high,
        mnq_low,
        nq_mid if nq_mid > 0 else None,
    )


def build_tick_stream(  # noqa: PLR0912, PLR0915
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    nq_instrument_id: int = 1,
    mnq_instrument_id: int = 2,
    progress: ProgressLike | None = None,
    emit_interval_ns: int | None = 1_000_000_000,
) -> TickStream:
    """يبني تسلسل tick موحّد من MBO خام (NQ + MNQ) مع دفتر حي وميزات inline.

    عندما ``nq is mnq`` (وضع ``nq_only``) يُبنى مسار أداة واحدة دون مضاعفة الأحداث.

    **أداء شهر-مقياس:** الحالة تُحدَّث على *كل* حدث (سببي)، لكن الصفوف تُصدَّر
    كـ snapshots عند حدود ``emit_interval_ns`` (افتراضي 1s) — ليس صفًا لكل حدث.
    ``emit_interval_ns=None`` يعيد السلوك القديم (صف لكل حدث) للاختبارات الدقيقة.
    """
    log = progress
    if emit_interval_ns is not None and emit_interval_ns < 1:
        raise ValueError(f"emit_interval_ns must be >= 1 or None, got {emit_interval_ns}")

    nq_only = nq is mnq
    nq_sorted = sort_causal(nq.with_columns(pl.lit(nq_instrument_id).alias("instrument_id")))
    if nq_only:
        if log is not None:
            log.op(f"tick_stream أحادي (nq_only) — {nq.height:,} حدث (بدون مضاعفة NQ كـ MNQ)")
        combined = nq_sorted
    else:
        if log is not None:
            log.op(f"دمج NQ+MNQ وترتيب سببي (NQ={nq.height:,} · MNQ={mnq.height:,})")
        mnq_sorted = sort_causal(mnq.with_columns(pl.lit(mnq_instrument_id).alias("instrument_id")))
        combined = pl.concat([nq_sorted, mnq_sorted], how="vertical").sort([EVENT_TS, SEQUENCE])

    nq_book = OrderBook()
    mnq_book = OrderBook()
    nq_profile = DevelopingVolumeProfile()
    regime_tracker = CausalRegimeTracker(
        min_samples=32,
        refit_interval=2500,
        fit_window=2048,
        seed=0,
    )
    nq_signed = 0
    mnq_signed = 0
    nq_high = 0.0
    nq_low = 0.0
    mnq_high = 0.0
    mnq_low = 0.0
    ref_price = _REF_PRICE
    current_session: str | None = None

    rows: list[dict[str, float | int]] = []
    actions = combined["action"].to_list()
    sides = combined["side"].to_list()
    prices = combined["price"].to_list()
    sizes = combined["size"].to_list()
    order_ids = combined["order_id"].to_list()
    event_times = combined[EVENT_TS].to_list()
    sequences = combined["sequence"].to_list()
    instruments = combined["instrument_id"].to_list()

    total = len(actions)
    # إن كان مدى الزمن أصغر من فاصل الإصدار (اختبارات/شرائح قصيرة) → أصدر كل حدث
    emit_every = emit_interval_ns is None
    if not emit_every and total > 0:
        t_min = int(min(event_times))
        t_max = int(max(event_times))
        if (t_max - t_min) < int(emit_interval_ns):  # type: ignore[arg-type]
            emit_every = True
            if log is not None:
                log.op(
                    "مدى الزمن < emit_interval — إصدار كل حدث "
                    f"(span={t_max - t_min:,} < {emit_interval_ns:,})"
                )
    if log is not None:
        mode = "كل حدث" if emit_every else f"snapshot كل {emit_interval_ns:,}ns"
        log.op(f"بدء آلة الحالة حدث-بحدث: {total:,} حدث · إصدار={mode}")
    hb_every = 50_000 if total > _HEARTBEAT_LARGE_EVENT_THRESHOLD else (500 if total else 1)
    next_hb = hb_every

    current_bucket: int | None = None
    last_ts = 0
    last_seq = 0
    last_inst = nq_instrument_id
    # آخر أعلام trap داخل البرميل (OR سببي للبرميل)
    bucket_nq_new_high = False
    bucket_nq_new_low = False
    bucket_mnq_new_high = False
    bucket_mnq_new_low = False
    bucket_mnq_delta = 0

    def _emit(*, availability_ts: int) -> None:
        rows.append(
            _pack_tick_row(
                ts=last_ts,
                seq=last_seq,
                inst=last_inst,
                availability_ts=availability_ts,
                nq_book=nq_book,
                mnq_book=mnq_book,
                nq_profile=nq_profile,
                regime_tracker=regime_tracker,
                nq_signed=nq_signed,
                mnq_signed=mnq_signed,
                event_mnq_delta=bucket_mnq_delta,
                nq_new_high=bucket_nq_new_high,
                nq_new_low=bucket_nq_new_low,
                mnq_new_high=bucket_mnq_new_high,
                mnq_new_low=bucket_mnq_new_low,
                ref_price=ref_price,
            )
        )

    for i, (action, side, price, size, order_id, ts, seq, inst) in enumerate(
        zip(
            actions,
            sides,
            prices,
            sizes,
            order_ids,
            event_times,
            sequences,
            instruments,
            strict=True,
        ),
        start=1,
    ):
        ts_i = int(ts)
        session = session_date_from_ns(ts_i)
        if current_session is None:
            current_session = session
        elif session != current_session:
            nq_signed = 0
            mnq_signed = 0
            nq_high = 0.0
            nq_low = 0.0
            mnq_high = 0.0
            mnq_low = 0.0
            current_session = session

        (
            nq_signed,
            mnq_signed,
            nq_high,
            nq_low,
            mnq_high,
            mnq_low,
            _enq,
            emnq,
            nq_nh,
            nq_nl,
            mnq_nh,
            mnq_nl,
        ) = _apply_event_state(
            action=str(action),
            side=str(side),
            price=int(price),
            size=int(size),
            order_id=int(order_id),
            inst=int(inst),
            nq_instrument_id=nq_instrument_id,
            nq_book=nq_book,
            mnq_book=mnq_book,
            nq_profile=nq_profile,
            nq_signed=nq_signed,
            mnq_signed=mnq_signed,
            nq_high=nq_high,
            nq_low=nq_low,
            mnq_high=mnq_high,
            mnq_low=mnq_low,
        )
        last_ts, last_seq, last_inst = ts_i, int(seq), int(inst)

        if emit_every:
            row = _pack_tick_row(
                ts=ts_i,
                seq=int(seq),
                inst=int(inst),
                availability_ts=ts_i,
                nq_book=nq_book,
                mnq_book=mnq_book,
                nq_profile=nq_profile,
                regime_tracker=regime_tracker,
                nq_signed=nq_signed,
                mnq_signed=mnq_signed,
                event_mnq_delta=emnq,
                nq_new_high=nq_nh,
                nq_new_low=nq_nl,
                mnq_new_high=mnq_nh,
                mnq_new_low=mnq_nl,
                ref_price=ref_price,
            )
            rows.append(row)
        else:
            assert emit_interval_ns is not None
            bucket = (ts_i // emit_interval_ns) * emit_interval_ns
            if current_bucket is None:
                current_bucket = bucket
            elif bucket != current_bucket:
                _emit(availability_ts=current_bucket + emit_interval_ns)
                current_bucket = bucket
                bucket_nq_new_high = False
                bucket_nq_new_low = False
                bucket_mnq_new_high = False
                bucket_mnq_new_low = False
                bucket_mnq_delta = 0
            bucket_nq_new_high = bucket_nq_new_high or nq_nh
            bucket_nq_new_low = bucket_nq_new_low or nq_nl
            bucket_mnq_new_high = bucket_mnq_new_high or mnq_nh
            bucket_mnq_new_low = bucket_mnq_new_low or mnq_nl
            bucket_mnq_delta += emnq

        if log is not None and (i >= next_hb or i == total):
            log.heartbeat(i, total, label="tick_stream", force=True, every=hb_every)
            next_hb = i + hb_every

    if not emit_every and current_bucket is not None and total > 0:
        assert emit_interval_ns is not None
        _emit(availability_ts=current_bucket + emit_interval_ns)

    if not rows:
        if log is not None:
            log.op("آلة الحالة: لا أحداث — إطار فارغ")
        return TickStream(frame=pl.DataFrame(schema=_TICK_SCHEMA))

    if log is not None:
        log.op(f"تجميع DataFrame من {len(rows):,} snapshot (من {total:,} حدث)")
    frame = pl.DataFrame(rows)
    if log is not None:
        log.op(f"اكتمل tick_stream: {frame.height:,} صف")
    return TickStream(frame=frame)


__all__ = [
    "TICK_FEATURE_NAMES",
    "MarketPhase",
    "MaskPath",
    "TickStream",
    "build_tick_stream",
]
