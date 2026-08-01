"""فلتر ضوضاء عمق سببي — عواصف إلغاء / وميض / سبوف.

يُصفّي أحداث MBO التي تُشوّه لقطة الدفتر دون إعادة كتابة تاريخ الماضي:

* ``cancel_storm`` — معدّل إلغاء مرتفع في نافذة زمنية ماضية فقط.
* ``flicker`` — إضافة ثم إلغاء سريع لنفس ``order_id`` (وميض).
* ``spoof`` — إضافة كبيرة بعيدة عن منتصف الدفتر ثم إلغاء دون تنفيذ.

كل قرار فلترة يستخدم فقط أحداثًا بـ ``event_ts <= t`` (سببي / PIT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from nq.contracts.mbo import PRICE_SCALE, MboAction
from nq.contracts.temporal import EVENT_TS
from nq.core.time import sort_causal

_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_MODIFY = MboAction.MODIFY.value

_DEFAULT_TICK: Final = 0.25
_TICK_FIXED: Final = round(_DEFAULT_TICK / PRICE_SCALE)


@dataclass(frozen=True, slots=True)
class DepthNoiseConfig:
    """عتبات فلتر الضوضاء (سببية)."""

    window_ns: int = 500_000_000  # 0.5s
    cancel_storm_ratio: float = 0.75
    cancel_storm_min_events: int = 8
    flicker_ns: int = 50_000_000  # 50ms
    spoof_ticks_from_mid: int = 4
    spoof_min_size: int = 5
    spoof_cancel_ns: int = 250_000_000  # 0.25s


def filter_depth_noise(  # noqa: PLR0912, PLR0915
    frame: pl.DataFrame,
    *,
    config: DepthNoiseConfig | None = None,
) -> pl.DataFrame:
    """يُعيد إطار MBO بعد إسقاط أحداث الضوضاء السببية.

    يحافظ على الصفقات/التنفيذات دائمًا. يُسقط الإلغاءات الملوّثة والاضافات
    المرتبطة بوميض/سبوف عند اكتشافها من الماضي فقط.
    """
    cfg = config if config is not None else DepthNoiseConfig()
    if frame.height == 0:
        return frame

    work = sort_causal(frame)
    actions = work["action"].cast(pl.Utf8).to_list()
    sides = work["side"].cast(pl.Utf8).to_list()
    prices = work["price"].to_list()
    sizes = work["size"].to_list()
    order_ids = work["order_id"].to_list()
    event_ts = work[EVENT_TS].to_list()
    n = len(actions)

    # حالة سببية للطلبات الظاهرة
    live: dict[int, tuple[int, int, str, int]] = {}  # oid -> (ts, price, side, size)
    executed_oids: set[int] = set()
    keep = [True] * n

    # نافذة أحداث حديثة للـ cancel storm: (ts, is_cancel)
    recent: list[tuple[int, bool]] = []
    head = 0

    best_bid: int | None = None
    best_ask: int | None = None

    for i in range(n):
        action = str(actions[i])
        side = str(sides[i])
        price = int(prices[i])
        size = int(sizes[i])
        oid = int(order_ids[i])
        ts = int(event_ts[i])

        # تنظيف نافذة العاصفة
        cutoff = ts - cfg.window_ns
        while head < len(recent) and recent[head][0] < cutoff:
            head += 1
        window = recent[head:]
        n_win = len(window)
        n_cancel = sum(1 for _, c in window if c)
        storm = (
            n_win >= cfg.cancel_storm_min_events
            and (n_cancel / float(n_win)) >= cfg.cancel_storm_ratio
        )

        if action in (_TRADE, _FILL):
            executed_oids.add(oid)
            # الصفقات تُبقى دائمًا — حقيقة الشريط
            recent.append((ts, False))
            keep[i] = True
            continue

        if action == _ADD:
            mid = None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2.0
            # وميض/سبوف يُحكمان عند الإلغاء لاحقًا؛ الإضافة تُسجَّل
            live[oid] = (ts, price, side, size)
            if side == "B":
                best_bid = price if best_bid is None else max(best_bid, price)
            elif side == "A":
                best_ask = price if best_ask is None else min(best_ask, price)
            recent.append((ts, False))
            keep[i] = True
            # إن كنا في عاصفة إلغاء، نتجاهل إضافات ضخمة بعيدة (سبوف محتمل)
            if storm and mid is not None and size >= cfg.spoof_min_size:
                dist_ticks = abs(price - mid) / float(_TICK_FIXED)
                if dist_ticks >= cfg.spoof_ticks_from_mid:
                    keep[i] = False
                    live.pop(oid, None)
            continue

        if action == _CANCEL:
            meta = live.get(oid)
            drop = storm
            if meta is not None:
                add_ts, add_px, _add_side, add_sz = meta
                dt = ts - add_ts
                if dt <= cfg.flicker_ns and oid not in executed_oids:
                    drop = True  # flicker
                mid = None
                if best_bid is not None and best_ask is not None:
                    mid = (best_bid + best_ask) / 2.0
                if (
                    mid is not None
                    and add_sz >= cfg.spoof_min_size
                    and dt <= cfg.spoof_cancel_ns
                    and oid not in executed_oids
                ):
                    dist_ticks = abs(add_px - mid) / float(_TICK_FIXED)
                    if dist_ticks >= cfg.spoof_ticks_from_mid:
                        drop = True  # spoof cancel
                live.pop(oid, None)
            recent.append((ts, True))
            keep[i] = not drop
            continue

        if action == _MODIFY:
            if oid in live:
                _, _, old_side, _ = live[oid]
                live[oid] = (ts, price, old_side, size)
            recent.append((ts, False))
            keep[i] = True
            continue

        recent.append((ts, False))
        keep[i] = True

    mask = pl.Series("_keep_noise", keep)
    return work.with_columns(mask).filter(pl.col("_keep_noise")).drop("_keep_noise")


__all__ = ["DepthNoiseConfig", "filter_depth_noise"]
