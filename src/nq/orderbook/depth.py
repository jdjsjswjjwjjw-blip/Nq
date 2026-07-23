"""لقطات عمق الدفتر ومسح المستويات للتنفيذ/الخروج (سببي).

كل اللقطات تُشتق من ``OrderBook`` بعد أحداث ≤ زمن القرار فقط.
مسح الدفتر (walk) يحوّل الكمية المطلوبة إلى سعر VWAP من المستويات
المعروضة عند ``availability_ts`` — بلا افتراض سيولة غير مرئية.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nq.contracts.mbo import PRICE_SCALE

_BID = "B"
_ASK = "A"


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    """لقطة عمق معلّق عند نقطة زمنية واحدة."""

    availability_ts: int
    best_bid: int | None
    bid_size: int
    best_ask: int | None
    ask_size: int
    bid_levels: tuple[tuple[int, int], ...]
    ask_levels: tuple[tuple[int, int], ...]
    cum_bid: int
    cum_ask: int
    imbalance: float
    trail_bid: int
    trail_ask: int
    n_levels: int

    def size_at(self, side: str, price: int) -> int:
        levels = self.bid_levels if side == _BID else self.ask_levels
        for px, sz in levels:
            if px == price:
                return int(sz)
        return 0


def walk_buy_vwap(
    ask_levels: Sequence[tuple[int, int]],
    qty: int,
    *,
    price_scale: float = PRICE_SCALE,
) -> float | None:
    """سعر شراء عدواني بمسح عروض معلّقة؛ ``None`` إن لم تكفِ السيولة."""
    if qty < 1:
        raise ValueError(f"qty must be >= 1, got {qty}")
    remaining = qty
    notional = 0
    filled = 0
    for px, sz in ask_levels:
        if sz <= 0:
            continue
        take = min(remaining, int(sz))
        notional += take * int(px)
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled < qty:
        return None
    return (notional / filled) * price_scale


def walk_sell_vwap(
    bid_levels: Sequence[tuple[int, int]],
    qty: int,
    *,
    price_scale: float = PRICE_SCALE,
) -> float | None:
    """سعر بيع عدواني بمسح طلبات معلّقة؛ ``None`` إن لم تكفِ السيولة."""
    if qty < 1:
        raise ValueError(f"qty must be >= 1, got {qty}")
    remaining = qty
    notional = 0
    filled = 0
    for px, sz in bid_levels:
        if sz <= 0:
            continue
        take = min(remaining, int(sz))
        notional += take * int(px)
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled < qty:
        return None
    return (notional / filled) * price_scale


def unpack_levels_from_row(
    row: dict[str, float | int | None],
    *,
    n_levels: int,
    side: str,
) -> list[tuple[int, int]]:
    """يعيد بناء مستويات من أعمدة ``depth_{side}_px_k`` / ``depth_{side}_sz_k``."""
    prefix = "bid" if side == _BID else "ask"
    levels: list[tuple[int, int]] = []
    for k in range(1, n_levels + 1):
        px = row.get(f"depth_{prefix}_px_{k}")
        sz = row.get(f"depth_{prefix}_sz_{k}")
        if px is None or sz is None:
            continue
        px_f = float(px)
        sz_i = int(sz)
        if sz_i <= 0:
            continue
        # الأعمدة مخزّنة بسعر حقيقي؛ نعيد للنقطة الثابتة للمسح الداخلي
        levels.append((int(round(px_f / PRICE_SCALE)), sz_i))
    return levels


__all__ = [
    "DepthSnapshot",
    "unpack_levels_from_row",
    "walk_buy_vwap",
    "walk_sell_vwap",
]
