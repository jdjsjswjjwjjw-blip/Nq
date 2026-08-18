"""دورة حياة الأمر من MBO: تنفيذ مقابل إلغاء سريع، عند امتداد تحدّده العين.

الصفقات داخل تدفق MBO (``T``/``F``) — لا ملف Trades منفصل. يوم جلسة واحد.
التصنيف حتى ``t`` فقط: أمر ما زال معلّقًا عند الامتداد = ``open`` وليس خادعًا.
``fleeting_unfilled`` حدس بحثي (إلغاء كامل بلا تنفيذ، عمر < ثانيتين)، ليس
حكمًا قانونيًا بالسبوفينج. ليست overlay حيّة.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl

from nq.contracts.mbo import MboAction
from nq.research.mbo_sequence_mlp import assert_no_future_events, assert_single_day_mbo

LAYER_ID = "order_lifecycle"
SECOND_NS: Final = 1_000_000_000
FLEETING_NS: Final = 2 * SECOND_NS
ULTRAFAST_NS: Final = 100_000_000
DEFAULT_WINDOWS_S: Final = (10, 20, 30)
_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_MODIFY = MboAction.MODIFY.value
_CLEAR = MboAction.CLEAR.value
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_KIND_OPEN = "open"
_KIND_GENUINE = "genuine"
_KIND_FLEETING = "fleeting_unfilled"
_KIND_CANCELLED = "cancelled_unfilled"


@dataclass(frozen=True, slots=True)
class ClosedOrder:
    """أمر أُغلق (إلغاء كامل) في أو قبل ``t``."""

    order_id: int
    add_ts: int
    close_ts: int
    add_size: int
    executed_size: int
    lifetime_ns: int
    price: int
    had_refill: bool

    @property
    def kind(self) -> str:
        if self.executed_size > 0:
            return _KIND_GENUINE
        if self.lifetime_ns < FLEETING_NS:
            return _KIND_FLEETING
        return _KIND_CANCELLED


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    """إحصاءات نافذة ``(t-h, t]`` سببية."""

    window_s: int
    n_add: int
    n_cancel: int
    n_trade: int
    add_size: int
    cancel_size: int
    trade_size: int
    n_fleeting: int
    fleeting_size: int
    n_ultrafast: int
    n_genuine_close: int
    n_iceberg: int
    mean_close_lifetime_ns: float
    execution_ratio: float
    cancel_ratio: float
    cancel_to_trade: float
    fleeting_to_trade: float


class _LiveOrder:
    __slots__ = ("add_size", "add_ts", "executed", "had_refill", "oid", "price", "size")

    def __init__(self, oid: int, ts: int, size: int, price: int) -> None:
        self.oid = oid
        self.add_ts = ts
        self.add_size = max(int(size), 0)
        self.size = self.add_size
        self.price = int(price)
        self.executed = 0
        self.had_refill = False


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _empty_counts() -> dict[str, int]:
    return {
        "n_add": 0,
        "n_cancel": 0,
        "n_trade": 0,
        "add_size": 0,
        "cancel_size": 0,
        "trade_size": 0,
    }


class _LifecycleWalk:
    """آلة حالات أمر حتى ``t``."""

    def __init__(self) -> None:
        self.live: dict[int, _LiveOrder] = {}
        self.at_price: dict[int, set[int]] = {}
        self.closed: list[ClosedOrder] = []
        self.counts = _empty_counts()

    def unlink(self, oid: int, price: int) -> None:
        bucket = self.at_price.get(price)
        if bucket is None:
            return
        bucket.discard(oid)
        if not bucket:
            self.at_price.pop(price, None)

    def link(self, oid: int, price: int) -> None:
        self.at_price.setdefault(price, set()).add(oid)

    def drop(self, oid: int) -> _LiveOrder | None:
        order = self.live.pop(oid, None)
        if order is not None:
            self.unlink(oid, order.price)
        return order

    def close(self, order: _LiveOrder, ts: int) -> None:
        self.closed.append(
            ClosedOrder(
                order_id=order.oid,
                add_ts=order.add_ts,
                close_ts=int(ts),
                add_size=order.add_size,
                executed_size=order.executed,
                lifetime_ns=int(ts) - order.add_ts,
                price=order.price,
                had_refill=order.had_refill,
            )
        )

    def on_clear(self) -> None:
        self.live.clear()
        self.at_price.clear()

    def on_trade(self, oid: int, price: int, size: int) -> None:
        self.counts["n_trade"] += 1
        self.counts["trade_size"] += max(size, 0)
        touched: list[_LiveOrder] = []
        if oid in self.live:
            touched.append(self.live[oid])
        elif price in self.at_price:
            touched.extend(self.live[i] for i in self.at_price[price] if i in self.live)
        share = max(size, 0)
        if len(touched) == 1:
            touched[0].executed += share
        elif len(touched) > 1 and share > 0:
            for order in touched:
                order.executed += 1

    def on_add(self, oid: int, ts: int, size: int, price: int) -> None:
        self.counts["n_add"] += 1
        self.counts["add_size"] += max(size, 0)
        if oid <= 0:
            return
        old = self.drop(oid)
        if old is not None:
            self.close(old, ts)
        order = _LiveOrder(oid, ts, size, price)
        self.live[oid] = order
        self.link(oid, price)

    def on_modify(self, order: _LiveOrder, size: int, price: int) -> None:
        if size > order.size:
            order.had_refill = True
        if price != order.price:
            self.unlink(order.oid, order.price)
            order.price = price
            self.link(order.oid, price)
        order.size = max(size, 0)

    def on_cancel(self, order: _LiveOrder, ts: int, size: int) -> None:
        self.counts["n_cancel"] += 1
        cancel_qty = size if size > 0 else order.size
        self.counts["cancel_size"] += max(min(cancel_qty, order.size), 0)
        remaining = max(order.size - min(cancel_qty, order.size), 0)
        if remaining > 0:
            order.size = remaining
            return
        self.drop(order.oid)
        self.close(order, ts)

    def on_event(self, action: str, ts: int, size: int, price: int, oid: int) -> None:
        if action == _CLEAR:
            self.on_clear()
            return
        if action in {_TRADE, _FILL}:
            self.on_trade(oid, price, size)
            return
        if action == _ADD:
            self.on_add(oid, ts, size, price)
            return
        if oid <= 0 or oid not in self.live:
            return
        order = self.live[oid]
        if action == _MODIFY:
            self.on_modify(order, size, price)
            return
        if action == _CANCEL:
            self.on_cancel(order, ts, size)


def close_orders_by_t(
    mbo: pl.DataFrame, availability_ts: int
) -> tuple[list[ClosedOrder], dict[str, int]]:
    """يغلق الأوامر حتى ``t``. الصفقات تُنسَب لـ ``order_id`` إن وُجد، وإلا لمستوى السعر."""
    assert_single_day_mbo(mbo)
    known = assert_no_future_events(mbo, availability_ts)
    walk = _LifecycleWalk()
    if known.height == 0:
        return [], walk.counts
    actions = known["action"].cast(pl.Utf8).fill_null("").to_list()
    times = known["event_ts"].to_list()
    sizes = known["size"].cast(pl.Int64).fill_null(0).to_list()
    prices = known["price"].cast(pl.Int64).fill_null(0).to_list()
    oids = (
        known["order_id"].cast(pl.Int64).fill_null(0).to_list()
        if "order_id" in known.columns
        else [0] * known.height
    )
    for action, ts_raw, size_raw, price_raw, oid_raw in zip(
        actions, times, sizes, prices, oids, strict=True
    ):
        walk.on_event(str(action), int(ts_raw), int(size_raw), int(price_raw), int(oid_raw))
    return walk.closed, walk.counts


def _window_event_totals(
    mbo: pl.DataFrame,
    availability_ts: int,
    window_ns: int,
) -> dict[str, int]:
    known = assert_no_future_events(mbo, availability_ts)
    if known.height == 0:
        return _empty_counts()
    lo = int(availability_ts) - int(window_ns)
    hi = int(availability_ts)
    work = known.filter((pl.col("event_ts") > lo) & (pl.col("event_ts") <= hi))
    if work.height == 0:
        return _empty_counts()
    actions = work["action"].cast(pl.Utf8).fill_null("").to_list()
    sizes = work["size"].cast(pl.Int64).fill_null(0).to_list()
    out = _empty_counts()
    for action, size in zip(actions, sizes, strict=True):
        qty = max(int(size), 0)
        if action == _ADD:
            out["n_add"] += 1
            out["add_size"] += qty
        elif action == _CANCEL:
            out["n_cancel"] += 1
            out["cancel_size"] += qty
        elif action in {_TRADE, _FILL}:
            out["n_trade"] += 1
            out["trade_size"] += qty
    return out


def window_metrics(
    mbo: pl.DataFrame,
    availability_ts: int,
    *,
    windows_s: Sequence[int] = DEFAULT_WINDOWS_S,
    prices: Sequence[int] | None = None,
) -> list[WindowMetrics]:
    """نسب النافذة قبل ``t``. ``prices`` اختياري لتقييد أوامر العين."""
    assert_single_day_mbo(mbo)
    closed, _ = close_orders_by_t(mbo, availability_ts)
    allowed = None if prices is None else set(int(p) for p in prices)
    rows: list[WindowMetrics] = []
    t = int(availability_ts)
    for seconds in windows_s:
        window_ns = int(seconds) * SECOND_NS
        lo = t - window_ns
        events = _window_event_totals(mbo, t, window_ns)
        in_win = [
            order
            for order in closed
            if lo < order.close_ts <= t and (allowed is None or order.price in allowed)
        ]
        fleeting = [o for o in in_win if o.kind == _KIND_FLEETING]
        genuine = [o for o in in_win if o.kind == _KIND_GENUINE]
        iceberg = [o for o in genuine if o.had_refill]
        ultra = [o for o in fleeting if o.lifetime_ns < ULTRAFAST_NS]
        fleeting_size = int(sum(o.add_size for o in fleeting))
        trade_size = events["trade_size"]
        lives = [float(o.lifetime_ns) for o in in_win]
        mean_life = float(sum(lives) / len(lives)) if lives else float("nan")
        rows.append(
            WindowMetrics(
                window_s=int(seconds),
                n_add=events["n_add"],
                n_cancel=events["n_cancel"],
                n_trade=events["n_trade"],
                add_size=events["add_size"],
                cancel_size=events["cancel_size"],
                trade_size=trade_size,
                n_fleeting=len(fleeting),
                fleeting_size=fleeting_size,
                n_ultrafast=len(ultra),
                n_genuine_close=len(genuine),
                n_iceberg=len(iceberg),
                mean_close_lifetime_ns=mean_life,
                execution_ratio=_ratio(events["n_trade"], events["n_add"]),
                cancel_ratio=_ratio(events["n_cancel"], events["n_add"]),
                cancel_to_trade=_ratio(events["cancel_size"], trade_size),
                fleeting_to_trade=_ratio(fleeting_size, trade_size),
            )
        )
    return rows


def metrics_frame(rows: Sequence[WindowMetrics]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "window_s": pl.Int64(),
                "n_add": pl.Int64(),
                "n_cancel": pl.Int64(),
                "n_trade": pl.Int64(),
                "add_size": pl.Int64(),
                "cancel_size": pl.Int64(),
                "trade_size": pl.Int64(),
                "n_fleeting": pl.Int64(),
                "fleeting_size": pl.Int64(),
                "n_ultrafast": pl.Int64(),
                "n_genuine_close": pl.Int64(),
                "n_iceberg": pl.Int64(),
                "mean_close_lifetime_ns": pl.Float64(),
                "execution_ratio": pl.Float64(),
                "cancel_ratio": pl.Float64(),
                "cancel_to_trade": pl.Float64(),
                "fleeting_to_trade": pl.Float64(),
            }
        )
    return pl.DataFrame(
        [
            {
                "window_s": r.window_s,
                "n_add": r.n_add,
                "n_cancel": r.n_cancel,
                "n_trade": r.n_trade,
                "add_size": r.add_size,
                "cancel_size": r.cancel_size,
                "trade_size": r.trade_size,
                "n_fleeting": r.n_fleeting,
                "fleeting_size": r.fleeting_size,
                "n_ultrafast": r.n_ultrafast,
                "n_genuine_close": r.n_genuine_close,
                "n_iceberg": r.n_iceberg,
                "mean_close_lifetime_ns": r.mean_close_lifetime_ns,
                "execution_ratio": r.execution_ratio,
                "cancel_ratio": r.cancel_ratio,
                "cancel_to_trade": r.cancel_to_trade,
                "fleeting_to_trade": r.fleeting_to_trade,
            }
            for r in rows
        ]
    )


def write_lifecycle_report(
    scored: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scored.height:
        scored.write_parquet(out / "order_lifecycle.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Order lifecycle before an eye-specified extension",
        "",
        "MBO-only. Trades are T/F in the same stream. One session day.",
        "fleeting_unfilled = full cancel, no fill by t, lifetime < 2s (heuristic).",
        "Open orders at t are not labelled fleeting. Not a live overlay.",
        "",
        "| win_s | n_add | n_cancel | n_trade | fleeting_sz | trade_sz | fleeting/trade |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scored.iter_rows(named=True):
        ratio = row["fleeting_to_trade"]
        ratio_s = "nan" if ratio is None else f"{float(ratio):.3f}"
        lines.append(
            f"| {row['window_s']} | {row['n_add']} | {row['n_cancel']} | "
            f"{row['n_trade']} | {row['fleeting_size']} | {row['trade_size']} | {ratio_s} |"
        )
    lines.append("")
    (out / "ORDER_LIFECYCLE.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "DEFAULT_WINDOWS_S",
    "FLEETING_NS",
    "LAYER_ID",
    "ULTRAFAST_NS",
    "ClosedOrder",
    "WindowMetrics",
    "close_orders_by_t",
    "metrics_frame",
    "window_metrics",
    "write_lifecycle_report",
]
