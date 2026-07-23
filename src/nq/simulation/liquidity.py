"""مُحاكي السيولة (Liquidity Simulator).

يقيس ديناميكية السيولة القائمة (resting) من أحداث الأوامر:

* ``added_volume``  — الحجم المُضاف عبر أحداث ``ADD`` لكل نافذة.
* ``pulled_volume`` — الحجم المسحوب عبر أحداث ``CANCEL`` لكل نافذة.
* ``net_liquidity`` = ``added_volume - pulled_volume`` (تدفّق صافٍ للسيولة).

كشف الآيسبرغ (Iceberg Detection) — heuristic سببي لكل سعر:

* ``executed``       — الحجم المُنفَّذ عند السعر (من الصفقات، tape).
* ``peak_display``   — أعلى حجم ظاهر (resting) لوحظ عند السعر.
* ``replenish_count``— مرّات إعادة التعبئة: إضافة عند سعر بعد أن نُفّذ حجم عنده.

يُرفع علم ``is_iceberg`` عندما يتجاوز المُنفَّذ الحجم الظاهر بمضاعف
``min_hidden_ratio`` مع إعادة تعبئة لا تقل عن ``min_refills`` — دلالة على أمر
مخفيّ يُظهر جزءًا صغيرًا ويُعاد تعبئته.
"""

from __future__ import annotations

import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.time import assert_sorted_causal
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket

_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_FILL = MboAction.FILL.value
_TRADE = MboAction.TRADE.value

_DEFAULT_MIN_REFILLS = 2
_DEFAULT_MIN_HIDDEN_RATIO = 2.0


def liquidity_summary(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يلخّص إضافة/سحب السيولة لكل نافذة (متاح عند ``bucket_end``)."""
    ev = add_time_bucket(frame, interval_ns=interval_ns)
    size = pl.col("size").cast(pl.Int64)
    added = pl.when(pl.col("action") == _ADD).then(size).otherwise(0)
    pulled = pl.when(pl.col("action") == _CANCEL).then(size).otherwise(0)
    return (
        ev.with_columns(added.alias("_added"), pulled.alias("_pulled"))
        .group_by(BUCKET_START)
        .agg(
            pl.col("_added").sum().alias("added_volume"),
            pl.col("_pulled").sum().alias("pulled_volume"),
            pl.col(BUCKET_END).first(),
            pl.col(AVAILABILITY_TS).first(),
        )
        .with_columns((pl.col("added_volume") - pl.col("pulled_volume")).alias("net_liquidity"))
        .sort(BUCKET_START)
    )


def detect_icebergs(
    frame: pl.DataFrame,
    *,
    min_refills: int = _DEFAULT_MIN_REFILLS,
    min_hidden_ratio: float = _DEFAULT_MIN_HIDDEN_RATIO,
) -> pl.DataFrame:
    """يكشف الأوامر المخفيّة (icebergs) لكل سعر عبر مسح سببي للأحداث.

    يفترض إطارًا لأداة واحدة مرتّبًا سببيًا. يُعيد لكل سعر: ``peak_display``،
    ``executed``، ``replenish_count``، و علم ``is_iceberg``.
    """
    if min_refills < 1:
        raise ValueError(f"min_refills must be >= 1, got {min_refills}")
    if min_hidden_ratio <= 0:
        raise ValueError(f"min_hidden_ratio must be > 0, got {min_hidden_ratio}")
    assert_sorted_causal(frame)

    actions: list[str] = frame["action"].cast(pl.Utf8).to_list()
    prices: list[int] = frame["price"].to_list()
    sizes: list[int] = frame["size"].to_list()

    resting: dict[int, int] = {}
    peak: dict[int, int] = {}
    executed: dict[int, int] = {}
    exec_at_last_add: dict[int, int] = {}
    replenish: dict[int, int] = {}

    for action, price, size in zip(actions, prices, sizes, strict=True):
        if action == _ADD:
            resting[price] = resting.get(price, 0) + size
            peak[price] = max(peak.get(price, 0), resting[price])
            done = executed.get(price, 0)
            if done > exec_at_last_add.get(price, 0):
                replenish[price] = replenish.get(price, 0) + 1
            exec_at_last_add[price] = done
        elif action == _CANCEL:
            resting[price] = max(0, resting.get(price, 0) - size)
        elif action in (_TRADE, _FILL):
            executed[price] = executed.get(price, 0) + size
            resting[price] = max(0, resting.get(price, 0) - size)

    prices_seen = sorted(peak)
    rows = [
        {
            "price": p,
            "peak_display": peak.get(p, 0),
            "executed": executed.get(p, 0),
            "replenish_count": replenish.get(p, 0),
        }
        for p in prices_seen
    ]
    if not rows:
        return pl.DataFrame(
            schema={
                "price": pl.Int64(),
                "peak_display": pl.Int64(),
                "executed": pl.Int64(),
                "replenish_count": pl.Int64(),
                "is_iceberg": pl.Boolean(),
            }
        )

    is_iceberg = (
        (pl.col("peak_display") > 0)
        & (pl.col("replenish_count") >= min_refills)
        & (pl.col("executed") >= min_hidden_ratio * pl.col("peak_display"))
    )
    return pl.DataFrame(rows).with_columns(is_iceberg.alias("is_iceberg"))
