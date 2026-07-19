"""مُحاكي البصمة السعرية (Footprint Simulator).

البصمة تُظهر الحجم العدواني المُنفَّذ عند كل سعر داخل كل نافذة زمنية، مفصولًا
إلى شراء/بيع، مع مقاييس الدلتا والاختلال والامتصاص.

التعريفات الرياضية (لكل خلية = نافذة × سعر):

* ``buy_volume``  = Σ حجم الصفقات ذات المُعتدي المشتري عند السعر.
* ``sell_volume`` = Σ حجم الصفقات ذات المُعتدي البائع عند السعر.
* ``delta``       = ``buy_volume - sell_volume``.
* ``imbalance``   = ``delta / total_volume`` ∈ [-1, 1] (0 عند غياب الحجم).

ملخّص النافذة يضيف:

* ``cumulative_delta`` = المجموع التراكمي للدلتا عبر النوافذ (سببي).
* ``absorption_ratio`` = ``total_volume / (price_range_ticks + 1)`` — حجم كبير
  مع مدى سعري ضيّق يدل على امتصاص السيولة (absorption) عند المستوى.

كل الميزات مُجمّعة على النافذة وتحمل ``availability_ts = bucket_end`` (point-in-time).
"""

from __future__ import annotations

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades


def _imbalance(delta: pl.Expr, total: pl.Expr) -> pl.Expr:
    return pl.when(total > 0).then(delta / total).otherwise(0.0)


def footprint_cells(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يحسب البصمة لكل خلية (نافذة × سعر).

    الأعمدة: ``bucket_start``, ``price``, ``buy_volume``, ``sell_volume``,
    ``delta``, ``total_volume``, ``imbalance``, ``bucket_end``, ``availability_ts``.
    """
    trades = extract_trades(add_time_bucket(frame, interval_ns=interval_ns))
    cells = trades.group_by([BUCKET_START, "price"]).agg(
        pl.col("buy_volume").sum(),
        pl.col("sell_volume").sum(),
        pl.col(BUCKET_END).first(),
        pl.col(AVAILABILITY_TS).first(),
    )
    delta = pl.col("buy_volume") - pl.col("sell_volume")
    total = pl.col("buy_volume") + pl.col("sell_volume")
    return (
        cells.with_columns(
            delta.alias("delta"),
            total.alias("total_volume"),
        )
        .with_columns(_imbalance(pl.col("delta"), pl.col("total_volume")).alias("imbalance"))
        .sort([BUCKET_START, "price"])
    )


def footprint_summary(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    price_tick: int = 1,
) -> pl.DataFrame:
    """يلخّص البصمة لكل نافذة زمنية مع الدلتا التراكمية ونسبة الامتصاص.

    الأعمدة: ``bucket_start``, ``buy_volume``, ``sell_volume``, ``delta``,
    ``total_volume``, ``cumulative_delta``, ``price_range``, ``absorption_ratio``,
    ``bucket_end``, ``availability_ts``.
    """
    if price_tick < 1:
        raise ValueError(f"price_tick must be >= 1, got {price_tick}")

    trades = extract_trades(add_time_bucket(frame, interval_ns=interval_ns))
    buckets = trades.group_by(BUCKET_START).agg(
        pl.col("buy_volume").sum(),
        pl.col("sell_volume").sum(),
        pl.col("price").max().alias("price_max"),
        pl.col("price").min().alias("price_min"),
        pl.col(BUCKET_END).first(),
        pl.col(AVAILABILITY_TS).first(),
    )
    delta = pl.col("buy_volume") - pl.col("sell_volume")
    total = pl.col("buy_volume") + pl.col("sell_volume")
    price_range = (pl.col("price_max") - pl.col("price_min")) // price_tick
    return (
        buckets.with_columns(
            delta.alias("delta"),
            total.alias("total_volume"),
            price_range.alias("price_range"),
        )
        .sort(BUCKET_START)
        .with_columns(
            pl.col("delta").cum_sum().alias("cumulative_delta"),
            (pl.col("total_volume") / (pl.col("price_range") + 1)).alias("absorption_ratio"),
        )
        .drop("price_max", "price_min")
    )
