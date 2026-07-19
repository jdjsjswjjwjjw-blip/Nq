"""مُحاكي المزاد (Auction Market Simulator).

يستند إلى نظرية المزاد ومنطقة القيمة لوصف حالة السوق لكل نافذة زمنية:

* التوازن/الاختلال (Balance / Imbalance): السوق **متوازن** حين يُغلق داخل منطقة
  القيمة (قبول القيمة، ``close_in_value``) دون تمدّد مدى؛ و**مختلّ** حين يُغلق
  خارج منطقة القيمة (رفض/قبول بعيدًا عن القيمة → اتجاه) أو مع تمدّد مدى. الانقلاب
  من متوازن إلى مختلّ يُكشَف بتغيّر ``is_balanced`` من ``True`` إلى ``False``.
  (تبقى ``in_value_fraction`` مقياسًا مُبلَّغًا مساعدًا.)
* التمدّد (Expansion): ``expansion_ratio = range_t / range_{t-1}`` حيث
  ``range = high - low`` للنافذة؛ علم ``is_expansion`` عند تجاوز العتبة.
* دفاع الارتداد (Pullback Defense): حين تصنع النافذة نهايةً جديدة (قمة/قاع) ثم
  يعود الإغلاق داخل منطقة القيمة — أي أن الامتداد لم يُدافَع عنه وارتد إلى القيمة.

كل الحالات سببية: كل صف يعتمد على نافذته والنوافذ السابقة فقط، ومتاح عند
``bucket_end``.
"""

from __future__ import annotations

import polars as pl

from nq.core.time import sort_causal
from nq.simulation.common import BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.volume_profile import developing_value_area

_DEFAULT_BALANCE_THRESHOLD = 0.6
_DEFAULT_EXPANSION_THRESHOLD = 1.5


def auction_states(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    fraction: float = 0.7,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
    expansion_threshold: float = _DEFAULT_EXPANSION_THRESHOLD,
) -> pl.DataFrame:
    """يصنّف حالة المزاد لكل نافذة زمنية (متاح عند ``bucket_end``).

    الأعمدة تشمل: ``poc``, ``vah``, ``val``, ``high``, ``low``, ``close``,
    ``range``, ``in_value_fraction``, ``is_balanced``, ``expansion_ratio``,
    ``is_expansion``, ``made_new_high``, ``made_new_low``, ``pullback_defended``.
    """
    dva = developing_value_area(frame, interval_ns=interval_ns, fraction=fraction)
    if dva.height == 0:
        return dva.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("high"),
            pl.lit(None, dtype=pl.Int64).alias("low"),
            pl.lit(None, dtype=pl.Int64).alias("close"),
        )

    trades = extract_trades(add_time_bucket(sort_causal(frame), interval_ns=interval_ns))

    stats = trades.group_by(BUCKET_START).agg(
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("size").cast(pl.Int64).sum().alias("bucket_volume"),
    )

    # حجم الصفقات داخل منطقة القيمة [val, vah] لكل نافذة.
    va_bounds = dva.select(BUCKET_START, "vah", "val")
    in_value = (
        trades.join(va_bounds, on=BUCKET_START, how="left")
        .filter((pl.col("price") >= pl.col("val")) & (pl.col("price") <= pl.col("vah")))
        .group_by(BUCKET_START)
        .agg(pl.col("size").cast(pl.Int64).sum().alias("in_value_volume"))
    )

    merged = (
        dva.join(stats, on=BUCKET_START, how="left")
        .join(in_value, on=BUCKET_START, how="left")
        .with_columns(pl.col("in_value_volume").fill_null(0))
        .sort(BUCKET_START)
    )

    price_range = pl.col("high") - pl.col("low")
    prev_range = price_range.shift(1)
    prev_high = pl.col("high").shift(1)
    prev_low = pl.col("low").shift(1)
    in_value_fraction = (
        pl.when(pl.col("bucket_volume") > 0)
        .then(pl.col("in_value_volume") / pl.col("bucket_volume"))
        .otherwise(0.0)
    )
    expansion_ratio = (
        pl.when((prev_range.is_not_null()) & (prev_range > 0))
        .then(price_range / prev_range)
        .otherwise(None)
    )
    made_new_high = (prev_high.is_not_null()) & (pl.col("high") > prev_high)
    made_new_low = (prev_low.is_not_null()) & (pl.col("low") < prev_low)
    closed_in_value = (pl.col("close") >= pl.col("val")) & (pl.col("close") <= pl.col("vah"))
    is_expansion = expansion_ratio.is_not_null() & (expansion_ratio >= expansion_threshold)

    return merged.with_columns(
        price_range.alias("range"),
        in_value_fraction.alias("in_value_fraction"),
        expansion_ratio.alias("expansion_ratio"),
        made_new_high.alias("made_new_high"),
        made_new_low.alias("made_new_low"),
        closed_in_value.alias("close_in_value"),
        is_expansion.alias("is_expansion"),
    ).with_columns(
        # التوازن (rotational): أُغلق داخل منطقة القيمة (قبول للقيمة) دون تمدّد مدى،
        # مع بقاء حصّة الحجم داخل القيمة فوق العتبة. غير ذلك = اختلال (اتجاه/رفض القيمة).
        (
            pl.col("close_in_value")
            & ~pl.col("is_expansion")
            & (pl.col("in_value_fraction") >= balance_threshold)
        ).alias("is_balanced"),
        ((pl.col("made_new_high") | pl.col("made_new_low")) & pl.col("close_in_value")).alias(
            "pullback_defended"
        ),
    )


__all__ = ["auction_states"]
