"""مُحاكي ملف الحجم (Volume Profile Simulator).

يوزّع الحجم المُنفَّذ على مستويات الأسعار، ويشتق من التوزيع:

* ``POC`` (Point of Control) — السعر ذو أعلى حجم.
* منطقة القيمة ``Value Area`` — أصغر مدى أسعار متّصل حول POC يحوي نسبة
  ``fraction`` من إجمالي الحجم (افتراضيًا 70%). حدّاها ``VAH`` (أعلى) و ``VAL`` (أدنى).
* ``HVN`` / ``LVN`` — عُقد الحجم المرتفع/المنخفض (قمم/قيعان محلية في التوزيع).
* هجرة القيمة ``Value Migration`` — إزاحة POC/VA عبر النوافذ المتتالية (سببي).

خوارزمية منطقة القيمة (Market-Profile): تبدأ من POC وتتوسّع في كل خطوة نحو
الجار المجاور الأعلى حجمًا حتى تبلغ الحصّة المطلوبة.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades

_DEFAULT_VALUE_AREA_FRACTION = 0.7


def build_volume_profile(frame: pl.DataFrame) -> pl.DataFrame:
    """يبني ملف الحجم: إجمالي الحجم المُنفَّذ لكل سعر (مرتّبًا تصاعديًا بالسعر)."""
    trades = extract_trades(frame)
    return (
        trades.group_by("price")
        .agg(pl.col("size").cast(pl.Int64).sum().alias("volume"))
        .sort("price")
    )


@dataclass(frozen=True, slots=True)
class ValueArea:
    """منطقة القيمة الناتجة عن ملف الحجم."""

    poc: int
    vah: int
    val: int
    poc_volume: int
    value_volume: int
    total_volume: int
    fraction: float


def value_area(
    profile: pl.DataFrame,
    *,
    fraction: float = _DEFAULT_VALUE_AREA_FRACTION,
) -> ValueArea | None:
    """يحسب POC و VAH/VAL من ملف حجم (يُفترض ترتيبه تصاعديًا بالسعر).

    يُعيد ``None`` لملف فارغ. يتوسّع من POC نحو الجار الأعلى حجمًا حتى بلوغ
    ``fraction`` من الإجمالي.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if profile.height == 0:
        return None

    prices: list[int] = profile["price"].to_list()
    volumes: list[int] = profile["volume"].to_list()
    n = len(prices)

    poc_idx = max(range(n), key=lambda i: volumes[i])
    total = sum(volumes)
    target = fraction * total
    acc = volumes[poc_idx]
    lo = hi = poc_idx
    while acc < target and (lo > 0 or hi < n - 1):
        up = volumes[hi + 1] if hi < n - 1 else -1
        down = volumes[lo - 1] if lo > 0 else -1
        if up >= down:
            hi += 1
            acc += volumes[hi]
        else:
            lo -= 1
            acc += volumes[lo]

    return ValueArea(
        poc=prices[poc_idx],
        vah=prices[hi],
        val=prices[lo],
        poc_volume=volumes[poc_idx],
        value_volume=acc,
        total_volume=total,
        fraction=fraction,
    )


def classify_nodes(profile: pl.DataFrame) -> pl.DataFrame:
    """يضيف علمَي ``is_hvn`` و ``is_lvn`` (قمم/قيعان محلية في التوزيع)."""
    vol = pl.col("volume")
    prev_vol = vol.shift(1)
    next_vol = vol.shift(-1)
    is_hvn = (vol > prev_vol) & (vol > next_vol)
    is_lvn = (vol < prev_vol) & (vol < next_vol)
    return profile.with_columns(
        is_hvn.fill_null(value=False).alias("is_hvn"),
        is_lvn.fill_null(value=False).alias("is_lvn"),
    )


def developing_value_area(
    frame: pl.DataFrame,
    *,
    interval_ns: int,
    fraction: float = _DEFAULT_VALUE_AREA_FRACTION,
) -> pl.DataFrame:
    """يحسب منطقة القيمة لكل نافذة زمنية ويقيس هجرة القيمة عبرها (سببي).

    الأعمدة: ``bucket_start``, ``poc``, ``vah``, ``val``, ``total_volume``,
    ``poc_migration`` (إزاحة POC عن النافذة السابقة), ``bucket_end``,
    ``availability_ts``. كل صف متاح فقط عند ``bucket_end``.
    """
    trades = extract_trades(add_time_bucket(frame, interval_ns=interval_ns))
    if trades.height == 0:
        return pl.DataFrame(
            schema={
                BUCKET_START: pl.Int64(),
                "poc": pl.Int64(),
                "vah": pl.Int64(),
                "val": pl.Int64(),
                "total_volume": pl.Int64(),
                "poc_migration": pl.Int64(),
                BUCKET_END: pl.Int64(),
                AVAILABILITY_TS: pl.Int64(),
            }
        )

    per_price = trades.group_by([BUCKET_START, "price"]).agg(
        pl.col("size").cast(pl.Int64).sum().alias("volume"),
        pl.col(BUCKET_END).first(),
    )

    rows: list[dict[str, int]] = []
    for (bucket_start,), group in per_price.group_by([BUCKET_START], maintain_order=True):
        va = value_area(group.sort("price"), fraction=fraction)
        if va is None:  # pragma: no cover - group is always non-empty here
            continue
        rows.append(
            {
                BUCKET_START: int(bucket_start),
                "poc": va.poc,
                "vah": va.vah,
                "val": va.val,
                "total_volume": va.total_volume,
                BUCKET_END: int(group[BUCKET_END][0]),
            }
        )

    result = pl.DataFrame(rows).sort(BUCKET_START)
    return result.with_columns(
        pl.col("poc").diff().fill_null(0).alias("poc_migration"),
        pl.col(BUCKET_END).alias(AVAILABILITY_TS),
    ).select(
        BUCKET_START,
        "poc",
        "vah",
        "val",
        "total_volume",
        "poc_migration",
        BUCKET_END,
        AVAILABILITY_TS,
    )
