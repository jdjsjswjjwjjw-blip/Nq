"""مُحاكي ملف الحجم (Volume Profile Simulator).

يوزّع الحجم المُنفَّذ على مستويات الأسعار، ويشتق من التوزيع **ثلاث حدود**:

* ``POC`` / mid (Point of Control) — السعر ذو أعلى حجم (الحد المتوسط / المركز).
* ``VAH`` / upper — أعلى منطقة القيمة.
* ``VAL`` / lower — أدنى منطقة القيمة.
* منطقة القيمة ``Value Area`` — أصغر مدى أسعار متّصل حول POC يحوي نسبة
  ``fraction`` من إجمالي الحجم (افتراضيًا 70%). حدّاها ``VAH`` و ``VAL``.
* ``HVN`` / ``LVN`` — عُقد الحجم المرتفع/المنخفض (قمم/قيعان محلية في التوزيع).
* هجرة القيمة ``Value Migration`` — إزاحة POC/VA عبر النوافذ المتتالية (سببي).

خوارزمية منطقة القيمة (Market-Profile): تبدأ من POC وتتوسّع في كل خطوة نحو
الجار المجاور الأعلى حجمًا حتى تبلغ الحصّة المطلوبة.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades

_DEFAULT_VALUE_AREA_FRACTION = 0.7
#: حجم تيك NQ بالوحدات الثابتة (0.25$ → fixed-point). كان 1e6 خطأً مقياسًا.
_NQ_TICK_SIZE: float = 0.25
_TICK_FIXED: float = float(round(_NQ_TICK_SIZE / PRICE_SCALE))


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


def value_area_from_levels(
    prices: list[int],
    volumes: list[int],
    *,
    fraction: float = _DEFAULT_VALUE_AREA_FRACTION,
) -> ValueArea | None:
    """نفس خوارزمية منطقة القيمة على قوائم مرتّبة تصاعديًا بالسعر (بدون Polars)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = len(prices)
    if n == 0:
        return None
    if n != len(volumes):
        raise ValueError("prices and volumes must have the same length")

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


def value_area(
    profile: pl.DataFrame,
    *,
    fraction: float = _DEFAULT_VALUE_AREA_FRACTION,
) -> ValueArea | None:
    """يحسب POC و VAH/VAL من ملف حجم (يُفترض ترتيبه تصاعديًا بالسعر).

    يُعيد ``None`` لملف فارغ. يتوسّع من POC نحو الجار الأعلى حجمًا حتى بلوغ
    ``fraction`` من الإجمالي.
    """
    if profile.height == 0:
        return None
    return value_area_from_levels(
        profile["price"].to_list(),
        profile["volume"].to_list(),
        fraction=fraction,
    )


@dataclass
class DevelopingVolumeProfile:
    """ملف حجم متطوّر event-by-event — يُحدَّث سببيًا مع كل صفقة.

    يُكمّل ``developing_value_area`` (نافذة زمنية) بتحديث فوري لكل tick.
    الكاش: يُعاد حساب VA فقط بعد ``add_trade`` (نفس الأرقام، أقل عمل).
    """

    fraction: float = _DEFAULT_VALUE_AREA_FRACTION
    _levels: dict[int, int] = field(default_factory=dict, repr=False)
    _cached_va: ValueArea | None = field(default=None, repr=False)
    _va_dirty: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

    def add_trade(self, price: int, size: int) -> None:
        """يُضيف حجم صفقة إلى المستوى السعري."""
        self._levels[price] = self._levels.get(price, 0) + size
        self._va_dirty = True

    def to_frame(self) -> pl.DataFrame:
        """يُعيد ملف الحجم الحالي كإطار polars مرتّب بالسعر."""
        if not self._levels:
            return pl.DataFrame(schema={"price": pl.Int64(), "volume": pl.Int64()})
        prices = sorted(self._levels)
        return pl.DataFrame({"price": prices, "volume": [self._levels[p] for p in prices]})

    def value_area(self) -> ValueArea | None:
        """POC/VAH/VAL من الحالة الحالية (مع كاش صالح حتى الصفقة التالية)."""
        if not self._va_dirty:
            return self._cached_va
        if not self._levels:
            self._cached_va = None
            self._va_dirty = False
            return None
        prices = sorted(self._levels)
        volumes = [self._levels[p] for p in prices]
        self._cached_va = value_area_from_levels(prices, volumes, fraction=self.fraction)
        self._va_dirty = False
        return self._cached_va

    def features_at_mid(
        self,
        mid: float,
        *,
        ref_price: float,
        near_ticks: int,
        va: ValueArea | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """ميزات VP سببية: مسافات POC/VAH/VAL + أعلام القرب/داخل المنطقة."""
        if not self._levels or mid <= 0 or ref_price <= 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        area = va if va is not None else self.value_area()
        if area is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # near_ticks = عدد تيكات NQ (0.25$) بوحدة السعر الثابتة — لا مقياس وهمي
        scale = float(near_ticks) * _TICK_FIXED
        poc_d = (mid - area.poc) / ref_price
        vah_d = (mid - area.vah) / ref_price
        val_d = (mid - area.val) / ref_price
        near_vah = 1.0 if abs(mid - area.vah) <= scale else 0.0
        near_val = 1.0 if abs(mid - area.val) <= scale else 0.0
        in_va = 1.0 if area.val <= mid <= area.vah else 0.0
        return (poc_d, vah_d, val_d, near_vah, near_val, in_va)


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
    cumulative: bool = False,
    progress: ProgressLike | None = None,
) -> pl.DataFrame:
    """يحسب منطقة القيمة لكل نافذة زمنية ويقيس هجرة القيمة عبرها (سببي).

    * ``cumulative=False``: ملف مستقل لكل نافذة (micro-profile).
    * ``cumulative=True``: ملف متطوّر تراكمي عبر النوافذ (قبول/رفض القيمة الجلسي)
      — الوضع الصحيح لمحاكي المزاد.

    الأعمدة: ``bucket_start``, ``poc``, ``vah``, ``val``, ``total_volume``,
    ``poc_migration``, ``bucket_end``, ``availability_ts``.
    """
    trades = extract_trades(add_time_bucket(frame, interval_ns=interval_ns))
    empty = pl.DataFrame(
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
    if trades.height == 0:
        return empty

    # ترتيب حتمي قبل التجميع — وإلا ترتيب group_by غير مستقر يكسر التراكم
    trades = trades.sort([BUCKET_START, "price"])
    per_price = (
        trades.group_by([BUCKET_START, "price"], maintain_order=True)
        .agg(
            pl.col("size").cast(pl.Int64).sum().alias("volume"),
            pl.col(BUCKET_END).first(),
        )
        .sort([BUCKET_START, "price"])
    )

    rows: list[dict[str, int]] = []
    groups = list(per_price.group_by([BUCKET_START], maintain_order=True))
    n_buckets = len(groups)
    if progress is not None:
        mode = "تراكمي" if cumulative else "لكل-نافذة"
        progress.op(
            f"developing_value_area: {n_buckets:,} نافذة · interval_ns={interval_ns} · {mode}"
        )

    running: DevelopingVolumeProfile | None = (
        DevelopingVolumeProfile(fraction=fraction) if cumulative else None
    )
    for i, ((bucket_start,), group) in enumerate(groups, start=1):
        if progress is not None:
            progress.heartbeat(i, n_buckets, label="value_area")
        sorted_group = group.sort("price")
        prices = sorted_group["price"].to_list()
        volumes = sorted_group["volume"].to_list()
        if cumulative and running is not None:
            for px, vol in zip(prices, volumes, strict=True):
                running.add_trade(int(px), int(vol))
            va = running.value_area()
        else:
            va = value_area_from_levels(prices, volumes, fraction=fraction)
        if va is None:  # pragma: no cover
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

    if not rows:
        return empty
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
