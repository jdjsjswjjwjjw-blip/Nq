"""فلتر دخول من مسار أحداث العمق — طبقة مرشّحين فوق إشارات FB/FVG.

لا يعيد كتابة قاعدة الاستراتيجية. يبني مقاييس مسار العمق داخل شمعة القرار
(``depth_event_path_at_bar_close``) + حافة أسفل الدفتر (L2–L5 / آيسبرغ)،
يلحقها بـ asof خلفي، ثم يولّد مرشّحين ``{signal}__depth__*`` بعتبات ماضية فقط
— يُختارون لاحقًا بـ walk-forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.simulation.bottom_book import (
    BOTTOM_BOOK_COLUMNS,
    attach_bottom_book_asof,
    bottom_book_features_at_bar_close,
)
from nq.simulation.depth_lifecycle import (
    DEPTH_PATH_COLUMNS,
    attach_depth_asof,
    depth_event_path_at_bar_close,
)
from nq.simulation.depth_noise import DepthNoiseConfig, filter_depth_noise

_GATE_WINDOW = 50
_GATE_MIN_SAMPLES = 10
_DEFAULT_QUANTILES = (0.6, 0.7, 0.8)
_PRESSURE_COL = "depth_path_pressure"
_IMB_DELTA_COL = "depth_path_imbalance_delta"
_QUEUE_COL = "bb_queue_pressure"
_ICEBERG_COL = "bb_iceberg_hit"
_L25_BID_DRAIN = "depth_path_l2_l5_bid_drain"
_L25_ASK_DRAIN = "depth_path_l2_l5_ask_drain"


@dataclass(frozen=True, slots=True)
class DepthEntrySpec:
    """وصف مرشّح فلتر عمق فوق عمود أساس."""

    base_column: str
    name: str
    kind: str  # pressure_abs_q | pressure_sign | imb_delta_abs_q | queue | iceberg | l25

    def column(self) -> str:
        return f"{self.base_column}__depth__{self.name}"


def _signals_all_zero(features: pl.DataFrame, signal_columns: Sequence[str]) -> bool:
    """True إذا لا توجد إشارة غير صفرية في الأعمدة المعطاة."""
    return count_signal_hits(features, signal_columns) == 0


def count_signal_hits(features: pl.DataFrame, signal_columns: Sequence[str]) -> int:
    """عدد الصفوف التي فيها إشارة غير صفرية في أي عمود من ``signal_columns``."""
    cols = [c for c in signal_columns if c in features.columns]
    if not cols or features.height == 0:
        return 0
    hit = pl.any_horizontal([(pl.col(c).fill_null(0.0).abs() > 0.0) for c in cols])
    return int(features.select(hit.sum().alias("_n"))["_n"][0])


def attach_depth_path_to_features(
    features: pl.DataFrame,
    mbo: pl.DataFrame,
    *,
    interval_ns: int,
    progress: ProgressLike | None = None,
    signal_columns: Sequence[str] | None = None,
    filter_noise: bool = True,
    include_bottom_book: bool = True,
) -> pl.DataFrame:
    """يحسب مسار أحداث العمق (+ أسفل الدفتر) لكل شمعة ويلحقه asof خلفي.

    إذا مُرِّرت ``signal_columns`` وكانت كلها صفرًا، يُتخطّى الحساب.
    ``filter_noise`` يصفّي عواصف الإلغاء/الوميض/السبوف سببيًا قبل المسار.
    """
    if features.height == 0 or mbo.height == 0:
        return features
    if signal_columns is not None and _signals_all_zero(features, signal_columns):
        if progress is not None:
            progress.op("depth_event_path: تخطّي — إشارات الأساس كلها صفر")
        return features

    cleaned = filter_depth_noise(mbo, config=DepthNoiseConfig()) if filter_noise else mbo
    if progress is not None and filter_noise:
        progress.op(
            f"depth_noise_filter: {mbo.height:,} → {cleaned.height:,} حدث "
            f"(أُسقط {mbo.height - cleaned.height:,})"
        )

    path = depth_event_path_at_bar_close(
        cleaned, interval_ns=interval_ns, progress=progress
    )
    out = attach_depth_asof(features, path, columns=DEPTH_PATH_COLUMNS, fill_missing=False)

    if include_bottom_book:
        bottom = bottom_book_features_at_bar_close(
            cleaned,
            interval_ns=interval_ns,
            filter_noise=False,  # سبق التنظيف
            progress=progress,
        )
        out = attach_bottom_book_asof(out, bottom)
        if progress is not None:
            progress.op(f"bottom_book مُلحق: {len(BOTTOM_BOOK_COLUMNS)} عمود")
    return out


def generate_depth_entry_candidates(  # noqa: PLR0912, PLR0915
    features: pl.DataFrame,
    base_columns: Sequence[str],
    *,
    quantiles: Sequence[float] = _DEFAULT_QUANTILES,
    include_sign_agree: bool = True,
    include_imbalance_delta: bool = True,
    include_bottom_edge: bool = True,
) -> tuple[pl.DataFrame, tuple[str, ...], tuple[DepthEntrySpec, ...]]:
    """يولّد مرشّحي دخول عمق فوق ``base_columns`` (إشارة × بوابة ماضية).

    * ``pressure_q{q}``: |pressure| ≥ كمّية ماضية لـ |pressure|
    * ``pressure_sign``: الإشارة فقط عند اتفاق الإشارة مع sign(pressure)
    * ``imb_delta_q{q}``: |imbalance_delta| ≥ كمّية ماضية
    * ``queue_q{q}`` / ``iceberg`` / ``l25_*``: حافة أسفل الدفتر
    """
    bases = [c for c in base_columns if c in features.columns]
    if not bases or _PRESSURE_COL not in features.columns:
        return features, tuple(), tuple()

    work = features.sort(AVAILABILITY_TS)
    specs: list[DepthEntrySpec] = []
    new_cols: list[str] = []
    exprs: list[pl.Expr] = []

    abs_pressure = pl.col(_PRESSURE_COL).abs().fill_null(0.0)
    for q in quantiles:
        past_q = abs_pressure.shift(1).rolling_quantile(
            float(q), window_size=_GATE_WINDOW, min_samples=_GATE_MIN_SAMPLES
        )
        gate = (abs_pressure >= past_q.fill_null(float("inf"))).cast(pl.Float64)
        q_tag = str(q).replace(".", "p")
        for base in bases:
            spec = DepthEntrySpec(
                base_column=base,
                name=f"pressure_q{q_tag}",
                kind="pressure_abs_q",
            )
            col = spec.column()
            specs.append(spec)
            new_cols.append(col)
            exprs.append((pl.col(base).fill_null(0.0) * gate).alias(col))

    if include_sign_agree:
        sign_p = pl.col(_PRESSURE_COL).fill_null(0.0).sign()
        for base in bases:
            spec = DepthEntrySpec(
                base_column=base,
                name="pressure_sign",
                kind="pressure_sign",
            )
            col = spec.column()
            specs.append(spec)
            new_cols.append(col)
            agree = ((pl.col(base).fill_null(0.0) * sign_p) > 0.0).cast(pl.Float64)
            exprs.append((pl.col(base).fill_null(0.0) * agree).alias(col))

    if include_imbalance_delta and _IMB_DELTA_COL in work.columns:
        abs_d = pl.col(_IMB_DELTA_COL).abs().fill_null(0.0)
        for q in quantiles:
            past_q = abs_d.shift(1).rolling_quantile(
                float(q), window_size=_GATE_WINDOW, min_samples=_GATE_MIN_SAMPLES
            )
            gate = (abs_d >= past_q.fill_null(float("inf"))).cast(pl.Float64)
            q_tag = str(q).replace(".", "p")
            for base in bases:
                spec = DepthEntrySpec(
                    base_column=base,
                    name=f"imb_delta_q{q_tag}",
                    kind="imb_delta_abs_q",
                )
                col = spec.column()
                specs.append(spec)
                new_cols.append(col)
                exprs.append((pl.col(base).fill_null(0.0) * gate).alias(col))

    if include_bottom_edge and _QUEUE_COL in work.columns:
        abs_q = pl.col(_QUEUE_COL).abs().fill_null(0.0)
        for q in quantiles:
            past_q = abs_q.shift(1).rolling_quantile(
                float(q), window_size=_GATE_WINDOW, min_samples=_GATE_MIN_SAMPLES
            )
            gate = (abs_q >= past_q.fill_null(float("inf"))).cast(pl.Float64)
            q_tag = str(q).replace(".", "p")
            for base in bases:
                spec = DepthEntrySpec(
                    base_column=base,
                    name=f"queue_q{q_tag}",
                    kind="queue",
                )
                col = spec.column()
                specs.append(spec)
                new_cols.append(col)
                exprs.append((pl.col(base).fill_null(0.0) * gate).alias(col))

    if include_bottom_edge and _ICEBERG_COL in work.columns:
        ice = (pl.col(_ICEBERG_COL).fill_null(0.0) > 0.0).cast(pl.Float64)
        for base in bases:
            spec = DepthEntrySpec(base_column=base, name="iceberg", kind="iceberg")
            col = spec.column()
            specs.append(spec)
            new_cols.append(col)
            exprs.append((pl.col(base).fill_null(0.0) * ice).alias(col))

    if include_bottom_edge and _L25_BID_DRAIN in work.columns and _L25_ASK_DRAIN in work.columns:
        l25_pressure = pl.col(_L25_ASK_DRAIN).fill_null(0.0) - pl.col(_L25_BID_DRAIN).fill_null(
            0.0
        )
        abs_l25 = l25_pressure.abs()
        past_q = abs_l25.shift(1).rolling_quantile(
            0.7, window_size=_GATE_WINDOW, min_samples=_GATE_MIN_SAMPLES
        )
        gate = (abs_l25 >= past_q.fill_null(float("inf"))).cast(pl.Float64)
        for base in bases:
            spec = DepthEntrySpec(base_column=base, name="l25_pressure_q0p7", kind="l25")
            col = spec.column()
            specs.append(spec)
            new_cols.append(col)
            exprs.append((pl.col(base).fill_null(0.0) * gate).alias(col))

    if not exprs:
        return work, tuple(), tuple()
    out = work.with_columns(exprs)
    return out, tuple(new_cols), tuple(specs)


__all__ = [
    "DepthEntrySpec",
    "attach_depth_path_to_features",
    "count_signal_hits",
    "generate_depth_entry_candidates",
]
