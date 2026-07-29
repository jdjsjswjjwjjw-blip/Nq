"""فلتر دخول من مسار أحداث العمق — طبقة مرشّحين فوق إشارات FB/FVG.

لا يعيد كتابة قاعدة الاستراتيجية. يبني مقاييس مسار العمق داخل شمعة القرار
(``depth_event_path_at_bar_close``)، يلحقها بـ asof خلفي، ثم يولّد مرشّحين
``{signal}__depth__*`` بعتبات ماضية فقط — يُختارون لاحقًا بـ walk-forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.progress import ProgressLike
from nq.simulation.depth_lifecycle import (
    DEPTH_PATH_COLUMNS,
    attach_depth_asof,
    depth_event_path_at_bar_close,
)

_GATE_WINDOW = 50
_GATE_MIN_SAMPLES = 10
_DEFAULT_QUANTILES = (0.6, 0.7, 0.8)
_PRESSURE_COL = "depth_path_pressure"
_IMB_DELTA_COL = "depth_path_imbalance_delta"


@dataclass(frozen=True, slots=True)
class DepthEntrySpec:
    """وصف مرشّح فلتر عمق فوق عمود أساس."""

    base_column: str
    name: str
    kind: str  # pressure_abs_q | pressure_sign | imb_delta_abs_q

    def column(self) -> str:
        return f"{self.base_column}__depth__{self.name}"


def _signals_all_zero(features: pl.DataFrame, signal_columns: Sequence[str]) -> bool:
    """True إذا لا توجد إشارة غير صفرية في الأعمدة المعطاة."""
    return count_signal_hits(features, signal_columns) == 0


def count_signal_hits(
    features: pl.DataFrame, signal_columns: Sequence[str]
) -> int:
    """عدد الصفوف التي فيها إشارة غير صفرية في أي عمود من ``signal_columns``."""
    cols = [c for c in signal_columns if c in features.columns]
    if not cols or features.height == 0:
        return 0
    hit = pl.any_horizontal(
        [(pl.col(c).fill_null(0.0).abs() > 0.0) for c in cols]
    )
    return int(features.select(hit.sum().alias("_n"))["_n"][0])


def attach_depth_path_to_features(
    features: pl.DataFrame,
    mbo: pl.DataFrame,
    *,
    interval_ns: int,
    progress: ProgressLike | None = None,
    signal_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """يحسب مسار أحداث العمق لكل شمعة ويلحقه بالإطار بـ asof خلفي.

    إذا مُرِّرت ``signal_columns`` وكانت كلها صفرًا، يُتخطّى حساب المسار
    (لا مرشّحي عمق بلا إشارة أساس) — تسريع آمن بلا تغيير قواعد التسريب.
    """
    if features.height == 0 or mbo.height == 0:
        return features
    if signal_columns is not None and _signals_all_zero(features, signal_columns):
        if progress is not None:
            progress.op("depth_event_path: تخطّي — إشارات الأساس كلها صفر")
        return features
    path = depth_event_path_at_bar_close(mbo, interval_ns=interval_ns, progress=progress)
    return attach_depth_asof(features, path, columns=DEPTH_PATH_COLUMNS)


def generate_depth_entry_candidates(
    features: pl.DataFrame,
    base_columns: Sequence[str],
    *,
    quantiles: Sequence[float] = _DEFAULT_QUANTILES,
    include_sign_agree: bool = True,
    include_imbalance_delta: bool = True,
) -> tuple[pl.DataFrame, tuple[str, ...], tuple[DepthEntrySpec, ...]]:
    """يولّد مرشّحي دخول عمق فوق ``base_columns`` (إشارة × بوابة ماضية).

    * ``pressure_q{q}``: |pressure| ≥ كمّية ماضية لـ |pressure|
    * ``pressure_sign``: الإشارة فقط عند اتفاق الإشارة مع sign(pressure)
    * ``imb_delta_q{q}``: |imbalance_delta| ≥ كمّية ماضية
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
