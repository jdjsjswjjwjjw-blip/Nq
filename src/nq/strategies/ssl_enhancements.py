"""مولّد تعزيزات علمية من SSL/السياق فوق إشارة أساس سببية.

المبدأ: التعلم العميق **لا يعيد كتابة** قاعدة الاستراتيجية.
يولّد مرشّحي شروط تعزيز (فلاتر تأكيد) تُختبر لاحقًا بـ walk-forward OOS.

كل بوابة زمنية:
* embeddings بـ ``join_asof(..., backward)``
* عتبات ``|z|`` من كمّية ماضية فقط (``shift(1).rolling_quantile``)
* فلاتر سياق من أعمدة متاحة point-in-time فقط
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS

_SSL_WINDOW = 50
_SSL_MIN_SAMPLES = 10
_DEFAULT_QUANTILES = (0.6, 0.7, 0.8)
_DEFAULT_Z_COLS = ("z0", "z1")
_CONTEXT_FILTERS = (
    ("phase_balance", 0.5),
    ("phase_expansion", 0.5),
    ("trap_setup", 0.0),  # |trap| > 0
    ("in_value_area", 0.5),
    ("near_vah", 0.5),
    ("near_val", 0.5),
    # فوليوم سببي (من Failed Breakout) — عتبات نسبية مقابل 0 أو متوسط ضعيف
    ("fb_effort_volume_ratio", 1.2),
    ("fb_effort_result_ratio", 1.2),
    ("fb_absorption", 0.0),  # |absorption| > 0 عند التفعيل لاحقًا عبر >= بعد abs
    ("fb_vol_imbalance", 0.0),  # |imbalance| > 0
)


@dataclass(frozen=True, slots=True)
class EnhancementSpec:
    """وصف مرشّح تعزيز فوق عمود أساس."""

    base_column: str
    name: str
    kind: str  # ssl_abs_q | ssl_sign_agree | context

    def column(self) -> str:
        return f"{self.base_column}__enh__{self.name}"


def _attach_embeddings(
    features: pl.DataFrame,
    embeddings: pl.DataFrame,
    z_cols: Sequence[str],
) -> pl.DataFrame:
    keep = [c for c in (AVAILABILITY_TS, *z_cols) if c in embeddings.columns]
    if AVAILABILITY_TS not in keep or len(keep) < 2:
        return features
    right = embeddings.select(keep).sort(AVAILABILITY_TS)
    left = features.sort(AVAILABILITY_TS)
    drop = [c for c in keep if c != AVAILABILITY_TS and c in left.columns]
    if drop:
        left = left.drop(drop)
    return left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")


def generate_ssl_enhancement_candidates(
    features: pl.DataFrame,
    embeddings: pl.DataFrame,
    base_columns: Sequence[str],
    *,
    z_cols: Sequence[str] = _DEFAULT_Z_COLS,
    quantiles: Sequence[float] = _DEFAULT_QUANTILES,
    include_context: bool = True,
    include_sign_agree: bool = True,
) -> tuple[pl.DataFrame, tuple[str, ...], tuple[EnhancementSpec, ...]]:
    """يبني أعمدة تعزيز فوق ``base_columns`` ويعيد (frame, columns, specs).

    المرشّحون:
    * ``ssl_abs_q{q}_{z}``: الإشارة × (|z| ≥ كمّية ماضية)
    * ``ssl_sign_{z}``: الإشارة فقط عند اتفاق الإشارة مع sign(z)
    * ``ctx_{name}``: فلاتر بنية السوق المتاحة سببيًا
    """
    bases = [c for c in base_columns if c in features.columns]
    if not bases:
        return features, tuple(), tuple()

    work = _attach_embeddings(features, embeddings, z_cols)
    specs: list[EnhancementSpec] = []
    new_cols: list[str] = []
    exprs: list[pl.Expr] = []

    for z_col in z_cols:
        if z_col not in work.columns:
            continue
        abs_z = pl.col(z_col).abs().fill_null(0.0)
        for q in quantiles:
            past_q = abs_z.shift(1).rolling_quantile(
                float(q), window_size=_SSL_WINDOW, min_samples=_SSL_MIN_SAMPLES
            )
            gate = (abs_z >= past_q.fill_null(float("inf"))).cast(pl.Float64)
            q_tag = str(q).replace(".", "p")
            for base in bases:
                spec = EnhancementSpec(
                    base_column=base,
                    name=f"ssl_abs_q{q_tag}_{z_col}",
                    kind="ssl_abs_q",
                )
                col = spec.column()
                specs.append(spec)
                new_cols.append(col)
                exprs.append((pl.col(base).fill_null(0.0) * gate).alias(col))

        if include_sign_agree:
            # اتفاق الاتجاه: long فقط إذا z>0، short فقط إذا z<0 (بعد معرفة الإشارة)
            sign_z = pl.col(z_col).fill_null(0.0).sign()
            for base in bases:
                spec = EnhancementSpec(
                    base_column=base,
                    name=f"ssl_sign_{z_col}",
                    kind="ssl_sign_agree",
                )
                col = spec.column()
                specs.append(spec)
                new_cols.append(col)
                # الإشارة تبقى كما هي عند الاتفاق، وإلا 0
                agree = (
                    (pl.col(base).fill_null(0.0) * sign_z) > 0.0
                ).cast(pl.Float64)
                exprs.append((pl.col(base).fill_null(0.0) * agree).alias(col))

    if include_context:
        for ctx_col, thresh in _CONTEXT_FILTERS:
            if ctx_col not in work.columns:
                continue
            if ctx_col in ("trap_setup", "fb_vol_imbalance", "fb_absorption"):
                gate = (pl.col(ctx_col).abs() > thresh).cast(pl.Float64)
            else:
                gate = (pl.col(ctx_col).fill_null(0.0) >= thresh).cast(pl.Float64)
            for base in bases:
                spec = EnhancementSpec(
                    base_column=base,
                    name=f"ctx_{ctx_col}",
                    kind="context",
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
    "EnhancementSpec",
    "generate_ssl_enhancement_candidates",
]
