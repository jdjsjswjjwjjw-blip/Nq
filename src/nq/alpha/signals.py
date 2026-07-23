"""إشارات الألفا وتقييمها (Alpha Signals & Evaluation).

الإشارة سببية (تُحسب من الماضي والحاضر فقط)، وتُقيَّم مقابل **العوائد الأمامية**
(labels مستقبلية تُستخدم للتقييم فقط لا كميزات). التقييم يشمل معامل المعلومات
(IC) مع دلالة بالتبديل، ونسبة شارب لعوائد استراتيجية بسيطة مبنية على الإشارة،
ثم فرزًا جماعيًا مع تصحيح التعدّد لعزل الألفا الحقيقي عن الصدفة.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.instruments import NQ_METADATA
from nq.simulation.execution.intraday import (
    directional_execution_returns,
    realistic_execution_forward_returns,
)
from nq.statistics.metrics import information_coefficient, sharpe_ratio
from nq.statistics.multiple_testing import benjamini_hochberg
from nq.statistics.resampling import block_permutation

FloatArray = npt.NDArray[np.float64]
ExecutionMode = Literal["mid", "intraday"]

_MIN_EVAL_SAMPLES = 8


def _default_permutation_block_size(n: int) -> int:
    return max(1, min(n, 3))


@dataclass(frozen=True, slots=True)
class AlphaSignal:
    """إشارة ألفا سببية مع طوابعها الزمنية ومصدرها (provenance)."""

    name: str
    times: npt.NDArray[np.int64]
    values: FloatArray
    version: str = "v1"
    provenance: str = ""


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    """نتيجة تقييم إشارة مقابل العوائد الأمامية."""

    name: str
    n: int
    ic: float
    ic_pvalue: float
    sharpe: float
    mean_strategy_return: float


def align_forward_returns(
    prices: npt.NDArray[np.floating] | Sequence[float], *, horizon: int = 1
) -> FloatArray:
    """يحسب العوائد الأمامية بأفق ``horizon`` (المواضع الأخيرة ``NaN``: لا مستقبل).

    ``r[t] = (price[t+horizon] - price[t]) / price[t]``. هذه أهداف مستقبلية
    للتقييم فقط، والإشارة نفسها تبقى سببية.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    p = np.asarray(prices, dtype=np.float64)
    n = p.shape[0]
    fwd = np.full(n, np.nan, dtype=np.float64)
    if n > horizon:
        base = p[: n - horizon]
        future = p[horizon:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(base != 0, (future - base) / base, np.nan)
        fwd[: n - horizon] = ret
    return fwd


def evaluate_signal(
    name: str,
    values: npt.NDArray[np.floating] | Sequence[float],
    forward_returns: npt.NDArray[np.floating] | Sequence[float],
    *,
    n_permutations: int = 2000,
    permutation_block_size: int | None = None,
    rng: np.random.Generator | None = None,
    min_samples: int = _MIN_EVAL_SAMPLES,
) -> SignalEvaluation:
    """يقيّم إشارة: IC (Spearman) مع دلالة بالتبديل، ونسبة شارب للاستراتيجية."""
    generator = rng if rng is not None else np.random.default_rng(0)
    v = np.asarray(values, dtype=np.float64)
    f = np.asarray(forward_returns, dtype=np.float64)
    if v.shape != f.shape:
        raise ValueError(f"values and forward_returns must align, got {v.shape} vs {f.shape}")

    mask = np.isfinite(v) & np.isfinite(f)
    v, f = v[mask], f[mask]
    n = int(v.shape[0])
    if n < min_samples or float(np.std(v)) == 0:
        return SignalEvaluation(
            name=name, n=n, ic=0.0, ic_pvalue=1.0, sharpe=0.0, mean_strategy_return=0.0
        )

    observed_ic = information_coefficient(v, f, method="spearman")
    block_size = (
        _default_permutation_block_size(n)
        if permutation_block_size is None
        else permutation_block_size
    )
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        null[i] = information_coefficient(
            v,
            block_permutation(f, block_size=block_size, rng=generator),
            method="spearman",
        )
    ic_pvalue = (int(np.sum(np.abs(null) >= abs(observed_ic))) + 1) / (n_permutations + 1)

    strategy = np.sign(v) * f
    return SignalEvaluation(
        name=name,
        n=n,
        ic=observed_ic,
        ic_pvalue=ic_pvalue,
        sharpe=sharpe_ratio(strategy),
        mean_strategy_return=float(np.mean(strategy)),
    )


def screen_signals(evaluations: Sequence[SignalEvaluation], *, alpha: float = 0.05) -> pl.DataFrame:
    """يفرز إشارات مرشّحة مع تصحيح التعدّد (BH) لعزل الألفا الحقيقي.

    يُعيد إطارًا: ``name``, ``n``, ``ic``, ``ic_pvalue``, ``adjusted_pvalue``,
    ``sharpe``, ``selected`` (هل تنجو الإشارة بعد التصحيح؟)، مرتّبًا بالقيمة المُعدّلة.
    """
    if not evaluations:
        return pl.DataFrame(
            schema={
                "name": pl.Utf8(),
                "n": pl.Int64(),
                "ic": pl.Float64(),
                "ic_pvalue": pl.Float64(),
                "adjusted_pvalue": pl.Float64(),
                "sharpe": pl.Float64(),
                "selected": pl.Boolean(),
            }
        )
    pvals = np.asarray([e.ic_pvalue for e in evaluations], dtype=np.float64)
    correction = benjamini_hochberg(pvals, alpha=alpha)
    return pl.DataFrame(
        {
            "name": [e.name for e in evaluations],
            "n": [e.n for e in evaluations],
            "ic": [e.ic for e in evaluations],
            "ic_pvalue": [e.ic_pvalue for e in evaluations],
            "adjusted_pvalue": correction.adjusted,
            "sharpe": [e.sharpe for e in evaluations],
            "selected": correction.reject,
        }
    ).sort("adjusted_pvalue")


def evaluate_signal_intraday(
    name: str,
    values: npt.NDArray[np.floating] | Sequence[float],
    bid: npt.NDArray[np.floating] | Sequence[float],
    ask: npt.NDArray[np.floating] | Sequence[float],
    *,
    horizon: int = 1,
    slippage_ticks: float = 0.5,
    tick_size: float = NQ_METADATA.tick_size,
    commission_bps: float = 0.0,
    n_permutations: int = 2000,
    permutation_block_size: int | None = None,
    latency_steps: int = 1,
    rng: np.random.Generator | None = None,
) -> SignalEvaluation:
    """يقيّم إشارة بعوائد أمامية بعد عبور spread وانزلاق intraday."""
    long_fwd, short_fwd = realistic_execution_forward_returns(
        bid,
        ask,
        horizon=horizon,
        latency_steps=latency_steps,
        slippage_ticks=slippage_ticks,
        tick_size=tick_size,
        commission_bps=commission_bps,
    )
    directional = directional_execution_returns(values, long_fwd, short_fwd)
    return evaluate_signal(
        name,
        values,
        directional,
        n_permutations=n_permutations,
        permutation_block_size=permutation_block_size,
        rng=rng,
    )
