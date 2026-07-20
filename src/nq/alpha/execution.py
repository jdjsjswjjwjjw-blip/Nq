"""تقييم الألفا بعد تنفيذ intraday (bid/ask + slippage)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from nq.alpha.signals import SignalEvaluation, evaluate_signal
from nq.simulation.execution.intraday import (
    directional_execution_returns,
    execution_forward_returns,
)

ExecutionMode = Literal["mid", "intraday"]


def evaluate_signal_intraday(
    name: str,
    values: npt.NDArray[np.floating] | Sequence[float],
    bid: npt.NDArray[np.floating] | Sequence[float],
    ask: npt.NDArray[np.floating] | Sequence[float],
    *,
    horizon: int = 1,
    slippage_ticks: float = 0.5,
    tick_size: float = 0.25,
    commission_bps: float = 0.0,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
) -> SignalEvaluation:
    """يقيّم إشارة بعوائد أمامية بعد عبور spread وانزلاق intraday."""
    long_fwd, short_fwd = execution_forward_returns(
        bid,
        ask,
        horizon=horizon,
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
        rng=rng,
    )


__all__ = ["ExecutionMode", "evaluate_signal_intraday"]
