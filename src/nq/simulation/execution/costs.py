"""انزلاق وتكاليف تنفيذ intraday مبسّطة (بدون محاكاة طابور)."""

from __future__ import annotations

from typing import Final

from nq.contracts.instruments import NQ_METADATA

_DEFAULT_SLIPPAGE_TICKS: Final = 0.5
_DEFAULT_TICK_SIZE: Final = NQ_METADATA.tick_size
_DEFAULT_COMMISSION_BPS: Final = 0.0


def slippage_amount(
    *,
    slippage_ticks: float = _DEFAULT_SLIPPAGE_TICKS,
    tick_size: float = _DEFAULT_TICK_SIZE,
) -> float:
    """قيمة الانزلاق المطلقة (نفس وحدات السعر)."""
    if slippage_ticks < 0:
        raise ValueError(f"slippage_ticks must be non-negative, got {slippage_ticks}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")
    return slippage_ticks * tick_size


def commission_rate(*, commission_bps: float = _DEFAULT_COMMISSION_BPS) -> float:
    """عمولة كنسبة مناسبة للضرب في العائد (bps / 10_000)."""
    if commission_bps < 0:
        raise ValueError(f"commission_bps must be non-negative, got {commission_bps}")
    return commission_bps / 10_000.0


__all__ = ["commission_rate", "slippage_amount"]
