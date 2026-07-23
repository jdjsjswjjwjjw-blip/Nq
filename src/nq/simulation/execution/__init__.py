"""محاكاة تنفيذ intraday مبسّطة (spread crossing + slippage، بدون طابور).

للتداول intraday على NQ/MNQ: التقييم على bid/ask لا mid، مع انزلاق
وتكلفة اختيارية — دون محاكاة أولوية الطابور.
"""

from __future__ import annotations

from nq.simulation.execution.costs import commission_rate, slippage_amount
from nq.simulation.execution.depth_fill import (
    depth_matrices_from_frame,
    execution_forward_returns_depth,
)
from nq.simulation.execution.intraday import (
    directional_execution_returns,
    execution_forward_returns,
)
from nq.simulation.execution.spread import buy_fill_price, sell_fill_price

__all__ = [
    "buy_fill_price",
    "commission_rate",
    "depth_matrices_from_frame",
    "directional_execution_returns",
    "execution_forward_returns",
    "execution_forward_returns_depth",
    "sell_fill_price",
    "slippage_amount",
]
