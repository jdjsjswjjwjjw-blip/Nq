"""طبقة المحاكاة (Simulation Layer) — المحطة 2.

تُشتق كل المحاكيات **حصريًا** من أحداث MBO ومن إعادة بناء دفتر الأوامر، وتُنتج
ميزات كمية بترتيب زمني سببي. كل ميزة مُجمّعة على نافذة/دفعة تحمل ``availability_ts``
(زمن اكتمال النافذة) لضمان الاستخدام point-in-time دون تسريب.

المحاكيات:

* ``footprint``       — البصمة السعرية (Bid/Ask volume، Delta، Imbalance، Absorption).
* ``volume_profile``  — ملف الحجم (POC، VAH/VAL، HVN/LVN، Value Migration).
* ``order_flow``      — تدفّق الأوامر (عدوانية الشراء/البيع، OFI، استهلاك السيولة).
* ``liquidity``       — السيولة (إضافة/سحب، أوامر قائمة، كشف الآيسبرغ).
* ``auction``         — نظرية المزاد (توازن/اختلال، تمدّد، دفاع الارتداد).
* ``cross_market``    — عبر السوقين (NQ↔MNQ، Lead/Lag، تباعد، مصيدة المتداولين).
* ``fvg``             — Fair Value Gap + Failed FVG / Effort-Without-Result (سببي).
"""

from __future__ import annotations

from nq.simulation.auction import auction_signal_frame, auction_states
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.cross_market import cross_market_features
from nq.simulation.execution import (
    directional_execution_returns,
    execution_forward_returns,
)
from nq.simulation.footprint import footprint_cells, footprint_summary
from nq.simulation.fvg import build_ohlcv_bars, detect_h1_fvgs, failed_fvg_features
from nq.simulation.liquidity import detect_icebergs, liquidity_summary
from nq.simulation.order_flow import ofi_by_bucket, order_flow_imbalance, order_flow_summary
from nq.simulation.volume_profile import (
    DevelopingVolumeProfile,
    ValueArea,
    build_volume_profile,
    classify_nodes,
    developing_value_area,
    value_area,
)

__all__ = [
    "BUCKET_END",
    "BUCKET_START",
    "DevelopingVolumeProfile",
    "ValueArea",
    "add_time_bucket",
    "auction_signal_frame",
    "auction_states",
    "build_ohlcv_bars",
    "build_volume_profile",
    "classify_nodes",
    "cross_market_features",
    "detect_h1_fvgs",
    "detect_icebergs",
    "developing_value_area",
    "directional_execution_returns",
    "execution_forward_returns",
    "extract_trades",
    "failed_fvg_features",
    "footprint_cells",
    "footprint_summary",
    "liquidity_summary",
    "ofi_by_bucket",
    "order_flow_imbalance",
    "order_flow_summary",
    "value_area",
]
