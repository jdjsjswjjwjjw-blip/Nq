"""طبقة المحاكاة (Simulation Layer) — المحطة 2.

تُشتق كل المحاكيات **حصريًا** من أحداث MBO ومن إعادة بناء دفتر الأوامر، وتُنتج
ميزات كمية بترتيب زمني سببي. كل ميزة مُجمّعة على نافذة/دفعة تحمل ``availability_ts``
(زمن اكتمال النافذة) لضمان الاستخدام point-in-time دون تسريب.

المحاكيات:

* ``footprint``       — البصمة السعرية (Bid/Ask volume، Delta، Imbalance، Absorption).
* ``volume_profile``  — ملف الحجم (POC، VAH/VAL، HVN/LVN، Value Migration).
* ``order_flow``      — تدفّق الأوامر (عدوانية، OFI، استهلاك، تسارع→اختلال مبكر).
* ``liquidity``       — السيولة (إضافة/سحب، أوامر قائمة، كشف الآيسبرغ).
* ``auction``         — نظرية المزاد (توازن/اختلال، تمدّد، دفاع الارتداد).
* ``cross_market``    — عبر السوقين (NQ↔MNQ، Lead/Lag، تباعد، مصيدة المتداولين).
* ``fvg``             — Fair Value Gap + Failed FVG / Effort-Without-Result (سببي).
"""

from __future__ import annotations

from nq.simulation.auction import (
    VP_FIXED_RANGE_COLUMNS,
    VP_PROFILE_INTERVAL_NS,
    VP_SIGNAL_INTERVAL_NS,
    VpFixedRangeConfig,
    attach_vp_fixed_range,
    auction_action_states,
    auction_fsm_columns,
    auction_signal_frame,
    auction_signals_from_states,
    auction_states,
)
from nq.simulation.bottom_book import (
    BOTTOM_BOOK_COLUMNS,
    attach_bottom_book_asof,
    bottom_book_features_at_bar_close,
)
from nq.simulation.breakout import failed_breakout_features, failed_breakout_from_bars
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades
from nq.simulation.cross_market import cross_market_features
from nq.simulation.deceptive_liquidity import (
    DECEPTIVE_FEATURE_COLUMNS,
    DeceptiveLiquidityConfig,
    deceptive_features_by_bucket,
    filter_deceptive_liquidity,
    score_deceptive_events,
)
from nq.simulation.depth_noise import DepthNoiseConfig, filter_depth_noise
from nq.simulation.edge_execution_plan import (
    EDGE_TRADE_COLUMNS,
    EdgeExecConfig,
    EdgeSearchSpec,
    default_edge_search_grid,
    run_edge_plan,
    search_best_edge_spec,
)
from nq.simulation.execution import (
    directional_execution_returns,
    execution_forward_returns,
)
from nq.simulation.footprint import footprint_cells, footprint_summary
from nq.simulation.fvg import (
    build_ohlcv_bars,
    detect_h1_fvgs,
    failed_fvg_features,
    failed_fvg_from_bars,
)
from nq.simulation.liquidity import detect_icebergs, liquidity_summary
from nq.simulation.market_truth import (
    MARKET_TRUTH_COLUMNS,
    MarketTruthConfig,
    build_market_truth_frame,
)
from nq.simulation.order_flow import (
    ORDER_ACCEL_COLUMNS,
    ofi_by_bucket,
    order_acceleration_columns,
    order_flow_imbalance,
    order_flow_summary,
)
from nq.simulation.volume_profile import (
    DevelopingVolumeProfile,
    ValueArea,
    build_volume_profile,
    classify_nodes,
    developing_value_area,
    value_area,
    value_area_from_levels,
)

__all__ = [
    "BOTTOM_BOOK_COLUMNS",
    "BUCKET_END",
    "BUCKET_START",
    "DECEPTIVE_FEATURE_COLUMNS",
    "EDGE_TRADE_COLUMNS",
    "MARKET_TRUTH_COLUMNS",
    "VP_FIXED_RANGE_COLUMNS",
    "VP_PROFILE_INTERVAL_NS",
    "VP_SIGNAL_INTERVAL_NS",
    "DeceptiveLiquidityConfig",
    "DepthNoiseConfig",
    "DevelopingVolumeProfile",
    "EdgeExecConfig",
    "EdgeSearchSpec",
    "MarketTruthConfig",
    "ValueArea",
    "VpFixedRangeConfig",
    "add_time_bucket",
    "attach_bottom_book_asof",
    "attach_vp_fixed_range",
    "auction_action_states",
    "auction_fsm_columns",
    "auction_signal_frame",
    "auction_signals_from_states",
    "auction_states",
    "bottom_book_features_at_bar_close",
    "build_market_truth_frame",
    "build_ohlcv_bars",
    "build_volume_profile",
    "classify_nodes",
    "cross_market_features",
    "deceptive_features_by_bucket",
    "default_edge_search_grid",
    "detect_h1_fvgs",
    "detect_icebergs",
    "developing_value_area",
    "directional_execution_returns",
    "execution_forward_returns",
    "extract_trades",
    "failed_breakout_features",
    "failed_breakout_from_bars",
    "failed_fvg_features",
    "failed_fvg_from_bars",
    "filter_deceptive_liquidity",
    "filter_depth_noise",
    "footprint_cells",
    "footprint_summary",
    "liquidity_summary",
    "ofi_by_bucket",
    "ORDER_ACCEL_COLUMNS",
    "order_acceleration_columns",
    "order_flow_imbalance",
    "order_flow_summary",
    "run_edge_plan",
    "score_deceptive_events",
    "search_best_edge_spec",
    "value_area",
    "value_area_from_levels",
]
