"""محرّك فهم سلوك المزاد — المرحلة الأولى (بدون قرارات تداول / بدون RL).

الهدف: احتمالات سلوك المزاد (توازن، كسر حقيقي/كاذب، ريتست، قبول/رفض)
مع درجة ثقة — فوق البنية السببية القائمة (``decision_*``، جلسات سيولة،
أوردرفلو خام + نية بدون حذف).

لا يُصدِر توصيات دخول/خروج ولا يُعيد تصميم دفتر الأوامر.
"""

from __future__ import annotations

from nq.auction_behavior.pipeline import (
    AuctionBehaviorResult,
    BehaviorConfig,
    behavior_probabilities_frame,
    behavior_probability_summary,
    behavior_state_frame,
    run_auction_behavior_analysis,
)
from nq.auction_behavior.state import latest_state_snapshot, state_matrix
from nq.auction_behavior.types import BehaviorProbabilities, BehaviorStateSnapshot
from nq.auction_behavior.validate import BehaviorValidationReport

__all__ = [
    "AuctionBehaviorResult",
    "BehaviorConfig",
    "BehaviorProbabilities",
    "BehaviorStateSnapshot",
    "BehaviorValidationReport",
    "behavior_probabilities_frame",
    "behavior_probability_summary",
    "behavior_state_frame",
    "latest_state_snapshot",
    "run_auction_behavior_analysis",
    "state_matrix",
]
