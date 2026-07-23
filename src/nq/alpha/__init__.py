"""المخرجات النهائية: إشارات ألفا وهياكل السوق (Alpha & Outputs) — المحطة 8.

يربط كامل المسار من MBO الخام إلى إشارات ألفا قابلة لإعادة الإنتاج ومُقيَّمة
إحصائيًا مع تصحيح التعدّد (لتفادي التنقيب عن البيانات / data-snooping)، وتقرير
بحثي موثّق بالأدلّة.

* ``signals``   — ``AlphaSignal``، العوائد الأمامية، وتقييم الإشارة (IC، Sharpe، دلالة).
* ``discovery`` — فرز الإشارات مع تصحيح التعدّد، وخط بحثي من MBO خام إلى مخرجات.
"""

from __future__ import annotations

from nq.alpha.discovery import (
    AlphaDiscovery,
    FullResearchResult,
    discover_alpha_from_features,
    run_full_research_pipeline,
    run_research_pipeline,
)
from nq.alpha.signals import (
    AlphaSignal,
    ExecutionMode,
    SignalEvaluation,
    align_forward_returns,
    evaluate_signal,
    evaluate_signal_intraday,
    screen_signals,
)

__all__ = [
    "AlphaDiscovery",
    "AlphaSignal",
    "ExecutionMode",
    "FullResearchResult",
    "SignalEvaluation",
    "align_forward_returns",
    "discover_alpha_from_features",
    "evaluate_signal",
    "evaluate_signal_intraday",
    "run_full_research_pipeline",
    "run_research_pipeline",
    "screen_signals",
]
