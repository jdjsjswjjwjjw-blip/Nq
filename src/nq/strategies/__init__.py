"""استراتيجيات بحثية منفصلة — تُركَّب على خط المشروع دون fork معماري.

كل استراتيجية:

* تشتق إشاراتها من MBO عبر ``nq.simulation``.
* تُقيَّم عبر ``nq.alpha`` (IC + دلالة + تصحيح تعدّد).
* تُوثَّق عبر ``nq.research`` (فرضيات بأدلّة قابلة للتتبع).
* قد تُفلتر اختياريًا بتمثيلات SSL الكامنة (بدون تسريب زمني).
"""

from __future__ import annotations

from nq.strategies.fail_fvg import FailFvgResearchResult, run_fail_fvg_research

__all__ = [
    "FailFvgResearchResult",
    "run_fail_fvg_research",
]
