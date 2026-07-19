"""الاختبار الإحصائي (Statistical Testing) — المحطة 6.

بروتوكول صارم للتحقق من كل فرضية/إشارة قبل اعتمادها علميًا:

* ``resampling``       — دلالة عبر permutation + فترات ثقة bootstrap و
  block-bootstrap (لاحترام الارتباط الزمني في السلاسل المالية).
* ``multiple_testing`` — تصحيح التعدّد (Benjamini-Hochberg، Holm، Bonferroni).
* ``metrics``          — مقاييس أداء/تنبّؤ (Sharpe، Information Coefficient، t).
* ``regime_tests``     — تحقّق اختلاف المقاييس عبر الحالات (permutation F-test).
* ``hypothesis``       — بطارية تحقّق فرضيات + تقرير موحّد مع تصحيح التعدّد.
"""

from __future__ import annotations

from nq.statistics.hypothesis import verify_hypotheses
from nq.statistics.metrics import information_coefficient, sharpe_ratio, t_statistic
from nq.statistics.multiple_testing import benjamini_hochberg, bonferroni, holm
from nq.statistics.regime_tests import regime_difference_test
from nq.statistics.resampling import (
    TestResult,
    bootstrap_ci,
    moving_block_bootstrap_ci,
    permutation_test,
)

__all__ = [
    "TestResult",
    "benjamini_hochberg",
    "bonferroni",
    "bootstrap_ci",
    "holm",
    "information_coefficient",
    "moving_block_bootstrap_ci",
    "permutation_test",
    "regime_difference_test",
    "sharpe_ratio",
    "t_statistic",
    "verify_hypotheses",
]
