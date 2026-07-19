"""مراقب التغطية البنيوية (Structural Coverage Monitor) — المحطة 9.

يكشف أنماطًا في بيانات MBO لا تلتقطها بنية المحاكيات/الميزات الحالية:

* ``MFIG``  — فجوة المعلومات الشرطية (MBO vs Features → Price).
* ``CER``   — بقايا التعرّض السببي لكل كتلة محاكاة.
* ``PSG``   — فجوة الكفاية التنبؤية (World Model surprise).
* ``CRS``   — كفاية إعادة البناء المُقنّعة لكل كتلة.
* ``LORI``  — الأنظمة اليتيمة + Transition Surprise.
* ``QDUF``  — نسبة ديناميكية الطابور غير المفسَّرة.

كل مقياس: walk-forward + embargo، p-value، ومخرج ``Evidence`` قابل للتتبّع.
"""

from __future__ import annotations

from nq.coverage.blocks import DEFAULT_FEATURE_BLOCKS, resolve_block_columns
from nq.coverage.distance import distance_correlation, max_axis_dependence
from nq.coverage.mbo_descriptors import mbo_window_descriptors
from nq.coverage.metrics import (
    measure_cer,
    measure_crs,
    measure_lori,
    measure_mfig,
    measure_psg,
    measure_qduf,
    metric_to_evidence,
    run_all_metrics,
)
from nq.coverage.monitor import (
    build_coverage_report,
    run_coverage_on_features,
    run_coverage_pipeline,
)
from nq.coverage.types import CoverageAlert, CoverageReport

__all__ = [
    "DEFAULT_FEATURE_BLOCKS",
    "CoverageAlert",
    "CoverageReport",
    "build_coverage_report",
    "distance_correlation",
    "max_axis_dependence",
    "mbo_window_descriptors",
    "measure_cer",
    "measure_crs",
    "measure_lori",
    "measure_mfig",
    "measure_psg",
    "measure_qduf",
    "metric_to_evidence",
    "resolve_block_columns",
    "run_all_metrics",
    "run_coverage_on_features",
    "run_coverage_pipeline",
]
