"""أنواع مخرجات مراقب التغطية (Coverage Monitor Types)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nq.research.assistant import ResearchReport
from nq.research.evidence import Evidence


@dataclass(frozen=True, slots=True)
class CoverageAlert:
    """تنبيه عمى بنيوي مؤكَّد إحصائيًا."""

    metric: str
    evidence_id: str
    severity: str
    detail: str


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """تقرير مراقبة التغطية الكامل."""

    alerts: list[CoverageAlert]
    evidence: list[Evidence]
    metrics: pl.DataFrame
    report: ResearchReport
