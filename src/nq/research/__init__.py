"""مساعد البحث المُؤسَّس على الأدلّة (Evidence-Grounded Research Assistant) — المحطة 7.

المبدأ الحاكم: **لا دليل مُختلَق (No hallucinated evidence)**. كل ادعاء يجب أن
يرتبط بدليل كمي قابل للتتبّع، وأي استنتاج غير مدعوم أو غير دال إحصائيًا يُرفَض
عبر بوابة التحقّق قبل أن يدخل أي تقرير.

المكوّنات:

* ``Evidence`` / ``EvidenceStore`` — أدلّة كمية قابلة للتتبّع مع مصدرها وإصدارها.
* ``Finding`` / ``verify_finding`` — استنتاج مرتبط بأدلّة + تحقّق صارم منها.
* ``ResearchAssistant`` — مقارنة الحالات، توليد الفرضيات، التخطيط، وكتابة تقرير
  موثّق حتمي. واجهة ``LanguageModel`` تسمح بتوصيل LLM لاحقًا شرط مرور مخرجه
  ببوابة التحقّق نفسها.
"""

from __future__ import annotations

from nq.research.assistant import LanguageModel, ResearchAssistant, ResearchReport
from nq.research.day_parallel import (
    DayInput,
    DayJobResult,
    DayParallelManifest,
    day_id_from_path,
    discover_day_inputs,
    run_fail_breakout_day_parallel,
    stable_day_seed,
)
from nq.research.evidence import Evidence, EvidenceStore
from nq.research.findings import Finding, VerificationOutcome, verify_finding, verify_report
from nq.research.progress import PipelineProgress, resolve_progress
from nq.research.unified import UnifiedResearchReport, build_unified_report

__all__ = [
    "DayInput",
    "DayJobResult",
    "DayParallelManifest",
    "Evidence",
    "EvidenceStore",
    "Finding",
    "LanguageModel",
    "PipelineProgress",
    "ResearchAssistant",
    "ResearchReport",
    "UnifiedResearchReport",
    "VerificationOutcome",
    "build_unified_report",
    "day_id_from_path",
    "discover_day_inputs",
    "resolve_progress",
    "run_fail_breakout_day_parallel",
    "stable_day_seed",
    "verify_finding",
    "verify_report",
]
