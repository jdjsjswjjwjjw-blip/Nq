"""التقرير الموحّد — دمج قنوات SSL و M9 (المراقب) و LLM (الألفا).

يجمع مخرجات القنوات الثلاث في تقرير Markdown واحد شامل مع سرد تنفيذي
اختياري من ``LanguageModel``.
"""

from __future__ import annotations

from dataclasses import dataclass

from nq.research.assistant import ResearchReport


@dataclass(frozen=True, slots=True)
class UnifiedResearchReport:
    """تقرير بحثي موحّد من القنوات الثلاث."""

    title: str
    ssl: ResearchReport
    coverage: ResearchReport
    alpha: ResearchReport
    narrative: str = ""

    def to_markdown(self) -> str:
        """يعرض التقرير الكامل بأقسام القنوات الثلاث."""
        lines = [f"# {self.title}", ""]
        if self.narrative:
            lines += ["## ملخّص تنفيذي", self.narrative, ""]

        lines += [
            "---",
            "",
            "## قناة 1 — SSL (النموذج التأسيسي ذاتي الإشراف)",
            "",
            self.ssl.to_markdown(),
            "",
            "---",
            "",
            "## قناة 2 — المراقب M9 (التغطية البنيوية / العمى البنيوي)",
            "",
            self.coverage.to_markdown(),
            "",
            "---",
            "",
            "## قناة 3 — LLM / الألفا (إشارات واستنتاجات موثّقة)",
            "",
            self.alpha.to_markdown(),
        ]
        return "\n".join(lines)

    @property
    def total_verified(self) -> int:
        return len(self.ssl.verified) + len(self.coverage.verified) + len(self.alpha.verified)

    @property
    def total_rejected(self) -> int:
        return len(self.ssl.rejected) + len(self.coverage.rejected) + len(self.alpha.rejected)


def build_unified_report(
    *,
    ssl_report: ResearchReport,
    coverage_report: ResearchReport,
    alpha_report: ResearchReport,
    title: str = "تقرير بحثي شامل — SSL + المراقب M9 + LLM",
    narrative: str = "",
) -> UnifiedResearchReport:
    """يدمج تقارير القنوات الثلاث في تقرير موحّد."""
    return UnifiedResearchReport(
        title=title,
        ssl=ssl_report,
        coverage=coverage_report,
        alpha=alpha_report,
        narrative=narrative,
    )
