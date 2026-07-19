"""مساعد البحث المُؤسَّس على الأدلّة (Evidence-Grounded Research Assistant).

يولّد استنتاجات وفرضيات **مرتبطة دومًا بأدلّة كمية** مُسجَّلة، ويكتب تقارير لا
تحوي إلا ما اجتاز بوابة التحقّق. واجهة ``LanguageModel`` تسمح بتوصيل نموذج لغوي
لصياغة السرد، لكن الأدلّة تبقى شرطًا صارمًا (النموذج يصوغ، لا يخترع).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from nq.research.evidence import Evidence, EvidenceStore
from nq.research.findings import Finding, VerificationOutcome, verify_report
from nq.statistics.metrics import sharpe_ratio
from nq.statistics.regime_tests import regime_difference_test

FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class LanguageModel(Protocol):
    """واجهة نموذج لغوي اختيارية لصياغة السرد (مخرجه لا يتجاوز بوابة الأدلّة)."""

    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """تقرير بحثي: استنتاجات موثّقة + استنتاجات مرفوضة (شفافية) + سرد اختياري."""

    title: str
    verified: list[VerificationOutcome]
    rejected: list[VerificationOutcome]
    store: EvidenceStore
    narrative: str = ""

    def to_markdown(self) -> str:
        """يعرض التقرير كـ Markdown موثّق بالأدلّة القابلة للتتبّع."""
        lines = [f"# {self.title}", ""]
        if self.narrative:
            lines += ["## ملخّص", self.narrative, ""]

        lines += ["## استنتاجات موثّقة (Verified Findings)", ""]
        if not self.verified:
            lines.append("_لا استنتاجات موثّقة._")
        for outcome in self.verified:
            lines.append(f"- **{outcome.finding.claim}**")
            for eid in outcome.finding.evidence_ids:
                ev = self.store.get(eid)
                p = f", p={ev.pvalue:.4g}" if ev.pvalue is not None else ""
                n = f", n={ev.sample_size}" if ev.sample_size is not None else ""
                lines.append(f"  - دليل `[{ev.id}]` {ev.source}.{ev.metric} = {ev.value:.4g}{p}{n}")
        lines.append("")

        if self.rejected:
            lines += ["## استنتاجات مرفوضة (Rejected — Unsupported)", ""]
            for outcome in self.rejected:
                lines.append(f"- ~~{outcome.finding.claim}~~ — {'; '.join(outcome.reasons)}")
        return "\n".join(lines)


class ResearchAssistant:
    """مساعد بحثي يُنتج استنتاجات مؤسَّسة على الأدلّة ويكتب تقارير موثّقة."""

    __slots__ = ("_counter", "alpha", "language_model", "store")

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        store: EvidenceStore | None = None,
        language_model: LanguageModel | None = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.store = store if store is not None else EvidenceStore()
        self.language_model = language_model
        self._counter = 0

    def _new_id(self, source: str) -> str:
        self._counter += 1
        return f"E{self._counter}:{source}"

    def compare_regimes(
        self,
        values: npt.NDArray[np.floating] | Sequence[float],
        labels: npt.NDArray[np.integer] | Sequence[int],
        *,
        metric_name: str,
        n_permutations: int = 10_000,
        rng: np.random.Generator | None = None,
        version: str | None = None,
    ) -> Finding:
        """يقارن مقياسًا عبر الحالات ويُنتج استنتاجًا مؤسَّسًا على اختبار إحصائي."""
        vals = np.asarray(values, dtype=np.float64)
        labs = np.asarray(labels, dtype=np.intp)
        result = regime_difference_test(vals, labs, n_permutations=n_permutations, rng=rng)
        eid = self._new_id("regime_difference_test")
        self.store.add(
            Evidence(
                id=eid,
                source="regime_difference_test",
                metric="F",
                value=result.statistic,
                pvalue=result.pvalue,
                sample_size=int(vals.shape[0]),
                version=version,
                detail=f"cross-regime difference in '{metric_name}'",
            )
        )
        claim = (
            f"'{metric_name}' يختلف عبر حالات السوق "
            f"(F={result.statistic:.3f}, p={result.pvalue:.4g})."
        )
        return Finding(claim=claim, evidence_ids=(eid,), alpha=self.alpha, category="regime")

    def assess_signal_significance(
        self,
        returns: npt.NDArray[np.floating] | Sequence[float],
        *,
        signal_name: str,
        n_permutations: int = 10_000,
        rng: np.random.Generator | None = None,
        version: str | None = None,
    ) -> Finding:
        """يقيّم دلالة عوائد إشارة (Sharpe + اختبار تبديل بقلب الإشارة)."""
        r = np.asarray(returns, dtype=np.float64)
        generator = rng if rng is not None else np.random.default_rng(0)
        observed = float(np.mean(r))
        null = np.empty(n_permutations, dtype=np.float64)
        for i in range(n_permutations):
            signs = generator.choice(np.array([-1.0, 1.0]), size=r.shape[0])
            null[i] = float(np.mean(signs * r))
        pvalue = (int(np.sum(np.abs(null) >= abs(observed))) + 1) / (n_permutations + 1)

        sharpe = sharpe_ratio(r)
        eid = self._new_id("sharpe_ratio")
        self.store.add(
            Evidence(
                id=eid,
                source="sharpe_ratio",
                metric="sharpe",
                value=sharpe,
                pvalue=pvalue,
                sample_size=int(r.shape[0]),
                version=version,
                detail=f"return significance of signal '{signal_name}' (sign-flip test)",
            )
        )
        claim = f"إشارة '{signal_name}' ذات عائد دالّ إحصائيًا (Sharpe={sharpe:.3f}, p={pvalue:.4g})."
        return Finding(claim=claim, evidence_ids=(eid,), alpha=self.alpha, category="signal")

    def generate_hypothesis(
        self,
        claim: str,
        evidence: Evidence | Sequence[Evidence],
        *,
        requires_significance: bool = True,
        category: str = "hypothesis",
    ) -> Finding:
        """يبني فرضية مرتبطة بأدلّة يُقدّمها المستخدم (تُسجَّل في السجلّ)."""
        items = [evidence] if isinstance(evidence, Evidence) else list(evidence)
        ids: list[str] = []
        for ev in items:
            if ev.id not in self.store:
                self.store.add(ev)
            ids.append(ev.id)
        return Finding(
            claim=claim,
            evidence_ids=tuple(ids),
            requires_significance=requires_significance,
            alpha=self.alpha,
            category=category,
        )

    @staticmethod
    def plan_research(question: str, steps: Sequence[str]) -> list[str]:
        """يبني خطة بحث مُرقّمة حتمية من سؤال وخطوات."""
        if not steps:
            raise ValueError("a research plan needs at least one step")
        return [f"سؤال البحث: {question}"] + [f"{i}. {s}" for i, s in enumerate(steps, start=1)]

    def write_report(self, findings: list[Finding], *, title: str) -> ResearchReport:
        """يتحقّق من الاستنتاجات ويكتب تقريرًا لا يحوي إلا الموثّق منها."""
        verified, rejected = verify_report(findings, self.store)
        narrative = ""
        if self.language_model is not None and verified:
            claims = " ".join(o.finding.claim for o in verified)
            narrative = self.language_model.complete(
                "لخّص الاستنتاجات الموثّقة التالية دون إضافة أي ادعاء جديد:\n" + claims
            )
        return ResearchReport(
            title=title,
            verified=verified,
            rejected=rejected,
            store=self.store,
            narrative=narrative,
        )
