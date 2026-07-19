"""الاستنتاجات وبوابة التحقّق (Findings & Verification Gate).

``Finding`` استنتاج/ادعاء مرتبط بمعرّفات أدلّة. ``verify_finding`` يفرض قاعدة
"لا دليل مُختلَق": يُرفَض الاستنتاج إذا لم يُشِر إلى أي دليل، أو أشار إلى دليل
غير موجود (غير قابل للتتبّع)، أو تطلّب دلالة إحصائية دون توفّر دليل دالّ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nq.research.evidence import EvidenceStore


@dataclass(frozen=True, slots=True)
class Finding:
    """ادعاء بحثي مرتبط بأدلّة كمية قابلة للتتبّع."""

    claim: str
    evidence_ids: tuple[str, ...]
    requires_significance: bool = True
    alpha: float = 0.05
    category: str = "general"


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """نتيجة تحقّق استنتاج: مقبول أم مرفوض ولماذا."""

    finding: Finding
    verified: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def verify_finding(finding: Finding, store: EvidenceStore) -> VerificationOutcome:
    """يتحقّق من استنتاج مقابل سجلّ الأدلّة (يفرض عدم اختلاق الأدلّة)."""
    reasons: list[str] = []

    if not finding.evidence_ids:
        reasons.append("unsupported claim: no evidence referenced")

    present = [eid for eid in finding.evidence_ids if eid in store]
    missing = [eid for eid in finding.evidence_ids if eid not in store]
    if missing:
        reasons.append(f"untraceable evidence (not found): {missing}")

    if finding.requires_significance and present:
        significant = any(store.get(eid).is_significant(finding.alpha) for eid in present)
        if not significant:
            reasons.append(
                f"no statistically significant supporting evidence at alpha={finding.alpha}"
            )

    return VerificationOutcome(finding=finding, verified=not reasons, reasons=tuple(reasons))


def verify_report(
    findings: list[Finding], store: EvidenceStore
) -> tuple[list[VerificationOutcome], list[VerificationOutcome]]:
    """يقسّم الاستنتاجات إلى مقبولة ومرفوضة بعد التحقّق."""
    verified: list[VerificationOutcome] = []
    rejected: list[VerificationOutcome] = []
    for finding in findings:
        outcome = verify_finding(finding, store)
        (verified if outcome.verified else rejected).append(outcome)
    return verified, rejected
