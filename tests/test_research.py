"""اختبارات المحطة 7: مساعد البحث المُؤسَّس على الأدلّة."""

from __future__ import annotations

import numpy as np
import pytest

from nq.core.determinism import make_generator
from nq.research import (
    Evidence,
    EvidenceStore,
    Finding,
    ResearchAssistant,
    verify_finding,
    verify_report,
)


def test_evidence_store_add_get_dup() -> None:
    store = EvidenceStore()
    ev = Evidence(id="E1", source="test", metric="F", value=3.0, pvalue=0.01)
    store.add(ev)
    assert store.get("E1") is ev
    assert "E1" in store
    assert len(store) == 1
    with pytest.raises(ValueError, match="duplicate"):
        store.add(ev)


def test_unsupported_claim_rejected() -> None:
    store = EvidenceStore()
    finding = Finding(claim="السوق سيرتفع", evidence_ids=())
    outcome = verify_finding(finding, store)
    assert not outcome.verified
    assert any("unsupported" in r for r in outcome.reasons)


def test_untraceable_evidence_rejected() -> None:
    store = EvidenceStore()
    finding = Finding(claim="ادعاء", evidence_ids=("ghost",))
    outcome = verify_finding(finding, store)
    assert not outcome.verified
    assert any("untraceable" in r for r in outcome.reasons)


def test_insignificant_evidence_rejected() -> None:
    store = EvidenceStore()
    store.add(Evidence(id="E1", source="t", metric="x", value=1.0, pvalue=0.5))
    finding = Finding(claim="ادعاء", evidence_ids=("E1",), requires_significance=True, alpha=0.05)
    outcome = verify_finding(finding, store)
    assert not outcome.verified
    assert any("significant" in r for r in outcome.reasons)


def test_significant_evidence_accepted() -> None:
    store = EvidenceStore()
    store.add(Evidence(id="E1", source="t", metric="x", value=5.0, pvalue=0.001))
    finding = Finding(claim="ادعاء مدعوم", evidence_ids=("E1",), alpha=0.05)
    assert verify_finding(finding, store).verified


def test_assistant_compare_regimes_grounded() -> None:
    rng = make_generator(0)
    values = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    labels = np.repeat([0, 1], 100)
    assistant = ResearchAssistant()
    finding = assistant.compare_regimes(
        values, labels, metric_name="delta", n_permutations=500, rng=rng
    )
    outcome = verify_finding(finding, assistant.store)
    assert outcome.verified
    assert finding.evidence_ids[0] in assistant.store


def test_assistant_signal_significance() -> None:
    rng = make_generator(1)
    returns = rng.normal(0.2, 0.3, 300)  # عائد إيجابي واضح
    assistant = ResearchAssistant()
    finding = assistant.assess_signal_significance(
        returns, signal_name="ofi_signal", n_permutations=500, rng=rng
    )
    assert verify_finding(finding, assistant.store).verified


def test_report_excludes_unsupported() -> None:
    rng = make_generator(2)
    values = np.concatenate([rng.normal(0, 1, 80), rng.normal(4, 1, 80)])
    labels = np.repeat([0, 1], 80)
    assistant = ResearchAssistant()
    good = assistant.compare_regimes(values, labels, metric_name="vol", n_permutations=500, rng=rng)
    bad = Finding(claim="ادعاء بلا دليل", evidence_ids=())
    report = assistant.write_report([good, bad], title="تقرير")
    assert len(report.verified) == 1
    assert len(report.rejected) == 1
    md = report.to_markdown()
    assert "دليل `[" in md
    assert "ادعاء بلا دليل" in md  # يظهر في قسم المرفوضة بشفافية


def test_plan_research() -> None:
    plan = ResearchAssistant.plan_research(
        "هل يقود NQ الـ MNQ؟", ["حساب lead/lag", "اختبار الدلالة"]
    )
    assert plan[0].startswith("سؤال البحث")
    assert plan[1].startswith("1.")


def test_generate_hypothesis_registers_evidence() -> None:
    assistant = ResearchAssistant()
    ev = Evidence(id="H1", source="cross_market", metric="lead_lag", value=1.0, pvalue=0.01)
    finding = assistant.generate_hypothesis("NQ يقود MNQ", ev)
    assert "H1" in assistant.store
    assert verify_finding(finding, assistant.store).verified


class _StubLM:
    def complete(self, prompt: str) -> str:
        return "ملخّص محايد للاستنتاجات الموثّقة."


def test_report_with_language_model_narrative() -> None:
    rng = make_generator(3)
    values = np.concatenate([rng.normal(0, 1, 80), rng.normal(4, 1, 80)])
    labels = np.repeat([0, 1], 80)
    assistant = ResearchAssistant(language_model=_StubLM())
    finding = assistant.compare_regimes(
        values, labels, metric_name="d", n_permutations=300, rng=rng
    )
    report = assistant.write_report([finding], title="T")
    assert "ملخّص" in report.to_markdown()


def test_verify_report_split() -> None:
    store = EvidenceStore()
    store.add(Evidence(id="E1", source="t", metric="x", value=1.0, pvalue=0.001))
    good = Finding(claim="مدعوم", evidence_ids=("E1",))
    bad = Finding(claim="غير مدعوم", evidence_ids=())
    verified, rejected = verify_report([good, bad], store)
    assert len(verified) == 1
    assert len(rejected) == 1
