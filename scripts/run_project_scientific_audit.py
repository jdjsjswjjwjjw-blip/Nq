#!/usr/bin/env python3
"""تحقق علمي كمّي شامل لكل المشروع — بوابات + بطاريات + تقرير.

استخدام:
    python scripts/run_project_scientific_audit.py
    python scripts/run_project_scientific_audit.py \\
        --out /opt/cursor/artifacts/project_scientific_audit.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SuiteResult:
    name: str
    cmd: list[str]
    returncode: int
    wall_s: float
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _run(name: str, cmd: list[str], *, cwd: Path) -> SuiteResult:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return SuiteResult(
        name=name,
        cmd=cmd,
        returncode=proc.returncode,
        wall_s=time.perf_counter() - t0,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _tail(text: str, n: int = 8_000) -> str:
    return text[-n:] if text else ""


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/project_scientific_audit.md"),
    )
    parser.add_argument("--skip-full", action="store_true", help="تخطّي pytest الكامل")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    suites: list[tuple[str, list[str]]] = [
        (
            "ruff",
            [py, "-m", "ruff", "check", "src", "tests", "scripts"],
        ),
        (
            "mypy",
            [py, "-m", "mypy", "src/nq"],
        ),
        (
            "audit_core",
            [
                py,
                "-m",
                "pytest",
                "tests/test_project_scientific_audit.py",
                "tests/test_vp_fixed_range.py",
                "tests/test_vp_session_reset.py",
                "tests/test_vp_scientific_intensive.py",
                "-q",
                "--tb=line",
            ],
        ),
        (
            "principles",
            [
                py,
                "-m",
                "pytest",
                "tests/test_leakage.py",
                "tests/test_invariants.py",
                "tests/test_integrity.py",
                "tests/test_reconstruction_causality.py",
                "tests/test_temporal_policy.py",
                "tests/test_shared_signal_semantics.py",
                "-q",
                "--tb=line",
            ],
        ),
        (
            "simulation_stack",
            [
                py,
                "-m",
                "pytest",
                "tests/test_auction.py",
                "tests/test_volume_profile.py",
                "tests/test_footprint.py",
                "tests/test_order_flow.py",
                "tests/test_liquidity.py",
                "tests/test_liquidity_edge.py",
                "tests/test_cross_market.py",
                "tests/test_reconstruction.py",
                "tests/test_book.py",
                "-q",
                "--tb=line",
            ],
        ),
        (
            "research_strategies",
            [
                py,
                "-m",
                "pytest",
                "tests/test_vp_auction_strategy.py",
                "tests/test_fail_fvg_strategy.py",
                "tests/test_breakout.py",
                "tests/test_orchestrator.py",
                "tests/test_research.py",
                "tests/test_alpha.py",
                "tests/test_models_ssl.py",
                "tests/test_models_pipeline.py",
                "tests/test_models_splitting.py",
                "-q",
                "--tb=line",
            ],
        ),
    ]
    if not args.skip_full:
        suites.append(("full_pytest", [py, "-m", "pytest", "-q", "--tb=line"]))

    results: list[SuiteResult] = []
    for name, cmd in suites:
        print(f"[run] {name}: {' '.join(cmd)}", flush=True)
        results.append(_run(name, cmd, cwd=root))
        status = "PASS" if results[-1].ok else "FAIL"
        print(f"[{status}] {name} ({results[-1].wall_s:.1f}s)", flush=True)

    by = {r.name: r for r in results}
    all_ok = all(r.ok for r in results)

    claim_rows = [
        ("Zero temporal leakage (suffix / availability / splits)", "principles", "audit_core"),
        ("Mathematical invariants (Hypothesis)", "principles", None),
        ("MBO-only contract + reconstruction causality", "audit_core", "simulation_stack"),
        ("Auction dual-TF + FSM scientific phases", "audit_core", "simulation_stack"),
        ("Liquidity-session VP reset (Asia/London/NY)", "audit_core", None),
        ("Fixed-Range five-rule lock + FR-only decisions", "audit_core", None),
        ("WF-before-exec + strategy ontology", "research_strategies", "audit_core"),
        ("Simulation stack (VP/OF/FP/book/liquidity)", "simulation_stack", None),
        ("Static gates (ruff + mypy)", "ruff", "mypy"),
        ("Full repository pytest", "full_pytest", None),
    ]

    lines: list[str] = [
        "# تقرير تحقق علمي كمّي شامل — مشروع Nq",
        "",
        "## النطاق",
        "مبادئ حاكمة (تسريب/سببية/MBO) · ثوابت رياضية · محاكاة · Fixed-Range ·",
        "جلسات سيولة · استراتيجيات/WF · بوابات ثابتة · pytest كامل.",
        "",
        "## نتائج الأجنحة",
        "",
        "| الجناح | الحكم | الزمن (ث) | exit |",
        "|---|---|---:|---:|",
    ]
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        lines.append(f"| `{r.name}` | **{mark}** | {r.wall_s:.2f} | {r.returncode} |")

    lines += ["", "## مصفوفة الادعاءات العلمية", "", "| الادعاء | الحكم | أدلة |", "|---|---|---|"]
    for claim, primary, secondary in claim_rows:
        keys = [primary] + ([secondary] if secondary else [])
        present = [k for k in keys if k in by]
        if not present:
            verdict = "SKIP"
            evidence = "—"
        elif all(by[k].ok for k in present):
            verdict = "PASS"
            evidence = ", ".join(f"`{k}`" for k in present)
        else:
            verdict = "FAIL"
            evidence = ", ".join(f"`{k}`={'PASS' if by[k].ok else 'FAIL'}" for k in present)
        lines.append(f"| {claim} | **{verdict}** | {evidence} |")

    lines += ["", "## مخرجات مفصّلة (ذيول)", ""]
    for r in results:
        lines.append(f"### `{r.name}`")
        lines.append(f"- cmd: `{' '.join(r.cmd)}`")
        lines.append(f"- exit: `{r.returncode}` · wall_s: `{r.wall_s:.2f}`")
        lines.append("")
        lines.append("```")
        lines.append(_tail(r.stdout, 6_000) or "(no stdout)")
        lines.append("```")
        if r.stderr.strip():
            lines.append("")
            lines.append("<details><summary>stderr</summary>")
            lines.append("")
            lines.append("```")
            lines.append(_tail(r.stderr, 3_000))
            lines.append("```")
            lines.append("</details>")
        lines.append("")

    lines.append("## الحكم النهائي")
    lines.append("")
    if all_ok:
        lines.append(
            "**PASS — التحقق العلمي الكمّي للمشروع أخضر على البيانات الاصطناعية "
            "المُتحكَّم بها والبوابات الثابتة.**"
        )
        lines.append("")
        lines.append(
            "هذا يثبت صحّة العقود السببية وطبقات المحاكاة/FR/الجلسات/WF على "
            "المسارات المختبرة. **لا يساوي إثبات ربحية على بيانات حية.**"
        )
        verdict = 0
    else:
        failed = [r.name for r in results if not r.ok]
        lines.append(f"**FAIL — أجنحة فاشلة: {', '.join(failed)}**")
        lines.append("")
        lines.append("راجع الذيول أعلاه قبل أي اعتماد تداولي أو دمج إضافي.")
        verdict = 1

    body = "\n".join(lines) + "\n"
    args.out.write_text(body, encoding="utf-8")
    print("\n" + body)
    print(f"[wrote] {args.out}")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
