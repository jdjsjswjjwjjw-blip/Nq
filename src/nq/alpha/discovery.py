"""اكتشاف الألفا من الميزات وخط البحث الكامل (Alpha Discovery & Pipeline).

يجمع كامل المسار: من إطار الميزات (المُشتق سببيًا من MBO) إلى إشارات مرشّحة،
تقييمها وفرزها إحصائيًا مع تصحيح التعدّد، ثم تقرير بحثي موثّق. كل شيء حتمي
وقابل لإعادة الإنتاج من البيانات الخام.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nq.alpha.signals import align_forward_returns, evaluate_signal, screen_signals
from nq.contracts.temporal import AVAILABILITY_TS
from nq.coverage import run_coverage_pipeline
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.findings import Finding
from nq.simulation.cross_market import cross_market_features

if TYPE_CHECKING:
    from nq.coverage.types import CoverageReport

_DEFAULT_SIGNAL_COLUMNS = ("nq_delta", "mnq_delta", "lead_lag", "trap_setup", "divergence")


@dataclass(frozen=True, slots=True)
class AlphaDiscovery:
    """مخرجات اكتشاف الألفا: تقييمات مفرزة، إشارات مختارة، وتقرير موثّق."""

    evaluations: pl.DataFrame
    selected: list[str]
    report: ResearchReport


def discover_alpha_from_features(
    frame: pl.DataFrame,
    *,
    signal_columns: Sequence[str],
    price_col: str,
    time_col: str = AVAILABILITY_TS,
    horizon: int = 1,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
) -> AlphaDiscovery:
    """يقيّم ويفرز إشارات مرشّحة من إطار ميزات، ويكتب تقريرًا موثّقًا."""
    generator = rng if rng is not None else np.random.default_rng(0)
    assistant = ResearchAssistant(alpha=alpha)

    if frame.height == 0:
        empty = screen_signals([], alpha=alpha)
        return AlphaDiscovery(empty, [], assistant.write_report([], title="Alpha Discovery"))

    prices = frame[price_col].to_numpy().astype(np.float64)
    forward = align_forward_returns(prices, horizon=horizon)

    evaluations = [
        evaluate_signal(
            col,
            frame[col].to_numpy().astype(np.float64),
            forward,
            n_permutations=n_permutations,
            rng=generator,
        )
        for col in signal_columns
    ]
    screened = screen_signals(evaluations, alpha=alpha)

    findings: list[Finding] = []
    selected: list[str] = []
    for row in screened.filter(pl.col("selected")).iter_rows(named=True):
        selected.append(row["name"])
        evidence = Evidence(
            id=f"alpha:{row['name']}",
            source="alpha_screen",
            metric="IC",
            value=float(row["ic"]),
            pvalue=float(row["adjusted_pvalue"]),
            sample_size=int(row["n"]),
            detail=f"predictive alpha of signal '{row['name']}' (horizon-forward IC)",
        )
        claim = (
            f"إشارة '{row['name']}' تحمل ألفا تنبّئيًا دالًّا "
            f"(IC={row['ic']:.3f}, adj_p={row['adjusted_pvalue']:.4g}, Sharpe={row['sharpe']:.3f})."
        )
        findings.append(assistant.generate_hypothesis(claim, evidence, category="alpha"))

    report = assistant.write_report(findings, title="Novel Alpha Signals — Research Report")
    return AlphaDiscovery(evaluations=screened, selected=selected, report=report)


@dataclass(frozen=True, slots=True)
class FullResearchResult:
    """مخرجات الخط البحثي الكامل: تغطية + ألفا."""

    coverage: CoverageReport
    alpha: AlphaDiscovery


def run_full_research_pipeline(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    horizon: int = 1,
    signal_columns: Sequence[str] | None = None,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_permutations: int = 2000,
    lead_lag_window: int = 2,
    coverage_splits: int = 3,
    coverage_embargo: int = 0,
    rng: np.random.Generator | None = None,
) -> FullResearchResult:
    """خط بحثي متكامل: مراقبة التغطية ثم اكتشاف الألفا (قابل لإعادة الإنتاج)."""
    coverage = run_coverage_pipeline(
        nq,
        mnq,
        interval_ns=interval_ns,
        price_col=price_col,
        alpha=alpha,
        n_splits=coverage_splits,
        embargo=coverage_embargo,
        n_permutations=n_permutations,
        lead_lag_window=lead_lag_window,
        rng=rng,
    )
    alpha_result = run_research_pipeline(
        nq,
        mnq,
        interval_ns=interval_ns,
        horizon=horizon,
        signal_columns=signal_columns,
        price_col=price_col,
        alpha=alpha,
        n_permutations=n_permutations,
        lead_lag_window=lead_lag_window,
        rng=rng,
    )
    return FullResearchResult(coverage=coverage, alpha=alpha_result)


def run_research_pipeline(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    interval_ns: int,
    horizon: int = 1,
    signal_columns: Sequence[str] | None = None,
    price_col: str = "nq_close",
    alpha: float = 0.05,
    n_permutations: int = 2000,
    lead_lag_window: int = 2,
    rng: np.random.Generator | None = None,
) -> AlphaDiscovery:
    """خط بحثي كامل قابل لإعادة الإنتاج: من MBO الخام (NQ/MNQ) إلى إشارات الألفا.

    يعيد بناء ميزات عبر السوقين ثم يكتشف الألفا ويفرزه ويكتب التقرير — وكله
    حتمي، فيُعاد إنتاج المخرجات نفسها من البيانات الخام نفسها.
    """
    features = cross_market_features(
        nq, mnq, interval_ns=interval_ns, lead_lag_window=lead_lag_window
    )
    columns = (
        list(signal_columns)
        if signal_columns is not None
        else [c for c in _DEFAULT_SIGNAL_COLUMNS if c in features.columns]
    )
    return discover_alpha_from_features(
        features,
        signal_columns=columns,
        price_col=price_col,
        time_col=AVAILABILITY_TS,
        horizon=horizon,
        alpha=alpha,
        n_permutations=n_permutations,
        rng=rng,
    )
