"""Failed FVG كطبقة بحث داخل الخط الموحّد.

أمر تشغيل منفصل (``run_fail_fvg`` / ``run_fail_fvg_research``) يمرّ بنفس
محرك المشروع كاملًا: ميزات + SSL ‖ M9 ‖ ألفا + مخرجات. يضيّق فقط أعمدة
الفرز على ``fail_fvg`` — ليس fork خارج المنظومة.

المسار الافتراضي شاشة عيّنة كاملة استكشافية؛ اختيار الفرضية بـ ``--search``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from nq.alpha.discovery import AlphaDiscovery
from nq.models.ssl_pipeline import SSLPipelineResult
from nq.research.assistant import ResearchReport
from nq.research.orchestrator import (
    PipelineConfig,
    UnifiedResearchResult,
    run_research_pipeline,
)
from nq.research.unified import UnifiedResearchReport

_FAIL_FVG_FOCUS = (
    "fail_fvg",
    "lead_lag",
    "trap_setup",
    "divergence",
    "nq_delta",
    "mnq_delta",
)

_DEFAULT_CONFIG = Path("configs/default.toml")


def _base_pipeline_config(config_path: Path | str | None) -> PipelineConfig:
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    if path.is_file():
        return PipelineConfig.from_toml(path)
    if _DEFAULT_CONFIG.is_file():
        return PipelineConfig.from_toml(_DEFAULT_CONFIG)
    return PipelineConfig()


@dataclass(frozen=True, slots=True)
class FailFvgResearchResult:
    """غلاف مريح فوق ``UnifiedResearchResult`` لتركيز Failed FVG."""

    features: pl.DataFrame
    alpha: AlphaDiscovery
    ssl: SSLPipelineResult | None
    report: ResearchReport
    unified: UnifiedResearchReport
    signal_columns: tuple[str, ...]

    @classmethod
    def from_unified(
        cls,
        result: UnifiedResearchResult,
        *,
        signal_columns: tuple[str, ...],
    ) -> FailFvgResearchResult:
        return cls(
            features=result.features,
            alpha=result.alpha,
            ssl=result.ssl,
            report=result.alpha.report,
            unified=result.report,
            signal_columns=signal_columns,
        )


def run_fail_fvg_research(
    nq: pl.DataFrame | str | Path,
    mnq: pl.DataFrame | str | Path | None = None,
    *,
    config_path: Path | str | None = None,
    use_ssl_gate: bool = True,
    ssl_window: int | None = None,
    ssl_components: int | None = None,
    horizon: int | None = None,
    alpha: float | None = None,
    n_permutations: int | None = None,
    max_rows: int | None = None,
    rng: np.random.Generator | None = None,
    output_dir: Path | str | None = None,
    quiet: bool = False,
) -> FailFvgResearchResult:
    """يشغّل Failed FVG عبر الخط الموحّد (أمر تشغيل منفصل — داخل المنظومة).

    يرث ``interval_ns`` / فلتر العمق / bottom_book من ``config_path`` (أو default.toml).
    ``use_ssl_gate`` اسم توافق؛ البوابة عبر ``ssl_mode`` داخل الخط الموحّد.
    """
    _ = use_ssl_gate  # التوافق مع الواجهة السابقة؛ البوابة عبر ssl_mode في الخط الموحّد
    base = _base_pipeline_config(config_path)
    cfg = replace(
        base,
        include_failed_fvg=True,
        include_auction_vp=False,  # تركيز فرز FVG؛ الخط العام ما زال يجمع الكل
        include_failed_breakout=False,
        cross_market_mode="nq_only" if mnq is None else "dual",
        max_rows=max_rows if max_rows is not None else base.max_rows,
        horizon=int(horizon) if horizon is not None else base.horizon,
        alpha=float(alpha) if alpha is not None else base.alpha,
        n_permutations=int(n_permutations) if n_permutations is not None else base.n_permutations,
        ssl_window=int(ssl_window) if ssl_window is not None else base.ssl_window,
        ssl_components=int(ssl_components) if ssl_components is not None else base.ssl_components,
        signal_columns=_FAIL_FVG_FOCUS,
        quiet=quiet,
    )
    partner = mnq if mnq is not None else nq
    result = run_research_pipeline(
        nq,
        partner,
        config=cfg,
        signal_columns=_FAIL_FVG_FOCUS,
        output_dir=output_dir,
        rng=rng,
    )
    return FailFvgResearchResult.from_unified(result, signal_columns=_FAIL_FVG_FOCUS)


__all__ = [
    "FailFvgResearchResult",
    "run_fail_fvg_research",
]
