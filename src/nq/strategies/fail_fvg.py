"""بحث Failed FVG — مسار استراتيجية منفصل فوق البنية الموحّدة.

التدفّق:

1. ``failed_fvg_features`` من MBO (سببي، بدون look-ahead).
2. تقييم ألفا عبر ``discover_alpha_from_features``.
3. فرضيات خاصة بالاستراتيجية + فرضيات أوسع (جهد/نظام السوق).
4. اختياريًا: بوابة SSL latent (``fail_fvg_ssl``) بـ asof-join خلفي فقط.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from nq.alpha.discovery import AlphaDiscovery, discover_alpha_from_features
from nq.contracts.temporal import AVAILABILITY_TS
from nq.ingestion.reader import load_mbo_frame
from nq.models.ssl_pipeline import SSLPipelineResult, run_ssl_pipeline
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.findings import Finding
from nq.simulation.fvg import failed_fvg_features
from nq.states.regimes import KMeansRegimes

_SSL_GATE_Z_COL = "z0"
_MIN_SSL_ROWS = 24
_MIN_REGIMES = 2


@dataclass(frozen=True, slots=True)
class FailFvgResearchResult:
    """مخرجات مسار Failed FVG: ميزات، ألفا، SSL اختياري، تقرير فرضيات."""

    features: pl.DataFrame
    alpha: AlphaDiscovery
    ssl: SSLPipelineResult | None
    report: ResearchReport
    signal_columns: tuple[str, ...]


def _asof_join_embeddings(features: pl.DataFrame, embeddings: pl.DataFrame) -> pl.DataFrame:
    """دمج خلفي (asof backward) — لا يُدخل تمثيلًا مستقبليًا."""
    if embeddings.height == 0 or AVAILABILITY_TS not in embeddings.columns:
        return features
    z_cols = [c for c in embeddings.columns if c.startswith("z")]
    if not z_cols:
        return features
    left = features.sort(AVAILABILITY_TS)
    right = embeddings.select([AVAILABILITY_TS, *z_cols]).sort(AVAILABILITY_TS)
    return left.join_asof(right, on=AVAILABILITY_TS, strategy="backward")


def _apply_ssl_gate(features: pl.DataFrame, *, seed: int = 0) -> pl.DataFrame:
    """``fail_fvg_ssl`` = الإشارة فقط عندما يكون النظام الكامن «غير متوازن»."""
    if _SSL_GATE_Z_COL not in features.columns:
        return features.with_columns(pl.lit(0.0).alias("fail_fvg_ssl"))

    z_cols = [c for c in features.columns if c.startswith("z")]
    mat = features.select(z_cols).fill_null(0).to_numpy().astype(np.float64)
    if mat.shape[0] < _MIN_SSL_ROWS or mat.shape[1] < 1:
        return features.with_columns(pl.col("fail_fvg").alias("fail_fvg_ssl"))

    n_regimes = min(3, mat.shape[0] // 8)
    if n_regimes < _MIN_REGIMES:
        return features.with_columns(pl.col("fail_fvg").alias("fail_fvg_ssl"))

    # fit على النصف الأول فقط ثم predict للجميع (منع تسريب مراكز المستقبل)
    split = max(n_regimes * 2, mat.shape[0] // 2)
    model = KMeansRegimes(n_regimes, seed=seed).fit(mat[:split])
    labels = model.predict(mat)
    # النظام ذو أعلى تشتت latent ≈ imbalance/expansion proxy
    spreads = []
    for k in range(n_regimes):
        members = mat[labels == k]
        spreads.append(float(np.std(members)) if members.shape[0] else 0.0)
    active = int(np.argmax(spreads))
    gate = (labels == active).astype(np.float64)
    signal = features["fail_fvg"].to_numpy().astype(np.float64) * gate
    return features.with_columns(
        pl.Series("fail_fvg_ssl", signal),
        pl.Series("ssl_regime", labels.astype(np.int64)),
    )


def _strategy_findings(
    alpha: AlphaDiscovery,
    features: pl.DataFrame,
    research: ResearchAssistant,
    signal_columns: tuple[str, ...],
) -> list[Finding]:
    """فرضيات خاصة بـ Failed FVG + فرضيات أوسع على الجهد/التغطية."""
    findings: list[Finding] = []
    n_signals = int((features["fail_fvg"] != 0).sum()) if "fail_fvg" in features.columns else 0

    # فرضية أوسع: وجود جهد مرتفع كافٍ لإنتاج إشارات
    effort = features.filter(pl.col("effort_range_ratio") > 0)
    mean_effort_val = effort["effort_range_ratio"].mean() if effort.height else 0.0
    mean_effort = float(mean_effort_val) if isinstance(mean_effort_val, (int, float)) else 0.0
    findings.append(
        research.generate_hypothesis(
            f"جهد سعري متوسط {mean_effort:.2f}×ATR يسبق إشارات Failed FVG ({n_signals} إشارة).",
            Evidence(
                id="fail_fvg:effort_coverage",
                source="fail_fvg_strategy",
                metric="effort_range_ratio",
                value=mean_effort,
                sample_size=max(effort.height, 1),
                detail="mean effort among bars with valid ATR baseline",
            ),
            category="fail_fvg",
            requires_significance=False,
        )
    )

    # فرضيات خاصة بكل عمود إشارة مُقيَّم
    for row in alpha.evaluations.iter_rows(named=True):
        name = str(row["name"])
        if name not in signal_columns:
            continue
        ic = float(row["ic"])
        pvalue = float(row["adjusted_pvalue"])
        findings.append(
            research.generate_hypothesis(
                f"إشارة '{name}' تحمل IC={ic:.3f} مقابل عوائد forward (Failed FVG).",
                Evidence(
                    id=f"fail_fvg:alpha:{name}",
                    source="fail_fvg_strategy",
                    metric="IC",
                    value=ic,
                    pvalue=pvalue,
                    sample_size=int(row["n"]),
                    detail=f"screened alpha for {name}",
                ),
                category="fail_fvg",
                requires_significance=True,
            )
        )

    if "fail_fvg_ssl" in signal_columns and "ssl_regime" in features.columns:
        gated = features.filter(pl.col("fail_fvg_ssl") != 0).height
        findings.append(
            research.generate_hypothesis(
                f"بوابة SSL أبقت {gated} إشارة Failed FVG داخل نظام latent نشط.",
                Evidence(
                    id="fail_fvg:ssl_gate",
                    source="fail_fvg_strategy",
                    metric="gated_signal_count",
                    value=float(gated),
                    sample_size=features.height,
                    detail="count of non-zero fail_fvg_ssl after causal SSL regime gate",
                ),
                category="fail_fvg",
                requires_significance=False,
            )
        )

    return findings


def run_fail_fvg_research(
    nq: pl.DataFrame | str | Path,
    *,
    use_ssl_gate: bool = True,
    ssl_window: int = 5,
    ssl_components: int = 4,
    horizon: int = 1,
    alpha: float = 0.05,
    n_permutations: int = 2000,
    max_rows: int | None = None,
    rng: np.random.Generator | None = None,
    output_dir: Path | str | None = None,
) -> FailFvgResearchResult:
    """يشغّل مسار Failed FVG الكامل: ميزات → ألفا → فرضيات → تقرير.

    مسار منفصل عن ``run_research_pipeline`` لكنه يعيد استخدام نفس طبقات
    alpha / research / models / states دون fork.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    research = ResearchAssistant(alpha=alpha)

    frame = nq if isinstance(nq, pl.DataFrame) else load_mbo_frame(nq, max_rows=max_rows)
    features = failed_fvg_features(frame)

    ssl_result: SSLPipelineResult | None = None
    signal_columns: list[str] = ["fail_fvg"]

    if use_ssl_gate and features.height >= _MIN_SSL_ROWS:
        ssl_result = run_ssl_pipeline(
            features,
            feature_columns=["fail_fvg", "effort_range_ratio", "effort_volume_ratio", "c"],
            window=ssl_window,
            n_components=ssl_components,
            alpha=alpha,
            rng=generator,
            assistant=research,
        )
        features = _asof_join_embeddings(features, ssl_result.embeddings)
        features = _apply_ssl_gate(features, seed=int(generator.integers(0, 2**31)))
        signal_columns.append("fail_fvg_ssl")
    else:
        features = features.with_columns(pl.lit(0.0).alias("fail_fvg_ssl"))

    alpha_result = discover_alpha_from_features(
        features,
        signal_columns=signal_columns,
        price_col="nq_close",
        horizon=horizon,
        execution_mode="mid",
        alpha=alpha,
        n_permutations=n_permutations,
        rng=generator,
        assistant=research,
    )

    findings = _strategy_findings(
        alpha_result, features, research, signal_columns=tuple(signal_columns)
    )
    report = research.write_report(
        findings,
        title="Failed FVG Strategy — Research Report",
    )

    result = FailFvgResearchResult(
        features=features,
        alpha=alpha_result,
        ssl=ssl_result,
        report=report,
        signal_columns=tuple(signal_columns),
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fail_fvg_report.md").write_text(report.to_markdown(), encoding="utf-8")
        features.write_parquet(out / "fail_fvg_features.parquet")
        if alpha_result.evaluations.height > 0:
            alpha_result.evaluations.write_parquet(out / "fail_fvg_alpha.parquet")

    return result


__all__ = [
    "FailFvgResearchResult",
    "run_fail_fvg_research",
]
