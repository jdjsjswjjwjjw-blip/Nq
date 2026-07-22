"""خط SSL التأسيسي الكامل (Self-Supervised Foundation Model Pipeline).

يشغّل walk-forward على الميزات: تطبيع سببي، PCAEncoder، نمذجة مُقنّعة،
World Model، وتعلّم تبايني — ويُنتج تمثيلات كامنة + تقرير موثّق بالأدلّة.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.contracts.temporal import AVAILABILITY_TS
from nq.core.temporal_policy import TemporalPolicy
from nq.models.contrastive import augment_windows, info_nce_loss
from nq.models.encoder import PCAEncoder
from nq.models.masking import mask_matrix, masked_reconstruction_error
from nq.models.masking_structural import batch_masked_mse, structural_mask_batch
from nq.models.preprocessing import CausalStandardScaler
from nq.models.splitting import WalkForwardFold, purged_walk_forward_split
from nq.models.tick_stream import TICK_FEATURE_NAMES, build_tick_stream
from nq.models.windowing import TickSequenceDataset, build_sequences, build_tick_sequences
from nq.models.world_model import NextStatePredictor, r2_score
from nq.research.assistant import ResearchAssistant, ResearchReport
from nq.research.evidence import Evidence
from nq.research.findings import Finding

FloatArray = npt.NDArray[np.float64]

_MIN_SSL_SAMPLES = 12
_MIN_CONTRASTIVE_BATCH = 4
_MIN_SSL_TEST = 4
_MIN_WM_SAMPLES = 3
_DEFAULT_WINDOW = 5

_METRICS_SCHEMA = {
    "fold": pl.Int64(),
    "masked_mse": pl.Float64(),
    "world_model_r2": pl.Float64(),
    "contrastive_loss": pl.Float64(),
}


@dataclass(frozen=True, slots=True)
class SSLPipelineResult:
    """مخرجات خط SSL: مقاييس، تمثيلات كامنة، وتقرير موثّق."""

    metrics: pl.DataFrame
    embeddings: pl.DataFrame
    report: ResearchReport


@dataclass(frozen=True, slots=True)
class _FoldMetrics:
    fold: int
    masked_mse: float
    world_model_r2: float
    contrastive_loss: float
    embeddings: list[dict[str, float | int]]


def _empty_ssl_result(research: ResearchAssistant) -> SSLPipelineResult:
    report = research.write_report([], title="SSL Foundation Model — Research Report")
    return SSLPipelineResult(
        metrics=pl.DataFrame(schema=_METRICS_SCHEMA),
        embeddings=pl.DataFrame({AVAILABILITY_TS: pl.Series([], dtype=pl.Int64())}),
        report=report,
    )


def _evaluate_ssl_fold(
    fold_idx: int,
    train: FloatArray,
    test: FloatArray,
    test_times: np.ndarray,
    *,
    n_components: int,
    mask_ratio: float,
    generator: np.random.Generator,
) -> _FoldMetrics | None:
    if train.shape[0] < _MIN_SSL_SAMPLES or test.shape[0] < _MIN_SSL_TEST:
        return None
    scaler = CausalStandardScaler().fit(train)
    x_train = scaler.transform(train)
    x_test = scaler.transform(test)
    k = min(n_components, x_train.shape[1], x_train.shape[0] - 1)
    encoder = PCAEncoder(k).fit(x_train)
    masked = mask_matrix(x_test, mask_ratio=mask_ratio, rng=generator)
    mse = masked_reconstruction_error(encoder.reconstruct(masked.masked), masked)
    z_train = encoder.transform(x_train)
    z_test = encoder.transform(x_test)
    wm_r2 = 0.0
    if z_train.shape[0] >= _MIN_WM_SAMPLES and z_test.shape[0] >= _MIN_WM_SAMPLES:
        y_train = z_train[1:]
        predictor = NextStatePredictor().fit(z_train[:-1], y_train)
        # OOS R²: خط الأساس = متوسط أهداف التدريب (ليس متوسط الاختبار)
        wm_r2 = max(
            r2_score(
                z_test[1:],
                predictor.predict(z_test[:-1]),
                baseline_mean=y_train.mean(axis=0),
            ),
            0.0,
        )
    contrastive = 0.0
    if z_test.shape[0] >= _MIN_CONTRASTIVE_BATCH:
        contrastive = info_nce_loss(z_test, augment_windows(z_test, rng=generator), temperature=0.1)
    emb_rows: list[dict[str, float | int]] = []
    for i, ts in enumerate(test_times):
        row: dict[str, float | int] = {AVAILABILITY_TS: int(ts)}
        for j, val in enumerate(z_test[i]):
            row[f"z{j}"] = float(val)
        emb_rows.append(row)
    return _FoldMetrics(fold_idx, mse, wm_r2, contrastive, emb_rows)


def _ssl_findings(metrics: pl.DataFrame, research: ResearchAssistant) -> list[Finding]:
    if metrics.height == 0:
        return []
    mean_mse = float(metrics.select(pl.col("masked_mse").mean()).item())
    mean_r2 = float(metrics.select(pl.col("world_model_r2").mean()).item())
    mean_nce = float(metrics.select(pl.col("contrastive_loss").mean()).item())
    n = int(metrics.height)
    return [
        research.generate_hypothesis(
            f"SSL يعيد بناء الميزات المُقنّعة (masked MSE={mean_mse:.4g}).",
            Evidence("ssl:masked_mse", "ssl_pipeline", "masked_mse", mean_mse, sample_size=n),
            category="ssl",
            requires_significance=False,
        ),
        research.generate_hypothesis(
            f"World Model يتنبّأ بالحالة الكامنة التالية (OOS R²={mean_r2:.3f}).",
            Evidence(
                "ssl:world_model_r2", "ssl_pipeline", "world_model_r2", mean_r2, sample_size=n
            ),
            category="ssl",
            requires_significance=False,
        ),
        research.generate_hypothesis(
            f"تمثيل SSL متماسك تباينيًا (InfoNCE={mean_nce:.3f}).",
            Evidence(
                "ssl:contrastive_loss", "ssl_pipeline", "contrastive_loss", mean_nce, sample_size=n
            ),
            category="ssl",
            requires_significance=False,
        ),
    ]


def _feature_columns(frame: pl.DataFrame, *, max_null_frac: float = 0.05) -> list[str]:
    if frame.height == 0:
        return []
    excluded = {
        AVAILABILITY_TS,
        "bucket_start",
        "bucket_end",
        "nq_close",
        "mnq_close",
        "session_phase",
        "minutes_since_rth_open",
    }
    cols: list[str] = []
    for c, dtype in frame.schema.items():
        if c in excluded or not dtype.is_numeric():
            continue
        if frame[c].null_count() / frame.height <= max_null_frac:
            cols.append(c)
    return cols


def _walk_forward_folds(
    times: np.ndarray,
    *,
    n_splits: int,
    embargo: int,
    purge_samples: int,
) -> list[WalkForwardFold]:
    n = times.shape[0]
    if n < _MIN_SSL_SAMPLES:
        return []
    for splits in range(min(n_splits, n - 1), 0, -1):
        if n < splits + 1:
            continue
        try:
            folds = purged_walk_forward_split(
                times,
                n_splits=splits,
                embargo=embargo,
                purge_samples=purge_samples,
            )
        except ValueError:
            continue
        if folds:
            return folds
    return []


def run_ssl_pipeline(
    features: pl.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    window: int = _DEFAULT_WINDOW,
    n_components: int = 4,
    n_splits: int = 3,
    embargo: int | None = None,
    purge_samples: int | None = None,
    interval_ns: int | None = None,
    mask_ratio: float = 0.2,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    assistant: ResearchAssistant | None = None,
    progress: object | None = None,
) -> SSLPipelineResult:
    """يشغّل SSL walk-forward على إطار ميزات ويكتب تقريرًا موثّقًا."""
    generator = rng if rng is not None else np.random.default_rng(0)
    research = assistant if assistant is not None else ResearchAssistant(alpha=alpha)
    log = progress

    cols = list(feature_columns) if feature_columns is not None else _feature_columns(features)
    if not cols or features.height < window:
        if log is not None:
            log.op(  # type: ignore[union-attr]
                f"SSL-bucket: تخطّي (cols={len(cols)} · rows={features.height} · window={window})"
            )
        return _empty_ssl_result(research)

    if log is not None:
        log.op(  # type: ignore[union-attr]
            f"SSL-bucket: تحضير نوافذ window={window} من {features.height:,} صف · "
            f"features={len(cols)}"
        )
    work = features.select([AVAILABILITY_TS, *cols])
    for col in cols:
        work = work.with_columns(pl.col(col).fill_null(0).alias(col))

    sequences = build_sequences(work, feature_columns=cols, window=window)
    if len(sequences) < _MIN_SSL_SAMPLES:
        if log is not None:
            log.op(f"SSL-bucket: عيّنات غير كافية ({len(sequences)})")  # type: ignore[union-attr]
        return _empty_ssl_result(research)

    policy = TemporalPolicy.for_run(
        interval_ns=interval_ns if interval_ns is not None else 1,
        window=window,
    )
    embargo_val = (
        embargo
        if embargo is not None
        else (
            policy.embargo_time_units(interval_ns=interval_ns, times=sequences.times)
            if interval_ns is not None
            else 0
        )
    )
    purge_val = purge_samples if purge_samples is not None else policy.purge_samples()

    flat = sequences.flatten()
    folds = _walk_forward_folds(
        sequences.times,
        n_splits=n_splits,
        embargo=embargo_val,
        purge_samples=purge_val,
    )
    if log is not None:
        log.op(  # type: ignore[union-attr]
            f"SSL-bucket: walk-forward {len(folds)} طيّات · sequences={len(sequences):,}"
        )
    fold_rows: list[dict[str, float | int]] = []
    embedding_rows: list[dict[str, float | int]] = []

    for fold_idx, fold in enumerate(folds):
        if log is not None:
            log.op(  # type: ignore[union-attr]
                f"SSL-bucket fold {fold_idx + 1}/{len(folds)} "
                f"(train={len(fold.train_idx):,} · test={len(fold.test_idx):,})"
            )
        result = _evaluate_ssl_fold(
            fold_idx,
            flat[fold.train_idx],
            flat[fold.test_idx],
            sequences.times[fold.test_idx],
            n_components=n_components,
            mask_ratio=mask_ratio,
            generator=generator,
        )
        if result is None:
            if log is not None:
                log.op(f"SSL-bucket fold {fold_idx + 1}: تخطّي (نتيجة فارغة)")  # type: ignore[union-attr]
            continue
        fold_rows.append(
            {
                "fold": result.fold,
                "masked_mse": result.masked_mse,
                "world_model_r2": result.world_model_r2,
                "contrastive_loss": result.contrastive_loss,
            }
        )
        embedding_rows.extend(result.embeddings)

    metrics = pl.DataFrame(fold_rows) if fold_rows else pl.DataFrame(schema=_METRICS_SCHEMA)
    embeddings = (
        pl.DataFrame(embedding_rows)
        if embedding_rows
        else pl.DataFrame({AVAILABILITY_TS: pl.Series([], dtype=pl.Int64())})
    )
    if log is not None:
        log.op(f"SSL-bucket انتهى — folds={metrics.height} · emb={embeddings.height:,}")  # type: ignore[union-attr]
    findings = _ssl_findings(metrics, research)
    report = research.write_report(findings, title="SSL Foundation Model — Research Report")
    return SSLPipelineResult(metrics=metrics, embeddings=embeddings, report=report)


def _evaluate_ssl_tick_fold(
    fold_idx: int,
    train: FloatArray,
    test: FloatArray,
    test_times: np.ndarray,
    test_paths: np.ndarray,
    test_phases: np.ndarray,
    *,
    window: int,
    n_components: int,
    generator: np.random.Generator,
) -> _FoldMetrics | None:
    """يقيّم طيّة tick SSL بإخفاء هيكلي (لا ``mask_matrix`` عشوائي)."""
    n_feat = len(TICK_FEATURE_NAMES)
    if train.shape[0] < _MIN_SSL_SAMPLES or test.shape[0] < _MIN_SSL_TEST:
        return None

    test_3d = test.reshape(test.shape[0], window, n_feat)
    scaler = CausalStandardScaler().fit(train)
    x_train = scaler.transform(train)
    x_test = scaler.transform(test)
    k = min(n_components, x_train.shape[1], x_train.shape[0] - 1)
    encoder = PCAEncoder(k).fit(x_train)

    masked_batch = structural_mask_batch(
        test_3d,
        mask_paths=test_paths,
        market_phases=test_phases,
    )
    recon_flat = encoder.reconstruct(x_test)
    recon_3d = recon_flat.reshape(test_3d.shape)
    mse = batch_masked_mse(recon_3d, masked_batch)

    z_train = encoder.transform(x_train)
    z_test = encoder.transform(x_test)
    wm_r2 = 0.0
    if z_train.shape[0] >= _MIN_WM_SAMPLES and z_test.shape[0] >= _MIN_WM_SAMPLES:
        y_train = z_train[1:]
        predictor = NextStatePredictor().fit(z_train[:-1], y_train)
        # OOS R²: خط الأساس = متوسط أهداف التدريب (ليس متوسط الاختبار)
        wm_r2 = max(
            r2_score(
                z_test[1:],
                predictor.predict(z_test[:-1]),
                baseline_mean=y_train.mean(axis=0),
            ),
            0.0,
        )
    contrastive = 0.0
    if z_test.shape[0] >= _MIN_CONTRASTIVE_BATCH:
        contrastive = info_nce_loss(z_test, augment_windows(z_test, rng=generator), temperature=0.1)
    emb_rows: list[dict[str, float | int]] = []
    for i, ts in enumerate(test_times):
        row: dict[str, float | int] = {AVAILABILITY_TS: int(ts)}
        for j, val in enumerate(z_test[i]):
            row[f"z{j}"] = float(val)
        emb_rows.append(row)
    return _FoldMetrics(fold_idx, mse, wm_r2, contrastive, emb_rows)


def run_ssl_tick_pipeline(
    nq: pl.DataFrame,
    mnq: pl.DataFrame,
    *,
    window: int = _DEFAULT_WINDOW,
    n_components: int = 4,
    n_splits: int = 3,
    embargo: int | None = None,
    purge_samples: int | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    assistant: ResearchAssistant | None = None,
    progress: object | None = None,
) -> SSLPipelineResult:
    """SSL على tick/event: دفتر حي + ميزات inline + إخفاء هيكلي (الأبعاد 1–6).

    يُكمّل ``run_ssl_pipeline`` (bucket) ولا يستبدله.
    """

    generator = rng if rng is not None else np.random.default_rng(0)
    research = assistant if assistant is not None else ResearchAssistant(alpha=alpha)
    log = progress

    if log is not None:
        log.op("SSL-tick: بناء tick_stream للتمثيلات")  # type: ignore[union-attr]
    stream = build_tick_stream(nq, mnq, progress=progress)
    if stream.height < window:
        if log is not None:
            log.op(f"SSL-tick: أحداث غير كافية ({stream.height} < window={window})")  # type: ignore[union-attr]
        return _empty_ssl_result(research)

    if log is not None:
        log.op(f"SSL-tick: بناء نوافذ window={window} من {stream.height:,} حدث")  # type: ignore[union-attr]
    sequences: TickSequenceDataset = build_tick_sequences(
        stream.frame,
        feature_columns=list(TICK_FEATURE_NAMES),
        window=window,
    )
    if len(sequences) < _MIN_SSL_SAMPLES:
        if log is not None:
            log.op(f"SSL-tick: عيّنات غير كافية ({len(sequences)})")  # type: ignore[union-attr]
        return _empty_ssl_result(research)

    policy = TemporalPolicy.for_run(interval_ns=1, window=window)
    embargo_val = (
        embargo
        if embargo is not None
        else policy.embargo_time_units(interval_ns=1, times=sequences.times)
    )
    purge_val = purge_samples if purge_samples is not None else policy.purge_samples()

    flat = sequences.flatten()
    folds = _walk_forward_folds(
        sequences.times,
        n_splits=n_splits,
        embargo=embargo_val,
        purge_samples=purge_val,
    )
    if log is not None:
        log.op(f"SSL-tick: walk-forward {len(folds)} طيّات · sequences={len(sequences):,}")  # type: ignore[union-attr]
    fold_rows: list[dict[str, float | int]] = []
    embedding_rows: list[dict[str, float | int]] = []

    for fold_idx, fold in enumerate(folds):
        if log is not None:
            log.op(  # type: ignore[union-attr]
                f"SSL-tick fold {fold_idx + 1}/{len(folds)} "
                f"(train={len(fold.train_idx):,} · test={len(fold.test_idx):,})"
            )
        result = _evaluate_ssl_tick_fold(
            fold_idx,
            flat[fold.train_idx],
            flat[fold.test_idx],
            sequences.times[fold.test_idx],
            sequences.mask_paths[fold.test_idx],
            sequences.market_phases[fold.test_idx],
            window=window,
            n_components=n_components,
            generator=generator,
        )
        if result is None:
            if log is not None:
                log.op(f"SSL-tick fold {fold_idx + 1}: تخطّي (نتيجة فارغة)")  # type: ignore[union-attr]
            continue
        fold_rows.append(
            {
                "fold": result.fold,
                "masked_mse": result.masked_mse,
                "world_model_r2": result.world_model_r2,
                "contrastive_loss": result.contrastive_loss,
            }
        )
        embedding_rows.extend(result.embeddings)

    metrics = pl.DataFrame(fold_rows) if fold_rows else pl.DataFrame(schema=_METRICS_SCHEMA)
    embeddings = (
        pl.DataFrame(embedding_rows)
        if embedding_rows
        else pl.DataFrame({AVAILABILITY_TS: pl.Series([], dtype=pl.Int64())})
    )
    findings = _ssl_findings(metrics, research)
    report = research.write_report(
        findings, title="SSL Tick/Event Foundation Model — Research Report"
    )
    return SSLPipelineResult(metrics=metrics, embeddings=embeddings, report=report)
