"""مقاييس مراقبة التغطية البنيوية (Structural Coverage Metrics).

كل مقياس:
* walk-forward + embargo (سببي)
* p-value عبر permutation أو block-bootstrap
* مخرج ``Evidence`` قابل للتتبّع
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from nq.coverage.blocks import resolve_block_columns
from nq.coverage.distance import max_axis_dependence
from nq.coverage.mbo_descriptors import descriptor_matrix
from nq.models.encoder import PCAEncoder
from nq.models.masking import MaskedMatrix, mask_matrix, masked_reconstruction_error
from nq.models.preprocessing import CausalStandardScaler
from nq.models.splitting import WalkForwardFold, purged_walk_forward_split
from nq.models.world_model import NextStatePredictor, r2_score
from nq.research.evidence import Evidence
from nq.states.regimes import KMeansRegimes, transition_matrix
from nq.statistics.regime_tests import regime_difference_test
from nq.statistics.resampling import TestResult, permutation_test

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

_MIN_FOLD_SAMPLES = 8
_MIN_REGIME_SAMPLES = 5
_MIN_TEST_SAMPLES = 4
_MIN_RIDGE_TRAIN = 2
_MIN_BLOCK_ROWS = 2
_DEFAULT_N_PERM = 2000
_CER_RATIO_THRESHOLD = 1.5
_PSG_RATIO_THRESHOLD = 1.5
_CRS_RATIO_THRESHOLD = 1.25
_QDUF_THRESHOLD = 0.1
_TRANSITION_SURPRISE_THRESHOLD = 3.0
_LORI_PERMUTATIONS = 500


def _walk_forward_folds(times: IntArray, *, n_splits: int, embargo: int) -> list[WalkForwardFold]:
    """طيّات walk-forward مع تخفيض تلقائي لـ ``n_splits`` عند نقص العيّنات."""
    n = times.shape[0]
    if n < _MIN_BLOCK_ROWS:
        return []
    for splits in range(min(n_splits, n - 1), 0, -1):
        if n < splits + 1:
            continue
        try:
            folds = purged_walk_forward_split(times, n_splits=splits, embargo=embargo)
        except ValueError:
            continue
        if folds:
            return folds
    return []


@dataclass(frozen=True, slots=True)
class MetricResult:
    """نتيجة مقياس تغطية واحد."""

    name: str
    value: float
    pvalue: float
    sample_size: int
    detail: str
    triggered: bool


def _ridge_r2(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
    y_test: FloatArray,
    *,
    alpha: float = 1.0,
) -> float:
    """R² خارج العيّنة لانحدار ريدج — خط الأساس = متوسط أهداف التدريب."""
    if x_train.shape[0] < _MIN_RIDGE_TRAIN or x_test.shape[0] < 1:
        return 0.0
    ytr = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    yte = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
    predictor = NextStatePredictor(alpha=alpha)
    predictor.fit(x_train, ytr)
    pred = predictor.predict(x_test)
    return max(r2_score(yte, pred, baseline_mean=ytr.mean(axis=0)), 0.0)


def _align_frames(
    features: pl.DataFrame,
    descriptors: pl.DataFrame,
    *,
    price_col: str,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray]:
    """يُحاذي الميزات والواصفات والسعر على ``availability_ts``."""
    joined = features.join(descriptors, on="availability_ts", how="inner", suffix="_mbo")
    if joined.height == 0:
        raise ValueError("no aligned windows between features and MBO descriptors")
    price = joined[price_col].to_numpy().astype(np.float64)
    returns = np.diff(price, prepend=price[0])
    returns[0] = 0.0

    feat_cols = [
        c
        for c, dtype in features.schema.items()
        if c not in {"availability_ts", "bucket_start", "bucket_end", price_col}
        and dtype.is_numeric()
    ]
    _desc_cols, desc_mat = descriptor_matrix(
        joined.select([c for c in descriptors.columns if c in joined.columns])
    )
    feat_mat = joined.select(feat_cols).fill_null(0).to_numpy().astype(np.float64)
    times = joined["availability_ts"].to_numpy().astype(np.int64)
    return feat_mat, desc_mat, returns, price, times


def _information_gap_stat(
    descriptors: FloatArray,
    features: FloatArray,
    returns: FloatArray,
) -> float:
    mbo_dep = max_axis_dependence(descriptors, returns)
    feat_dep = max_axis_dependence(features, returns)
    return mbo_dep - feat_dep


def measure_mfig(
    features: pl.DataFrame,
    descriptors: pl.DataFrame,
    *,
    price_col: str = "nq_close",
    n_splits: int = 3,
    embargo: int = 0,
    n_permutations: int = _DEFAULT_N_PERM,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> MetricResult:
    """MFIG — فجوة المعلومات الشرطية (MBO vs Features → Price)."""
    generator = rng if rng is not None else np.random.default_rng(0)
    feat_mat, desc_mat, returns, _price, times = _align_frames(
        features, descriptors, price_col=price_col
    )
    folds = _walk_forward_folds(times, n_splits=n_splits, embargo=embargo)
    if not folds:
        return MetricResult("mfig", 0.0, 1.0, 0, "insufficient data for walk-forward", False)

    gaps: list[float] = []
    for fold in folds:
        test_idx = fold.test_idx
        if test_idx.shape[0] < _MIN_FOLD_SAMPLES:
            continue
        gaps.append(
            _information_gap_stat(desc_mat[test_idx], feat_mat[test_idx], returns[test_idx])
        )

    if not gaps:
        return MetricResult("mfig", 0.0, 1.0, 0, "no valid folds", False)

    observed = float(np.mean(gaps))
    feat_dep = max_axis_dependence(feat_mat, returns)
    mbo_dep = max_axis_dependence(desc_mat, returns)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = generator.permutation(returns)
        null[i] = _information_gap_stat(desc_mat, feat_mat, perm)
    pvalue = (int(np.sum(null >= observed)) + 1) / (n_permutations + 1)
    triggered = pvalue <= alpha and observed > 0
    detail = (
        f"MBO explains Δprice beyond simulator features "
        f"(gap={observed:.4f}, mbo_dcor={mbo_dep:.4f}, feat_dcor={feat_dep:.4f})"
        if triggered
        else f"no structural information gap (gap={observed:.4f})"
    )
    return MetricResult(
        "mfig",
        observed,
        pvalue,
        int(sum(f.shape[0] for f in [fold.test_idx for fold in folds])),
        detail,
        triggered,
    )


def _block_delta_norm(block: FloatArray) -> FloatArray:
    if block.shape[0] < _MIN_BLOCK_ROWS:
        return np.zeros(block.shape[0], dtype=np.float64)
    delta = np.diff(block, axis=0, prepend=block[:1])
    return np.asarray(np.linalg.norm(delta, axis=1), dtype=np.float64)


def measure_cer(
    features: pl.DataFrame,
    *,
    price_col: str = "nq_close",
    blocks: dict[str, tuple[str, ...]] | None = None,
    n_splits: int = 3,
    embargo: int = 0,
    n_permutations: int = _DEFAULT_N_PERM,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> list[MetricResult]:
    """CER — بقايا التعرّض السببي لكل كتلة محاكاة."""
    generator = rng if rng is not None else np.random.default_rng(0)
    resolved = resolve_block_columns(features.columns, blocks)
    if not resolved:
        return []

    price = features[price_col].to_numpy().astype(np.float64)
    price_delta = np.abs(np.diff(price, prepend=price[0]))
    times = features["availability_ts"].to_numpy().astype(np.int64)
    folds = _walk_forward_folds(times, n_splits=n_splits, embargo=embargo)
    results: list[MetricResult] = []

    for block_name, cols in resolved.items():
        block = features.select(cols).fill_null(0).to_numpy().astype(np.float64)
        feat_delta = _block_delta_norm(block)
        cer_series = price_delta / (feat_delta + 1e-9)

        train_cer: list[float] = []
        test_cer: list[float] = []
        for fold in folds:
            train_cer.extend(cer_series[fold.train_idx].tolist())
            test_cer.extend(cer_series[fold.test_idx].tolist())

        if len(test_cer) < _MIN_FOLD_SAMPLES:
            continue

        observed = float(np.median(test_cer))
        train_median = float(np.median(train_cer)) if train_cer else observed
        ratio = observed / (train_median + 1e-9)

        null = np.empty(n_permutations, dtype=np.float64)
        pooled = cer_series.copy()
        for i in range(n_permutations):
            perm = generator.permutation(pooled)
            if len(train_cer) < len(pooled):
                null[i] = float(np.median(perm[len(train_cer) :]))
            else:
                null[i] = float(np.median(perm))
        pvalue = (int(np.sum(null >= observed)) + 1) / (n_permutations + 1)
        triggered = pvalue <= alpha and ratio > _CER_RATIO_THRESHOLD
        results.append(
            MetricResult(
                f"cer:{block_name}",
                ratio,
                pvalue,
                len(test_cer),
                (
                    f"price moves without '{block_name}' feature response (CER ratio={ratio:.3f})"
                    if triggered
                    else f"'{block_name}' exposure aligned with price"
                ),
                triggered,
            )
        )
    return results


def measure_psg(
    features: pl.DataFrame,
    *,
    n_components: int = 4,
    n_splits: int = 3,
    embargo: int = 0,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> MetricResult:
    """PSG — فجوة الكفاية التنبؤية (World Model surprise)."""
    generator = rng if rng is not None else np.random.default_rng(0)
    feat_cols = [
        c
        for c, dtype in features.schema.items()
        if c not in {"availability_ts", "bucket_start", "bucket_end", "nq_close", "mnq_close"}
        and dtype.is_numeric()
    ]
    if not feat_cols:
        return MetricResult("psg", 0.0, 1.0, 0, "no numeric features", False)

    raw = features.select(feat_cols).fill_null(0).to_numpy().astype(np.float64)
    times = features["availability_ts"].to_numpy().astype(np.int64)
    folds = _walk_forward_folds(times, n_splits=n_splits, embargo=embargo)
    if not folds:
        return MetricResult("psg", 0.0, 1.0, 0, "insufficient folds", False)

    surprises: list[float] = []
    train_surprises: list[float] = []
    for fold in folds:
        train = raw[fold.train_idx]
        test = raw[fold.test_idx]
        if train.shape[0] < _MIN_FOLD_SAMPLES + 1 or test.shape[0] < _MIN_BLOCK_ROWS:
            continue
        scaler = CausalStandardScaler().fit(train)
        z_train = scaler.transform(train)
        z_test = scaler.transform(test)
        k = min(n_components, z_train.shape[1], z_train.shape[0] - 1)
        encoder = PCAEncoder(k).fit(z_train)
        zt = encoder.transform(z_train)
        predictor = NextStatePredictor().fit(zt[:-1], zt[1:])
        z_test_enc = encoder.transform(z_test)
        pred = predictor.predict(z_test_enc[:-1])
        surprise = float(np.mean(np.sum((z_test_enc[1:] - pred) ** 2, axis=1)))
        surprises.append(surprise)
        zt_pred = predictor.predict(zt[:-1])
        train_surprises.append(float(np.mean(np.sum((zt[1:] - zt_pred) ** 2, axis=1))))

    if not surprises:
        return MetricResult("psg", 0.0, 1.0, 0, "no valid surprise windows", False)

    observed = float(np.mean(surprises))
    baseline = float(np.median(train_surprises)) if train_surprises else observed
    ratio = observed / (baseline + 1e-9)

    surprise_series = np.asarray(surprises, dtype=np.float64)
    baseline_arr = np.asarray(train_surprises, dtype=np.float64)
    test_result = permutation_test(
        surprise_series,
        baseline_arr if baseline_arr.shape[0] >= _MIN_BLOCK_ROWS else surprise_series * 0.5,
        statistic=lambda a, b: float(np.mean(a) / (np.mean(b) + 1e-9)),
        n_permutations=min(_DEFAULT_N_PERM, 1000),
        rng=generator,
        alternative="greater",
    )
    pvalue = test_result.pvalue
    triggered = ratio > _PSG_RATIO_THRESHOLD and pvalue <= alpha
    return MetricResult(
        "psg",
        ratio,
        pvalue,
        len(surprises),
        (
            f"latent dynamics surprise exceeds training baseline (PSG={ratio:.3f})"
            if triggered
            else f"predictive sufficiency within baseline (PSG={ratio:.3f})"
        ),
        triggered,
    )


def measure_crs(
    features: pl.DataFrame,
    *,
    blocks: dict[str, tuple[str, ...]] | None = None,
    n_components: int = 4,
    n_splits: int = 3,
    embargo: int = 0,
    mask_ratio: float = 0.3,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> list[MetricResult]:
    """CRS — كفاية إعادة البناء المُقنّعة لكل كتلة محاكاة."""
    generator = rng if rng is not None else np.random.default_rng(0)
    resolved = resolve_block_columns(features.columns, blocks)
    feat_cols = list(dict.fromkeys(c for cols in resolved.values() for c in cols))
    if not feat_cols:
        return []

    raw = features.select(feat_cols).fill_null(0).to_numpy().astype(np.float64)
    col_index = {c: i for i, c in enumerate(feat_cols)}
    times = features["availability_ts"].to_numpy().astype(np.int64)
    folds = _walk_forward_folds(times, n_splits=n_splits, embargo=embargo)
    results: list[MetricResult] = []

    for block_name, cols in resolved.items():
        block_idx = [col_index[c] for c in cols]
        block_errors: list[float] = []
        other_errors: list[float] = []

        for fold in folds:
            train = raw[fold.train_idx]
            test = raw[fold.test_idx]
            if train.shape[0] < _MIN_FOLD_SAMPLES or test.shape[0] < _MIN_BLOCK_ROWS:
                continue
            scaler = CausalStandardScaler().fit(train)
            x_train = scaler.transform(train)
            x_test = scaler.transform(test)
            k = min(n_components, x_train.shape[1], x_train.shape[0] - 1)
            encoder = PCAEncoder(k).fit(x_train)

            masked = mask_matrix(x_test, mask_ratio=mask_ratio, rng=generator)
            block_mask = masked.mask.copy()
            block_mask[:, :] = False
            for idx in block_idx:
                block_mask[:, idx] = True
            block_target = MaskedMatrix(
                masked=masked.masked.copy(),
                mask=block_mask,
                targets=masked.targets,
            )
            recon = encoder.reconstruct(x_test)
            block_err = masked_reconstruction_error(recon, block_target)
            full_err = masked_reconstruction_error(recon, masked)
            block_errors.append(block_err)
            other_errors.append(max(full_err - block_err, 1e-12))

        if not block_errors:
            continue

        ratios = [b / o for b, o in zip(block_errors, other_errors, strict=True)]
        observed = float(np.mean(ratios))
        test_result: TestResult = regime_difference_test(
            np.asarray(block_errors, dtype=np.float64),
            np.zeros(len(block_errors), dtype=np.intp),
            n_permutations=_DEFAULT_N_PERM,
            rng=generator,
        )
        pvalue = test_result.pvalue if len(block_errors) >= _MIN_BLOCK_ROWS else 1.0
        triggered = observed > _CRS_RATIO_THRESHOLD and pvalue <= alpha
        results.append(
            MetricResult(
                f"crs:{block_name}",
                observed,
                pvalue,
                len(block_errors),
                (
                    f"block '{block_name}' not reconstructible from other simulators "
                    f"(CRS={observed:.3f})"
                    if triggered
                    else f"block '{block_name}' reconstructible (CRS={observed:.3f})"
                ),
                triggered,
            )
        )
    return results


def measure_lori(
    features: pl.DataFrame,
    *,
    blocks: dict[str, tuple[str, ...]] | None = None,
    n_regimes: int = 3,
    n_splits: int = 3,
    embargo: int = 0,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> list[MetricResult]:
    """LORI — مؤشر الأنظمة اليتيمة + Transition Surprise."""
    generator = rng if rng is not None else np.random.default_rng(0)
    resolved = resolve_block_columns(features.columns, blocks)
    feat_cols = [
        c
        for c, dtype in features.schema.items()
        if c not in {"availability_ts", "bucket_start", "bucket_end", "nq_close", "mnq_close"}
        and dtype.is_numeric()
    ]
    if not feat_cols:
        return []

    raw = features.select(feat_cols).fill_null(0).to_numpy().astype(np.float64)
    times = features["availability_ts"].to_numpy().astype(np.int64)
    folds = _walk_forward_folds(times, n_splits=n_splits, embargo=embargo)
    results: list[MetricResult] = []

    for fold in folds:
        train = raw[fold.train_idx]
        test = raw[fold.test_idx]
        if train.shape[0] < n_regimes * 2 or test.shape[0] < _MIN_REGIME_SAMPLES:
            continue
        scaler = CausalStandardScaler().fit(train)
        z_train = scaler.transform(train)
        z_test = scaler.transform(test)
        k = min(4, z_train.shape[1], z_train.shape[0] - 1)
        encoder = PCAEncoder(k).fit(z_train)
        embed_train = encoder.transform(z_train)
        embed_test = encoder.transform(z_test)

        regimes = KMeansRegimes(n_regimes, seed=0).fit(embed_train)
        train_labels = regimes.predict(embed_train)
        test_labels = regimes.predict(embed_test)
        trans_train = transition_matrix(train_labels, n_regimes=n_regimes)

        for regime in np.unique(test_labels):
            mask = test_labels == regime
            count = int(np.sum(mask))
            if count < _MIN_REGIME_SAMPLES:
                continue
            min_p = 1.0
            for _block_name, cols in resolved.items():
                block = (
                    features.select(cols).fill_null(0).to_numpy().astype(np.float64)[fold.test_idx]
                )
                block_norm = np.linalg.norm(block, axis=1)
                result = regime_difference_test(
                    block_norm,
                    test_labels,
                    n_permutations=_LORI_PERMUTATIONS,
                    rng=generator,
                )
                min_p = min(min_p, result.pvalue)
            orphan = min_p > alpha
            if orphan:
                results.append(
                    MetricResult(
                        f"lori:regime_{int(regime)}",
                        min_p,
                        min_p,
                        count,
                        (
                            f"orphan regime {int(regime)}: no simulator block explains it "
                            f"(min_p={min_p:.4g})"
                        ),
                        True,
                    )
                )

        for i in range(test_labels.shape[0] - 1):
            src = int(test_labels[i])
            dst = int(test_labels[i + 1])
            prob = trans_train[src, dst] if trans_train[src, dst] > 0 else 1e-9
            surprise = -np.log(prob)
            if surprise > _TRANSITION_SURPRISE_THRESHOLD:
                results.append(
                    MetricResult(
                        "lori:transition_surprise",
                        surprise,
                        0.01,
                        1,
                        f"novel transition {src}->{dst} (TS={surprise:.2f})",
                        True,
                    )
                )
    return results


def measure_qduf(
    features: pl.DataFrame,
    descriptors: pl.DataFrame,
    *,
    price_col: str = "nq_close",
    n_splits: int = 3,
    embargo: int = 0,
    n_permutations: int = _DEFAULT_N_PERM,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> MetricResult:
    """QDUF — نسبة ديناميكية الطابور غير المفسَّرة."""
    generator = rng if rng is not None else np.random.default_rng(0)
    feat_mat, desc_mat, returns, _price, times = _align_frames(
        features, descriptors, price_col=price_col
    )
    folds = _walk_forward_folds(times, n_splits=n_splits, embargo=embargo)
    qduf_values: list[float] = []

    for fold in folds:
        train_idx = fold.train_idx
        test_idx = fold.test_idx
        if train_idx.shape[0] < _MIN_FOLD_SAMPLES or test_idx.shape[0] < _MIN_TEST_SAMPLES:
            continue
        y_train = returns[train_idx]
        y_test = returns[test_idx]
        r2_mbo = _ridge_r2(desc_mat[train_idx], y_train, desc_mat[test_idx], y_test)
        r2_feat = _ridge_r2(feat_mat[train_idx], y_train, feat_mat[test_idx], y_test)
        if r2_mbo <= r2_feat:
            qduf_values.append(0.0)
        else:
            qduf_values.append(1.0 - r2_feat / max(r2_mbo, 1e-9))

    if not qduf_values:
        return MetricResult("qduf", 0.0, 1.0, 0, "insufficient folds", False)

    observed = float(np.mean(qduf_values))
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = generator.permutation(returns)
        perm_vals: list[float] = []
        for fold in folds:
            train_idx = fold.train_idx
            test_idx = fold.test_idx
            if train_idx.shape[0] < _MIN_FOLD_SAMPLES or test_idx.shape[0] < _MIN_TEST_SAMPLES:
                continue
            r2_mbo = _ridge_r2(
                desc_mat[train_idx], perm[train_idx], desc_mat[test_idx], perm[test_idx]
            )
            r2_feat = _ridge_r2(
                feat_mat[train_idx], perm[train_idx], feat_mat[test_idx], perm[test_idx]
            )
            if r2_mbo <= r2_feat:
                perm_vals.append(0.0)
            else:
                perm_vals.append(1.0 - r2_feat / max(r2_mbo, 1e-9))
        null[i] = float(np.mean(perm_vals)) if perm_vals else 0.0

    pvalue = (int(np.sum(null >= observed)) + 1) / (n_permutations + 1)
    triggered = pvalue <= alpha and observed > _QDUF_THRESHOLD
    return MetricResult(
        "qduf",
        observed,
        pvalue,
        len(qduf_values),
        (
            f"order-book dynamics explain price better than simulators (QDUF={observed:.3f})"
            if triggered
            else f"simulator features sufficient vs MBO queue (QDUF={observed:.3f})"
        ),
        triggered,
    )


def metric_to_evidence(result: MetricResult, *, version: str = "m9") -> Evidence:
    """يحوّل نتيجة مقياس إلى ``Evidence`` قابل للتتبّع."""
    return Evidence(
        id=f"coverage:{result.name}",
        source="coverage_monitor",
        metric=result.name,
        value=result.value,
        pvalue=result.pvalue,
        sample_size=result.sample_size,
        version=version,
        detail=result.detail,
    )


def run_all_metrics(
    features: pl.DataFrame,
    descriptors: pl.DataFrame,
    *,
    price_col: str = "nq_close",
    n_splits: int = 3,
    embargo: int = 0,
    alpha: float = 0.05,
    n_permutations: int = _DEFAULT_N_PERM,
    rng: np.random.Generator | None = None,
    progress: object | None = None,
) -> list[MetricResult]:
    """يشغّل كل مقاييس التغطية الستة."""
    generator = rng if rng is not None else np.random.default_rng(0)
    log = progress
    results: list[MetricResult] = []

    def _emit(name: str) -> None:
        if log is not None:
            log.op(f"M9 مقياس: {name}")  # type: ignore[union-attr]

    _emit("mfig")
    results.append(
        measure_mfig(
            features,
            descriptors,
            price_col=price_col,
            n_splits=n_splits,
            embargo=embargo,
            n_permutations=n_permutations,
            alpha=alpha,
            rng=generator,
        )
    )
    _emit("cer")
    results.extend(
        measure_cer(
            features,
            price_col=price_col,
            n_splits=n_splits,
            embargo=embargo,
            n_permutations=n_permutations,
            alpha=alpha,
            rng=generator,
        )
    )
    _emit("psg")
    results.append(
        measure_psg(
            features,
            n_splits=n_splits,
            embargo=embargo,
            alpha=alpha,
            rng=generator,
        )
    )
    _emit("crs")
    results.extend(
        measure_crs(
            features,
            n_splits=n_splits,
            embargo=embargo,
            alpha=alpha,
            rng=generator,
        )
    )
    _emit("lori")
    results.extend(
        measure_lori(
            features,
            n_splits=n_splits,
            embargo=embargo,
            alpha=alpha,
            rng=generator,
        )
    )
    _emit("qduf")
    results.append(
        measure_qduf(
            features,
            descriptors,
            price_col=price_col,
            n_splits=n_splits,
            embargo=embargo,
            n_permutations=n_permutations,
            alpha=alpha,
            rng=generator,
        )
    )
    if log is not None:
        triggered = sum(1 for r in results if r.triggered)
        log.op(f"M9 انتهى — metrics={len(results)} · triggered={triggered}")  # type: ignore[union-attr]
    return results
