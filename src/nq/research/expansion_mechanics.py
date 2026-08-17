"""ميكانيكا الامتداد: سبق الحجم مقابل السعر، توازن→اختلال→امتداد، وحماية المركز.

يقرأ حالات ``science_labeled`` (ميزات القرار عند ``setup_availability_ts`` فقط).
لا تحميل MBO، لا إعادة بناء دفتر، لا إعادة حساب طبقات، ولا لمس الـholdout.

السؤال: في إعدادات الامتداد الناجح، هل ``proj_outside_volume_share`` و
``path_depth_follow`` سبقا الحركة عند القرار، أم السعر خرج والحجم/العمق لحق؟
هذا يفرّق تنبؤًا حقيقيًا من زخم ظاهر (امتداد مقبول أصلًا يستمر).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.holdout import carve_frozen_holdout
from nq.auction_behavior.outcomes import (
    OUTCOME_AVAILABLE_TS,
    SETUP_AVAILABILITY_TS,
    filter_resolved_outcomes,
)
from nq.core.determinism import make_generator
from nq.research.progress import ProgressLike
from nq.statistics.hypothesis import verify_hypotheses
from nq.statistics.metrics import information_coefficient
from nq.statistics.resampling import temporal_block_permutation
from nq.validation.leakage import assert_availability_not_before_event, assert_causal_order

_RAW_MBO_SIGNATURE = frozenset({"order_id", "action"})
_ACTIVE = 0.5
_MIN_GROUP = 3
_MAJORITY_SHARE = 0.5
_MEASURABLE_SHARE = 0.2
_RARE_LEAD_ABS = 20.0
_RARE_LEAD_FRAC = 0.05
_LEAD_CLASSES = ("both_already", "volume_lead", "price_lead", "neither")
_DEPTH_CLASSES = ("both_already", "depth_lead", "price_lead", "neither")
_SEQ_CLASSES = (
    "balance_imbalance_expansion",
    "imbalance_then_expansion",
    "balance_then_expansion",
    "already_expanding",
    "still_balanced",
    "unclassified",
)
_PROTECT_FEATURES = (
    "path_depth_follow",
    "path_depth_defend",
    "path_held_frac",
    "proj_outside_volume_share",
    "proj_poc_shift_ticks",
    "lf_liquidity_migration",
    "lf_absorption_proxy",
    "path_beyond_asia_ticks",
    "proj_expansion_testing",
    "proj_expansion_accepting",
)

LEAD_CLASS_COL = "lead_class"
DEPTH_LEAD_CLASS_COL = "depth_lead_class"
SEQUENCE_CLASS_COL = "sequence_class"


@dataclass(frozen=True, slots=True)
class ExpansionMechanicsConfig:
    """عتبات وصفية للسبق/التسلسل/الحماية — ليست بوابات تداول."""

    outcome_further: str = "y_path_further_beyond"
    outcome_reverse: str = "y_path_reverse"
    lag: int = 5
    imbalance_lag: int = 3
    volume_threshold: float = 0.35
    beyond_threshold: float = 4.0
    follow_threshold: float = 1.0
    acceptance_share: float = 0.55
    holdout_months: int | None = 4
    n_permutations: int = 199
    seed: int = 2025


@dataclass(frozen=True, slots=True)
class ExpansionMechanicsReport:
    """تقرير ميكانيكا الامتداد على التطوير/OOF فقط."""

    develop: pl.DataFrame
    primary: pl.DataFrame
    lead_lag: pl.DataFrame
    depth_lead_lag: pl.DataFrame
    sequence: pl.DataFrame
    protection: pl.DataFrame
    snapshot_by_y: pl.DataFrame
    hypotheses: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def assert_not_raw_mbo_stream(frame: pl.DataFrame, *, source: str = "") -> None:
    """يرفض تدفق MBO خام — البحث يقرأ حالات مكتملة فقط."""
    present = _RAW_MBO_SIGNATURE.intersection(frame.columns)
    if present != _RAW_MBO_SIGNATURE:
        return
    where = f" in {source}" if source else ""
    raise ValueError(
        "expansion mechanics refuses raw MBO streams "
        f"(found {sorted(present)}{where}); "
        "it reads labeled blended states only — no book reconstruction"
    )


def _f64(frame: pl.DataFrame, name: str, default: float = 0.0) -> pl.Expr:
    if name in frame.columns:
        return pl.col(name).cast(pl.Float64).fill_null(default)
    return pl.lit(default, dtype=pl.Float64)


def _lag_name(base: str, lag: int) -> str:
    return f"{base}__lag{lag}"


def classify_volume_price_lead(
    frame: pl.DataFrame,
    *,
    lag: int = 5,
    volume_threshold: float = 0.35,
    beyond_threshold: float = 4.0,
) -> pl.DataFrame:
    """يصنّف سبق الحجم مقابل السعر من lags سببية (ماضي صارم) إن وُجدت."""
    if frame.height == 0:
        return frame.with_columns(pl.lit("neither").alias(LEAD_CLASS_COL))
    vol_past = _f64(frame, _lag_name("proj_outside_volume_share", lag))
    px_past = _f64(frame, _lag_name("path_beyond_asia_ticks", lag))
    vol_hi = vol_past >= float(volume_threshold)
    px_out = px_past >= float(beyond_threshold)
    lead = (
        pl.when(vol_hi & px_out)
        .then(pl.lit("both_already"))
        .when(vol_hi)
        .then(pl.lit("volume_lead"))
        .when(px_out)
        .then(pl.lit("price_lead"))
        .otherwise(pl.lit("neither"))
    )
    return frame.with_columns(lead.alias(LEAD_CLASS_COL))


def classify_depth_price_lead(
    frame: pl.DataFrame,
    *,
    lag: int = 5,
    follow_threshold: float = 1.0,
    beyond_threshold: float = 4.0,
) -> pl.DataFrame:
    """يصنّف سبق عمق المتابعة مقابل السعر من lags سببية."""
    if frame.height == 0:
        return frame.with_columns(pl.lit("neither").alias(DEPTH_LEAD_CLASS_COL))
    follow_lag_col = _lag_name("path_depth_follow", lag)
    px_lag_col = _lag_name("path_beyond_asia_ticks", lag)
    follow_past = _f64(frame, follow_lag_col)
    px_past = _f64(frame, px_lag_col)
    follow_hi = follow_past >= float(follow_threshold)
    px_out = px_past >= float(beyond_threshold)
    lead = (
        pl.when(follow_hi & px_out)
        .then(pl.lit("both_already"))
        .when(follow_hi)
        .then(pl.lit("depth_lead"))
        .when(px_out)
        .then(pl.lit("price_lead"))
        .otherwise(pl.lit("neither"))
    )
    return frame.with_columns(lead.alias(DEPTH_LEAD_CLASS_COL))


def classify_auction_sequence(
    frame: pl.DataFrame,
    *,
    balance_lag: int = 5,
    imbalance_lag: int = 3,
) -> pl.DataFrame:
    """تسلسل سببي: توازن → اختلال → امتداد، من lags الماضي فقط + حالة t."""
    if frame.height == 0:
        return frame.with_columns(pl.lit("unclassified").alias(SEQUENCE_CLASS_COL))
    beyond_now = _f64(frame, "path_beyond_asia_ticks")
    beyond_bal = _f64(frame, _lag_name("path_beyond_asia_ticks", balance_lag))
    bal_past = _f64(frame, _lag_name("vp_balance", balance_lag))
    imb_past = _f64(frame, _lag_name("vp_imbalance", imbalance_lag))
    brk_past = _f64(frame, _lag_name("vp_fsm_break", imbalance_lag))
    test_past = _f64(frame, _lag_name("proj_expansion_testing", imbalance_lag))
    accepting = _f64(frame, "proj_expansion_accepting")
    testing = _f64(frame, "proj_expansion_testing")
    expanding_now = (
        (beyond_now > _ACTIVE)
        | (accepting > _ACTIVE)
        | (testing > _ACTIVE)
        | (_f64(frame, "vp_fsm_expand") > _ACTIVE)
    )
    was_balanced = (bal_past > _ACTIVE) & (beyond_bal <= _ACTIVE)
    was_imbalanced = (imb_past > _ACTIVE) | (brk_past > _ACTIVE) | (test_past > _ACTIVE)
    already_out = beyond_bal > _ACTIVE
    seq = (
        pl.when(expanding_now & already_out)
        .then(pl.lit("already_expanding"))
        .when(expanding_now & was_balanced & was_imbalanced)
        .then(pl.lit("balance_imbalance_expansion"))
        .when(expanding_now & was_balanced)
        .then(pl.lit("balance_then_expansion"))
        .when(expanding_now & was_imbalanced)
        .then(pl.lit("imbalance_then_expansion"))
        .when(expanding_now)
        .then(pl.lit("already_expanding"))
        .when(was_balanced | (_f64(frame, "vp_balance") > _ACTIVE))
        .then(pl.lit("still_balanced"))
        .otherwise(pl.lit("unclassified"))
    )
    return frame.with_columns(seq.alias(SEQUENCE_CLASS_COL))


def _exclude_holdout(
    labeled: pl.DataFrame,
    *,
    holdout_cut_ts: int | None,
    holdout_months: int | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """يعزل التطوير. الـholdout لا يُمسّ ولا يُقاس."""
    meta: dict[str, Any] = {
        "holdout_scored": False,
        "holdout_excluded": False,
        "holdout_cut_ts": -1,
        "holdout_n_rows": 0,
    }
    if labeled.height == 0 or SETUP_AVAILABILITY_TS not in labeled.columns:
        return labeled, meta
    if holdout_cut_ts is not None and int(holdout_cut_ts) >= 0:
        cut = int(holdout_cut_ts)
        develop = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) <= cut)
        held = labeled.filter(pl.col(SETUP_AVAILABILITY_TS) > cut)
        meta.update(
            holdout_excluded=held.height > 0,
            holdout_cut_ts=cut,
            holdout_n_rows=int(held.height),
        )
        return develop, meta
    if holdout_months is None:
        return labeled, meta
    try:
        pack = carve_frozen_holdout(
            labeled, holdout_months=int(holdout_months), ts_col=SETUP_AVAILABILITY_TS
        )
    except ValueError:
        meta["holdout_carve"] = "insufficient_months_used_all_labeled_as_develop"
        return labeled, meta
    meta.update(
        holdout_excluded=pack.holdout.height > 0,
        holdout_cut_ts=int(pack.cut_ts),
        holdout_n_rows=int(pack.holdout.height),
        holdout_months=int(holdout_months),
    )
    return pack.develop, meta


def _outcome_frame(labeled: pl.DataFrame, name: str) -> pl.DataFrame:
    if labeled.height == 0 or "outcome_name" not in labeled.columns:
        return labeled.head(0)
    work = labeled.filter(pl.col("outcome_name") == name)
    if "label_status" in work.columns:
        work = filter_resolved_outcomes(work)
    elif "y" in work.columns:
        work = work.filter(pl.col("y").is_not_null())
    return work


def _class_table(
    frame: pl.DataFrame,
    class_col: str,
    *,
    scope: str,
    classes: tuple[str, ...],
) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={
            "scope": pl.Utf8(),
            class_col: pl.Utf8(),
            "n": pl.Int64(),
            "n_pos": pl.Float64(),
            "n_neg": pl.Float64(),
            "pos_rate": pl.Float64(),
            "share_of_successes": pl.Float64(),
            "share_of_failures": pl.Float64(),
            "mean_outside_vol_t": pl.Float64(),
            "mean_beyond_t": pl.Float64(),
            "mean_follow_t": pl.Float64(),
            "mean_defend_t": pl.Float64(),
            "mean_held_frac_t": pl.Float64(),
        }
    )
    if frame.height == 0 or class_col not in frame.columns or "y" not in frame.columns:
        return empty
    total_pos = float(frame["y"].sum())
    total_neg = float(frame.height) - total_pos
    stats = frame.group_by(class_col).agg(
        pl.len().cast(pl.Int64).alias("n"),
        pl.col("y").sum().alias("n_pos"),
        (pl.len() - pl.col("y").sum()).alias("n_neg"),
        pl.col("y").mean().alias("pos_rate"),
        _f64(frame, "proj_outside_volume_share").mean().alias("mean_outside_vol_t"),
        _f64(frame, "path_beyond_asia_ticks").mean().alias("mean_beyond_t"),
        _f64(frame, "path_depth_follow").mean().alias("mean_follow_t"),
        _f64(frame, "path_depth_defend").mean().alias("mean_defend_t"),
        _f64(frame, "path_held_frac").mean().alias("mean_held_frac_t"),
    )
    stats = stats.with_columns(
        pl.lit(scope).alias("scope"),
        pl.when(pl.lit(total_pos) > 0.0)
        .then(pl.col("n_pos") / total_pos)
        .otherwise(pl.lit(0.0))
        .alias("share_of_successes"),
        pl.when(pl.lit(total_neg) > 0.0)
        .then(pl.col("n_neg") / total_neg)
        .otherwise(pl.lit(0.0))
        .alias("share_of_failures"),
    )
    present = set(stats[class_col].to_list())
    missing = [c for c in classes if c not in present]
    if missing:
        zeros = pl.DataFrame(
            {
                class_col: missing,
                "n": [0] * len(missing),
                "n_pos": [0.0] * len(missing),
                "n_neg": [0.0] * len(missing),
                "pos_rate": [0.0] * len(missing),
                "mean_outside_vol_t": [0.0] * len(missing),
                "mean_beyond_t": [0.0] * len(missing),
                "mean_follow_t": [0.0] * len(missing),
                "mean_defend_t": [0.0] * len(missing),
                "mean_held_frac_t": [0.0] * len(missing),
                "scope": [scope] * len(missing),
                "share_of_successes": [0.0] * len(missing),
                "share_of_failures": [0.0] * len(missing),
            }
        )
        stats = pl.concat([stats, zeros], how="diagonal_relaxed")
    return stats.select(list(empty.schema.keys())).sort(class_col)


def _snapshot_by_y(frame: pl.DataFrame, *, scope: str, outcome: str) -> pl.DataFrame:
    cols = [
        c
        for c in (
            "proj_outside_volume_share",
            "path_beyond_asia_ticks",
            "path_depth_follow",
            "path_depth_defend",
            "path_held_frac",
            "proj_poc_shift_ticks",
            "proj_expansion_testing",
            "proj_expansion_accepting",
            "lf_liquidity_migration",
            "lf_absorption_proxy",
            "vp_balance",
            "vp_imbalance",
            "vp_fsm_break",
            "vp_fsm_expand",
        )
        if c in frame.columns
    ]
    if frame.height == 0 or "y" not in frame.columns or not cols:
        return pl.DataFrame(
            schema={
                "scope": pl.Utf8(),
                "outcome_name": pl.Utf8(),
                "y": pl.Float64(),
                "n": pl.Int64(),
                **{c: pl.Float64() for c in cols},
            }
        )
    aggs = [pl.col(c).cast(pl.Float64).fill_null(0.0).mean().alias(c) for c in cols]
    out = frame.group_by("y").agg(pl.len().cast(pl.Int64).alias("n"), *aggs)
    return out.with_columns(
        pl.lit(scope).alias("scope"),
        pl.lit(outcome).alias("outcome_name"),
    ).sort("y")


def _already_expanded_mask(frame: pl.DataFrame, cfg: ExpansionMechanicsConfig) -> pl.Expr:
    return (_f64(frame, "path_beyond_asia_ticks") >= float(cfg.beyond_threshold)) & (
        _f64(frame, "proj_outside_volume_share") >= float(cfg.volume_threshold)
    )


def _feature_y_perm_p(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """فرق متوسط الميزة عند y=1 مقابل y=0 مع تبديل كتلي لـ y (يحافظ على الزمن)."""
    pos = y > _ACTIVE
    n_pos = int(pos.sum())
    n_neg = int(y.size - n_pos)
    if n_pos < _MIN_GROUP or n_neg < _MIN_GROUP:
        return 0.0, 1.0
    observed = float(np.mean(x[pos]) - np.mean(x[~pos]))
    hits = 0
    for _ in range(n_permutations):
        y_p = temporal_block_permutation(y, rng=rng)
        pmask = y_p > _ACTIVE
        if int(pmask.sum()) < 1 or int((~pmask).sum()) < 1:
            continue
        stat = float(np.mean(x[pmask]) - np.mean(x[~pmask]))
        if abs(stat) >= abs(observed):
            hits += 1
    pvalue = float((hits + 1) / (n_permutations + 1))
    return observed, pvalue


def _protection_table(
    frame: pl.DataFrame,
    *,
    scope: str,
    outcome: str,
    cfg: ExpansionMechanicsConfig,
    rng: np.random.Generator,
    already_expanded: bool,
) -> pl.DataFrame:
    schema = {
        "scope": pl.Utf8(),
        "outcome_name": pl.Utf8(),
        "already_expanded": pl.Boolean(),
        "feature": pl.Utf8(),
        "n": pl.Int64(),
        "n_pos": pl.Int64(),
        "mean_pos": pl.Float64(),
        "mean_neg": pl.Float64(),
        "delta_pos_minus_neg": pl.Float64(),
        "spearman_ic": pl.Float64(),
        "pvalue": pl.Float64(),
    }
    if frame.height == 0 or "y" not in frame.columns:
        return pl.DataFrame(schema=schema)
    work = frame
    if already_expanded:
        work = work.filter(_already_expanded_mask(work, cfg))
    y = work["y"].fill_null(0.0).to_numpy().astype(np.float64)
    rows: list[dict[str, Any]] = []
    for feat in _PROTECT_FEATURES:
        if feat not in work.columns:
            continue
        x = work[feat].cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64)
        pos = y > _ACTIVE
        n_pos = int(pos.sum())
        n_neg = int(y.size - n_pos)
        if n_pos < _MIN_GROUP or n_neg < _MIN_GROUP:
            continue
        delta, pvalue = _feature_y_perm_p(x, y, n_permutations=int(cfg.n_permutations), rng=rng)
        rows.append(
            {
                "scope": scope,
                "outcome_name": outcome,
                "already_expanded": already_expanded,
                "feature": feat,
                "n": int(y.size),
                "n_pos": n_pos,
                "mean_pos": float(np.mean(x[pos])),
                "mean_neg": float(np.mean(x[~pos])),
                "delta_pos_minus_neg": delta,
                "spearman_ic": information_coefficient(x, y, method="spearman"),
                "pvalue": pvalue,
            }
        )
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _annotate(frame: pl.DataFrame, cfg: ExpansionMechanicsConfig) -> pl.DataFrame:
    work = classify_volume_price_lead(
        frame,
        lag=cfg.lag,
        volume_threshold=cfg.volume_threshold,
        beyond_threshold=cfg.beyond_threshold,
    )
    work = classify_depth_price_lead(
        work,
        lag=cfg.lag,
        follow_threshold=cfg.follow_threshold,
        beyond_threshold=cfg.beyond_threshold,
    )
    work = classify_auction_sequence(work, balance_lag=cfg.lag, imbalance_lag=cfg.imbalance_lag)
    accepted = (
        (_f64(work, "proj_outside_volume_share") >= float(cfg.acceptance_share))
        & (_f64(work, "path_beyond_asia_ticks") >= float(cfg.beyond_threshold))
        & (_f64(work, "path_depth_follow") >= float(cfg.follow_threshold))
    )
    return work.with_columns(accepted.cast(pl.Float64).alias("already_accepted_expansion"))


def _scope_slice(
    develop: pl.DataFrame,
    *,
    oof_availability_ts: Sequence[int] | None,
) -> tuple[pl.DataFrame, str]:
    if not oof_availability_ts:
        return develop, "develop"
    oof_set = {int(t) for t in oof_availability_ts}
    if SETUP_AVAILABILITY_TS not in develop.columns:
        return develop, "develop"
    oof = develop.filter(pl.col(SETUP_AVAILABILITY_TS).is_in(list(oof_set)))
    if oof.height == 0:
        return develop, "develop"
    return oof, "oof_develop"


def run_expansion_mechanics(
    labeled: pl.DataFrame,
    *,
    config: ExpansionMechanicsConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    progress: ProgressLike | None = None,
) -> ExpansionMechanicsReport:
    """يحسب سبق الحجم/العمق، تسلسل المزاد، وحماية الامتداد على التطوير فقط."""
    cfg = config or ExpansionMechanicsConfig()
    assert_not_raw_mbo_stream(labeled, source="labeled")
    empty = ExpansionMechanicsReport(
        develop=pl.DataFrame(),
        primary=pl.DataFrame(),
        lead_lag=pl.DataFrame(),
        depth_lead_lag=pl.DataFrame(),
        sequence=pl.DataFrame(),
        protection=pl.DataFrame(),
        snapshot_by_y=pl.DataFrame(),
        hypotheses=pl.DataFrame(),
        diagnostics={"empty": True, "holdout_scored": False},
    )
    if labeled.height == 0:
        return empty
    if SETUP_AVAILABILITY_TS in labeled.columns:
        assert_causal_order(labeled.sort(SETUP_AVAILABILITY_TS)[SETUP_AVAILABILITY_TS].to_list())
    if SETUP_AVAILABILITY_TS in labeled.columns and OUTCOME_AVAILABLE_TS in labeled.columns:
        assert_availability_not_before_event(
            labeled[SETUP_AVAILABILITY_TS].to_list(),
            labeled[OUTCOME_AVAILABLE_TS].to_list(),
        )
    if progress is not None:
        progress.op(f"expansion_mechanics labeled={labeled.height:,}")
    develop, holdout_meta = _exclude_holdout(
        labeled, holdout_cut_ts=holdout_cut_ts, holdout_months=cfg.holdout_months
    )
    further = _annotate(_outcome_frame(develop, cfg.outcome_further), cfg)
    reverse = _annotate(_outcome_frame(develop, cfg.outcome_reverse), cfg)
    primary, primary_scope = _scope_slice(further, oof_availability_ts=oof_availability_ts)
    reverse_primary, _ = _scope_slice(reverse, oof_availability_ts=oof_availability_ts)
    if progress is not None:
        progress.op(
            f"expansion_mechanics develop_further={further.height:,} "
            f"primary={primary.height:,} scope={primary_scope}"
        )

    rng = make_generator(cfg.seed)
    extra_scope = primary_scope != "develop"
    lead_parts = [_class_table(further, LEAD_CLASS_COL, scope="develop", classes=_LEAD_CLASSES)]
    depth_parts = [
        _class_table(further, DEPTH_LEAD_CLASS_COL, scope="develop", classes=_DEPTH_CLASSES)
    ]
    seq_parts = [_class_table(further, SEQUENCE_CLASS_COL, scope="develop", classes=_SEQ_CLASSES)]
    snap_parts = [
        _snapshot_by_y(further, scope="develop", outcome=cfg.outcome_further),
        _snapshot_by_y(reverse, scope="develop", outcome=cfg.outcome_reverse),
    ]
    prot_parts = [
        _protection_table(
            further,
            scope="develop",
            outcome=cfg.outcome_further,
            cfg=cfg,
            rng=rng,
            already_expanded=False,
        ),
        _protection_table(
            further,
            scope="develop",
            outcome=cfg.outcome_further,
            cfg=cfg,
            rng=rng,
            already_expanded=True,
        ),
        _protection_table(
            reverse,
            scope="develop",
            outcome=cfg.outcome_reverse,
            cfg=cfg,
            rng=rng,
            already_expanded=True,
        ),
    ]
    if extra_scope:
        lead_parts.append(
            _class_table(primary, LEAD_CLASS_COL, scope=primary_scope, classes=_LEAD_CLASSES)
        )
        depth_parts.append(
            _class_table(primary, DEPTH_LEAD_CLASS_COL, scope=primary_scope, classes=_DEPTH_CLASSES)
        )
        seq_parts.append(
            _class_table(primary, SEQUENCE_CLASS_COL, scope=primary_scope, classes=_SEQ_CLASSES)
        )
        snap_parts.extend(
            [
                _snapshot_by_y(primary, scope=primary_scope, outcome=cfg.outcome_further),
                _snapshot_by_y(reverse_primary, scope=primary_scope, outcome=cfg.outcome_reverse),
            ]
        )
        prot_parts.extend(
            [
                _protection_table(
                    primary,
                    scope=primary_scope,
                    outcome=cfg.outcome_further,
                    cfg=cfg,
                    rng=rng,
                    already_expanded=True,
                ),
                _protection_table(
                    reverse_primary,
                    scope=primary_scope,
                    outcome=cfg.outcome_reverse,
                    cfg=cfg,
                    rng=rng,
                    already_expanded=True,
                ),
            ]
        )
    lead_lag = pl.concat(lead_parts, how="diagonal_relaxed")
    depth_lead = pl.concat(depth_parts, how="diagonal_relaxed")
    sequence = pl.concat(seq_parts, how="diagonal_relaxed")
    snapshot = pl.concat(snap_parts, how="diagonal_relaxed")
    protection = pl.concat(prot_parts, how="diagonal_relaxed")

    pvalues: dict[str, float] = {}
    if protection.height and "pvalue" in protection.columns:
        for row in protection.iter_rows(named=True):
            key = (
                f"{row['scope']}:{row['outcome_name']}:"
                f"{'expanded' if row['already_expanded'] else 'all'}:{row['feature']}"
            )
            pvalues[key] = float(row["pvalue"])
    hypotheses = verify_hypotheses(pvalues) if pvalues else verify_hypotheses({})

    lag_vol = _lag_name("proj_outside_volume_share", cfg.lag)
    lag_px = _lag_name("path_beyond_asia_ticks", cfg.lag)
    lag_follow = _lag_name("path_depth_follow", cfg.lag)
    n_accepted = (
        int(primary.filter(pl.col("already_accepted_expansion") > _ACTIVE).height)
        if primary.height and "already_accepted_expansion" in primary.columns
        else 0
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "holdout_scored": False,
        "raw_mbo_not_loaded": True,
        "book_not_reconstructed": True,
        "features_not_recomputed_from_mbo": True,
        "prediction_uses_oos_labels": False,
        "primary_scope": primary_scope,
        "n_labeled": int(labeled.height),
        "n_develop": int(develop.height),
        "n_further_develop": int(further.height),
        "n_further_primary": int(primary.height),
        "n_reverse_primary": int(reverse_primary.height),
        "n_already_accepted_primary": n_accepted,
        "lag": int(cfg.lag),
        "imbalance_lag": int(cfg.imbalance_lag),
        "volume_threshold": float(cfg.volume_threshold),
        "beyond_threshold": float(cfg.beyond_threshold),
        "follow_threshold": float(cfg.follow_threshold),
        "has_volume_lag": lag_vol in further.columns,
        "has_beyond_lag": lag_px in further.columns,
        "has_follow_lag": lag_follow in further.columns,
        "lag_classification_uses_contemporaneous_fallback": not (
            lag_vol in further.columns and lag_px in further.columns
        ),
        "n_permutations": int(cfg.n_permutations),
        "seed": int(cfg.seed),
        "outcome_further": cfg.outcome_further,
        "outcome_reverse": cfg.outcome_reverse,
        "principles": (
            "labeled join is exact on setup_availability_ts; Y is future path, not a feature",
            "lags are causal past only (__lag k with k>=1); missing lags fall back to zeros",
            "holdout rows after cut_ts are excluded and never scored",
            "OOF timestamps (walk-forward test months) are the primary claim scope when present",
            "volume-lead vs price-lead separates true lead from already-accepted continuation",
            "protection is measured only among already-expanded states at decision t",
        ),
        **holdout_meta,
    }
    return ExpansionMechanicsReport(
        develop=further,
        primary=primary,
        lead_lag=lead_lag,
        depth_lead_lag=depth_lead,
        sequence=sequence,
        protection=protection,
        snapshot_by_y=snapshot,
        hypotheses=hypotheses,
        diagnostics=diagnostics,
    )


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _md_table(frame: pl.DataFrame, columns: Sequence[str], *, max_rows: int = 16) -> list[str]:
    cols = [c for c in columns if c in frame.columns]
    if not cols or frame.height == 0:
        return ["_(empty)_", ""]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in frame.select(cols).head(max_rows).iter_rows(named=True):
        cells: list[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _rate(frame: pl.DataFrame, class_col: str, name: str, *, scope: str) -> dict[str, float]:
    if frame.height == 0 or class_col not in frame.columns:
        return {"n": 0.0, "pos_rate": 0.0, "share_of_successes": 0.0, "share_of_failures": 0.0}
    work = frame.filter((pl.col("scope") == scope) & (pl.col(class_col) == name))
    if work.height == 0:
        return {"n": 0.0, "pos_rate": 0.0, "share_of_successes": 0.0, "share_of_failures": 0.0}
    row = work.row(0, named=True)
    return {
        "n": float(row["n"]),
        "pos_rate": float(row["pos_rate"] or 0.0),
        "share_of_successes": float(row["share_of_successes"] or 0.0),
        "share_of_failures": float(row["share_of_failures"] or 0.0),
    }


def _reading(report: ExpansionMechanicsReport) -> tuple[str, ...]:
    """قراءة مبنية على الأرقام — لا سرد جاهز."""
    scope = str(report.diagnostics.get("primary_scope", "develop"))
    both = _rate(report.lead_lag, LEAD_CLASS_COL, "both_already", scope=scope)
    vol = _rate(report.lead_lag, LEAD_CLASS_COL, "volume_lead", scope=scope)
    px = _rate(report.lead_lag, LEAD_CLASS_COL, "price_lead", scope=scope)
    neither = _rate(report.lead_lag, LEAD_CLASS_COL, "neither", scope=scope)
    seq_full = _rate(
        report.sequence, SEQUENCE_CLASS_COL, "balance_imbalance_expansion", scope=scope
    )
    seq_already = _rate(report.sequence, SEQUENCE_CLASS_COL, "already_expanding", scope=scope)
    lag_fallback = bool(report.diagnostics.get("lag_classification_uses_contemporaneous_fallback"))
    claims: list[str] = []
    if lag_fallback:
        claims.append(
            "lags of volume/price were missing — classes used zeros for the past, "
            "so contemporaneous state at t dominates (not a true lead-lag test)."
        )
    if both["share_of_successes"] >= _MAJORITY_SHARE and both["pos_rate"] > neither["pos_rate"]:
        claims.append(
            "successful further-beyond is mostly already-accepted expansion at lag "
            f"(both_already share_of_successes={both['share_of_successes']:.2f}, "
            f"pos_rate={both['pos_rate']:.2f} vs neither={neither['pos_rate']:.2f})."
        )
    total_n = both["n"] + vol["n"] + px["n"] + neither["n"]
    if total_n > 0 and vol["n"] < max(_RARE_LEAD_ABS, _RARE_LEAD_FRAC * total_n):
        claims.append(
            f"pure volume-before-price is uncommon (n={vol['n']:.0f}, "
            f"pos_rate={vol['pos_rate']:.2f})."
        )
    if px["n"] > vol["n"] and px["pos_rate"] >= vol["pos_rate"]:
        claims.append(
            "price typically leads; volume confirms "
            f"(price_lead n={px['n']:.0f} pos_rate={px['pos_rate']:.2f} vs "
            f"volume_lead n={vol['n']:.0f} pos_rate={vol['pos_rate']:.2f})."
        )
    if seq_already["share_of_successes"] >= _MAJORITY_SHARE:
        claims.append(
            "the visible auction at decision is already expanding "
            f"(already_expanding share_of_successes={seq_already['share_of_successes']:.2f}); "
            "this is continuation after a realized setup, not quiet-balance detection."
        )
    if seq_full["share_of_successes"] >= _MEASURABLE_SHARE:
        claims.append(
            "a measurable share of successes still shows balance→imbalance→expansion "
            f"in causal lags (share_of_successes={seq_full['share_of_successes']:.2f})."
        )
    prot = report.protection
    if prot.height:
        follow = prot.filter(
            (pl.col("scope") == scope)
            & (pl.col("feature") == "path_depth_follow")
            & (pl.col("already_expanded"))
            & (pl.col("outcome_name") == str(report.diagnostics.get("outcome_further")))
        )
        defend = prot.filter(
            (pl.col("scope") == scope)
            & (pl.col("feature") == "path_depth_defend")
            & (pl.col("already_expanded"))
            & (pl.col("outcome_name") == str(report.diagnostics.get("outcome_further")))
        )
        if follow.height:
            row = follow.row(0, named=True)
            claims.append(
                "among already-expanded setups, depth follow at t "
                f"mean_pos={row['mean_pos']:.3f} vs mean_neg={row['mean_neg']:.3f} "
                f"(IC={row['spearman_ic']:.3f}, p={row['pvalue']:.3f})."
            )
        if defend.height:
            row = defend.row(0, named=True)
            claims.append(
                "among already-expanded setups, depth defend at t "
                f"mean_pos={row['mean_pos']:.3f} vs mean_neg={row['mean_neg']:.3f} "
                f"(IC={row['spearman_ic']:.3f}, p={row['pvalue']:.3f})."
            )
    if not claims:
        claims.append(
            "insufficient contrast in the available labeled slice to support a lead-lag claim."
        )
    return tuple(claims)


def render_expansion_mechanics_markdown(report: ExpansionMechanicsReport) -> str:
    """بيان Markdown من الأرقام فقط."""
    d = report.diagnostics
    scope = str(d.get("primary_scope", "develop"))
    lines = [
        "# expansion mechanics — volume vs price, auction sequence, protection",
        "",
        "Labeled states only. No MBO reload. No book reconstruction. Holdout never scored.",
        "Y is the next realized path after setup t. Features/lags are known at t.",
        "",
        f"- primary scope: `{scope}` (OOF walk-forward months when provided; else develop)",
        (
            f"- further-beyond rows: develop={d.get('n_further_develop')} · "
            f"primary={d.get('n_further_primary')}"
        ),
        f"- reverse rows (primary): {d.get('n_reverse_primary')}",
        f"- already-accepted expansion at t (primary): {d.get('n_already_accepted_primary')}",
        f"- lag={d.get('lag')} · imbalance_lag={d.get('imbalance_lag')} · "
        f"vol_thr={d.get('volume_threshold')} · beyond_thr={d.get('beyond_threshold')} · "
        f"follow_thr={d.get('follow_threshold')}",
        f"- volume lag present={d.get('has_volume_lag')} · "
        f"beyond lag present={d.get('has_beyond_lag')} · "
        f"follow lag present={d.get('has_follow_lag')}",
        f"- holdout excluded={d.get('holdout_excluded')} · scored={d.get('holdout_scored')} · "
        f"cut_ts={d.get('holdout_cut_ts')} · holdout_rows={d.get('holdout_n_rows')}",
        "",
        "## Isolation",
        "",
    ]
    for p in d.get("principles", ()):
        lines.append(f"- {p}")
    lines.extend(["", "## Reading", ""])
    for claim in _reading(report):
        lines.append(f"- {claim}")
    lines.extend(["", "## Volume vs price at lag (further-beyond)", ""])
    lead = (
        report.lead_lag.filter(pl.col("scope") == scope)
        if report.lead_lag.height and "scope" in report.lead_lag.columns
        else report.lead_lag
    )
    lines.extend(
        _md_table(
            lead,
            (
                "scope",
                LEAD_CLASS_COL,
                "n",
                "n_pos",
                "pos_rate",
                "share_of_successes",
                "share_of_failures",
                "mean_outside_vol_t",
                "mean_beyond_t",
                "mean_follow_t",
            ),
        )
    )
    lines.extend(["## Depth follow vs price at lag (further-beyond)", ""])
    depth = (
        report.depth_lead_lag.filter(pl.col("scope") == scope)
        if report.depth_lead_lag.height and "scope" in report.depth_lead_lag.columns
        else report.depth_lead_lag
    )
    lines.extend(
        _md_table(
            depth,
            (
                "scope",
                DEPTH_LEAD_CLASS_COL,
                "n",
                "n_pos",
                "pos_rate",
                "share_of_successes",
                "share_of_failures",
                "mean_follow_t",
                "mean_beyond_t",
            ),
        )
    )
    lines.extend(["## Balance → imbalance → expansion", ""])
    seq = (
        report.sequence.filter(pl.col("scope") == scope)
        if report.sequence.height and "scope" in report.sequence.columns
        else report.sequence
    )
    lines.extend(
        _md_table(
            seq,
            (
                "scope",
                SEQUENCE_CLASS_COL,
                "n",
                "n_pos",
                "pos_rate",
                "share_of_successes",
                "share_of_failures",
            ),
        )
    )
    lines.extend(["## Protection of expansion positions (already expanded at t)", ""])
    prot = (
        report.protection.filter((pl.col("scope") == scope) & (pl.col("already_expanded")))
        if report.protection.height and "already_expanded" in report.protection.columns
        else report.protection
    )
    lines.extend(
        _md_table(
            prot,
            (
                "outcome_name",
                "feature",
                "n",
                "n_pos",
                "mean_pos",
                "mean_neg",
                "delta_pos_minus_neg",
                "spearman_ic",
                "pvalue",
            ),
        )
    )
    lines.extend(["## Snapshot at decision t by y", ""])
    snap = (
        report.snapshot_by_y.filter(pl.col("scope") == scope)
        if report.snapshot_by_y.height and "scope" in report.snapshot_by_y.columns
        else report.snapshot_by_y
    )
    snap_cols = [
        c
        for c in (
            "scope",
            "outcome_name",
            "y",
            "n",
            "proj_outside_volume_share",
            "path_beyond_asia_ticks",
            "path_depth_follow",
            "path_depth_defend",
            "path_held_frac",
            "lf_liquidity_migration",
            "lf_absorption_proxy",
        )
        if c in snap.columns
    ]
    lines.extend(_md_table(snap, snap_cols))
    return "\n".join(lines).rstrip() + "\n"


def write_expansion_mechanics_report(
    report: ExpansionMechanicsReport,
    output_dir: Path | str,
) -> Path:
    """يكتب باركيه + JSON + EXPANSION.md — بدون لمس holdout."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.lead_lag.height:
        report.lead_lag.write_parquet(out / "expansion_lead_lag.parquet")
    if report.depth_lead_lag.height:
        report.depth_lead_lag.write_parquet(out / "expansion_depth_lead_lag.parquet")
    if report.sequence.height:
        report.sequence.write_parquet(out / "expansion_sequence.parquet")
    if report.protection.height:
        report.protection.write_parquet(out / "expansion_protection.parquet")
    if report.snapshot_by_y.height:
        report.snapshot_by_y.write_parquet(out / "expansion_snapshot_by_y.parquet")
    if report.hypotheses.height:
        report.hypotheses.write_parquet(out / "expansion_hypotheses.parquet")
    payload = {
        "diagnostics": _jsonable(report.diagnostics),
        "reading": list(_reading(report)),
        "holdout_scored": False,
    }
    (out / "expansion_mechanics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "EXPANSION.md").write_text(render_expansion_mechanics_markdown(report), encoding="utf-8")
    return out


def run_expansion_mechanics_from_period_dir(
    period_dir: Path | str,
    *,
    config: ExpansionMechanicsConfig | None = None,
    progress: ProgressLike | None = None,
) -> ExpansionMechanicsReport:
    """يشغّل البحث من مخرجات المرحلة 2 الجاهزة (باركيه فقط)."""
    root = Path(period_dir)
    labeled_path = root / "science_labeled.parquet"
    if not labeled_path.is_file():
        raise FileNotFoundError(f"science_labeled.parquet not found under {root.resolve()}")
    labeled = pl.read_parquet(labeled_path)
    oof_ts: list[int] | None = None
    oof_path = root / "oof_predictions.parquet"
    if oof_path.is_file():
        oof = pl.read_parquet(oof_path)
        ts_col = (
            SETUP_AVAILABILITY_TS if SETUP_AVAILABILITY_TS in oof.columns else "availability_ts"
        )
        if ts_col in oof.columns:
            oof_ts = [int(t) for t in oof[ts_col].to_list()]
    cut_ts: int | None = None
    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        diag = summary.get("diagnostics", {})
        science = diag.get("science", diag)
        raw_cut = science.get("holdout_cut_ts", summary.get("holdout_cut_ts"))
        if raw_cut is not None and int(raw_cut) >= 0:
            cut_ts = int(raw_cut)
        if bool(science.get("holdout_touched")):
            # even a previously touched year-holdout is excluded by cut_ts; never scored here
            pass
    cfg = config or ExpansionMechanicsConfig()
    if progress is not None:
        progress.op(f"expansion_mechanics from {root}")
    return run_expansion_mechanics(
        labeled,
        config=cfg,
        oof_availability_ts=oof_ts,
        holdout_cut_ts=cut_ts,
        progress=progress,
    )


__all__ = [
    "DEPTH_LEAD_CLASS_COL",
    "LEAD_CLASS_COL",
    "SEQUENCE_CLASS_COL",
    "ExpansionMechanicsConfig",
    "ExpansionMechanicsReport",
    "assert_not_raw_mbo_stream",
    "classify_auction_sequence",
    "classify_depth_price_lead",
    "classify_volume_price_lead",
    "render_expansion_mechanics_markdown",
    "run_expansion_mechanics",
    "run_expansion_mechanics_from_period_dir",
    "write_expansion_mechanics_report",
]
