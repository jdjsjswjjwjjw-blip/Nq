#!/usr/bin/env python3
"""تشغيل الملف الكامل كما هو — بلا max_rows وبلا قص زمني."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from nq.auction_behavior import (
    BehaviorConfig,
    behavior_prediction_frame,
    behavior_probability_summary,
    run_auction_behavior_analysis,
)
from nq.auction_behavior.path_confirm import PATH_CONFIRM_COLUMNS
from nq.contracts.temporal import AVAILABILITY_TS
from nq.ingestion.reader import load_mbo_frame
from nq.research.progress import PipelineProgress

SRC = "/workspace/GLBX-20260815-A7C5K4A6UE/MNQ10H.continuous.clean.parquet"
OUT = Path("/workspace/out/mnq10h")


def _summarize(name: str, result, *, n_raw: int) -> dict:
    probs = result.probabilities
    val = result.validation
    diag = result.diagnostics
    latest = None
    if result.blended.height and AVAILABILITY_TS in result.blended.columns:
        last = result.blended.sort(AVAILABILITY_TS).tail(1)
        latest = {
            "availability_ts": int(last[AVAILABILITY_TS][0]),
            "session": (
                int(last["vp_liquidity_session"][0])
                if "vp_liquidity_session" in last.columns
                else None
            ),
            "phase": (
                str(last["auction_phase"][0])
                if "auction_phase" in last.columns and last["auction_phase"][0] is not None
                else None
            ),
            "signal_quality": (
                float(last["signal_quality"][0]) if "signal_quality" in last.columns else None
            ),
        }
        for col in PATH_CONFIRM_COLUMNS:
            if col in last.columns and last[col][0] is not None:
                latest[col] = float(last[col][0])
    science = result.science
    return {
        "name": name,
        "src": SRC,
        "n_raw_parquet": n_raw,
        "n_mbo": diag.get("n_mbo_rows"),
        "n_state_bars": diag.get("n_state_bars"),
        "n_signal_bars": diag.get("n_signal_bars"),
        "n_events": diag.get("n_events"),
        "n_projection_bars": diag.get("n_projection_bars"),
        "deceptive_scored_rows": diag.get("deceptive_scored_rows"),
        "deceptive_filtered": diag.get("deceptive_filtered"),
        "validation_ok": val.ok,
        "causal_ok": val.causal_ok,
        "no_trade_outputs": val.no_trade_outputs,
        "cut_file": False,
        "max_rows": None,
        "probabilities": {
            "probability_source": probs.probability_source,
            "probabilities_are_joint_distribution": (probs.probabilities_are_joint_distribution),
            "p_expansion_accepting": probs.p_expansion_accepting,
            "p_rejection_return_to_asia": probs.p_rejection_return_to_asia,
            "p_repriced_balance": probs.p_repriced_balance,
            "p_residual": probs.p_residual,
            "p_balanced": probs.p_balanced,
            "p_imbalanced": probs.p_imbalanced,
            "p_true_break": probs.p_true_break,
            "p_false_break": probs.p_false_break,
            "p_retest_success": probs.p_retest_success,
            "p_retest_fail": probs.p_retest_fail,
            "p_expansion_continue": probs.p_expansion_continue,
            "p_return_to_value": probs.p_return_to_value,
            "confidence": probs.confidence,
            "n_samples": probs.n_samples,
            "n_oof_rows": probs.n_oof_rows,
            "detail": probs.detail,
        },
        "latest_state": latest,
        "oos": None
        if science is None
        else {
            "probability_source": probs.probability_source,
            "conditional_probability_semantics": science.diagnostics.get(
                "conditional_probability_semantics"
            ),
            "probabilities_are_joint_distribution": science.diagnostics.get(
                "probabilities_are_joint_distribution"
            ),
            "n_labeled_rows": science.diagnostics.get("n_labeled"),
            "n_unique_setups": science.diagnostics.get("n_unique_setups"),
            "n_competing_setups": science.diagnostics.get("n_competing_setups"),
            "n_develop": science.diagnostics.get("n_develop"),
            "n_holdout": science.diagnostics.get("n_holdout"),
            "n_features": science.diagnostics.get("n_features"),
            "sample_size_caution": science.diagnostics.get("sample_size_caution"),
            "n_oof_prediction_rows": science.diagnostics.get("n_oof_prediction_rows"),
            "n_live_prediction_rows": science.diagnostics.get("n_live_prediction_rows"),
            "n_competing_oof_rows": science.diagnostics.get("n_competing_oof_rows"),
            "holdout_touched": science.diagnostics.get("holdout_touched"),
            "competing_oof_metrics": science.diagnostics.get("competing_oof_metrics"),
            "competing_stability": science.diagnostics.get("competing_stability"),
            "ablation": science.diagnostics.get("ablation"),
            "binary_ablation": science.diagnostics.get("binary_ablation"),
            "n_by_outcome": science.diagnostics.get("n_by_outcome"),
            "oos_skill_probability_column": science.diagnostics.get(
                "oos_skill_probability_column"
            ),
            "calibration_by_outcome": science.diagnostics.get("calibration_by_outcome"),
            "calibration_by_outcome_calibrated": science.diagnostics.get(
                "calibration_by_outcome_calibrated"
            ),
            "live_eligible_for_backtest": science.diagnostics.get(
                "live_predictions_eligible_for_backtest"
            ),
            "oof_eligible_for_backtest": science.diagnostics.get(
                "oof_predictions_eligible_for_backtest"
            ),
        },
        "science": None if science is None else science.diagnostics,
        "causality": diag.get("causality"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "progress.log"
    progress = PipelineProgress(enabled=True, heartbeat_seconds=1.0)
    progress.attach_log(log_path)
    progress.begin("mnq10h-full-file")
    progress.op(f"load full file (no max_rows, no time cut): {SRC}")
    n_raw = int(pq.ParquetFile(SRC).metadata.num_rows)
    progress.op(f"parquet rows on disk={n_raw:,}")
    mbo = load_mbo_frame(SRC, progress=progress)
    progress.op(f"loaded {mbo.height:,} rows (raw={n_raw:,})")
    cfg = BehaviorConfig(
        include_deceptive_scores=True,
        include_level_flow=True,
        include_reliability_evidence=True,
        include_science=True,
        include_asia_london_projection=True,
        evaluate_holdout=False,
        quiet=False,
        n_splits=3,
        min_train_size=8,
    )
    result = run_auction_behavior_analysis(mbo, config=cfg, progress=progress, quiet=False)
    summary = _summarize("mnq10h", result, n_raw=n_raw)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    keep = {
        AVAILABILITY_TS,
        "vp_liquidity_session",
        "auction_phase",
        "signal_quality",
        "vp_balance",
        "vp_imbalance",
        "vp_expansion",
        "vp_fsm_break",
        "vp_fsm_retest",
        "deceptive_score",
        "real_liquidity_ratio",
        *PATH_CONFIRM_COLUMNS,
    }
    if result.blended.height:
        cols = [c for c in result.blended.columns if c in keep]
        result.blended.select(cols).write_parquet(OUT / "state.parquet")
    if result.projection.height:
        result.projection.write_parquet(OUT / "projection.parquet")
    pred = behavior_prediction_frame(result)
    if pred.height:
        pred.write_parquet(OUT / "predictions.parquet")
    behavior_probability_summary(result).write_parquet(OUT / "prob_summary.parquet")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    progress.done(f"mnq10h ok={result.validation.ok} bars={result.blended.height} mbo={mbo.height}")


if __name__ == "__main__":
    main()
