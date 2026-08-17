"""طبقة ب: خروج سببي من الفيتشز — قابلة للخلع.

لا أوامر حد رقمية (وقف/هدف). الخروج عندما يضعف العمق أو يعود التوازن،
أو عند سقف إمساك بحثي (عدد بارميل، ليس سعرًا).

لا تغيّر Y العلمي. احذف هذا الملف + السكربت + ``_write_feature_exit`` للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.research.causal_entry import assert_not_raw_mbo_stream
from nq.research.hold_horizon import (
    causal_fires,
    jsonable,
    load_overlay_period_inputs,
    median_mean,
    oof_timestamps,
    walk_hold_windows,
)
from nq.research.progress import ProgressLike

LAYER_ID = "feature_exit"
_EPS = 1e-9
_PATH_COLS = (
    "path_depth_follow",
    "path_depth_confirm",
    "path_change_fail",
    "vp_balance",
)


@dataclass(frozen=True, slots=True)
class FeatureExitConfig:
    """عتبات ثابتة غير مُقدَّرة على العينة. احذف الطبقة بدل ضبطها على OOF."""

    min_p: float = 0.5
    expansion_start_ticks: float = 16.0
    max_hold_bars: int = 30
    depth_follow_drop: float = 0.35
    balance_return: float = 0.5
    fail_on: float = 0.5
    round_trip_cost_pts: float = 0.75
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class FeatureExitReport:
    trades: pl.DataFrame
    summaries: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _decide_feature_exit(
    window: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    *,
    drop: float,
    balance_on: float,
    fail_on: float,
) -> tuple[int, str]:
    follow = window["path_depth_follow"]
    confirm = window["path_depth_confirm"]
    fail = window["path_change_fail"]
    balance = window["vp_balance"]
    n = int(follow.size)
    if n == 0:
        return 0, "empty"
    follow0 = float(scalars.get("path_depth_follow_0", follow[0]))
    confirm0 = float(scalars.get("path_depth_confirm_0", confirm[0]))
    balance0 = float(scalars.get("vp_balance_0", 0.0))
    for k in range(n):
        if follow0 > _EPS and follow[k] <= follow0 * (1.0 - drop):
            return k, "depth_follow_drop"
        if confirm0 > _EPS and confirm[k] <= confirm0 * (1.0 - drop):
            return k, "depth_confirm_drop"
        if balance0 < balance_on and balance[k] >= balance_on:
            return k, "balance_return"
        if fail[k] >= fail_on:
            return k, "path_change_fail"
    return n - 1, "max_hold"


def run_feature_exit(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: FeatureExitConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> FeatureExitReport:
    """خروج فيتشز بعد إطلاق OOF على أفق إمساك ثابت — بلا ذروة وبلا holdout."""
    cfg = config or FeatureExitConfig()
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    fires = causal_fires(
        labeled,
        blended,
        predictions=predictions,
        oof_availability_ts=oof_availability_ts,
        holdout_cut_ts=holdout_cut_ts,
        min_p=cfg.min_p,
        expansion_start_ticks=cfg.expansion_start_ticks,
        holdout_months=cfg.holdout_months,
        progress=progress,
    )
    empty = FeatureExitReport(
        trades=pl.DataFrame(),
        summaries=pl.DataFrame(),
        diagnostics={
            "empty": True,
            "layer_id": LAYER_ID,
            "removable_layer": True,
            "does_not_modify_science_y": True,
            "holdout_scored": False,
            "numeric_stop_take_not_used": True,
        },
    )
    if fires.height == 0:
        return empty

    def _decide(window: Mapping[str, np.ndarray], scalars: Mapping[str, float]) -> tuple[int, str]:
        return _decide_feature_exit(
            window,
            scalars,
            drop=cfg.depth_follow_drop,
            balance_on=cfg.balance_return,
            fail_on=cfg.fail_on,
        )

    trades = walk_hold_windows(
        fires,
        blended,
        max_hold_bars=cfg.max_hold_bars,
        path_cols=_PATH_COLS,
        decide_exit=_decide,
        round_trip_cost_pts=cfg.round_trip_cost_pts,
    )
    reasons = (
        trades.group_by("exit_reason").agg(pl.len().cast(pl.Int64).alias("n")).sort("exit_reason")
        if trades.height
        else pl.DataFrame()
    )
    net_med, net_mean = median_mean(trades, "net_pts")
    real_med, real_mean = median_mean(trades, "realized_beyond_pts")
    mae_med, mae_mean = median_mean(trades, "mae_beyond_pts")
    mfe_med, mfe_mean = median_mean(trades, "mfe_beyond_pts")
    summaries = pl.DataFrame(
        [
            {
                "layer_id": LAYER_ID,
                "n_trades": int(trades.height),
                "net_pts_median": net_med,
                "net_pts_mean": net_mean,
                "realized_pts_median": real_med,
                "realized_pts_mean": real_mean,
                "mfe_pts_median": mfe_med,
                "mfe_pts_mean": mfe_mean,
                "mae_pts_median": mae_med,
                "mae_pts_mean": mae_mean,
                "round_trip_cost_pts": float(cfg.round_trip_cost_pts),
                "max_hold_bars": int(cfg.max_hold_bars),
            }
        ]
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "holdout_scored": False,
        "numeric_stop_take_not_used": True,
        "live_predictions_not_used": True,
        "hold_horizon_is_bars_not_label_window": True,
        "n_fires": int(fires.height),
        "n_trades": int(trades.height),
        "exit_reasons": (
            {str(r["exit_reason"]): int(r["n"]) for r in reasons.iter_rows(named=True)}
            if reasons.height
            else {}
        ),
        "net_pts_median": net_med,
        "net_pts_mean": net_mean,
        "principles": (
            "removable overlay — delete feature_exit.py to remove",
            "exit is depth/balance/fail at t, not a price stop or take-profit",
            "max_hold_bars is a research cap, not a live numeric stop",
            "cost is subtracted from realized points; 1-tick label window is not used",
            "holdout never scored; completed-wave peak never used",
        ),
    }
    return FeatureExitReport(trades=trades, summaries=summaries, diagnostics=diagnostics)


def render_feature_exit_markdown(report: FeatureExitReport) -> str:
    d = report.diagnostics
    lines = [
        "# Feature-exit overlay (removable layer B)",
        "",
        "Causal exit from live features after an OOF `p` fire.",
        "No numeric stop/take. Hold horizon is N bars, not the 1-tick label window.",
        "Delete this layer without touching science Y.",
        "",
        f"- layer_id={d.get('layer_id')} · removable={d.get('removable_layer')}",
        f"- trades={d.get('n_trades')} · net median={d.get('net_pts_median')} · "
        f"net mean={d.get('net_pts_mean')}",
        f"- exit_reasons={d.get('exit_reasons')}",
        "",
        "## Principles",
        "",
    ]
    for item in d.get("principles", ()):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_feature_exit_report(report: FeatureExitReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.trades.height:
        report.trades.write_parquet(out / "feature_exit_trades.parquet")
    if report.summaries.height:
        report.summaries.write_parquet(out / "feature_exit_summaries.parquet")
    payload = {
        "diagnostics": jsonable(report.diagnostics),
        "holdout_scored": False,
        "removable_layer": True,
    }
    (out / "feature_exit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "FEATURE_EXIT.md").write_text(render_feature_exit_markdown(report), encoding="utf-8")
    return out


def run_feature_exit_from_period_dir(
    period_dir: Path | str,
    *,
    config: FeatureExitConfig | None = None,
    progress: ProgressLike | None = None,
) -> FeatureExitReport:
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(period_dir)
    return run_feature_exit(
        labeled,
        blended,
        config=config,
        oof_availability_ts=oof_timestamps(oof),
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=progress,
    )


__all__ = [
    "LAYER_ID",
    "FeatureExitConfig",
    "FeatureExitReport",
    "render_feature_exit_markdown",
    "run_feature_exit",
    "run_feature_exit_from_period_dir",
    "write_feature_exit_report",
]
