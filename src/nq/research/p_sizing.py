"""طبقة ج: حجم من ``p`` وخروج عند انقلاب ``p`` — قابلة للخلع.

الدخول عند ``p ≥ enter_p`` بحجم يتناسب مع ``p``. الخروج عند أول درجة OOF
لاحقة ``p ≤ exit_p``، أو سقف الإمساك. بلا هدف رقمي.

يُستخدم OOF فقط — ``live_predictions`` ممنوع في الباك تست.
لا تغيّر Y العلمي. احذف هذا الملف + السكربت + ``_write_p_sizing`` للإزالة.
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
    attach_oof_p,
    causal_fires,
    jsonable,
    load_overlay_period_inputs,
    median_mean,
    oof_timestamps,
    walk_hold_windows,
)
from nq.research.progress import ProgressLike

LAYER_ID = "p_sizing"
_FRESH = 0.5


@dataclass(frozen=True, slots=True)
class PSizingConfig:
    """عتبات ثابتة. الحجم عند ``enter_p`` = ``min_size``، وعند 1.0 = 1."""

    enter_p: float = 0.5
    exit_p: float = 0.3
    min_size: float = 0.2
    expansion_start_ticks: float = 16.0
    max_hold_bars: int = 30
    round_trip_cost_pts: float = 0.75
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class PSizingReport:
    trades: pl.DataFrame
    summaries: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def position_size(
    p: float,
    *,
    enter_p: float,
    min_size: float,
) -> float:
    """حجم ∈ [min_size, 1] عندما ``p ≥ enter_p``، وإلا 0."""
    if p < float(enter_p) or not np.isfinite(p):
        return 0.0
    span = max(1e-12, 1.0 - float(enter_p))
    return float(
        min(1.0, float(min_size) + (float(p) - float(enter_p)) / span * (1.0 - float(min_size)))
    )


def _decide_p_exit(
    window: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
    *,
    exit_p: float,
) -> tuple[int, str]:
    fresh = window.get("oof_p_at_bar")
    p_hat = window.get("model_p")
    n = int(window["path_beyond_asia_ticks"].size)
    if n == 0 or fresh is None or p_hat is None:
        return max(n - 1, 0), "max_hold"
    _ = scalars
    for k in range(n):
        if fresh[k] >= _FRESH and np.isfinite(p_hat[k]) and p_hat[k] <= float(exit_p):
            return k, "p_flip"
    return n - 1, "max_hold"


def run_p_sizing(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: PSizingConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> PSizingReport:
    """حجم من p وخروج انقلاب OOF على أفق إمساك ثابت."""
    cfg = config or PSizingConfig()
    if cfg.exit_p >= cfg.enter_p:
        raise ValueError("exit_p must be < enter_p")
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    fires = causal_fires(
        labeled,
        blended,
        predictions=predictions,
        oof_availability_ts=oof_availability_ts,
        holdout_cut_ts=holdout_cut_ts,
        min_p=cfg.enter_p,
        expansion_start_ticks=cfg.expansion_start_ticks,
        holdout_months=cfg.holdout_months,
        progress=progress,
    )
    empty = PSizingReport(
        trades=pl.DataFrame(),
        summaries=pl.DataFrame(),
        diagnostics={
            "empty": True,
            "layer_id": LAYER_ID,
            "removable_layer": True,
            "does_not_modify_science_y": True,
            "holdout_scored": False,
            "live_predictions_not_used": True,
        },
    )
    if fires.height == 0:
        return empty
    path = attach_oof_p(blended, predictions)

    def _size(row: Mapping[str, Any]) -> float:
        raw = row.get("model_p")
        p = float(raw) if raw is not None else float("nan")
        return position_size(p, enter_p=cfg.enter_p, min_size=cfg.min_size)

    def _decide(window: Mapping[str, np.ndarray], scalars: Mapping[str, float]) -> tuple[int, str]:
        return _decide_p_exit(window, scalars, exit_p=cfg.exit_p)

    trades = walk_hold_windows(
        fires,
        path,
        max_hold_bars=cfg.max_hold_bars,
        path_cols=("model_p", "oof_p_at_bar"),
        decide_exit=_decide,
        round_trip_cost_pts=cfg.round_trip_cost_pts,
        size_of=_size,
    )
    reasons = (
        trades.group_by("exit_reason").agg(pl.len().cast(pl.Int64).alias("n")).sort("exit_reason")
        if trades.height
        else pl.DataFrame()
    )
    net_med, net_mean = median_mean(trades, "net_pts")
    real_med, real_mean = median_mean(trades, "realized_beyond_pts")
    size_med, size_mean = median_mean(trades, "size")
    summaries = pl.DataFrame(
        [
            {
                "layer_id": LAYER_ID,
                "n_trades": int(trades.height),
                "size_median": size_med,
                "size_mean": size_mean,
                "net_pts_median": net_med,
                "net_pts_mean": net_mean,
                "realized_pts_median": real_med,
                "realized_pts_mean": real_mean,
                "round_trip_cost_pts": float(cfg.round_trip_cost_pts),
                "enter_p": float(cfg.enter_p),
                "exit_p": float(cfg.exit_p),
            }
        ]
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "holdout_scored": False,
        "live_predictions_not_used": True,
        "p_flip_uses_fresh_oof_only": True,
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
        "size_median": size_med,
        "principles": (
            "removable overlay — delete p_sizing.py to remove",
            "size comes from entry OOF p; exit is the next fresh OOF p <= exit_p",
            "bars without an OOF score cannot flip — p is not invented",
            "live_predictions are never used for backtest",
            "no numeric take-profit; cost scales with size",
            "holdout never scored; completed-wave peak never used",
        ),
    }
    return PSizingReport(trades=trades, summaries=summaries, diagnostics=diagnostics)


def render_p_sizing_markdown(report: PSizingReport) -> str:
    d = report.diagnostics
    lines = [
        "# P-sizing overlay (removable layer C)",
        "",
        "Position size from OOF `p` at entry. Exit on the next **fresh** OOF score",
        "`p <= exit_p`, or at the research hold cap. Live predictions are not used.",
        "Delete this layer without touching science Y.",
        "",
        f"- layer_id={d.get('layer_id')} · removable={d.get('removable_layer')}",
        f"- trades={d.get('n_trades')} · size median={d.get('size_median')}",
        f"- net median={d.get('net_pts_median')} · net mean={d.get('net_pts_mean')}",
        f"- exit_reasons={d.get('exit_reasons')}",
        "",
        "## Principles",
        "",
    ]
    for item in d.get("principles", ()):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_p_sizing_report(report: PSizingReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.trades.height:
        report.trades.write_parquet(out / "p_sizing_trades.parquet")
    if report.summaries.height:
        report.summaries.write_parquet(out / "p_sizing_summaries.parquet")
    payload = {
        "diagnostics": jsonable(report.diagnostics),
        "holdout_scored": False,
        "removable_layer": True,
        "live_predictions_not_used": True,
    }
    (out / "p_sizing.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "P_SIZING.md").write_text(render_p_sizing_markdown(report), encoding="utf-8")
    return out


def run_p_sizing_from_period_dir(
    period_dir: Path | str,
    *,
    config: PSizingConfig | None = None,
    progress: ProgressLike | None = None,
) -> PSizingReport:
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(period_dir)
    return run_p_sizing(
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
    "PSizingConfig",
    "PSizingReport",
    "position_size",
    "render_p_sizing_markdown",
    "run_p_sizing",
    "run_p_sizing_from_period_dir",
    "write_p_sizing_report",
]
