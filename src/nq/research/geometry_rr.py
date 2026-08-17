"""طبقة د: هندسة 1:4 إلى مستوى معروف عند ``t`` — قابلة للخلع.

نفس إطلاق OOF السببي، ونفس شريط التيك/30ث. السؤال ليس Y العلمي
(``y_path_further_beyond``) ولا فريم الشارت، بل: عند ``t``، نحو أقرب مستوى
آسيا/لندن المجمّد، هل يتحقق الهدف قبل وقف 1R أو الإبطال الهندسي
(العودة داخل قيمة آسيا)؟

الهدف = مستوى مجمّد عند ``t`` (لا ذروة موجة، لا مركّب لاحق).
الوقف = ``المسافة إلى الهدف / reward_multiple`` (افتراضي 4).
لا FVG: العمود غير موجود على الحالات المكتملة، ولا إعادة بناء MBO.

لا تغيّر Y العلمي. احذف هذا الملف + السكربت + ``_write_geometry_rr`` للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.research.causal_entry import TICKS_PER_NQ_POINT, assert_not_raw_mbo_stream
from nq.research.hold_horizon import (
    causal_fires,
    jsonable,
    load_overlay_period_inputs,
    median_mean,
    oof_timestamps,
    walk_hold_windows,
)
from nq.research.progress import ProgressLike

LAYER_ID = "geometry_rr"
_EPS = 1e-9
_PATH_COLS = (
    "close",
    "path_inside_asia_va",
    "target_ticks",
    "risk_ticks",
    "direction",
    "target_price",
    "stop_price",
)
PRICE_TICK = float(round(0.25 / PRICE_SCALE))

# أقرب مستوى أمام الاتجاه يفوز. عند التعادل يُفضَّل آسيا ثم المركّب ثم القرار.
GEOMETRY_LEVELS: tuple[str, ...] = (
    "asia_vah",
    "asia_val",
    "asia_poc",
    "asia_primary_hvn",
    "composite_vah",
    "composite_val",
    "composite_poc",
    "composite_primary_hvn",
    "decision_vah",
    "decision_val",
    "decision_poc",
)


@dataclass(frozen=True, slots=True)
class GeometryRRConfig:
    """عتبات ثابتة غير مُقدَّرة على العينة. سقف الإمساك عدد بارميل 30ث."""

    min_p: float = 0.5
    expansion_start_ticks: float = 16.0
    reward_multiple: float = 4.0
    min_ahead_ticks: float = 16.0  # 4 نقاط NQ — أضيق من ضوضاء التيك
    max_hold_bars: int = 120  # ساعة على فريم 30ث الممزوج — ليس تغيير شارت
    round_trip_cost_pts: float = 0.75
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class GeometryRRReport:
    trades: pl.DataFrame
    skipped: pl.DataFrame
    summaries: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _f64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    if frame.height == 0 or name not in frame.columns:
        return np.zeros(int(frame.height), dtype=np.float64)
    return frame[name].cast(pl.Float64).fill_null(0.0).to_numpy().astype(np.float64, copy=False)


def infer_direction(
    *,
    break_dir: np.ndarray,
    close: np.ndarray,
    asia_vah: np.ndarray,
    asia_val: np.ndarray,
) -> np.ndarray:
    """+1 كسر أعلى آسيا، −1 أدنى VAL. لا نظرة أمامية."""
    out = np.sign(break_dir).astype(np.float64)
    unk = np.abs(out) < _EPS
    out[unk & (close > asia_vah)] = 1.0
    out[unk & (close < asia_val)] = -1.0
    return out


def attach_geometry_at_t(
    entries: pl.DataFrame,
    *,
    min_ahead_ticks: float,
    reward_multiple: float,
    price_tick: float = PRICE_TICK,
) -> pl.DataFrame:
    """يجمّد أقرب مستوى أمام ``t`` والوقف 1R. لا يقرأ ذروة ولا مركّبًا لاحقًا."""
    n = int(entries.height)
    empty_extra = {
        "direction": pl.Float64(),
        "target_name": pl.Utf8(),
        "target_price": pl.Float64(),
        "target_ticks": pl.Float64(),
        "risk_ticks": pl.Float64(),
        "stop_price": pl.Float64(),
        "rr_multiple": pl.Float64(),
        "geometry_ok": pl.Boolean(),
        "skip_reason": pl.Utf8(),
    }
    if n == 0:
        return entries.with_columns(
            pl.lit(None, dtype=dtype).alias(name) for name, dtype in empty_extra.items()
        )
    if min_ahead_ticks < 1.0:
        raise ValueError("min_ahead_ticks must be >= 1")
    if reward_multiple <= 0.0:
        raise ValueError("reward_multiple must be > 0")
    if price_tick <= 0.0:
        raise ValueError("price_tick must be > 0")

    close = _f64_arr(entries, "close")
    asia_vah = _f64_arr(entries, "asia_vah")
    asia_val = _f64_arr(entries, "asia_val")
    break_dir = _f64_arr(entries, "proj_break_direction")
    direction = infer_direction(
        break_dir=break_dir, close=close, asia_vah=asia_vah, asia_val=asia_val
    )
    best_ahead = np.full(n, np.inf, dtype=np.float64)
    best_price = np.full(n, np.nan, dtype=np.float64)
    best_name: list[str | None] = [None] * n
    for name in GEOMETRY_LEVELS:
        level = _f64_arr(entries, name)
        valid = np.isfinite(level) & (np.abs(level) > _EPS) & np.isfinite(close)
        ahead = np.where(
            direction > 0.0,
            (level - close) / price_tick,
            np.where(direction < 0.0, (close - level) / price_tick, np.nan),
        )
        take = valid & np.isfinite(ahead) & (ahead >= float(min_ahead_ticks)) & (ahead < best_ahead)
        best_ahead = np.where(take, ahead, best_ahead)
        best_price = np.where(take, level, best_price)
        for i, flag in enumerate(take.tolist()):
            if flag:
                best_name[i] = name

    has_level = np.isfinite(best_ahead) & (best_ahead < np.inf)
    risk = np.where(has_level, best_ahead / float(reward_multiple), np.nan)
    stop_price = np.where(
        has_level,
        close - direction * risk * price_tick,
        np.nan,
    )
    skip: list[str | None] = [None] * n
    ok = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if abs(float(direction[i])) < _EPS:
            skip[i] = "no_direction"
        elif not bool(has_level[i]):
            skip[i] = "no_level_ahead"
        else:
            ok[i] = True
            skip[i] = None
    return entries.with_columns(
        pl.Series("direction", direction),
        pl.Series("target_name", best_name, dtype=pl.Utf8()),
        pl.Series("target_price", best_price),
        pl.Series("target_ticks", np.where(has_level, best_ahead, np.nan)),
        pl.Series("risk_ticks", risk),
        pl.Series("stop_price", stop_price),
        pl.lit(float(reward_multiple), dtype=pl.Float64).alias("rr_multiple"),
        pl.Series("geometry_ok", ok),
        pl.Series("skip_reason", skip, dtype=pl.Utf8()),
    )


def _hit_stop_or_take(
    *,
    px: float | None,
    beyond: float,
    beyond0: float,
    direction: float,
    target: float,
    risk: float,
    target_px: float,
    stop_px: float,
) -> str | None:
    """وقف 1R قبل الهدف. سعر مجمّد إن وُجد، وإلا مسار ما بعد آسيا."""
    if px is not None:
        adverse = (stop_px - px) * direction
        favorable = (px - target_px) * direction
        if adverse >= 0.0:
            return "stop_1r"
        if favorable >= 0.0:
            return "take_4r"
        return None
    if np.isfinite(risk) and beyond <= beyond0 - risk:
        return "stop_1r"
    if np.isfinite(target) and beyond >= beyond0 + target:
        return "take_4r"
    return None


def _decide_geometry_exit(
    window: Mapping[str, np.ndarray],
    scalars: Mapping[str, float],
) -> tuple[int, str]:
    beyond = window["path_beyond_asia_ticks"]
    n = int(beyond.size)
    if n == 0:
        return 0, "empty"
    beyond0 = float(scalars.get("beyond0", 0.0))
    target = float(scalars.get("target_ticks_0", 0.0))
    risk = float(scalars.get("risk_ticks_0", 0.0))
    direction = float(scalars.get("direction_0", 0.0))
    target_px = float(scalars.get("target_price_0", float("nan")))
    stop_px = float(scalars.get("stop_price_0", float("nan")))
    close_arr = window.get("close")
    inside = window.get("path_inside_asia_va")
    use_price = (
        close_arr is not None
        and np.isfinite(target_px)
        and np.isfinite(stop_px)
        and abs(direction) > _EPS
    )
    for k in range(n):
        if inside is not None and float(inside[k]) > _EPS:
            return k, "asia_return"
        px = float(close_arr[k]) if use_price and close_arr is not None else None
        hit = _hit_stop_or_take(
            px=px,
            beyond=float(beyond[k]),
            beyond0=beyond0,
            direction=direction,
            target=target,
            risk=risk,
            target_px=target_px,
            stop_px=stop_px,
        )
        if hit is not None:
            return k, hit
    return n - 1, "max_hold"


def _reason_counts(frame: pl.DataFrame, col: str) -> dict[str, int]:
    if frame.height == 0 or col not in frame.columns:
        return {}
    grouped = frame.group_by(col).agg(pl.len().cast(pl.Int64).alias("n")).sort(col)
    return {str(row[col]): int(row["n"]) for row in grouped.iter_rows(named=True)}


def run_geometry_rr(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: GeometryRRConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> GeometryRRReport:
    """1:4 إلى مستوى مجمّد عند t بعد إطلاق OOF — بلا ذروة وبلا holdout."""
    cfg = config or GeometryRRConfig()
    if cfg.reward_multiple <= 0.0:
        raise ValueError("reward_multiple must be > 0")
    if cfg.min_ahead_ticks < 1.0:
        raise ValueError("min_ahead_ticks must be >= 1")
    if cfg.max_hold_bars < 1:
        raise ValueError("max_hold_bars must be >= 1")
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    empty = GeometryRRReport(
        trades=pl.DataFrame(),
        skipped=pl.DataFrame(),
        summaries=pl.DataFrame(),
        diagnostics={
            "empty": True,
            "layer_id": LAYER_ID,
            "removable_layer": True,
            "does_not_modify_science_y": True,
            "chart_timeframe_unchanged": True,
            "holdout_scored": False,
            "completed_wave_peak_not_used": True,
            "fvg_not_on_completed_states": True,
        },
    )
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
    if fires.height == 0:
        return empty
    geo = attach_geometry_at_t(
        fires,
        min_ahead_ticks=cfg.min_ahead_ticks,
        reward_multiple=cfg.reward_multiple,
    )
    traded = geo.filter(pl.col("geometry_ok"))
    skipped = geo.filter(~pl.col("geometry_ok"))
    if traded.height == 0:
        summaries = pl.DataFrame(
            [
                {
                    "layer_id": LAYER_ID,
                    "n_fires": int(geo.height),
                    "n_traded": 0,
                    "n_skipped": int(skipped.height),
                    "take_4r_rate": None,
                    "stop_1r_rate": None,
                    "net_pts_median": None,
                    "net_pts_mean": None,
                    "reward_multiple": float(cfg.reward_multiple),
                    "min_ahead_ticks": float(cfg.min_ahead_ticks),
                    "max_hold_bars": int(cfg.max_hold_bars),
                    "round_trip_cost_pts": float(cfg.round_trip_cost_pts),
                }
            ]
        )
        return GeometryRRReport(
            trades=pl.DataFrame(),
            skipped=skipped,
            summaries=summaries,
            diagnostics={
                "empty": False,
                "layer_id": LAYER_ID,
                "removable_layer": True,
                "does_not_modify_science_y": True,
                "chart_timeframe_unchanged": True,
                "holdout_scored": False,
                "completed_wave_peak_not_used": True,
                "fvg_not_on_completed_states": True,
                "live_predictions_not_used": True,
                "levels_frozen_at_t": True,
                "n_fires": int(geo.height),
                "n_traded": 0,
                "n_skipped": int(skipped.height),
                "skip_reasons": _reason_counts(skipped, "skip_reason"),
                "exit_reasons": {},
                "principles": _principles(),
            },
        )

    trades = walk_hold_windows(
        traded,
        blended,
        max_hold_bars=cfg.max_hold_bars,
        path_cols=_PATH_COLS,
        decide_exit=_decide_geometry_exit,
        round_trip_cost_pts=cfg.round_trip_cost_pts,
    )
    extras = traded.with_row_index("entry_i").select(
        "entry_i",
        "target_name",
        "target_price",
        "target_ticks",
        "risk_ticks",
        "stop_price",
        "direction",
        "rr_multiple",
    )
    if trades.height:
        trades = trades.join(extras, on="entry_i", how="left")
        trades = trades.with_columns(
            (pl.col("target_ticks") / TICKS_PER_NQ_POINT).alias("target_pts"),
            (pl.col("risk_ticks") / TICKS_PER_NQ_POINT).alias("risk_pts"),
        )
    n_traded = int(trades.height)
    take_n = int((trades["exit_reason"] == "take_4r").sum()) if n_traded else 0
    stop_n = int((trades["exit_reason"] == "stop_1r").sum()) if n_traded else 0
    net_med, net_mean = median_mean(trades, "net_pts")
    real_med, real_mean = median_mean(trades, "realized_beyond_pts")
    mfe_med, mfe_mean = median_mean(trades, "mfe_beyond_pts")
    mae_med, mae_mean = median_mean(trades, "mae_beyond_pts")
    tgt_med, tgt_mean = median_mean(trades, "target_pts")
    risk_med, risk_mean = median_mean(trades, "risk_pts")
    summaries = pl.DataFrame(
        [
            {
                "layer_id": LAYER_ID,
                "n_fires": int(geo.height),
                "n_traded": n_traded,
                "n_skipped": int(skipped.height),
                "take_4r_n": take_n,
                "stop_1r_n": stop_n,
                "take_4r_rate": (take_n / n_traded) if n_traded else None,
                "stop_1r_rate": (stop_n / n_traded) if n_traded else None,
                "net_pts_median": net_med,
                "net_pts_mean": net_mean,
                "realized_pts_median": real_med,
                "realized_pts_mean": real_mean,
                "mfe_pts_median": mfe_med,
                "mfe_pts_mean": mfe_mean,
                "mae_pts_median": mae_med,
                "mae_pts_mean": mae_mean,
                "target_pts_median": tgt_med,
                "target_pts_mean": tgt_mean,
                "risk_pts_median": risk_med,
                "risk_pts_mean": risk_mean,
                "reward_multiple": float(cfg.reward_multiple),
                "min_ahead_ticks": float(cfg.min_ahead_ticks),
                "max_hold_bars": int(cfg.max_hold_bars),
                "round_trip_cost_pts": float(cfg.round_trip_cost_pts),
            }
        ]
    )
    diagnostics: dict[str, Any] = {
        "empty": False,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "chart_timeframe_unchanged": True,
        "holdout_scored": False,
        "completed_wave_peak_not_used": True,
        "wave_frac_not_used": True,
        "fvg_not_on_completed_states": True,
        "live_predictions_not_used": True,
        "levels_frozen_at_t": True,
        "hold_horizon_is_bars_not_label_window": True,
        "n_fires": int(geo.height),
        "n_traded": n_traded,
        "n_skipped": int(skipped.height),
        "skip_reasons": _reason_counts(skipped, "skip_reason"),
        "exit_reasons": _reason_counts(trades, "exit_reason"),
        "target_names": _reason_counts(trades, "target_name"),
        "take_4r_rate": (take_n / n_traded) if n_traded else None,
        "stop_1r_rate": (stop_n / n_traded) if n_traded else None,
        "net_pts_median": net_med,
        "net_pts_mean": net_mean,
        "target_pts_median": tgt_med,
        "risk_pts_median": risk_med,
        "principles": _principles(),
    }
    return GeometryRRReport(
        trades=trades,
        skipped=skipped,
        summaries=summaries,
        diagnostics=diagnostics,
    )


def _principles() -> tuple[str, ...]:
    return (
        "removable overlay — delete geometry_rr.py to remove",
        "chart timeframe is unchanged; hold cap is N bars on the 30s blended clock",
        "science Y is unchanged; this is not a new label horizon",
        "target is the nearest Asia/London/decision level frozen at t",
        "stop is target_distance / 4; asia VA return is extra geometric invalidation",
        "completed-wave peak / remaining-to-peak / FVG-from-MBO are never used",
        "holdout never scored; live_predictions never used",
    )


def render_geometry_rr_markdown(report: GeometryRRReport) -> str:
    d = report.diagnostics
    lines = [
        "# Geometry 1:4 overlay (removable layer D)",
        "",
        "Same causal OOF fire at `t`, same 30-second states. **Not** a chart",
        "timeframe change and **not** a new science Y.",
        "Target = nearest Asia / composite / decision level **frozen at t**.",
        "Stop = that distance / 4. Extra invalidation = return inside Asia VA.",
        "FVG is omitted: it is not on completed states and MBO is not reloaded.",
        "Delete this layer without touching science Y.",
        "",
        f"- layer_id={d.get('layer_id')} · removable={d.get('removable_layer')}",
        f"- chart_timeframe_unchanged={d.get('chart_timeframe_unchanged')}",
        f"- levels_frozen_at_t={d.get('levels_frozen_at_t')}",
        f"- fires={d.get('n_fires')} · traded={d.get('n_traded')} · skipped={d.get('n_skipped')}",
        f"- take_4r_rate={d.get('take_4r_rate')} · stop_1r_rate={d.get('stop_1r_rate')}",
        f"- target pts median={d.get('target_pts_median')} · "
        f"risk pts median={d.get('risk_pts_median')}",
        f"- net median={d.get('net_pts_median')} · net mean={d.get('net_pts_mean')}",
        f"- skip_reasons={d.get('skip_reasons')}",
        f"- exit_reasons={d.get('exit_reasons')}",
        f"- target_names={d.get('target_names')}",
        "",
        "## Principles",
        "",
    ]
    for item in d.get("principles", ()):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_geometry_rr_report(report: GeometryRRReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.trades.height:
        report.trades.write_parquet(out / "geometry_rr_trades.parquet")
    if report.skipped.height:
        report.skipped.write_parquet(out / "geometry_rr_skipped.parquet")
    if report.summaries.height:
        report.summaries.write_parquet(out / "geometry_rr_summaries.parquet")
    payload = {
        "diagnostics": jsonable(report.diagnostics),
        "holdout_scored": False,
        "removable_layer": True,
        "chart_timeframe_unchanged": True,
        "does_not_modify_science_y": True,
    }
    (out / "geometry_rr.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "GEOMETRY.md").write_text(render_geometry_rr_markdown(report), encoding="utf-8")
    return out


def run_geometry_rr_from_period_dir(
    period_dir: Path | str,
    *,
    config: GeometryRRConfig | None = None,
    progress: ProgressLike | None = None,
) -> GeometryRRReport:
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(period_dir)
    return run_geometry_rr(
        labeled,
        blended,
        config=config,
        oof_availability_ts=oof_timestamps(oof),
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=progress,
    )


__all__ = [
    "GEOMETRY_LEVELS",
    "LAYER_ID",
    "PRICE_TICK",
    "GeometryRRConfig",
    "GeometryRRReport",
    "attach_geometry_at_t",
    "infer_direction",
    "render_geometry_rr_markdown",
    "run_geometry_rr",
    "run_geometry_rr_from_period_dir",
    "write_geometry_rr_report",
]
