"""طبقة فلترة قابلة للخلع: تيك المسار ثم طور الامتداد — بلا إعادة تدريب.

دخول = ``p_path ≥ 0.5`` **و** ``p_phase_extend ≥ 0.5``.
التيك بوابة حيّة؛ الطور فلتر الجودة العلوي. لا يغيّر Y العلمي ولا يُصدَّر
من ``nq.research.__init__``. احذف هذا الملف + السكربت + الاختبار للإزالة.

القياس على OOF الموسوم فقط (``fold_scores``). تنبؤات live ممنوعة.
بلا خروج آلي: MAE / MFE / إصابة 20 نقطة خلال 50 برميلًا بعد الإعداد.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.auction_behavior.realized_path import Y_PHASE_EXTEND
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.behavior_period import assert_not_raw_mbo_stream
from nq.validation.leakage import assert_causal_order

LAYER_ID = "path_phase_cascade"
Y_PATH = "y_path_further_beyond"
MIN_P_PATH = 0.5
MIN_P_PHASE = 0.5
QUALITY_HORIZON_BARS = 50
FAR_TARGET_POINTS = 20.0
_GROUP = "_behavior_story_run"
_DAY = "_period_day_id"
_ONSET = 0.5
_FIXED_POINT_FLOOR = 1.0 / float(PRICE_SCALE)


def _price_to_points(px: float) -> float:
    value = float(px)
    if abs(value) >= _FIXED_POINT_FLOOR:
        return value * float(PRICE_SCALE)
    return value


_COHORT_PATH = "path_only"
_COHORT_CASCADE = "path_and_phase"


def _p_col(frame: pl.DataFrame) -> str:
    if "p_cal" in frame.columns:
        return "p_cal"
    if "p_hat" in frame.columns:
        return "p_hat"
    raise ValueError("fold_scores must have p_cal or p_hat")


def assert_oof_fold_scores(frame: pl.DataFrame) -> None:
    """يرفض التنبؤ الحي والإطار العريض لكل بار."""
    if frame.height == 0:
        return
    if "outcome_name" not in frame.columns:
        raise ValueError("cascade uses labeled OOF fold_scores, not wide state p_y_*")
    names = {str(x) for x in frame["outcome_name"].unique().to_list()}
    if Y_PATH not in names:
        raise ValueError(f"fold_scores missing {Y_PATH}")
    if Y_PHASE_EXTEND not in names:
        raise ValueError(
            f"fold_scores missing {Y_PHASE_EXTEND}; "
            "use the period OOF that includes the phase-extend head"
        )
    if "eligible_for_backtest" in frame.columns:
        flags = [bool(x) for x in frame["eligible_for_backtest"].to_list()]
        if flags and not all(flags):
            raise ValueError("live / ineligible rows are not allowed")
    if "prediction_is_oof" in frame.columns:
        flags = [bool(x) for x in frame["prediction_is_oof"].to_list()]
        if flags and not all(flags):
            raise ValueError("live predictions are not allowed")


def _score_map(frame: pl.DataFrame, outcome: str) -> dict[int, float]:
    part = frame.filter(pl.col("outcome_name") == outcome)
    if part.height == 0:
        return {}
    col = _p_col(part)
    ts = part[SETUP_AVAILABILITY_TS].to_list()
    p = part[col].cast(pl.Float64).to_list()
    out: dict[int, float] = {}
    for t, prob in zip(ts, p, strict=True):
        if prob is None:
            continue
        value = float(prob)
        if np.isfinite(value):
            out[int(t)] = value
    return out


def _infer_direction(
    *,
    brk_dir: float,
    close_pts: float,
    asia_vah: float,
    asia_val: float,
    has_asia: bool,
) -> float:
    if abs(float(brk_dir)) >= _ONSET:
        return float(np.sign(brk_dir))
    if not has_asia:
        return 0.0
    if close_pts >= asia_vah > 0.0:
        return 1.0
    if asia_val > 0.0 and close_pts <= asia_val:
        return -1.0
    return 0.0


def _quality_row(
    *,
    i: int,
    direction: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    groups: np.ndarray,
    window: int,
    far_pts: float,
) -> dict[str, float | bool]:
    entry = float(close[i])
    n = close.size
    visible = 0
    mae = 0.0
    mfe = 0.0
    for j in range(i + 1, min(n, i + window + 1)):
        if groups[j] != groups[i]:
            break
        visible += 1
        if direction > 0:
            mae = max(mae, entry - float(low[j]))
            mfe = max(mfe, float(high[j]) - entry)
        else:
            mae = max(mae, float(high[j]) - entry)
            mfe = max(mfe, entry - float(low[j]))
    complete = visible >= int(window)
    return {
        "mae_pts": float(mae),
        "mfe_pts": float(mfe),
        "hit_far": bool(mfe >= float(far_pts)),
        "window_complete": bool(complete),
        "horizon_bars": float(visible),
    }


def score_entry_quality(
    blended: pl.DataFrame,
    fires: pl.DataFrame,
    *,
    window: int = QUALITY_HORIZON_BARS,
    far_pts: float = FAR_TARGET_POINTS,
) -> pl.DataFrame:
    """MAE / MFE / إصابة الهدف البعيد — بلا وقف أو هدف للتنفيذ."""
    schema = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        "cohort": pl.Utf8(),
        "direction": pl.Float64(),
        "p_path": pl.Float64(),
        "p_phase": pl.Float64(),
        "mae_pts": pl.Float64(),
        "mfe_pts": pl.Float64(),
        "hit_far": pl.Boolean(),
        "window_complete": pl.Boolean(),
        "horizon_bars": pl.Int64(),
        "story_run": pl.Int64(),
        "day_id": pl.Utf8(),
    }
    if fires.height == 0 or blended.height == 0:
        return pl.DataFrame(schema=schema)
    work = blended.sort(AVAILABILITY_TS)
    ts = work[AVAILABILITY_TS].to_numpy().astype(np.int64)
    assert_causal_order(ts)
    index = {int(t): i for i, t in enumerate(ts.tolist())}
    n = work.height
    groups = (
        work[_GROUP].fill_null(-1).to_numpy().astype(np.int64)
        if _GROUP in work.columns
        else np.zeros(n, dtype=np.int64)
    )
    close = np.array(
        [_price_to_points(float(v)) for v in work["close"].fill_null(0.0).to_list()],
        dtype=np.float64,
    )
    high = np.array(
        [_price_to_points(float(v)) for v in work["high"].fill_null(0.0).to_list()],
        dtype=np.float64,
    )
    low = np.array(
        [_price_to_points(float(v)) for v in work["low"].fill_null(0.0).to_list()],
        dtype=np.float64,
    )
    brk = (
        work["proj_break_direction"].fill_null(0.0).to_numpy().astype(np.float64)
        if "proj_break_direction" in work.columns
        else np.zeros(n, dtype=np.float64)
    )
    has_asia = "asia_vah" in work.columns and "asia_val" in work.columns
    asia_vah = (
        np.array(
            [_price_to_points(float(v)) for v in work["asia_vah"].fill_null(0.0).to_list()],
            dtype=np.float64,
        )
        if has_asia
        else np.zeros(n, dtype=np.float64)
    )
    asia_val = (
        np.array(
            [_price_to_points(float(v)) for v in work["asia_val"].fill_null(0.0).to_list()],
            dtype=np.float64,
        )
        if has_asia
        else np.zeros(n, dtype=np.float64)
    )
    has_day = _DAY in work.columns
    day_ids = (
        [str(x) if x is not None else "" for x in work[_DAY].to_list()] if has_day else [""] * n
    )
    rows: list[dict[str, object]] = []
    for rec in fires.iter_rows(named=True):
        setup = int(rec[SETUP_AVAILABILITY_TS])
        i = index.get(setup)
        if i is None:
            continue
        direction = _infer_direction(
            brk_dir=float(brk[i]),
            close_pts=float(close[i]),
            asia_vah=float(asia_vah[i]),
            asia_val=float(asia_val[i]),
            has_asia=has_asia,
        )
        if abs(direction) < _ONSET:
            continue
        q = _quality_row(
            i=i,
            direction=direction,
            close=close,
            high=high,
            low=low,
            groups=groups,
            window=window,
            far_pts=far_pts,
        )
        rows.append(
            {
                SETUP_AVAILABILITY_TS: setup,
                "cohort": rec["cohort"],
                "direction": direction,
                "p_path": float(rec["p_path"]),
                "p_phase": float(rec["p_phase"]),
                "mae_pts": q["mae_pts"],
                "mfe_pts": q["mfe_pts"],
                "hit_far": q["hit_far"],
                "window_complete": q["window_complete"],
                "horizon_bars": int(q["horizon_bars"]),
                "story_run": int(groups[i]),
                "day_id": day_ids[i],
            }
        )
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def build_cascade_fires(
    fold_scores: pl.DataFrame,
    *,
    min_p_path: float = MIN_P_PATH,
    min_p_phase: float = MIN_P_PHASE,
) -> pl.DataFrame:
    """مسارين: التيك وحده، والتيك ثم الطور. عتبة 0.5 معلنة وليست معايرة OOF."""
    assert_oof_fold_scores(fold_scores)
    schema = {
        SETUP_AVAILABILITY_TS: pl.Int64(),
        "cohort": pl.Utf8(),
        "p_path": pl.Float64(),
        "p_phase": pl.Float64(),
    }
    if fold_scores.height == 0:
        return pl.DataFrame(schema=schema)
    path_p = _score_map(fold_scores, Y_PATH)
    phase_p = _score_map(fold_scores, Y_PHASE_EXTEND)
    rows: list[dict[str, object]] = []
    for ts, p_path in path_p.items():
        p_phase = phase_p.get(ts)
        if p_path < float(min_p_path):
            continue
        phase_val = float(p_phase) if p_phase is not None and np.isfinite(p_phase) else float("nan")
        rows.append(
            {
                SETUP_AVAILABILITY_TS: int(ts),
                "cohort": _COHORT_PATH,
                "p_path": float(p_path),
                "p_phase": phase_val,
            }
        )
        if p_phase is not None and float(p_phase) >= float(min_p_phase):
            rows.append(
                {
                    SETUP_AVAILABILITY_TS: int(ts),
                    "cohort": _COHORT_CASCADE,
                    "p_path": float(p_path),
                    "p_phase": float(p_phase),
                }
            )
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def summarize_cohort(quality: pl.DataFrame, cohort: str) -> dict[str, Any]:
    part = quality.filter(pl.col("cohort") == cohort) if quality.height else quality
    n = int(part.height)
    mae = part["mae_pts"].to_list() if n else []
    mfe = part["mfe_pts"].to_list() if n else []
    hits = int(part.filter(pl.col("hit_far")).height) if n else 0
    n_stories = int(part["story_run"].n_unique()) if n and "story_run" in part.columns else 0
    if n and "day_id" in part.columns:
        days = [str(x) for x in part["day_id"].to_list() if str(x)]
        n_days = len(set(days))
    else:
        n_days = 0
    return {
        "cohort": cohort,
        "n_fires": n,
        "n_unique_stories": n_stories,
        "n_unique_days": n_days,
        "mean_mae_pts": _finite_mean(mae) if n else float("nan"),
        "median_mae_pts": _finite_median(mae) if n else float("nan"),
        "mean_mfe_pts": _finite_mean(mfe) if n else float("nan"),
        "median_mfe_pts": _finite_median(mfe) if n else float("nan"),
        "hit_far_rate": (hits / n) if n else float("nan"),
        "n_hit_far": hits,
        "far_target_points": FAR_TARGET_POINTS,
        "horizon_bars": QUALITY_HORIZON_BARS,
    }


def run_path_phase_cascade(
    *,
    blended: pl.DataFrame,
    fold_scores: pl.DataFrame,
    min_p_path: float = MIN_P_PATH,
    min_p_phase: float = MIN_P_PHASE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """يقارن التيك وحده بالتيك+الطور على نفس إعدادات OOF."""
    assert_not_raw_mbo_stream(blended, source="cascade blended")
    assert_oof_fold_scores(fold_scores)
    fires = build_cascade_fires(fold_scores, min_p_path=min_p_path, min_p_phase=min_p_phase)
    quality = score_entry_quality(blended, fires)
    n_gate_path = int(fires.filter(pl.col("cohort") == _COHORT_PATH).height)
    n_gate_cas = int(fires.filter(pl.col("cohort") == _COHORT_CASCADE).height)
    path_s = summarize_cohort(quality, _COHORT_PATH)
    cas_s = summarize_cohort(quality, _COHORT_CASCADE)
    diagnostics: dict[str, Any] = {
        "layer_id": LAYER_ID,
        "is_live_overlay": False,
        "overlay_fire_set_used": False,
        "retrained": False,
        "holdout_touched": False,
        "thresholds_tuned_on_oof": False,
        "auto_exit": False,
        "min_p_path": float(min_p_path),
        "min_p_phase": float(min_p_phase),
        "n_fold_score_rows": int(fold_scores.height),
        "n_path_only_gate": n_gate_path,
        "n_path_and_phase_gate": n_gate_cas,
        "n_path_only_unscored": n_gate_path - int(path_s["n_fires"]),
        "n_path_and_phase_unscored": n_gate_cas - int(cas_s["n_fires"]),
        "path_only": path_s,
        "path_and_phase": cas_s,
    }
    return quality, diagnostics


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.2f}"


def write_cascade_report(
    quality: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if quality.height:
        quality.write_parquet(out / "cascade_quality.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    path_s = diagnostics.get("path_only", {})
    cas_s = diagnostics.get("path_and_phase", {})
    min_path = diagnostics.get("min_p_path")
    min_phase = diagnostics.get("min_p_phase")
    holdout = diagnostics.get("holdout_touched")
    retrained = diagnostics.get("retrained")
    n_path_gate = diagnostics.get("n_path_only_gate")
    n_cas_gate = diagnostics.get("n_path_and_phase_gate")
    lines = [
        "# path + phase cascade (removable filter)",
        "",
        "Not a retrain. Not live execution. Labeled OOF fold_scores only.",
        "Enter iff `p_path ≥ 0.5` and `p_phase_extend ≥ 0.5`. No automatic exit.",
        "Thresholds are declared, not tuned on this OOF. Not the 5642 overlay fire set.",
        "Unscored rows are undirected or missing from blended. Small n is not an edge.",
        "",
        f"- min_p_path={min_path} · min_p_phase={min_phase}",
        f"- horizon={QUALITY_HORIZON_BARS} bars · far target={FAR_TARGET_POINTS} pts",
        f"- holdout_touched={holdout} · retrained={retrained}",
        f"- path gate={n_path_gate} · cascade gate={n_cas_gate}",
        "",
        "| cohort | n gate | n scored | stories | mean MAE | median MAE | mean MFE | hit 20+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| path_only | {int(diagnostics.get('n_path_only_gate', 0))} | "
            f"{int(path_s.get('n_fires', 0))} | {int(path_s.get('n_unique_stories', 0))} | "
            f"{_fmt(path_s.get('mean_mae_pts'))} | {_fmt(path_s.get('median_mae_pts'))} | "
            f"{_fmt(path_s.get('mean_mfe_pts'))} | {_fmt(path_s.get('hit_far_rate'))} |"
        ),
        (
            f"| path_and_phase | {int(diagnostics.get('n_path_and_phase_gate', 0))} | "
            f"{int(cas_s.get('n_fires', 0))} | {int(cas_s.get('n_unique_stories', 0))} | "
            f"{_fmt(cas_s.get('mean_mae_pts'))} | {_fmt(cas_s.get('median_mae_pts'))} | "
            f"{_fmt(cas_s.get('mean_mfe_pts'))} | {_fmt(cas_s.get('hit_far_rate'))} |"
        ),
        "",
    ]
    (out / "CASCADE.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "FAR_TARGET_POINTS",
    "LAYER_ID",
    "MIN_P_PATH",
    "MIN_P_PHASE",
    "QUALITY_HORIZON_BARS",
    "Y_PATH",
    "assert_oof_fold_scores",
    "build_cascade_fires",
    "run_path_phase_cascade",
    "score_entry_quality",
    "write_cascade_report",
]
