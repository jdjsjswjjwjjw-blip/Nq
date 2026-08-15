"""معايرة احتمالات حقيقية: Brier + ECE (reliability)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """ملخص معايرة لهدف واحد أو مجمّع."""

    n: int
    brier: float
    ece: float
    mae: float
    base_rate: float
    detail: str = ""


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(np.square(y.astype(np.float64) - p.astype(np.float64))))


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """ECE على صناديق احتمالية متساوية العرض في [0,1]."""
    if y.size == 0:
        return 0.0
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y = y.astype(np.float64)
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (float(np.sum(mask)) / float(n)) * abs(acc - conf)
    return float(ece)


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    """جدول موثوقية: لكل صندوق mean(p), mean(y), count."""
    if y.size == 0:
        return pl.DataFrame(
            schema={
                "bin": pl.Int64(),
                "p_mean": pl.Float64(),
                "y_mean": pl.Float64(),
                "count": pl.Int64(),
            }
        )
    p = np.clip(p.astype(np.float64), 0.0, 1.0)
    y = y.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if not np.any(mask):
            rows.append({"bin": i, "p_mean": 0.0, "y_mean": 0.0, "count": 0})
            continue
        rows.append(
            {
                "bin": i,
                "p_mean": float(np.mean(p[mask])),
                "y_mean": float(np.mean(y[mask])),
                "count": int(np.sum(mask)),
            }
        )
    return pl.DataFrame(rows)


def evaluate_calibration(scored: pl.DataFrame, *, n_bins: int = 10) -> CalibrationReport:
    """معايرة من إطار فيه ``y`` و ``p_hat``."""
    if scored.height == 0 or "y" not in scored.columns or "p_hat" not in scored.columns:
        return CalibrationReport(n=0, brier=0.0, ece=0.0, mae=0.0, base_rate=0.0, detail="empty")
    y = scored["y"].fill_null(0.0).to_numpy().astype(np.float64)
    p = scored["p_hat"].fill_null(0.5).to_numpy().astype(np.float64)
    return CalibrationReport(
        n=int(y.size),
        brier=brier_score(y, p),
        ece=expected_calibration_error(y, p, n_bins=n_bins),
        mae=float(np.mean(np.abs(y - p))),
        base_rate=float(np.mean(y)),
        detail=f"reliability_bins={n_bins}",
    )


def evaluate_calibration_by_outcome(
    scored: pl.DataFrame,
    *,
    n_bins: int = 10,
) -> pl.DataFrame:
    if scored.height == 0 or "outcome_name" not in scored.columns:
        return pl.DataFrame(
            schema={
                "outcome_name": pl.Utf8(),
                "n": pl.Int64(),
                "brier": pl.Float64(),
                "ece": pl.Float64(),
                "mae": pl.Float64(),
                "base_rate": pl.Float64(),
            }
        )
    rows: list[dict[str, float | int | str]] = []
    for name, g in scored.group_by("outcome_name", maintain_order=True):
        outcome = name[0] if isinstance(name, tuple) else name
        rep = evaluate_calibration(g, n_bins=n_bins)
        rows.append(
            {
                "outcome_name": str(outcome),
                "n": rep.n,
                "brier": rep.brier,
                "ece": rep.ece,
                "mae": rep.mae,
                "base_rate": rep.base_rate,
            }
        )
    return pl.DataFrame(rows)


__all__ = [
    "CalibrationReport",
    "brier_score",
    "evaluate_calibration",
    "evaluate_calibration_by_outcome",
    "expected_calibration_error",
    "reliability_table",
]
