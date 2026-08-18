"""مسار اختلال NQ/MNQ لخمس دقائق بعد القمة. بلا نموذج.

نافذة الهبوط السابقة كانت ``[H+30s, H+60s)`` ففاتت ``[H, H+30s)`` وباقي الخمس
دقائق حتى القاع. هنا شرائح 30ث من ``H`` إلى ``H+300s``. مسح اليوم في
``triple_tape`` كان أصلًا أفق 5 دقائق للهبوط السعري؛ نضيف ضابط ``T_rate>50``
واختلال NQ الأمامي على نفس الأفق.
NQ بلا MBO: Fill لا يُختلق. ليست overlay ولا LSTM.
احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS
from nq.research.cross_nq_mnq import MNQ_MULT, NQ_MULT, _first_t_ts, _with_notional
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import WINDOW_S, NamedWindow
from nq.research.peak_flow import score_flow_window
from nq.research.peak_pattern import HORIZON_S
from nq.research.triple_tape import NQ_IMB_NEAR_ZERO

LAYER_ID = "horizon_flow"
BIN_S: Final = 30
_TRADE = MboAction.TRADE.value


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def post_peak_windows(
    high_ts: int,
    *,
    window_s: int = WINDOW_S,
    horizon_s: int = HORIZON_S,
    bin_s: int = BIN_S,
) -> tuple[NamedWindow, ...]:
    """قمة ``[H-30s, H)`` ثم شرائح ``bin_s`` على ``[H, H+horizon)``."""

    if horizon_s % bin_s != 0:
        raise ValueError("horizon_s must be divisible by bin_s")
    peak = NamedWindow("peak", int(high_ts) - window_s * SECOND_NS, int(high_ts))
    bins: list[NamedWindow] = []
    for i in range(int(horizon_s) // int(bin_s)):
        start = int(high_ts) + i * bin_s * SECOND_NS
        end = start + bin_s * SECOND_NS
        bins.append(NamedWindow(f"+{i * bin_s}-{i * bin_s + bin_s}s", start, end))
    whole = NamedWindow("+0-300s", int(high_ts), int(high_ts) + horizon_s * SECOND_NS)
    return (peak, *bins, whole)


def _t_extrema(book: pl.DataFrame, window: NamedWindow) -> tuple[float, float]:
    chunk = book.filter(
        (pl.col("action") == _TRADE)
        & (pl.col(EVENT_TS) >= window.start_ts)
        & (pl.col(EVENT_TS) < window.end_ts)
    )
    if chunk.height == 0:
        return float("nan"), float("nan")
    lo = chunk.select(pl.col("price").min()).item()
    hi = chunk.select(pl.col("price").max()).item()
    return (
        float("nan") if lo is None else float(lo),
        float("nan") if hi is None else float(hi),
    )


def _bin_row(
    mnq_mbo: pl.DataFrame,
    mnq_tape: pl.DataFrame,
    nq_tape: pl.DataFrame,
    window: NamedWindow,
    high_ts: int,
) -> dict[str, Any]:
    mbo = _with_notional(
        score_flow_window(mnq_mbo, window), contract="MNQ", multiplier=MNQ_MULT, window=window
    )
    tape = _with_notional(
        score_flow_window(mnq_tape, window), contract="MNQ", multiplier=MNQ_MULT, window=window
    )
    nq = _with_notional(
        score_flow_window(nq_tape, window), contract="NQ", multiplier=NQ_MULT, window=window
    )
    min_px, max_px = _t_extrema(mnq_mbo, window)
    return {
        "name": window.name,
        "start_ts": window.start_ts,
        "end_ts": window.end_ts,
        "offset_s": (window.start_ts - high_ts) / float(SECOND_NS),
        "mnq_mbo_t_per_s": mbo["t_per_s"],
        "mnq_mbo_t_imbalance": mbo["t_imbalance"],
        "mnq_fill_ratio": mbo["ask_hit_share"],
        "mnq_tape_t_per_s": tape["t_per_s"],
        "mnq_tape_t_imbalance": tape["t_imbalance"],
        "nq_t_per_s": nq["t_per_s"],
        "nq_t_imbalance": nq["t_imbalance"],
        "nq_n_t": nq["n_t"],
        "nq_t_notional": nq["t_notional"],
        "nq_fill_ratio": float("nan"),
        "min_px": min_px,
        "max_px": max_px,
    }


def _path_flags(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["name"]: r for r in rows}
    step = [r for r in rows if r["name"].startswith("+") and r["name"] != "+0-300s"]
    nq_vals = [float(r["nq_t_imbalance"]) for r in step]
    finite = [v for v in nq_vals if not math.isnan(v)]
    min_imb = min(finite) if finite else float("nan")
    whole = by.get("+0-300s", {})
    first = by.get("+0-30s", {})
    old_drop = by.get("+30-60s", {})
    last = by.get("+270-300s", {})
    peak = by.get("peak", {})
    whole_imb = float(whole.get("nq_t_imbalance", float("nan")))
    px_vals = [float(r["min_px"]) for r in step if not math.isnan(float(r["min_px"]))]
    return {
        "peak_nq_imbalance": peak.get("nq_t_imbalance"),
        "first_30s_nq_imbalance": first.get("nq_t_imbalance"),
        "old_drop_30_60s_nq_imbalance": old_drop.get("nq_t_imbalance"),
        "last_30s_nq_imbalance": last.get("nq_t_imbalance"),
        "horizon_5m_nq_imbalance": whole_imb,
        "min_step_nq_imbalance": min_imb,
        "nq_faded_near_zero": bool(
            (not math.isnan(whole_imb) and abs(whole_imb) < NQ_IMB_NEAR_ZERO)
            or any(not math.isnan(v) and abs(v) < NQ_IMB_NEAR_ZERO for v in nq_vals)
        ),
        "nq_faded_nonpos": bool(
            (not math.isnan(whole_imb) and whole_imb <= 0.0)
            or any(not math.isnan(v) and v <= 0.0 for v in nq_vals)
        ),
        "fwd_min_px": min(px_vals) if px_vals else float("nan"),
    }


def compare_post_peak_horizon(
    mnq_mbo: pl.DataFrame,
    mnq_trades: pl.DataFrame,
    nq_trades: pl.DataFrame,
    *,
    price_hi: float,
    window_s: int = WINDOW_S,
    horizon_s: int = HORIZON_S,
    bin_s: int = BIN_S,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """من أول ``T`` لـ MNQ عند ``price_hi`` إلى ``+horizon_s``. النتيجة بعد ``H`` فقط في الشرائح."""

    book = prepare_mbo_events(mnq_mbo)
    mnq_tape = prepare_trades_tape(mnq_trades)
    nq_tape = prepare_trades_tape(nq_trades)
    high_ts = _first_t_ts(book, price_hi)
    if high_ts is None:
        raise ValueError(f"no MNQ T print at price >= {price_hi}")
    windows = post_peak_windows(high_ts, window_s=window_s, horizon_s=horizon_s, bin_s=bin_s)
    rows = [_bin_row(book, mnq_tape, nq_tape, w, high_ts) for w in windows]
    table = pl.DataFrame(rows)
    diagnostics = {
        "layer": LAYER_ID,
        "price_hi": price_hi,
        "mnq_high_ts": high_ts,
        "horizon_s": horizon_s,
        "bin_s": bin_s,
        "clock": "from_mnq_first_T_forward_300s",
        "old_drop_window": "+30-60s skipped [H, H+30s) and minutes 1-5",
        "nq_source": "trades_tape_T_only",
        "nq_fill_ratio": "unavailable_without_mbo_F_C",
        "day_scan_drop_already_5m": True,
        "not_spoofing": True,
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
        **_path_flags(rows),
    }
    return table, diagnostics


def write_horizon_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "horizon_bins.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Post-peak 5-minute NQ/MNQ flow path",
        "",
        "Bins are 30s from H (first MNQ T at price_hi) through H+300s.",
        "The old drop window was only +30-60s. Day-scan price drop was already 5m.",
        "NQ Fill_Ratio unavailable. Not spoofing, not a model.",
        f"H={diagnostics.get('mnq_high_ts')} 5m NQ imb="
        f"{diagnostics.get('horizon_5m_nq_imbalance')} faded_nonpos="
        f"{diagnostics.get('nq_faded_nonpos')} faded_near_zero="
        f"{diagnostics.get('nq_faded_near_zero')}.",
        "",
        "| name | off_s | MNQ T/s | NQ T/s | MNQ imb | NQ imb | MNQ fill | min_px | max_px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.iter_rows(named=True):
        lines.append(
            f"| {row['name']} | {float(row['offset_s']):.0f} | "
            f"{float(row['mnq_mbo_t_per_s']):.3f} | {float(row['nq_t_per_s']):.3f} | "
            f"{float(row['mnq_mbo_t_imbalance']):.3f} | {float(row['nq_t_imbalance']):.3f} | "
            f"{float(row['mnq_fill_ratio']):.3f} | {float(row['min_px']):.2f} | "
            f"{float(row['max_px']):.2f} |"
        )
    lines.append("")
    (out / "HORIZON_FLOW.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "BIN_S",
    "LAYER_ID",
    "compare_post_peak_horizon",
    "post_peak_windows",
    "write_horizon_report",
]
