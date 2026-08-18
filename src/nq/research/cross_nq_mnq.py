"""شريط NQ مقابل MNQ على نفس نوافذ PR #116. بلا نموذج.

NQ هنا ملف Trades (``T`` فقط). ``Fill_Ratio`` يحتاج ``F``/``C`` من MBO —
لا يوجد MBO لـ NQ في هذه الجلسة، فالنسبة تُترك NaN ولا تُختلق.
الساعة مقفلة على أول ``T`` لـ MNQ عند ``price_hi``. نقيس تأخر أول لمسة NQ.
ليس حكم سبوفينج عبر العقود، ليست overlay، لا LSTM. احذف الملف + السكربت +
الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS, SEQUENCE
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import WINDOW_S, NamedWindow, default_windows
from nq.research.peak_flow import score_flow_window, session_t_imbalance_medians

LAYER_ID = "cross_nq_mnq"
MNQ_MULT: Final = 2.0
NQ_MULT: Final = 20.0
_TRADE = MboAction.TRADE.value


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _duration_s(window: NamedWindow) -> float:
    return (window.end_ts - window.start_ts) / float(SECOND_NS)


def _first_t_ts(book: pl.DataFrame, price_hi: float) -> int | None:
    high = (
        book.filter((pl.col("action") == _TRADE) & (pl.col("price") >= price_hi))
        .sort([EVENT_TS, SEQUENCE])
        .head(1)
    )
    if high.height == 0:
        return None
    return int(high[EVENT_TS][0])


def _first_t_in_window(book: pl.DataFrame, window: NamedWindow) -> int | None:
    chunk = (
        book.filter(
            (pl.col("action") == _TRADE)
            & (pl.col(EVENT_TS) >= window.start_ts)
            & (pl.col(EVENT_TS) < window.end_ts)
        )
        .sort([EVENT_TS, SEQUENCE])
        .head(1)
    )
    if chunk.height == 0:
        return None
    return int(chunk[EVENT_TS][0])


def _with_notional(
    row: Mapping[str, Any],
    *,
    contract: str,
    multiplier: float,
    window: NamedWindow,
) -> dict[str, Any]:
    out = dict(row)
    size = float(out["t_buy_size"] + out["t_sell_size"])
    out["contract"] = contract
    out["multiplier"] = multiplier
    out["t_notional"] = size * multiplier
    out["t_notional_per_s"] = _ratio(out["t_notional"], _duration_s(window))
    return out


def _lag_ns(later: int | None, earlier: int | None) -> float:
    if later is None or earlier is None:
        return float("nan")
    return float(later - earlier)


def _diff_row(
    mnq: Mapping[str, Any],
    nq: Mapping[str, Any],
    *,
    first_print_lag_ns: float,
) -> dict[str, Any]:
    return {
        "name": mnq["name"],
        "mnq_t_per_s": mnq["t_per_s"],
        "nq_t_per_s": nq["t_per_s"],
        "d_t_per_s": float(nq["t_per_s"]) - float(mnq["t_per_s"]),
        "mnq_t_imbalance": mnq["t_imbalance"],
        "nq_t_imbalance": nq["t_imbalance"],
        "d_t_imbalance": float(nq["t_imbalance"]) - float(mnq["t_imbalance"]),
        "mnq_ask_hit_share": mnq["ask_hit_share"],
        "nq_ask_hit_share": nq["ask_hit_share"],
        "d_ask_hit_share": float(nq["ask_hit_share"]) - float(mnq["ask_hit_share"]),
        "mnq_t_notional": mnq["t_notional"],
        "nq_t_notional": nq["t_notional"],
        "d_t_notional": float(nq["t_notional"]) - float(mnq["t_notional"]),
        "nq_over_mnq_notional": _ratio(float(nq["t_notional"]), float(mnq["t_notional"])),
        "first_print_lag_ns": first_print_lag_ns,
    }


def compare_nq_mnq_windows(
    mnq_mbo: pl.DataFrame,
    nq_trades: pl.DataFrame,
    *,
    price_hi: float,
    window_s: int = WINDOW_S,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """نفس ساعات MNQ في PR #116. NQ = شريط ``T`` فقط على تلك النوافذ."""

    mnq = prepare_mbo_events(mnq_mbo)
    nq = prepare_trades_tape(nq_trades)
    mnq_high = _first_t_ts(mnq, price_hi)
    if mnq_high is None:
        raise ValueError(f"no MNQ T print at price >= {price_hi}")
    nq_high = _first_t_ts(nq, price_hi)
    windows = default_windows(mnq_high, window_s=window_s)
    mnq_rows: list[dict[str, Any]] = []
    nq_rows: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []
    for window in windows:
        mnq_row = _with_notional(
            score_flow_window(mnq, window),
            contract="MNQ",
            multiplier=MNQ_MULT,
            window=window,
        )
        nq_row = _with_notional(
            score_flow_window(nq, window),
            contract="NQ",
            multiplier=NQ_MULT,
            window=window,
        )
        lag = _lag_ns(_first_t_in_window(nq, window), _first_t_in_window(mnq, window))
        mnq_rows.append(mnq_row)
        nq_rows.append(nq_row)
        diffs.append(_diff_row(mnq_row, nq_row, first_print_lag_ns=lag))
    high_lag = _lag_ns(nq_high, mnq_high)
    leader = "unknown"
    if nq_high is None:
        leader = "mnq_only"
    elif high_lag > 0:
        leader = "mnq"
    elif high_lag < 0:
        leader = "nq"
    else:
        leader = "tie"
    stacked = pl.DataFrame(mnq_rows + nq_rows)
    diff_table = pl.DataFrame(diffs)
    diagnostics = {
        "layer": LAYER_ID,
        "price_hi": price_hi,
        "mnq_high_ts": mnq_high,
        "nq_high_ts": nq_high,
        "high_lag_ns": high_lag,
        "high_leader": leader,
        "window_s": window_s,
        "mnq_multiplier": MNQ_MULT,
        "nq_multiplier": NQ_MULT,
        "mnq_day_30s": session_t_imbalance_medians(mnq, bin_s=window_s),
        "nq_day_30s": session_t_imbalance_medians(nq, bin_s=window_s),
        "nq_source": "trades_tape_T_only",
        "nq_fill_ratio": "unavailable_without_mbo_F_C",
        "clock": "locked_to_mnq_first_T_at_price_hi",
        "diff": "NQ minus MNQ on the same [start, end)",
        "not_spoofing": True,
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
        "phantom_closed": True,
    }
    return stacked, diff_table, diagnostics


def write_cross_report(
    stacked: pl.DataFrame,
    diffs: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if stacked.height:
        stacked.write_parquet(out / "cross_nq_mnq.parquet")
    if diffs.height:
        diffs.write_parquet(out / "cross_nq_mnq_diff.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# NQ trades vs MNQ MBO on locked peak windows",
        "",
        "Clock locked to MNQ first T at price_hi (PR #116 windows).",
        "NQ is a trades tape (T only). Fill_Ratio needs MBO F/C — not invented.",
        "Not spoofing, not a model, not LSTM.",
        f"MNQ high_ts={diagnostics.get('mnq_high_ts')} NQ high_ts={diagnostics.get('nq_high_ts')} "
        f"lag_ns={diagnostics.get('high_lag_ns')} leader={diagnostics.get('high_leader')}.",
        "",
        "| name | MNQ T/s | NQ T/s | d T/s | MNQ imb | NQ imb | d imb | MNQ hit | "
        "NQ hit | NQ$/MNQ$ | lag_ns |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diffs.iter_rows(named=True):
        lines.append(
            f"| {row['name']} | {float(row['mnq_t_per_s']):.3f} | {float(row['nq_t_per_s']):.3f} | "
            f"{float(row['d_t_per_s']):.3f} | {float(row['mnq_t_imbalance']):.3f} | "
            f"{float(row['nq_t_imbalance']):.3f} | {float(row['d_t_imbalance']):.3f} | "
            f"{float(row['mnq_ask_hit_share']):.3f} | {float(row['nq_ask_hit_share']):.3f} | "
            f"{float(row['nq_over_mnq_notional']):.3f} | {float(row['first_print_lag_ns']):.0f} |"
        )
    lines.append("")
    (out / "CROSS_NQ_MNQ.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "LAYER_ID",
    "MNQ_MULT",
    "NQ_MULT",
    "compare_nq_mnq_windows",
    "write_cross_report",
]
