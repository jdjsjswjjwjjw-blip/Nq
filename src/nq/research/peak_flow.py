"""سلوك ``T``/``F`` واختلال التدفق في نوافذ القمة الضابطة. بلا نموذج.

نفس ساعات PR #115. الضوضاء العكسية أُغلقت: نسبة القمة لم تكن استثنائية.
هنا السؤال تغيّر: هل العدوان (``T``) والسيولة المستهلكة (``F``) انقلبا حول القمة؟
ليس LSTM، ليست overlay، لا باك تست. احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from nq.contracts.mbo import MboAction, MboSide
from nq.contracts.temporal import EVENT_TS, SEQUENCE
from nq.research.mbo_trade_overlap import prepare_mbo_events
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import (
    WINDOW_S,
    NamedWindow,
    default_windows,
)

LAYER_ID = "peak_flow"
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_CANCEL = MboAction.CANCEL.value
_BID = MboSide.BID.value
_ASK = MboSide.ASK.value


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _isum(frame: pl.DataFrame, col: str) -> int:
    if frame.height == 0:
        return 0
    val = frame.select(pl.col(col).sum()).item()
    return 0 if val is None else int(val)


def _imb(buy: float, sell: float) -> float:
    return _ratio(buy - sell, buy + sell)


def _slice(book: pl.DataFrame, start_ts: int, end_ts: int) -> pl.DataFrame:
    return book.filter((pl.col(EVENT_TS) >= start_ts) & (pl.col(EVENT_TS) < end_ts))


def _side_size(frame: pl.DataFrame, action: str, side: str) -> int:
    return _isum(frame.filter((pl.col("action") == action) & (pl.col("side") == side)), "size")


def _side_n(frame: pl.DataFrame, action: str, side: str) -> int:
    return frame.filter((pl.col("action") == action) & (pl.col("side") == side)).height


def score_flow_window(book: pl.DataFrame, window: NamedWindow) -> dict[str, Any]:
    """عدوان ``T`` وملء ``F`` وإلغاء داخل ``[start, end)``."""

    chunk = _slice(book, window.start_ts, window.end_ts)
    t_buy = _side_size(chunk, _TRADE, _BID)
    t_sell = _side_size(chunk, _TRADE, _ASK)
    f_ask = _side_size(chunk, _FILL, _ASK)
    f_bid = _side_size(chunk, _FILL, _BID)
    c_ask = _side_size(chunk, _CANCEL, _ASK)
    c_bid = _side_size(chunk, _CANCEL, _BID)
    mid = window.start_ts + (window.end_ts - window.start_ts) // 2
    early = _slice(chunk, window.start_ts, mid)
    late = _slice(chunk, mid, window.end_ts)
    t_buy_e = _side_size(early, _TRADE, _BID)
    t_sell_e = _side_size(early, _TRADE, _ASK)
    t_buy_l = _side_size(late, _TRADE, _BID)
    t_sell_l = _side_size(late, _TRADE, _ASK)
    t_n = chunk.filter(pl.col("action") == _TRADE).height
    t_size = t_buy + t_sell
    f_size = f_ask + f_bid
    return {
        "name": window.name,
        "start_ts": window.start_ts,
        "end_ts": window.end_ts,
        "n_t": t_n,
        "t_size": t_size,
        "t_buy_n": _side_n(chunk, _TRADE, _BID),
        "t_sell_n": _side_n(chunk, _TRADE, _ASK),
        "t_buy_size": t_buy,
        "t_sell_size": t_sell,
        "t_imbalance": _imb(t_buy, t_sell),
        "t_imbalance_early": _imb(t_buy_e, t_sell_e),
        "t_imbalance_late": _imb(t_buy_l, t_sell_l),
        "f_size": f_size,
        "f_ask_size": f_ask,
        "f_bid_size": f_bid,
        "f_imbalance": _imb(f_ask, f_bid),
        "c_ask_size": c_ask,
        "c_bid_size": c_bid,
        "ask_hit_share": _ratio(f_ask, f_ask + c_ask),
        "bid_hit_share": _ratio(f_bid, f_bid + c_bid),
        "t_per_s": _ratio(t_n, (window.end_ts - window.start_ts) / float(SECOND_NS)),
    }


def _median_col(frame: pl.DataFrame, col: str) -> float:
    if frame.height == 0:
        return float("nan")
    val = frame.select(pl.col(col).median()).item()
    return float("nan") if val is None else float(val)


def session_t_imbalance_medians(book: pl.DataFrame, *, bin_s: int = WINDOW_S) -> dict[str, Any]:
    """وسيط اختلال ``T`` لكل شريحة ``bin_s`` فيها تنفيذ."""

    width = int(bin_s) * SECOND_NS
    prints = book.filter(pl.col("action") == _TRADE).with_columns(
        (pl.col(EVENT_TS) // width).alias("bin")
    )
    bins = (
        prints.group_by("bin")
        .agg(
            pl.col("size").filter(pl.col("side") == _BID).sum().alias("t_buy"),
            pl.col("size").filter(pl.col("side") == _ASK).sum().alias("t_sell"),
            pl.len().alias("n_t"),
        )
        .with_columns(
            pl.col("t_buy").fill_null(0),
            pl.col("t_sell").fill_null(0),
        )
        .filter((pl.col("t_buy") + pl.col("t_sell")) > 0)
        .with_columns(
            ((pl.col("t_buy") - pl.col("t_sell")) / (pl.col("t_buy") + pl.col("t_sell"))).alias(
                "t_imbalance"
            )
        )
    )
    return {
        "n_bins": bins.height,
        "median_t_imbalance": _median_col(bins, "t_imbalance"),
        "p10_t_imbalance": (
            float("nan")
            if bins.height == 0
            else float(bins.select(pl.col("t_imbalance").quantile(0.1)).item() or float("nan"))
        ),
        "p90_t_imbalance": (
            float("nan")
            if bins.height == 0
            else float(bins.select(pl.col("t_imbalance").quantile(0.9)).item() or float("nan"))
        ),
        "median_n_t": _median_col(bins, "n_t"),
    }


def compare_peak_flow(
    mbo: pl.DataFrame,
    *,
    price_hi: float,
    window_s: int = WINDOW_S,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """نفس نوافذ القمة/الصعود/الهبوط: اختلال ``T`` وملء ``F``."""

    book = prepare_mbo_events(mbo)
    high = (
        book.filter((pl.col("action") == _TRADE) & (pl.col("price") >= price_hi))
        .sort([EVENT_TS, SEQUENCE])
        .head(1)
    )
    if high.height == 0:
        raise ValueError(f"no T print at price >= {price_hi}")
    high_ts = int(high[EVENT_TS][0])
    windows = default_windows(high_ts, window_s=window_s)
    rows = [score_flow_window(book, w) for w in windows]
    day = session_t_imbalance_medians(book, bin_s=window_s)
    diagnostics = {
        "layer": LAYER_ID,
        "price_hi": price_hi,
        "high_ts": high_ts,
        "window_s": window_s,
        "day_30s": day,
        "t_side": "B=buy aggressor, A=sell aggressor",
        "f_ask": "resting offer filled (buy taking)",
        "ask_hit_share": "F_ask / (F_ask + C_ask); high = wall consumed not pulled",
        "heuristic": "named 30s flow; not a model, not a filter, not LSTM",
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
        "phantom_closed": True,
    }
    return pl.DataFrame(rows), diagnostics


def write_flow_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "peak_flow.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    day = diagnostics.get("day_30s", {})
    lines = [
        "# Peak flow: aggressive T and consumed F",
        "",
        "Phantom/cancel-noise chapter is closed (peak opp/T was not exceptional).",
        "This is T/F imbalance on the same named 30s windows. Not a model.",
        f"High ts={diagnostics.get('high_ts')} price_hi={diagnostics.get('price_hi')}.",
        "",
        "| name | n_T | buy T | sell T | T imb | early | late | F ask | F bid | F imb | ask hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.iter_rows(named=True):
        lines.append(
            f"| {row['name']} | {row['n_t']} | {row['t_buy_size']} | {row['t_sell_size']} | "
            f"{float(row['t_imbalance']):.3f} | {float(row['t_imbalance_early']):.3f} | "
            f"{float(row['t_imbalance_late']):.3f} | {row['f_ask_size']} | {row['f_bid_size']} | "
            f"{float(row['f_imbalance']):.3f} | {float(row['ask_hit_share']):.3f} |"
        )
    if isinstance(day, Mapping) and day:
        lines.extend(
            [
                "",
                "Day 30s T-imbalance bins:",
                f"- n_bins={day.get('n_bins')} median={day.get('median_t_imbalance')} "
                f"p10={day.get('p10_t_imbalance')} p90={day.get('p90_t_imbalance')}",
            ]
        )
    lines.append("")
    (out / "PEAK_FLOW.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "LAYER_ID",
    "compare_peak_flow",
    "score_flow_window",
    "session_t_imbalance_medians",
    "write_flow_report",
]
