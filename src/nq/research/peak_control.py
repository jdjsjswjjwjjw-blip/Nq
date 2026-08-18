"""مقارنة نافذة القمة بنوافذ ضابطة: ضوضاء عكسية ملغاة، بلا نموذج.

يوم جلسة واحد. النوافذ تُسمّى مسبقًا (قمة / صعود قبل ساعة / هبوط بعد 30ث).
ليس حكم سبوفينج، ليست overlay، لا LSTM، لا فلتر، لا باك تست.
احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS, SEQUENCE
from nq.research.mbo_trade_overlap import prepare_mbo_events
from nq.research.opposite_phantom import SECOND_NS, closed_unfilled_orders

LAYER_ID = "peak_control"
INSTANT_NS: Final = 100_000_000
LARGE_MIN: Final = 3
WINDOW_S: Final = 30
HOUR_NS: Final = 3600 * SECOND_NS
_TRADE = MboAction.TRADE.value


@dataclass(frozen=True, slots=True)
class NamedWindow:
    """شريحة زمنية مسمّاة ``[start_ts, end_ts)``."""

    name: str
    start_ts: int
    end_ts: int


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _fmean(frame: pl.DataFrame, col: str) -> float:
    if frame.height == 0:
        return float("nan")
    val = frame.select(pl.col(col).mean()).item()
    return float("nan") if val is None else float(val)


def _isum(frame: pl.DataFrame, col: str) -> int:
    if frame.height == 0:
        return 0
    val = frame.select(pl.col(col).sum()).item()
    return 0 if val is None else int(val)


def _first_high_from_book(book: pl.DataFrame, price_hi: float) -> int:
    prints = (
        book.filter((pl.col("action") == _TRADE) & (pl.col("price") >= price_hi))
        .sort([EVENT_TS, SEQUENCE])
        .head(1)
    )
    if prints.height == 0:
        raise ValueError(f"no T print at price >= {price_hi}")
    return int(prints[EVENT_TS][0])


def first_high_ts(mbo: pl.DataFrame, price_hi: float) -> int:
    """أول ``T`` بسعر ≥ ``price_hi``."""

    return _first_high_from_book(prepare_mbo_events(mbo), price_hi)


def default_windows(high_ts: int, *, window_s: int = WINDOW_S) -> tuple[NamedWindow, ...]:
    """قمة = 30ث قبل أول لمسة. صعود = نفس الساعة قبلها. هبوط = 30ث بعد +30ث."""

    w = int(window_s) * SECOND_NS
    peak_end = int(high_ts)
    peak_start = peak_end - w
    climb_end = peak_end - HOUR_NS
    climb_start = climb_end - w
    drop_start = peak_end + w
    drop_end = drop_start + w
    return (
        NamedWindow("peak", peak_start, peak_end),
        NamedWindow("climb", climb_start, climb_end),
        NamedWindow("drop", drop_start, drop_end),
    )


def _in_window(closed: pl.DataFrame, start_ts: int, end_ts: int) -> pl.DataFrame:
    return closed.filter(
        (pl.col("add_ts") >= start_ts)
        & (pl.col("add_ts") < end_ts)
        & (pl.col("cancel_ts") >= start_ts)
        & (pl.col("cancel_ts") < end_ts)
    )


def score_window(
    book: pl.DataFrame,
    closed: pl.DataFrame,
    window: NamedWindow,
    *,
    large_min: int = LARGE_MIN,
    instant_ns: int = INSTANT_NS,
) -> dict[str, Any]:
    """مؤشرات نافذة واحدة. الجانب العكسي = عكس آخر ``T`` في النافذة."""

    prints = book.filter(
        (pl.col("action") == _TRADE)
        & (pl.col(EVENT_TS) >= window.start_ts)
        & (pl.col(EVENT_TS) < window.end_ts)
    ).sort([EVENT_TS, SEQUENCE])
    n_t = prints.height
    t_size = _isum(prints, "size")
    last_side = str(prints["side"][-1]) if n_t else ""
    inside = _in_window(closed, window.start_ts, window.end_ts)
    opposite = inside.filter(pl.col("side") != last_side) if last_side else inside.head(0)
    large = opposite.filter(pl.col("add_size") >= large_min)
    instant = opposite.filter(pl.col("lifetime_ns") < instant_ns)
    opp_n = opposite.height
    opp_size = _isum(opposite, "add_size")
    return {
        "name": window.name,
        "start_ts": window.start_ts,
        "end_ts": window.end_ts,
        "last_t_side": last_side,
        "n_t": n_t,
        "t_size": t_size,
        "opp_n": opp_n,
        "opp_size": opp_size,
        "opp_over_t": _ratio(opp_size, t_size),
        "mean_life_ms": _fmean(opposite, "lifetime_ns") / 1e6,
        "instant_lt_100ms": _ratio(instant.height, opp_n),
        "large_n": large.height,
        "large_size": _isum(large, "add_size"),
        "n_unfilled_in_window": inside.height,
    }


def _median_col(frame: pl.DataFrame, col: str) -> float:
    if frame.height == 0:
        return float("nan")
    val = frame.select(pl.col(col).median()).item()
    return float("nan") if val is None else float(val)


def _quantile_col(frame: pl.DataFrame, col: str, q: float) -> float:
    if frame.height == 0:
        return float("nan")
    val = frame.select(pl.col(col).quantile(q)).item()
    return float("nan") if val is None else float(val)


def session_bin_medians(
    book: pl.DataFrame,
    closed: pl.DataFrame,
    *,
    bin_s: int = WINDOW_S,
    large_min: int = LARGE_MIN,
    instant_ns: int = INSTANT_NS,
) -> dict[str, Any]:
    """وسيط كل شرائح ``bin_s`` التي فيها ``T``. وصف لليوم، ليس نموذجًا."""

    width = int(bin_s) * SECOND_NS
    prints = book.filter(pl.col("action") == _TRADE).with_columns(
        (pl.col(EVENT_TS) // width).alias("bin")
    )
    last_t = (
        prints.sort([EVENT_TS, SEQUENCE])
        .unique(subset=["bin"], keep="last")
        .select("bin", pl.col("side").alias("last_t_side"))
    )
    t_agg = prints.group_by("bin").agg(
        pl.len().alias("n_t"),
        pl.col("size").sum().alias("t_size"),
    )
    same_bin = closed.filter(
        (pl.col("add_ts") // width) == (pl.col("cancel_ts") // width)
    ).with_columns((pl.col("add_ts") // width).alias("bin"))
    tagged = same_bin.join(last_t, on="bin", how="inner")
    opp = tagged.filter(pl.col("side") != pl.col("last_t_side"))
    opp_agg = opp.group_by("bin").agg(
        pl.len().alias("opp_n"),
        pl.col("add_size").sum().alias("opp_size"),
        (pl.col("lifetime_ns") < instant_ns).sum().alias("instant_n"),
        (pl.col("add_size") >= large_min).sum().alias("large_n"),
    )
    bins = (
        t_agg.join(opp_agg, on="bin", how="left")
        .with_columns(
            pl.col("opp_n").fill_null(0),
            pl.col("opp_size").fill_null(0),
            pl.col("instant_n").fill_null(0),
            pl.col("large_n").fill_null(0),
        )
        .filter(pl.col("t_size") > 0)
        .with_columns(
            (pl.col("opp_size") / pl.col("t_size")).alias("opp_over_t"),
            pl.when(pl.col("opp_n") > 0)
            .then(pl.col("instant_n") / pl.col("opp_n"))
            .otherwise(float("nan"))
            .alias("instant_lt_100ms"),
        )
    )
    return {
        "n_bins": bins.height,
        "median_opp_n": _median_col(bins, "opp_n"),
        "median_opp_over_t": _median_col(bins, "opp_over_t"),
        "p90_opp_over_t": _quantile_col(bins, "opp_over_t", 0.9),
        "median_life_ms": _median_col(opp, "lifetime_ns") / 1e6 if opp.height else float("nan"),
        "median_instant_lt_100ms": _median_col(bins, "instant_lt_100ms"),
        "median_large_n": _median_col(bins, "large_n"),
    }


def compare_peak_controls(
    mbo: pl.DataFrame,
    *,
    price_hi: float,
    window_s: int = WINDOW_S,
    large_min: int = LARGE_MIN,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """جدول القمة مقابل صعود/هبوط + وسيط شرائح اليوم."""

    book = prepare_mbo_events(mbo)
    high_ts = _first_high_from_book(book, price_hi)
    windows = default_windows(high_ts, window_s=window_s)
    closed = closed_unfilled_orders(book)
    rows = [score_window(book, closed, w, large_min=large_min) for w in windows]
    day = session_bin_medians(book, closed, bin_s=window_s, large_min=large_min)
    frame = pl.DataFrame(rows)
    diagnostics = {
        "layer": LAYER_ID,
        "price_hi": price_hi,
        "high_ts": high_ts,
        "window_s": window_s,
        "large_min": large_min,
        "instant_ns": INSTANT_NS,
        "day_30s": day,
        "heuristic": "named 30s slices; not spoofing, not a model, not a filter",
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
    }
    return frame, diagnostics


def write_control_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "peak_control.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    day = diagnostics.get("day_30s", {})
    lines = [
        "# Peak vs climb vs drop (30s named windows)",
        "",
        "Diagnostic only. Not spoofing, not a model, not a live overlay, not LSTM.",
        f"High ts={diagnostics.get('high_ts')} price_hi={diagnostics.get('price_hi')}.",
        "",
        "| name | n_T | T sz | opp n | opp sz / T | life_ms | instant<100ms | large n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.iter_rows(named=True):
        lines.append(
            f"| {row['name']} | {row['n_t']} | {row['t_size']} | {row['opp_n']} | "
            f"{float(row['opp_over_t']):.4f} | {float(row['mean_life_ms']):.3f} | "
            f"{float(row['instant_lt_100ms']):.4f} | {row['large_n']} |"
        )
    if isinstance(day, Mapping) and day:
        lines.extend(
            [
                "",
                "Day 30s bins (median over bins with T>0):",
                f"- n_bins={day.get('n_bins')}",
                f"- median opp/T={day.get('median_opp_over_t')}",
                f"- p90 opp/T={day.get('p90_opp_over_t')}",
                f"- median opp_n={day.get('median_opp_n')}",
                f"- median large_n={day.get('median_large_n')}",
                f"- median instant<100ms={day.get('median_instant_lt_100ms')}",
            ]
        )
    lines.append("")
    (out / "PEAK_CONTROL.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "HOUR_NS",
    "INSTANT_NS",
    "LARGE_MIN",
    "LAYER_ID",
    "WINDOW_S",
    "NamedWindow",
    "compare_peak_controls",
    "default_windows",
    "first_high_ts",
    "score_window",
    "session_bin_medians",
    "write_control_report",
]
