"""استكشاف ثلاثة أشرطة على ساعة مسمّاة: MNQ MBO وMNQ Trades وNQ Trades.

مثل أول قياس للقمة: اختلال ``T`` وملء ``F`` على MNQ، وشريط ``T`` لكل عقد.
بلا عتبة نمط وبلا إشارة. NQ بلا MBO فـ Fill لا يُختلق.
``price_lo`` اختياري: قاع الشريحة + أول لمسة للمستوى بعد بداية الشريحة.
احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import polars as pl

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS, SEQUENCE
from nq.research.cross_nq_mnq import MNQ_MULT, NQ_MULT, _with_notional
from nq.research.horizon_flow import _bin_row
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import HOUR_NS, WINDOW_S, NamedWindow
from nq.research.peak_flow import score_flow_window
from nq.research.peak_pattern import HORIZON_S

LAYER_ID = "clock_flow"
BIN_S: Final = 60
AFTER_S: Final = HORIZON_S
TZ_NAME: Final = "America/New_York"
_TRADE = MboAction.TRADE.value
_ET: Final = ZoneInfo(TZ_NAME)


def clock_to_ns(day: str, clock: str, tz_name: str) -> int:
    """``2026-08-17`` + ``03:12:00`` في ``tz_name`` → نانوثانية."""

    hour, minute, *rest = clock.split(":")
    second = int(rest[0]) if rest else 0
    yyyy, mm, dd = (int(p) for p in day.split("-"))
    stamp = dt.datetime(yyyy, mm, dd, int(hour), int(minute), second, tzinfo=ZoneInfo(tz_name))
    return int(stamp.timestamp() * 1_000_000_000)


def et_clock_to_ns(day: str, clock: str) -> int:
    """``2026-08-17`` + ``11:00:00`` (America/New_York) → نانوثانية."""

    return clock_to_ns(day, clock, TZ_NAME)


def _fmt_ts(ts: int | None, tz_name: str) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts / 1_000_000_000, tz=ZoneInfo(tz_name)).isoformat()


def clock_windows(
    day: str,
    start_clock: str,
    end_clock: str,
    *,
    bin_s: int = BIN_S,
    after_s: int = AFTER_S,
    tz_name: str = TZ_NAME,
) -> tuple[int, tuple[NamedWindow, ...]]:
    """``[start, end)`` في ``tz_name`` + شرائح ``bin_s`` + ``after_s`` بعدها."""

    start_ts = clock_to_ns(day, start_clock, tz_name)
    end_ts = clock_to_ns(day, end_clock, tz_name)
    if end_ts <= start_ts:
        raise ValueError("end_clock must be after start_clock")
    if bin_s <= 0:
        raise ValueError("bin_s must be positive")
    whole = NamedWindow("range", start_ts, end_ts)
    bins: list[NamedWindow] = []
    stamp = start_ts
    idx = 0
    while stamp < end_ts:
        nxt = min(stamp + bin_s * SECOND_NS, end_ts)
        bins.append(
            NamedWindow(f"+{idx * bin_s}-{idx * bin_s + (nxt - stamp) // SECOND_NS}s", stamp, nxt)
        )
        stamp = nxt
        idx += 1
    after = NamedWindow(
        f"after-0-{after_s}s",
        end_ts,
        end_ts + after_s * SECOND_NS,
    )
    return start_ts, (whole, *bins, after)


def _around(
    origin_ts: int,
    prefix: str,
    *,
    window_s: int = WINDOW_S,
    horizon_s: int = HORIZON_S,
) -> tuple[NamedWindow, ...]:
    """مثل نوافذ القمة: 30ث قبل الأصل، صعود قبل ساعة، 30ث/60ث/5د بعد."""

    w = int(window_s) * SECOND_NS
    h = int(horizon_s) * SECOND_NS
    return (
        NamedWindow(prefix, origin_ts - w, origin_ts),
        NamedWindow(f"{prefix}-climb", origin_ts - HOUR_NS - w, origin_ts - HOUR_NS),
        NamedWindow(f"{prefix}+0-{window_s}s", origin_ts, origin_ts + w),
        NamedWindow(f"{prefix}+{window_s}-{window_s * 2}s", origin_ts + w, origin_ts + 2 * w),
        NamedWindow(f"{prefix}+0-{horizon_s}s", origin_ts, origin_ts + h),
    )


def _first_min_t(book: pl.DataFrame, window: NamedWindow) -> tuple[int | None, float]:
    chunk = book.filter(
        (pl.col("action") == _TRADE)
        & (pl.col(EVENT_TS) >= window.start_ts)
        & (pl.col(EVENT_TS) < window.end_ts)
    )
    if chunk.height == 0:
        return None, float("nan")
    min_px = chunk.select(pl.col("price").min()).item()
    if min_px is None:
        return None, float("nan")
    first = chunk.filter(pl.col("price") == min_px).sort([EVENT_TS, SEQUENCE]).head(1)
    return int(first[EVENT_TS][0]), float(min_px)


def _first_t_ge(book: pl.DataFrame, price: float, start_ts: int) -> int | None:
    hit = (
        book.filter(
            (pl.col("action") == _TRADE)
            & (pl.col("price") >= price)
            & (pl.col(EVENT_TS) >= start_ts)
        )
        .sort([EVENT_TS, SEQUENCE])
        .head(1)
    )
    if hit.height == 0:
        return None
    return int(hit[EVENT_TS][0])


def _source_row(
    book: pl.DataFrame,
    window: NamedWindow,
    *,
    source: str,
    contract: str,
    multiplier: float,
    fill_ok: bool,
) -> dict[str, Any]:
    row = _with_notional(
        score_flow_window(book, window),
        contract=contract,
        multiplier=multiplier,
        window=window,
    )
    row["source"] = source
    if not fill_ok:
        row["ask_hit_share"] = float("nan")
        row["bid_hit_share"] = float("nan")
        row["f_ask_size"] = 0
        row["f_bid_size"] = 0
        row["c_ask_size"] = 0
        row["c_bid_size"] = 0
        row["f_imbalance"] = float("nan")
    return row


def _stack_window(
    mnq_mbo: pl.DataFrame,
    mnq_tape: pl.DataFrame,
    nq_tape: pl.DataFrame,
    window: NamedWindow,
) -> list[dict[str, Any]]:
    return [
        _source_row(
            mnq_mbo,
            window,
            source="mnq_mbo",
            contract="MNQ",
            multiplier=MNQ_MULT,
            fill_ok=True,
        ),
        _source_row(
            mnq_tape,
            window,
            source="mnq_trades",
            contract="MNQ",
            multiplier=MNQ_MULT,
            fill_ok=False,
        ),
        _source_row(
            nq_tape,
            window,
            source="nq_trades",
            contract="NQ",
            multiplier=NQ_MULT,
            fill_ok=False,
        ),
    ]


def compare_clock_range(
    mnq_mbo: pl.DataFrame,
    mnq_trades: pl.DataFrame,
    nq_trades: pl.DataFrame,
    *,
    day: str,
    start_clock: str,
    end_clock: str,
    bin_s: int = BIN_S,
    after_s: int = AFTER_S,
    tz_name: str = TZ_NAME,
    price_lo: float | None = None,
    window_s: int = WINDOW_S,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """MNQ MBO + شريط MNQ + شريط NQ على الساعة المسمّاة."""

    book = prepare_mbo_events(mnq_mbo)
    mnq_tape = prepare_trades_tape(mnq_trades)
    nq_tape = prepare_trades_tape(nq_trades)
    origin, windows = clock_windows(
        day, start_clock, end_clock, bin_s=bin_s, after_s=after_s, tz_name=tz_name
    )
    extra: list[NamedWindow] = []
    low_ts, low_px = _first_min_t(book, windows[0])
    level_ts: int | None = None
    if price_lo is not None:
        if low_ts is not None:
            extra.extend(_around(low_ts, "low", window_s=window_s, horizon_s=after_s))
        level_ts = _first_t_ge(book, price_lo, origin)
        if level_ts is not None:
            extra.extend(_around(level_ts, "level", window_s=window_s, horizon_s=after_s))
    slim_windows = (*windows, *extra)
    rows = [_bin_row(book, mnq_tape, nq_tape, w, origin) for w in slim_windows]
    table = pl.DataFrame(rows)
    focus = [windows[0], windows[-1], *extra]
    sources = pl.DataFrame(
        [row for w in focus for row in _stack_window(book, mnq_tape, nq_tape, w)]
    )
    by = {r["name"]: r for r in rows}
    rng = by.get("range", {})
    after = by.get(f"after-0-{after_s}s", {})
    diagnostics = {
        "layer": LAYER_ID,
        "day": day,
        "tz": tz_name,
        "start_clock": start_clock,
        "end_clock": end_clock,
        "start_ts": origin,
        "bin_s": bin_s,
        "after_s": after_s,
        "window_s": window_s,
        "price_lo": price_lo,
        "low_ts": low_ts,
        "low_px": low_px,
        "low_clock": _fmt_ts(low_ts, tz_name),
        "level_ts": level_ts,
        "level_clock": _fmt_ts(level_ts, tz_name),
        "range_nq_imbalance": rng.get("nq_t_imbalance"),
        "range_mnq_imbalance": rng.get("mnq_mbo_t_imbalance"),
        "range_mnq_fill_ratio": rng.get("mnq_fill_ratio"),
        "range_nq_t_per_s": rng.get("nq_t_per_s"),
        "range_mnq_t_per_s": rng.get("mnq_mbo_t_per_s"),
        "range_min_px": rng.get("min_px"),
        "range_max_px": rng.get("max_px"),
        "after_nq_imbalance": after.get("nq_t_imbalance"),
        "after_min_px": after.get("min_px"),
        "sources": sources.to_dicts() if sources.height else [],
        "nq_source": "trades_tape_T_only",
        "nq_fill_ratio": "unavailable_without_mbo_F_C",
        "not_pattern": True,
        "not_spoofing": True,
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
    }
    return table, diagnostics


def write_clock_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "clock_bins.parquet")
    sources = diagnostics.get("sources", [])
    if isinstance(sources, list) and sources:
        pl.DataFrame(sources).write_parquet(out / "clock_sources.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Three tapes on a named clock window",
        "",
        f"{diagnostics.get('day')} {diagnostics.get('start_clock')}–"
        f"{diagnostics.get('end_clock')} {diagnostics.get('tz')}.",
        "Explore MNQ MBO vs MNQ trades vs NQ trades. No pattern lock. NQ has no Fill.",
        f"Range min={diagnostics.get('range_min_px')} max={diagnostics.get('range_max_px')} "
        f"low={diagnostics.get('low_clock')} px={diagnostics.get('low_px')} "
        f"level={diagnostics.get('price_lo')} at {diagnostics.get('level_clock')}.",
        "",
        "| window | source | n_T | buy T | sell T | T imb | early | late | F ask | C ask | "
        "ask hit | T/s | $ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if isinstance(sources, list):
        for row in sources:
            lines.append(
                f"| {row['name']} | {row['source']} | {row['n_t']} | {row['t_buy_size']} | "
                f"{row['t_sell_size']} | {float(row['t_imbalance']):.3f} | "
                f"{float(row['t_imbalance_early']):.3f} | {float(row['t_imbalance_late']):.3f} | "
                f"{row['f_ask_size']} | {row['c_ask_size']} | {float(row['ask_hit_share']):.3f} | "
                f"{float(row['t_per_s']):.3f} | {float(row['t_notional']):.0f} |"
            )
    lines += [
        "",
        "Minute path (same three tapes, slim):",
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
    (out / "CLOCK_FLOW.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "AFTER_S",
    "BIN_S",
    "LAYER_ID",
    "clock_to_ns",
    "clock_windows",
    "compare_clock_range",
    "et_clock_to_ns",
    "write_clock_report",
]
