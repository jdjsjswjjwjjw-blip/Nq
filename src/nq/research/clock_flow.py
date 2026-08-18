"""شريحة ساعة نيويورك يسمّيها المراقب: تدفق MNQ/NQ داخلها وبعدها.

``11:00–11:30`` America/New_York على يوم الجلسة. النافذة الكاملة + شرائح
داخلها + 5 دقائق بعد النهاية (حتى لا نكرر خطأ أول 30ث). NQ بلا MBO:
Fill لا يُختلق. ليست overlay ولا LSTM.
احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import polars as pl

from nq.research.horizon_flow import _bin_row
from nq.research.manual_zone_depth import et_clock_to_ns
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import NamedWindow
from nq.research.peak_pattern import HORIZON_S

LAYER_ID = "clock_flow"
BIN_S: Final = 60
AFTER_S: Final = HORIZON_S
TZ_NAME: Final = "America/New_York"


def clock_windows(
    day: str,
    start_clock: str,
    end_clock: str,
    *,
    bin_s: int = BIN_S,
    after_s: int = AFTER_S,
) -> tuple[int, tuple[NamedWindow, ...]]:
    """``[start, end)`` بتوقيت نيويورك + شرائح ``bin_s`` + ``after_s`` بعدها."""

    start_ts = et_clock_to_ns(day, start_clock)
    end_ts = et_clock_to_ns(day, end_clock)
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
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """MNQ MBO + شريط MNQ + شريط NQ على ساعة نيويورك المسمّاة."""

    book = prepare_mbo_events(mnq_mbo)
    mnq_tape = prepare_trades_tape(mnq_trades)
    nq_tape = prepare_trades_tape(nq_trades)
    origin, windows = clock_windows(day, start_clock, end_clock, bin_s=bin_s, after_s=after_s)
    rows = [_bin_row(book, mnq_tape, nq_tape, w, origin) for w in windows]
    table = pl.DataFrame(rows)
    by = {r["name"]: r for r in rows}
    rng = by.get("range", {})
    after = by.get(f"after-0-{after_s}s", {})
    diagnostics = {
        "layer": LAYER_ID,
        "day": day,
        "tz": TZ_NAME,
        "start_clock": start_clock,
        "end_clock": end_clock,
        "start_ts": origin,
        "bin_s": bin_s,
        "after_s": after_s,
        "range_nq_imbalance": rng.get("nq_t_imbalance"),
        "range_mnq_imbalance": rng.get("mnq_mbo_t_imbalance"),
        "range_mnq_fill_ratio": rng.get("mnq_fill_ratio"),
        "range_nq_t_per_s": rng.get("nq_t_per_s"),
        "range_mnq_t_per_s": rng.get("mnq_mbo_t_per_s"),
        "range_min_px": rng.get("min_px"),
        "range_max_px": rng.get("max_px"),
        "after_nq_imbalance": after.get("nq_t_imbalance"),
        "after_min_px": after.get("min_px"),
        "nq_source": "trades_tape_T_only",
        "nq_fill_ratio": "unavailable_without_mbo_F_C",
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
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# New York clock window: MNQ MBO + MNQ/NQ trades",
        "",
        f"{diagnostics.get('day')} {diagnostics.get('start_clock')}–"
        f"{diagnostics.get('end_clock')} {diagnostics.get('tz')}.",
        "After-window is 5 minutes past end. NQ Fill_Ratio unavailable. Not a model.",
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
    "clock_windows",
    "compare_clock_range",
    "write_clock_report",
]
