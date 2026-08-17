"""أداة مشاهدة: كل إطلاق OOF على شريط 30ث، 10 دقائق قبل ``t`` و15 بعدها.

ليست فريم شارت جديدًا، وليست Y جديدًا، وليست فلتر دخول. المسار بعد ``t``
للعين فقط — نظرة أمامية للفحص، لا تُستخدم في الباك تست كبوابة.

احذف هذا الملف + القالب + السكربت للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nq.auction_behavior.outcomes import SETUP_AVAILABILITY_TS
from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.causal_entry import TICKS_PER_NQ_POINT, assert_not_raw_mbo_stream
from nq.research.hold_horizon import (
    causal_fires,
    jsonable,
    load_overlay_period_inputs,
    oof_timestamps,
)
from nq.research.progress import ProgressLike

LAYER_ID = "entry_replay"
NS_PER_MIN = 60 * 1_000_000_000
DEFAULT_LOOKBACK_NS = 10 * NS_PER_MIN
DEFAULT_LOOKAHEAD_NS = 15 * NS_PER_MIN
_TEMPLATE = Path(__file__).with_name("entry_replay.html")
_GROUP = "_behavior_story_run"
_SCALED_PRICE = 1_000_000.0
_LOOKAHEAD = frozenset(
    {
        "wave_frac",
        "wave_peak_ticks",
        "expansion_frac",
        "ticks_remaining_to_peak",
        "wave_bin",
    }
)
_BAR_COLS = (
    "close",
    "path_beyond_asia_ticks",
    "path_inside_asia_va",
    "path_depth_confirm",
)


@dataclass(frozen=True, slots=True)
class EntryReplayConfig:
    """نافذة مشاهدة ثابتة على ساعة البارميل 30ث."""

    min_p: float = 0.5
    expansion_start_ticks: float = 16.0
    lookback_ns: int = DEFAULT_LOOKBACK_NS
    lookahead_ns: int = DEFAULT_LOOKAHEAD_NS
    holdout_months: int | None = 4


@dataclass(frozen=True, slots=True)
class EntryReplayReport:
    trades: pl.DataFrame
    bars: pl.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def to_nq_price(raw: float) -> float:
    """سعر NQ بالدولار من وحدات الملف أو من سعر جاهز."""
    if not np.isfinite(raw):
        return float("nan")
    value = float(raw)
    if abs(value) >= _SCALED_PRICE:
        return value * float(PRICE_SCALE)
    return value


def _f64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    if frame.height == 0 or name not in frame.columns:
        return np.full(int(frame.height), np.nan, dtype=np.float64)
    return frame[name].cast(pl.Float64).to_numpy().astype(np.float64, copy=False)


def _i64_arr(frame: pl.DataFrame, name: str) -> np.ndarray:
    return frame[name].cast(pl.Int64).to_numpy().astype(np.int64, copy=False)


def _path_by_story(path: pl.DataFrame, group_col: str) -> dict[Any, pl.DataFrame]:
    grouped: dict[Any, pl.DataFrame] = {}
    for key, group in path.group_by(group_col, maintain_order=True):
        story = key[0] if isinstance(key, tuple) else key
        grouped[story] = group
    return grouped


def _px_or_nan(frame: pl.DataFrame, i: int, name: str) -> float:
    if name not in frame.columns:
        return float("nan")
    val = frame[name][i]
    if val is None:
        return float("nan")
    return to_nq_price(float(val))


def extract_entry_windows(  # noqa: PLR0915
    entries: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    lookback_ns: int = DEFAULT_LOOKBACK_NS,
    lookahead_ns: int = DEFAULT_LOOKAHEAD_NS,
    group_col: str = _GROUP,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """براميل 30ث من ``t - lookback`` إلى ``t + lookahead``. لا يقصّ نافذة Y."""
    if lookback_ns < 0 or lookahead_ns < 0:
        raise ValueError("lookback_ns and lookahead_ns must be >= 0")
    trade_schema = {
        "trade_id": pl.Int64(),
        group_col: pl.Int64(),
        SETUP_AVAILABILITY_TS: pl.Int64(),
        "entry_utc": pl.Utf8(),
        "model_p": pl.Float64(),
        "direction": pl.Float64(),
        "printed_at_entry_pts": pl.Float64(),
        "close_pts": pl.Float64(),
        "asia_vah_pts": pl.Float64(),
        "asia_val_pts": pl.Float64(),
        "asia_poc_pts": pl.Float64(),
        "decision_vah_pts": pl.Float64(),
        "decision_val_pts": pl.Float64(),
        "decision_poc_pts": pl.Float64(),
        "y_tick": pl.Float64(),
        "n_bars": pl.Int64(),
        "n_bars_before": pl.Int64(),
        "n_bars_after": pl.Int64(),
        "first_in_story": pl.Boolean(),
    }
    bar_schema = {
        "trade_id": pl.Int64(),
        AVAILABILITY_TS: pl.Int64(),
        "minutes_from_entry": pl.Float64(),
        "close_pts": pl.Float64(),
        "path_beyond_asia_ticks": pl.Float64(),
        "path_inside_asia_va": pl.Float64(),
        "path_depth_confirm": pl.Float64(),
        "is_entry_bar": pl.Boolean(),
    }
    if entries.height == 0:
        return pl.DataFrame(schema=trade_schema), pl.DataFrame(schema=bar_schema)
    entries = entries.sort(SETUP_AVAILABILITY_TS)
    drop_look = [c for c in _LOOKAHEAD if c in blended.columns]
    work_b = blended.drop(drop_look) if drop_look else blended
    cols = [group_col, AVAILABILITY_TS, *[c for c in _BAR_COLS if c in work_b.columns]]
    stories = entries[group_col].unique().to_list()
    path = (
        work_b.filter(pl.col(group_col).is_in(stories))
        .select(cols)
        .sort([group_col, AVAILABILITY_TS])
    )
    grouped = _path_by_story(path, group_col)
    t0 = _i64_arr(entries, SETUP_AVAILABILITY_TS)
    story_vals = entries[group_col].to_list()
    beyond0 = _f64_arr(entries, "path_beyond_asia_ticks")
    n_ent = t0.size
    p_entry = (
        _f64_arr(entries, "model_p") if "model_p" in entries.columns else np.full(n_ent, np.nan)
    )
    y_tick = _f64_arr(entries, "y") if "y" in entries.columns else np.full(n_ent, np.nan)
    break_dir = _f64_arr(entries, "proj_break_direction")
    trade_rows: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    seen_story: set[Any] = set()
    for i, story in enumerate(story_vals):
        first = story not in seen_story
        seen_story.add(story)
        story_path = grouped.get(story)
        utc = datetime.fromtimestamp(int(t0[i]) / 1e9, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        base = {
            "trade_id": i,
            group_col: int(story) if story is not None else -1,
            SETUP_AVAILABILITY_TS: int(t0[i]),
            "entry_utc": utc,
            "model_p": float(p_entry[i]),
            "direction": float(break_dir[i]) if np.isfinite(break_dir[i]) else 0.0,
            "printed_at_entry_pts": float(beyond0[i]) / TICKS_PER_NQ_POINT
            if np.isfinite(beyond0[i])
            else float("nan"),
            "close_pts": _px_or_nan(entries, i, "close"),
            "asia_vah_pts": _px_or_nan(entries, i, "asia_vah"),
            "asia_val_pts": _px_or_nan(entries, i, "asia_val"),
            "asia_poc_pts": _px_or_nan(entries, i, "asia_poc"),
            "decision_vah_pts": _px_or_nan(entries, i, "decision_vah"),
            "decision_val_pts": _px_or_nan(entries, i, "decision_val"),
            "decision_poc_pts": _px_or_nan(entries, i, "decision_poc"),
            "y_tick": float(y_tick[i]),
            "n_bars": 0,
            "n_bars_before": 0,
            "n_bars_after": 0,
            "first_in_story": first,
        }
        if story_path is None or story_path.height == 0:
            trade_rows.append(base)
            continue
        ts = _i64_arr(story_path, AVAILABILITY_TS)
        lo = int(np.searchsorted(ts, int(t0[i]) - int(lookback_ns), side="left"))
        hi = int(np.searchsorted(ts, int(t0[i]) + int(lookahead_ns), side="right"))
        if lo >= hi:
            trade_rows.append(base)
            continue
        window = story_path[lo:hi]
        w_ts = ts[lo:hi]
        w_close = _f64_arr(window, "close")
        w_beyond = _f64_arr(window, "path_beyond_asia_ticks")
        w_inside = _f64_arr(window, "path_inside_asia_va")
        w_confirm = _f64_arr(window, "path_depth_confirm")
        at_or_before = w_ts[w_ts <= int(t0[i])]
        entry_stamp = int(at_or_before[-1]) if at_or_before.size else int(t0[i])
        n_before = 0
        n_after = 0
        for k, raw_ts in enumerate(w_ts.tolist()):
            stamp = int(raw_ts)
            if stamp < int(t0[i]):
                n_before += 1
            elif stamp > int(t0[i]):
                n_after += 1
            bar_rows.append(
                {
                    "trade_id": i,
                    AVAILABILITY_TS: stamp,
                    "minutes_from_entry": (stamp - int(t0[i])) / NS_PER_MIN,
                    "close_pts": to_nq_price(float(w_close[k])),
                    "path_beyond_asia_ticks": float(w_beyond[k]),
                    "path_inside_asia_va": float(w_inside[k]),
                    "path_depth_confirm": float(w_confirm[k]),
                    "is_entry_bar": stamp == entry_stamp,
                }
            )
        base["n_bars"] = int(hi - lo)
        base["n_bars_before"] = n_before
        base["n_bars_after"] = n_after
        trade_rows.append(base)
    return pl.DataFrame(trade_rows, schema=trade_schema), pl.DataFrame(bar_rows, schema=bar_schema)


def run_entry_replay(
    labeled: pl.DataFrame,
    blended: pl.DataFrame,
    *,
    config: EntryReplayConfig | None = None,
    oof_availability_ts: Sequence[int] | None = None,
    holdout_cut_ts: int | None = None,
    predictions: pl.DataFrame | None = None,
    progress: ProgressLike | None = None,
) -> EntryReplayReport:
    """يجمع نوافذ المشاهدة لكل إطلاق OOF سببي."""
    cfg = config or EntryReplayConfig()
    assert_not_raw_mbo_stream(labeled, source="labeled")
    assert_not_raw_mbo_stream(blended, source="blended")
    empty = EntryReplayReport(
        trades=pl.DataFrame(),
        bars=pl.DataFrame(),
        diagnostics={
            "empty": True,
            "layer_id": LAYER_ID,
            "removable_layer": True,
            "does_not_modify_science_y": True,
            "chart_timeframe_unchanged": True,
            "post_entry_path_is_inspection_only": True,
            "holdout_scored": False,
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
    if progress is not None:
        progress.op(f"entry_replay fires={fires.height:,}")
    trades, bars = extract_entry_windows(
        fires,
        blended,
        lookback_ns=cfg.lookback_ns,
        lookahead_ns=cfg.lookahead_ns,
    )
    n_first = int(trades["first_in_story"].sum()) if trades.height else 0
    diagnostics: dict[str, Any] = {
        "empty": False,
        "layer_id": LAYER_ID,
        "removable_layer": True,
        "does_not_modify_science_y": True,
        "chart_timeframe_unchanged": True,
        "post_entry_path_is_inspection_only": True,
        "holdout_scored": False,
        "live_predictions_not_used": True,
        "n_trades": int(trades.height),
        "n_bars": int(bars.height),
        "n_stories": n_first,
        "lookback_minutes": int(cfg.lookback_ns) / 60e9,
        "lookahead_minutes": int(cfg.lookahead_ns) / 60e9,
        "bar_clock": "30s blended states",
        "principles": (
            "eye tool — 10 minutes before t and 15 minutes after on the 30s clock",
            "every OOF causal fire is listed; not a new Y and not a timeframe change",
            "bars after t are inspection look-ahead, never a live entry filter",
            "holdout never scored; completed-wave peak never plotted as a target",
        ),
    }
    return EntryReplayReport(trades=trades, bars=bars, diagnostics=diagnostics)


def _payload(report: EntryReplayReport) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    if report.trades.height == 0:
        return {"trades": [], "diagnostics": jsonable(report.diagnostics)}
    bars_by: dict[int, list[dict[str, float | bool | None]]] = {}
    if report.bars.height:
        for row in report.bars.iter_rows(named=True):
            tid = int(row["trade_id"])
            bars_by.setdefault(tid, []).append(
                {
                    "m": round(float(row["minutes_from_entry"]), 4),
                    "c": round(float(row["close_pts"]), 4)
                    if row["close_pts"] is not None
                    else None,
                    "x": round(float(row["path_beyond_asia_ticks"]), 3)
                    if row["path_beyond_asia_ticks"] is not None
                    else None,
                    "e": bool(row["is_entry_bar"]),
                }
            )
    keep = (
        "trade_id",
        _GROUP,
        SETUP_AVAILABILITY_TS,
        "entry_utc",
        "model_p",
        "direction",
        "printed_at_entry_pts",
        "close_pts",
        "asia_vah_pts",
        "asia_val_pts",
        "asia_poc_pts",
        "decision_vah_pts",
        "decision_val_pts",
        "decision_poc_pts",
        "y_tick",
        "n_bars",
        "n_bars_before",
        "n_bars_after",
        "first_in_story",
    )
    present = [c for c in keep if c in report.trades.columns]
    for row in report.trades.select(present).iter_rows(named=True):
        tid = int(row["trade_id"])
        item = {k: jsonable(row[k]) for k in present}
        packed = bars_by.get(tid, [])
        item["m"] = [b["m"] for b in packed]
        item["c"] = [b["c"] for b in packed]
        item["x"] = [b["x"] for b in packed]
        trades.append(item)
    return {"trades": trades, "diagnostics": jsonable(report.diagnostics)}


def render_entry_replay_html(report: EntryReplayReport) -> str:
    if not _TEMPLATE.is_file():
        raise FileNotFoundError(f"entry replay template missing: {_TEMPLATE}")
    blob = json.dumps(_payload(report), ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("<", "\\u003c")
    template = _TEMPLATE.read_text(encoding="utf-8")
    return template.replace("%%DATA%%", blob)


def render_entry_replay_markdown(report: EntryReplayReport) -> str:
    d = report.diagnostics
    lines = [
        "# Entry replay (eye tool)",
        "",
        "Every causal OOF fire on the **same 30-second states**.",
        "Window = 10 minutes before `t` through 15 minutes after `t`.",
        "Open `ENTRY_REPLAY.html` and step through trades. Path after `t` is",
        "for your eyes only — not an entry filter and not a new science Y.",
        "",
        f"- trades={d.get('n_trades')} · stories (first fire)={d.get('n_stories')} · "
        f"bars={d.get('n_bars')}",
        f"- lookback={d.get('lookback_minutes')} min · lookahead={d.get('lookahead_minutes')} min",
        f"- clock={d.get('bar_clock')}",
        "",
        "## Principles",
        "",
    ]
    for item in d.get("principles", ()):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def limit_replay(report: EntryReplayReport, n: int) -> EntryReplayReport:
    """أول N إطلاقات زمنياً — للرسم والمعاينة، ليس عينة إحصائية."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if report.trades.height <= n:
        return report
    trades = report.trades.head(n)
    ids = trades["trade_id"].to_list()
    bars = report.bars.filter(pl.col("trade_id").is_in(ids)) if report.bars.height else report.bars
    diagnostics = dict(report.diagnostics)
    diagnostics.update(
        n_trades=int(trades.height),
        n_bars=int(bars.height),
        limited_to=int(n),
        n_stories=int(trades["first_in_story"].sum()) if trades.height else 0,
    )
    return EntryReplayReport(trades=trades, bars=bars, diagnostics=diagnostics)


def _xml_esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _finite(values: Sequence[float | None]) -> list[float]:
    out: list[float] = []
    for raw in values:
        if raw is None:
            continue
        val = float(raw)
        if np.isfinite(val):
            out.append(val)
    return out


def trade_chart_svg(
    minutes: Sequence[float],
    close: Sequence[float | None],
    *,
    asia_vah: float | None,
    asia_val: float | None,
    decision_vah: float | None,
    decision_val: float | None,
    title: str,
    width: int = 720,
    height: int = 200,
) -> str:
    """رسم SVG لصفقة واحدة: −10د … +15د على الإغلاق."""
    xs = [float(x) for x in minutes]
    ys = [float(y) if y is not None and np.isfinite(float(y)) else float("nan") for y in close]
    extras = _finite((asia_vah, asia_val, decision_vah, decision_val, *ys))
    left, right, top, bottom = 44, 12, 22, 24
    xmin, xmax = -10.0, 15.0
    if extras:
        ymin, ymax = min(extras), max(extras)
    else:
        ymin, ymax = 0.0, 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad

    def _x(minute: float) -> float:
        return left + (minute - xmin) / (xmax - xmin) * (width - left - right)

    def _y(price: float) -> float:
        return top + (ymax - price) / (ymax - ymin) * (height - top - bottom)

    def _hline(price: float | None, color: str, dash: str) -> str:
        if price is None or not np.isfinite(price):
            return ""
        y = _y(float(price))
        return (
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="{dash}"/>'
        )

    pts: list[str] = []
    for minute, price in zip(xs, ys, strict=False):
        if not np.isfinite(price):
            continue
        pts.append(f"{_x(minute):.2f},{_y(price):.2f}")
    poly = " ".join(pts)
    x0 = _x(0.0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="#10141c"/>'
        f'<rect x="{left}" y="{top}" width="{x0 - left:.2f}" '
        f'height="{height - top - bottom}" fill="#6ea8fe14"/>'
        f'<line x1="{x0:.2f}" y1="{top}" x2="{x0:.2f}" y2="{height - bottom}" '
        f'stroke="#f5c542" stroke-width="1.4"/>'
        f"{_hline(asia_vah, '#6ea8fe', '6 4')}"
        f"{_hline(asia_val, '#6ea8fe', '2 4')}"
        f"{_hline(decision_vah, '#c084fc', '6 4')}"
        f"{_hline(decision_val, '#c084fc', '2 4')}"
        f'<polyline fill="none" stroke="#3dd68c" stroke-width="1.6" points="{poly}"/>'
        f'<text x="8" y="14" fill="#e8edf4" font-size="11">{_xml_esc(title)}</text>'
        f'<text x="{_x(-10):.1f}" y="{height - 8}" fill="#8b97a8" font-size="10">-10د</text>'
        f'<text x="{x0:.1f}" y="{height - 8}" fill="#f5c542" font-size="10">دخول</text>'
        f'<text x="{_x(15) - 28:.1f}" y="{height - 8}" fill="#8b97a8" font-size="10">+15د</text>'
        "</svg>"
    )


def render_entry_replay_gallery_html(report: EntryReplayReport) -> str:
    """صفحة واحدة: كل الصفقات مرسومة تحت بعض (معاينة بالعين)."""
    cards: list[str] = []
    bars_by: dict[int, pl.DataFrame] = {}
    if report.bars.height:
        for key, group in report.bars.group_by("trade_id", maintain_order=True):
            tid = key[0] if isinstance(key, tuple) else key
            bars_by[int(tid)] = group
    for i, row in enumerate(report.trades.iter_rows(named=True), start=1):
        tid = int(row["trade_id"])
        window = bars_by.get(tid)
        minutes = window["minutes_from_entry"].to_list() if window is not None else []
        close = window["close_pts"].to_list() if window is not None else []
        direction = float(row["direction"]) if row["direction"] is not None else 0.0
        arrow = "↑" if direction > 0 else "↓" if direction < 0 else "·"
        p_val = row["model_p"]
        p_txt = "—" if p_val is None else f"{float(p_val):.2f}"
        title = f"{i}. {arrow} {row['entry_utc']}  p={p_txt}"
        svg = trade_chart_svg(
            minutes,
            close,
            asia_vah=row.get("asia_vah_pts"),
            asia_val=row.get("asia_val_pts"),
            decision_vah=row.get("decision_vah_pts"),
            decision_val=row.get("decision_val_pts"),
            title=title,
        )
        cards.append(f'<figure class="card">{svg}</figure>')
    body = "\n".join(cards) if cards else "<p>لا صفقات.</p>"
    n = int(report.trades.height)
    return (
        "<!DOCTYPE html><html lang='ar' dir='rtl'><head><meta charset='utf-8'/>"
        f"<title>NQ {n} trades — 30s gallery</title>"
        "<style>body{margin:0;background:#0e1116;color:#e8edf4;"
        "font:14px/1.4 ui-sans-serif,system-ui,sans-serif}"
        "header{padding:12px 18px;border-bottom:1px solid #2a3140}"
        ".grid{display:grid;gap:12px;padding:12px 18px}"
        ".card{margin:0;background:#171b22;border:1px solid #2a3140;"
        "border-radius:8px;padding:8px}"
        "h1{margin:0;font-size:16px}.mut{color:#8b97a8}</style></head><body>"
        f"<header><h1>رسم {n} صفقة — 30 ثانية</h1>"
        "<p class='mut'>10 دقائق قبل الدخول · 15 دقيقة بعده · "
        "ذهبي = الإشارة · أزرق = آسيا · بنفسجي = قرار لندن</p></header>"
        f"<div class='grid'>{body}</div></body></html>"
    )


def write_entry_replay_png_sheets(
    report: EntryReplayReport,
    output_dir: Path | str,
    *,
    trades_per_sheet: int = 25,
    cols: int = 5,
) -> list[Path]:
    """صحائف PNG للمعاينة. يحتاج matplotlib عند الاستدعاء فقط."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bars_by: dict[int, pl.DataFrame] = {}
    if report.bars.height:
        for key, group in report.bars.group_by("trade_id", maintain_order=True):
            tid = key[0] if isinstance(key, tuple) else key
            bars_by[int(tid)] = group
    rows = list(report.trades.iter_rows(named=True))
    written: list[Path] = []
    per = max(1, int(trades_per_sheet))
    n_cols = max(1, int(cols))
    for sheet, start in enumerate(range(0, len(rows), per), start=1):
        chunk = rows[start : start + per]
        n_rows = int(np.ceil(len(chunk) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 2.2), squeeze=False
        )
        fig.patch.set_facecolor("#0e1116")
        for ax in axes.ravel():
            ax.set_visible(False)
        for i, row in enumerate(chunk):
            ax = axes[i // n_cols][i % n_cols]
            ax.set_visible(True)
            ax.set_facecolor("#10141c")
            window = bars_by.get(int(row["trade_id"]))
            if window is not None and window.height:
                ax.plot(
                    window["minutes_from_entry"].to_list(),
                    window["close_pts"].to_list(),
                    color="#3dd68c",
                    lw=1.1,
                )
            ax.axvline(0.0, color="#f5c542", lw=1.0)
            ax.axvspan(-10.0, 0.0, color="#6ea8fe", alpha=0.08)
            for name, color, ls in (
                ("asia_vah_pts", "#6ea8fe", "--"),
                ("asia_val_pts", "#6ea8fe", ":"),
                ("decision_vah_pts", "#c084fc", "--"),
                ("decision_val_pts", "#c084fc", ":"),
            ):
                val = row.get(name)
                if val is not None and np.isfinite(float(val)):
                    ax.axhline(float(val), color=color, ls=ls, lw=0.8)
            ax.set_xlim(-10.0, 15.0)
            direction = float(row["direction"]) if row["direction"] is not None else 0.0
            arrow = "up" if direction > 0 else "dn" if direction < 0 else "."
            ax.set_title(f"{start + i + 1} {arrow}", color="#e8edf4", fontsize=8, loc="left")
            ax.tick_params(colors="#8b97a8", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#2a3140")
        fig.suptitle(
            f"OOF fires {start + 1}–{start + len(chunk)} · 30s · −10m to +15m",
            color="#e8edf4",
            fontsize=11,
        )
        fig.tight_layout()
        path = out / f"entry_replay_sheet_{sheet:02d}.png"
        fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(path)
    return written


def write_entry_replay_report(report: EntryReplayReport, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report.trades.height:
        report.trades.write_parquet(out / "entry_replay_trades.parquet")
    if report.bars.height:
        report.bars.write_parquet(out / "entry_replay_bars.parquet")
    payload = {
        "diagnostics": jsonable(report.diagnostics),
        "holdout_scored": False,
        "post_entry_path_is_inspection_only": True,
    }
    (out / "entry_replay.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out / "ENTRY_REPLAY.md").write_text(render_entry_replay_markdown(report), encoding="utf-8")
    (out / "ENTRY_REPLAY.html").write_text(render_entry_replay_html(report), encoding="utf-8")
    (out / "ENTRY_REPLAY_GALLERY.html").write_text(
        render_entry_replay_gallery_html(report), encoding="utf-8"
    )
    return out


def run_entry_replay_from_period_dir(
    period_dir: Path | str,
    *,
    config: EntryReplayConfig | None = None,
    progress: ProgressLike | None = None,
) -> EntryReplayReport:
    labeled, blended, oof, cut_ts = load_overlay_period_inputs(period_dir)
    return run_entry_replay(
        labeled,
        blended,
        config=config,
        oof_availability_ts=oof_timestamps(oof),
        holdout_cut_ts=cut_ts,
        predictions=oof,
        progress=progress,
    )


__all__ = [
    "DEFAULT_LOOKAHEAD_NS",
    "DEFAULT_LOOKBACK_NS",
    "LAYER_ID",
    "EntryReplayConfig",
    "EntryReplayReport",
    "extract_entry_windows",
    "limit_replay",
    "render_entry_replay_gallery_html",
    "render_entry_replay_html",
    "render_entry_replay_markdown",
    "run_entry_replay",
    "run_entry_replay_from_period_dir",
    "to_nq_price",
    "trade_chart_svg",
    "write_entry_replay_png_sheets",
    "write_entry_replay_report",
]
