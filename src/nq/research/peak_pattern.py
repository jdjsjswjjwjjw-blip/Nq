"""مسح النمط الثلاثي على شرائح اليوم: هل يتبعه هبوط أكثر من العشوائي؟

شروط مقفلة من قمة PR #116 (ليست عتبة مُلاءَمة بعد المسح):
``T_rate > 50``، ``T_imbalance > 0.10``، ``ask_hit_share < 0.20``.
النتيجة بعد نهاية النافذة فقط (5 دقائق). 0.5%/1% تُحسب كما طُلب؛ هبوط
القمة نفسها كان ~0.16% في 5 دقائق فـ 50bps قد لا يمسك حتى ذلك الحدث.
ليس نموذجًا حيًّا ولا LSTM. احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from nq.contracts.mbo import PRICE_SCALE, MboAction, MboSide
from nq.contracts.temporal import EVENT_TS
from nq.core.determinism import make_generator
from nq.research.mbo_trade_overlap import prepare_mbo_events
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import WINDOW_S, default_windows
from nq.research.peak_flow import score_flow_window

LAYER_ID = "peak_pattern"
T_RATE_MIN: Final = 50.0
T_IMB_MIN: Final = 0.10
FILL_RATIO_MAX: Final = 0.20
STRIDE_S: Final = 5
HORIZON_S: Final = 300
REV_BPS: Final = (10, 50, 100)
REV_POINTS: Final = (20.0, 40.0)
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_CANCEL = MboAction.CANCEL.value
_BID = MboSide.BID.value
_ASK = MboSide.ASK.value
_FIXED_POINT_PRICE: Final = 1_000_000.0


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _point_unit(price: float) -> float:
    return float(round(1.0 / PRICE_SCALE)) if abs(price) >= _FIXED_POINT_PRICE else 1.0


def _forward_min(min_px: np.ndarray, horizon: int) -> np.ndarray:
    n = int(min_px.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or horizon < 1:
        return out
    pad = np.concatenate([min_px.astype(np.float64, copy=False), np.full(horizon, np.nan)])
    view = sliding_window_view(pad[1:], horizon)[:n]
    with np.errstate(all="ignore"):
        filled = np.where(np.isnan(view), np.inf, view)
        out = np.min(filled, axis=1)
    out[np.isinf(out)] = np.nan
    return out


def _slot_bars(book: pl.DataFrame, slot_ns: int) -> pl.DataFrame:
    events = book.filter(pl.col("action").is_in([_TRADE, _FILL, _CANCEL])).with_columns(
        (pl.col(EVENT_TS) // slot_ns).alias("slot")
    )
    t = events.filter(pl.col("action") == _TRADE)
    f = events.filter(pl.col("action") == _FILL)
    c = events.filter(pl.col("action") == _CANCEL)
    t_agg = t.group_by("slot").agg(
        pl.len().alias("n_t"),
        pl.col("size").filter(pl.col("side") == _BID).sum().alias("t_buy"),
        pl.col("size").filter(pl.col("side") == _ASK).sum().alias("t_sell"),
        pl.col("price").max().alias("max_px"),
        pl.col("price").min().alias("min_px"),
    )
    f_agg = f.group_by("slot").agg(
        pl.col("size").filter(pl.col("side") == _ASK).sum().alias("f_ask"),
        pl.col("size").filter(pl.col("side") == _BID).sum().alias("f_bid"),
    )
    c_agg = c.group_by("slot").agg(
        pl.col("size").filter(pl.col("side") == _ASK).sum().alias("c_ask"),
    )
    parts = [s.select("slot") for s in (t_agg, f_agg, c_agg) if s.height]
    if not parts:
        return pl.DataFrame(schema={"slot": pl.Int64()})
    keys = pl.concat(parts, how="vertical").unique()
    bars = (
        keys.join(t_agg, on="slot", how="left")
        .join(f_agg, on="slot", how="left")
        .join(c_agg, on="slot", how="left")
    )
    lo = int(bars.select(pl.col("slot").min()).item())
    hi = int(bars.select(pl.col("slot").max()).item())
    grid = pl.DataFrame({"slot": list(range(lo, hi + 1))})
    return grid.join(bars, on="slot", how="left").sort("slot")


def _roll_complete(bars: pl.DataFrame, n_slots: int) -> pl.DataFrame:
    filled = bars.with_columns(
        pl.col("n_t").fill_null(0),
        pl.col("t_buy").fill_null(0),
        pl.col("t_sell").fill_null(0),
        pl.col("f_ask").fill_null(0),
        pl.col("c_ask").fill_null(0),
    )
    return filled.with_columns(
        pl.col("n_t").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("n_t_w"),
        pl.col("t_buy").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("t_buy_w"),
        pl.col("t_sell").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("t_sell_w"),
        pl.col("f_ask").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("f_ask_w"),
        pl.col("c_ask").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("c_ask_w"),
        pl.col("max_px").rolling_max(window_size=n_slots, min_samples=1).alias("high_w"),
    ).filter(pl.col("n_t_w").is_not_null())


def _window_ns(window_s: int) -> int:
    return int(window_s) * SECOND_NS


def _attach_times(
    rolled: pl.DataFrame, bars: pl.DataFrame, horizon_slots: int, slot_ns: int, window_s: int
) -> pl.DataFrame:
    min_px = bars["min_px"].cast(pl.Float64).to_numpy()
    fwd = _forward_min(min_px, horizon_slots)
    fwd_df = bars.select("slot").with_columns(pl.Series("fwd_min", fwd))
    win_ns = _window_ns(window_s)
    return rolled.join(fwd_df, on="slot", how="left").with_columns(
        ((pl.col("slot") + 1) * slot_ns).alias("end_ts"),
        ((pl.col("slot") + 1) * slot_ns - win_ns).alias("start_ts"),
    )


def _features(rolled: pl.DataFrame, window_s: int) -> pl.DataFrame:
    tot = pl.col("t_buy_w") + pl.col("t_sell_w")
    hit_den = pl.col("f_ask_w") + pl.col("c_ask_w")
    return rolled.with_columns(
        (pl.col("n_t_w") / float(window_s)).alias("t_rate"),
        pl.when(tot > 0)
        .then((pl.col("t_buy_w") - pl.col("t_sell_w")) / tot)
        .otherwise(None)
        .alias("t_imbalance"),
        pl.when(hit_den > 0).then(pl.col("f_ask_w") / hit_den).otherwise(None).alias("fill_ratio"),
        (pl.col("high_w") - pl.col("fwd_min")).alias("drop_native"),
    )


def _pattern_mask(
    frame: pl.DataFrame, t_rate_min: float, t_imb_min: float, fill_ratio_max: float
) -> pl.Expr:
    return (
        (pl.col("t_rate") > t_rate_min)
        & (pl.col("t_imbalance") > t_imb_min)
        & (pl.col("fill_ratio") < fill_ratio_max)
    )


def _reversal_flags(frame: pl.DataFrame, point_unit: float) -> pl.DataFrame:
    drop = pl.col("drop_native")
    high = pl.col("high_w")
    cols = [
        (drop / high).alias("drop_frac"),
        (drop / point_unit).alias("drop_points"),
    ]
    for bps in REV_BPS:
        cols.append((drop >= high * (bps / 10_000.0)).alias(f"rev_{bps}bps"))
    for pts in REV_POINTS:
        cols.append((drop >= point_unit * pts).alias(f"rev_{int(pts)}pt"))
    return frame.with_columns(cols)


def _episode_ids(slots: np.ndarray) -> np.ndarray:
    n = slots.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    ids = np.empty(n, dtype=np.int64)
    eid = 0
    ids[0] = 0
    for i in range(1, n):
        if int(slots[i]) - int(slots[i - 1]) > 1:
            eid += 1
        ids[i] = eid
    return ids


def _hour_utc_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.height == 0:
        return {}
    hours = (
        frame.select(pl.from_epoch(pl.col("end_ts"), time_unit="ns").dt.hour().alias("h"))
        .group_by("h")
        .len()
        .sort("h")
    )
    return {
        str(int(h)): int(n)
        for h, n in zip(hours["h"].to_list(), hours["len"].to_list(), strict=True)
    }


def _rate(frame: pl.DataFrame, col: str) -> float:
    if frame.height == 0:
        return float("nan")
    return float(frame.select(pl.col(col).mean()).item() or 0.0)


def _summarize_group(frame: pl.DataFrame, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "n": frame.height}
    if frame.height == 0:
        return out
    out["mean_t_rate"] = float(frame.select(pl.col("t_rate").mean()).item() or 0.0)
    out["mean_t_imbalance"] = float(frame.select(pl.col("t_imbalance").mean()).item() or 0.0)
    out["mean_fill_ratio"] = float(frame.select(pl.col("fill_ratio").mean()).item() or 0.0)
    drop_pts = frame.select(pl.col("drop_points").median()).item()
    drop_frac = frame.select(pl.col("drop_frac").median()).item()
    out["median_drop_points"] = float("nan") if drop_pts is None else float(drop_pts)
    out["median_drop_frac"] = float("nan") if drop_frac is None else float(drop_frac)
    for bps in REV_BPS:
        out[f"rate_{bps}bps"] = _rate(frame, f"rev_{bps}bps")
    for pts in REV_POINTS:
        out[f"rate_{int(pts)}pt"] = _rate(frame, f"rev_{int(pts)}pt")
    return out


def scan_peak_pattern(
    mbo: pl.DataFrame,
    *,
    price_hi: float | None = None,
    t_rate_min: float = T_RATE_MIN,
    t_imb_min: float = T_IMB_MIN,
    fill_ratio_max: float = FILL_RATIO_MAX,
    window_s: int = WINDOW_S,
    stride_s: int = STRIDE_S,
    horizon_s: int = HORIZON_S,
    seed: int = 0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """شرائح متحركة بخطوة ``stride_s``. النتيجة بعد ``end_ts`` فقط."""

    if window_s % stride_s != 0:
        raise ValueError("window_s must be divisible by stride_s")
    if horizon_s % stride_s != 0:
        raise ValueError("horizon_s must be divisible by stride_s")
    book = prepare_mbo_events(mbo)
    slot_ns = int(stride_s) * SECOND_NS
    n_slots = int(window_s) // int(stride_s)
    horizon_slots = int(horizon_s) // int(stride_s)
    bars = _slot_bars(book, slot_ns)
    if bars.height == 0:
        empty = pl.DataFrame()
        return empty, {"layer": LAYER_ID, "n_windows": 0, "note": "no T/F/C events"}
    rolled = _roll_complete(bars, n_slots)
    scored = _features(_attach_times(rolled, bars, horizon_slots, slot_ns, window_s), window_s)
    sample_px = scored.filter(pl.col("high_w").is_not_null())
    px0 = float(sample_px["high_w"][0]) if sample_px.height else 0.0
    scored = _reversal_flags(scored, _point_unit(px0)).filter(pl.col("n_t_w") > 0)
    mask = _pattern_mask(scored, t_rate_min, t_imb_min, fill_ratio_max)
    hits = scored.filter(mask).sort("slot")
    ctrl = scored.filter(~mask)
    episodes = hits
    n_ep = 0
    if hits.height:
        eids = _episode_ids(hits["slot"].to_numpy())
        episodes = hits.with_columns(pl.Series("episode", eids)).unique(
            subset=["episode"], keep="first"
        )
        n_ep = episodes.height
    rng = make_generator(seed)
    n_rand = min(hits.height, ctrl.height)
    random_ctrl = ctrl.head(0)
    if n_rand > 0 and ctrl.height > 0:
        idx = np.sort(rng.choice(ctrl.height, size=n_rand, replace=False))
        random_ctrl = ctrl.gather(idx.tolist())
    busy = scored.filter(pl.col("t_rate") > t_rate_min)
    busy_ctrl = busy.filter(~mask)
    summary = {
        "pattern_windows": _summarize_group(hits, "pattern_windows"),
        "pattern_episodes": _summarize_group(episodes, "pattern_episodes"),
        "control": _summarize_group(ctrl, "control"),
        "random_control": _summarize_group(random_ctrl, "random_control"),
        "busy_control": _summarize_group(busy_ctrl, "busy_control"),
    }
    named: dict[str, Any] = {}
    if price_hi is not None:
        tcol = book.filter(pl.col("action") == _TRADE)
        high_row = tcol.filter(pl.col("price") >= price_hi).sort(EVENT_TS).head(1)
        if high_row.height:
            high_ts = int(high_row[EVENT_TS][0])
            peak_w = default_windows(high_ts, window_s=window_s)[0]
            flow = score_flow_window(book, peak_w)
            event_high = float(high_row["price"][0])
            after = tcol.filter(
                (pl.col(EVENT_TS) >= peak_w.end_ts)
                & (pl.col(EVENT_TS) < peak_w.end_ts + horizon_s * SECOND_NS)
            )
            fwd = after.select(pl.col("price").min()).item() if after.height else None
            drop_n = float("nan") if fwd is None else event_high - float(fwd)
            named = {
                "high_ts": high_ts,
                "t_rate": flow["t_per_s"],
                "t_imbalance": flow["t_imbalance"],
                "fill_ratio": flow["ask_hit_share"],
                "matches_pattern": bool(
                    flow["t_per_s"] > t_rate_min
                    and flow["t_imbalance"] > t_imb_min
                    and flow["ask_hit_share"] < fill_ratio_max
                ),
                "peak_high": event_high,
                "fwd_min": None if fwd is None else float(fwd),
                "drop_native": drop_n,
                "drop_frac": _ratio(drop_n, event_high),
            }
    diagnostics = {
        "layer": LAYER_ID,
        "t_rate_min": t_rate_min,
        "t_imb_min": t_imb_min,
        "fill_ratio_max": fill_ratio_max,
        "window_s": window_s,
        "stride_s": stride_s,
        "horizon_s": horizon_s,
        "seed": seed,
        "n_windows": scored.height,
        "n_pattern_windows": hits.height,
        "n_pattern_episodes": n_ep,
        "n_control": ctrl.height,
        "n_busy_control": busy_ctrl.height,
        "n_busy": busy.height,
        "pattern_hour_utc": _hour_utc_counts(hits),
        "summary": summary,
        "named_peak": named,
        "note_50bps": (
            "0.5% of MNQ ~150pts; the 10:29 high fell ~0.16% in 5m, so 50bps "
            "may miss even that event"
        ),
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
        "thresholds_locked_from_pr116": True,
    }
    return scored, diagnostics


def write_pattern_report(
    scored: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hits = (
        scored.filter(
            (pl.col("t_rate") > float(diagnostics["t_rate_min"]))
            & (pl.col("t_imbalance") > float(diagnostics["t_imb_min"]))
            & (pl.col("fill_ratio") < float(diagnostics["fill_ratio_max"]))
        )
        if scored.height
        else scored
    )
    if hits.height:
        hits.write_parquet(out / "pattern_hits.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    summ = diagnostics.get("summary", {})
    lines = [
        "# Peak pattern scan vs control (same day)",
        "",
        "Locked from PR #116: T_rate>50, T_imb>0.10, ask_hit<0.20.",
        "Outcome is the 5 minutes after window end only. Not a live overlay.",
        "busy_control = T_rate above the lock, but imb/fill fail (activity-matched).",
        "",
    ]
    if isinstance(summ, Mapping):
        lines += [
            "| group | n | med pts | med frac | 10bps | 50bps | 100bps | 20pt | 40pt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in (
            "pattern_windows",
            "pattern_episodes",
            "control",
            "random_control",
            "busy_control",
        ):
            row = summ.get(key, {})
            if not isinstance(row, Mapping) or not row:
                continue
            n = row.get("n", 0)
            if n == 0:
                lines.append(f"| {key} | 0 | nan | nan | nan | nan | nan | nan | nan |")
                continue
            lines.append(
                f"| {key} | {n} | {float(row.get('median_drop_points', float('nan'))):.3f} | "
                f"{float(row.get('median_drop_frac', float('nan'))):.6f} | "
                f"{float(row.get('rate_10bps', float('nan'))):.4f} | "
                f"{float(row.get('rate_50bps', float('nan'))):.4f} | "
                f"{float(row.get('rate_100bps', float('nan'))):.4f} | "
                f"{float(row.get('rate_20pt', float('nan'))):.4f} | "
                f"{float(row.get('rate_40pt', float('nan'))):.4f} |"
            )
    peak = diagnostics.get("named_peak", {})
    if isinstance(peak, Mapping) and peak:
        lines += ["", f"Named peak window: {json.dumps(peak, default=str)}"]
    hours = diagnostics.get("pattern_hour_utc", {})
    if isinstance(hours, Mapping) and hours:
        lines += ["", f"Pattern windows by UTC hour: {json.dumps(dict(hours), default=str)}"]
    lines.append("")
    (out / "PEAK_PATTERN.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "FILL_RATIO_MAX",
    "HORIZON_S",
    "LAYER_ID",
    "STRIDE_S",
    "T_IMB_MIN",
    "T_RATE_MIN",
    "scan_peak_pattern",
    "write_pattern_report",
]
