"""دمج MNQ MBO + MNQ Trades + NQ Trades على نوافذ PR #116 ثم مسح اليوم.

فرضية مقفلة قبل المسح (ليست مُلاءَمة بعد الأرقام):
عند القمة ``Fill_Ratio(MNQ) < 0.20`` و ``T_imbalance(NQ) > 0.20``؛
بعدها يُفترض أن يختفي مشترون NQ (``|imb| < 0.05``) مع بقاء الملء منخفضًا.
NQ بلا MBO: Fill_Ratio لـ NQ لا يُختلق. ليست overlay ولا LSTM.
احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from nq.contracts.mbo import MboAction, MboSide
from nq.contracts.temporal import EVENT_TS
from nq.core.determinism import make_generator
from nq.research.cross_nq_mnq import MNQ_MULT, NQ_MULT, _first_t_ts, _with_notional
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape
from nq.research.opposite_phantom import SECOND_NS
from nq.research.peak_control import WINDOW_S, NamedWindow, default_windows
from nq.research.peak_flow import score_flow_window
from nq.research.peak_pattern import (
    HORIZON_S,
    STRIDE_S,
    _attach_times,
    _episode_ids,
    _features,
    _hour_utc_counts,
    _point_unit,
    _reversal_flags,
    _roll_complete,
    _slot_bars,
    _summarize_group,
)

LAYER_ID = "triple_tape"
FILL_RATIO_MAX: Final = 0.20
NQ_IMB_MIN: Final = 0.20
NQ_IMB_NEAR_ZERO: Final = 0.05
_TRADE = MboAction.TRADE.value
_ADD = MboAction.ADD.value
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


def _slice(book: pl.DataFrame, start_ts: int, end_ts: int) -> pl.DataFrame:
    return book.filter((pl.col(EVENT_TS) >= start_ts) & (pl.col(EVENT_TS) < end_ts))


def _side_size(frame: pl.DataFrame, action: str, side: str) -> int:
    return _isum(frame.filter((pl.col("action") == action) & (pl.col("side") == side)), "size")


def _cancel_extras(
    book: pl.DataFrame, window: NamedWindow, flow: Mapping[str, Any]
) -> dict[str, Any]:
    chunk = _slice(book, window.start_ts, window.end_ts)
    a_ask = _side_size(chunk, _ADD, _ASK)
    a_bid = _side_size(chunk, _ADD, _BID)
    a_size = a_ask + a_bid
    c_size = int(flow["c_ask_size"]) + int(flow["c_bid_size"])
    f_ask = float(flow["f_ask_size"])
    c_ask = float(flow["c_ask_size"])
    return {
        "a_ask_size": a_ask,
        "a_bid_size": a_bid,
        "a_size": a_size,
        "c_size": c_size,
        "cancel_over_add": _ratio(c_size, a_size),
        "ask_cancel_share": _ratio(c_ask, f_ask + c_ask),
    }


def _hypotheses(fill_ratio: float, nq_imb: float) -> dict[str, bool]:
    return {
        "peak_hypothesis": bool(fill_ratio < FILL_RATIO_MAX and nq_imb > NQ_IMB_MIN),
        "drop_hypothesis": bool(fill_ratio < FILL_RATIO_MAX and abs(nq_imb) < NQ_IMB_NEAR_ZERO),
    }


def _score_window(
    mnq_mbo: pl.DataFrame,
    mnq_tape: pl.DataFrame,
    nq_tape: pl.DataFrame,
    window: NamedWindow,
) -> dict[str, Any]:
    mbo = _with_notional(
        score_flow_window(mnq_mbo, window),
        contract="MNQ",
        multiplier=MNQ_MULT,
        window=window,
    )
    extras = _cancel_extras(mnq_mbo, window, mbo)
    tape = _with_notional(
        score_flow_window(mnq_tape, window),
        contract="MNQ",
        multiplier=MNQ_MULT,
        window=window,
    )
    nq = _with_notional(
        score_flow_window(nq_tape, window),
        contract="NQ",
        multiplier=NQ_MULT,
        window=window,
    )
    fill = float(mbo["ask_hit_share"])
    nq_imb = float(nq["t_imbalance"])
    row = {
        "name": window.name,
        "start_ts": window.start_ts,
        "end_ts": window.end_ts,
        "mnq_mbo_n_t": mbo["n_t"],
        "mnq_mbo_t_per_s": mbo["t_per_s"],
        "mnq_mbo_t_imbalance": mbo["t_imbalance"],
        "mnq_fill_ratio": fill,
        "mnq_ask_cancel_share": extras["ask_cancel_share"],
        "mnq_cancel_over_add": extras["cancel_over_add"],
        "mnq_a_size": extras["a_size"],
        "mnq_c_size": extras["c_size"],
        "mnq_tape_n_t": tape["n_t"],
        "mnq_tape_t_per_s": tape["t_per_s"],
        "mnq_tape_t_imbalance": tape["t_imbalance"],
        "mnq_tape_t_notional": tape["t_notional"],
        "nq_n_t": nq["n_t"],
        "nq_t_per_s": nq["t_per_s"],
        "nq_t_imbalance": nq_imb,
        "nq_t_notional": nq["t_notional"],
        "nq_over_mnq_notional": _ratio(float(nq["t_notional"]), float(tape["t_notional"])),
        "mbo_tape_n_t_diff": int(tape["n_t"]) - int(mbo["n_t"]),
        "nq_fill_ratio": float("nan"),
    }
    row.update(_hypotheses(fill, nq_imb))
    return row


def compare_triple_windows(
    mnq_mbo: pl.DataFrame,
    mnq_trades: pl.DataFrame,
    nq_trades: pl.DataFrame,
    *,
    price_hi: float,
    window_s: int = WINDOW_S,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """نفس ساعات #116. MNQ دفتر + شريط، NQ شريط ``T`` فقط."""

    book = prepare_mbo_events(mnq_mbo)
    mnq_tape = prepare_trades_tape(mnq_trades)
    nq_tape = prepare_trades_tape(nq_trades)
    high_ts = _first_t_ts(book, price_hi)
    if high_ts is None:
        raise ValueError(f"no MNQ T print at price >= {price_hi}")
    windows = default_windows(high_ts, window_s=window_s)
    rows = [_score_window(book, mnq_tape, nq_tape, w) for w in windows]
    table = pl.DataFrame(rows)
    by = {r["name"]: r for r in rows}
    diagnostics = {
        "layer": LAYER_ID,
        "price_hi": price_hi,
        "mnq_high_ts": high_ts,
        "window_s": window_s,
        "fill_ratio_max": FILL_RATIO_MAX,
        "nq_imb_min": NQ_IMB_MIN,
        "nq_imb_near_zero": NQ_IMB_NEAR_ZERO,
        "peak_hypothesis_holds": bool(by.get("peak", {}).get("peak_hypothesis")),
        "drop_hypothesis_holds": bool(by.get("drop", {}).get("drop_hypothesis")),
        "nq_source": "trades_tape_T_only",
        "nq_fill_ratio": "unavailable_without_mbo_F_C",
        "clock": "locked_to_mnq_first_T_at_price_hi",
        "thresholds_locked_before_scan": True,
        "not_spoofing": True,
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
        "phantom_closed": True,
    }
    return table, diagnostics


def _tape_bars(tape: pl.DataFrame, slot_ns: int) -> pl.DataFrame:
    t = tape.filter(pl.col("action") == _TRADE).with_columns(
        (pl.col(EVENT_TS) // slot_ns).alias("slot")
    )
    if t.height == 0:
        return pl.DataFrame(
            schema={
                "slot": pl.Int64(),
                "n_t": pl.Int64(),
                "t_buy": pl.Int64(),
                "t_sell": pl.Int64(),
            }
        )
    return t.group_by("slot").agg(
        pl.len().alias("n_t"),
        pl.col("size").filter(pl.col("side") == _BID).sum().alias("t_buy"),
        pl.col("size").filter(pl.col("side") == _ASK).sum().alias("t_sell"),
    )


def _roll_tape_on_grid(
    grid: pl.DataFrame, tape: pl.DataFrame, slot_ns: int, n_slots: int
) -> pl.DataFrame:
    bars = grid.join(_tape_bars(tape, slot_ns), on="slot", how="left").with_columns(
        pl.col("n_t").fill_null(0),
        pl.col("t_buy").fill_null(0),
        pl.col("t_sell").fill_null(0),
    )
    return bars.with_columns(
        pl.col("n_t").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("n_t_w"),
        pl.col("t_buy").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("t_buy_w"),
        pl.col("t_sell").rolling_sum(window_size=n_slots, min_samples=n_slots).alias("t_sell_w"),
    )


def _tape_imb_rate(rolled: pl.DataFrame, window_s: int, prefix: str) -> pl.DataFrame:
    tot = pl.col("t_buy_w") + pl.col("t_sell_w")
    return rolled.select(
        "slot",
        (pl.col("n_t_w") / float(window_s)).alias(f"{prefix}_t_rate"),
        pl.when(tot > 0)
        .then((pl.col("t_buy_w") - pl.col("t_sell_w")) / tot)
        .otherwise(None)
        .alias(f"{prefix}_t_imbalance"),
        pl.col("n_t_w").alias(f"{prefix}_n_t"),
    )


def _joint_mask(fill_max: float, nq_imb_min: float) -> pl.Expr:
    return (pl.col("fill_ratio") < fill_max) & (pl.col("nq_t_imbalance") > nq_imb_min)


def _summarize_joint(frame: pl.DataFrame, label: str) -> dict[str, Any]:
    out = _summarize_group(frame, label)
    if frame.height == 0:
        return out
    out["mean_nq_t_rate"] = float(frame.select(pl.col("nq_t_rate").mean()).item() or 0.0)
    out["mean_nq_t_imbalance"] = float(frame.select(pl.col("nq_t_imbalance").mean()).item() or 0.0)
    out["mean_tape_t_rate"] = float(frame.select(pl.col("tape_t_rate").mean()).item() or 0.0)
    return out


def _groups(
    scored: pl.DataFrame,
    mask: pl.Expr,
    seed: int,
    fill_max: float,
    nq_min: float,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    hits = scored.filter(mask).sort("slot")
    ctrl = scored.filter(~mask)
    fill_only = scored.filter((pl.col("fill_ratio") < fill_max) & ~mask)
    nq_buy_only = scored.filter((pl.col("nq_t_imbalance") > nq_min) & ~mask)
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
    summary = {
        "pattern_windows": _summarize_joint(hits, "pattern_windows"),
        "pattern_episodes": _summarize_joint(episodes, "pattern_episodes"),
        "control": _summarize_joint(ctrl, "control"),
        "random_control": _summarize_joint(random_ctrl, "random_control"),
        "fill_only": _summarize_joint(fill_only, "fill_only"),
        "nq_buy_only": _summarize_joint(nq_buy_only, "nq_buy_only"),
    }
    return (
        hits,
        episodes,
        {
            "n_pattern_windows": hits.height,
            "n_pattern_episodes": n_ep,
            "n_control": ctrl.height,
            "n_fill_only": fill_only.height,
            "n_nq_buy_only": nq_buy_only.height,
            "summary": summary,
            "pattern_hour_utc": _hour_utc_counts(hits),
        },
    )


def scan_triple_pattern(
    mnq_mbo: pl.DataFrame,
    mnq_trades: pl.DataFrame,
    nq_trades: pl.DataFrame,
    *,
    fill_ratio_max: float = FILL_RATIO_MAX,
    nq_imb_min: float = NQ_IMB_MIN,
    window_s: int = WINDOW_S,
    stride_s: int = STRIDE_S,
    horizon_s: int = HORIZON_S,
    seed: int = 0,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """شرائح 30ث: ``fill_ratio < 0.20`` و ``nq_imb > 0.20``. النتيجة بعد النهاية فقط."""

    if window_s % stride_s != 0 or horizon_s % stride_s != 0:
        raise ValueError("window_s and horizon_s must be divisible by stride_s")
    book = prepare_mbo_events(mnq_mbo)
    mnq_tape = prepare_trades_tape(mnq_trades)
    nq_tape = prepare_trades_tape(nq_trades)
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
    grid = bars.select("slot")
    nq_feat = _tape_imb_rate(_roll_tape_on_grid(grid, nq_tape, slot_ns, n_slots), window_s, "nq")
    tape_feat = _tape_imb_rate(
        _roll_tape_on_grid(grid, mnq_tape, slot_ns, n_slots), window_s, "tape"
    )
    scored = scored.join(nq_feat, on="slot", how="left").join(tape_feat, on="slot", how="left")
    mask = _joint_mask(fill_ratio_max, nq_imb_min)
    _hits, _episodes, group = _groups(scored, mask, seed, fill_ratio_max, nq_imb_min)
    diagnostics = {
        "layer": LAYER_ID,
        "fill_ratio_max": fill_ratio_max,
        "nq_imb_min": nq_imb_min,
        "window_s": window_s,
        "stride_s": stride_s,
        "horizon_s": horizon_s,
        "seed": seed,
        "n_windows": scored.height,
        "nq_source": "trades_tape_T_only",
        "nq_fill_ratio": "unavailable_without_mbo_F_C",
        "thresholds_locked_before_scan": True,
        "not_spoofing": True,
        "not_lstm": True,
        "not_live_overlay": True,
        "not_backtest": True,
        "note_50bps": (
            "0.5% of MNQ ~150pts; the 10:29 high fell ~0.16% in 5m, so 50bps "
            "may miss even that event"
        ),
        **group,
    }
    return scored, diagnostics


def write_triple_report(
    named: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
    *,
    scored: pl.DataFrame | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if named.height:
        named.write_parquet(out / "triple_named.parquet")
    if scored is not None and scored.height:
        hits = scored.filter(
            _joint_mask(
                float(diagnostics.get("fill_ratio_max", FILL_RATIO_MAX)),
                float(diagnostics.get("nq_imb_min", NQ_IMB_MIN)),
            )
        )
        if hits.height:
            hits.write_parquet(out / "triple_hits.parquet")
    payload = dict(diagnostics)
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Triple tape: MNQ MBO + MNQ trades + NQ trades",
        "",
        "Locked hypothesis: MNQ fill_ratio<0.20 and NQ T_imbalance>0.20.",
        "NQ Fill_Ratio is unavailable (trades tape). Not spoofing, not a model.",
        "",
        "| name | MNQ MBO T/s | MNQ tape T/s | NQ T/s | MNQ imb | NQ imb | fill | "
        "ask cancel | C/A | NQ$/MNQ$ | peak_h | drop_h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in named.iter_rows(named=True):
        lines.append(
            f"| {row['name']} | {float(row['mnq_mbo_t_per_s']):.3f} | "
            f"{float(row['mnq_tape_t_per_s']):.3f} | {float(row['nq_t_per_s']):.3f} | "
            f"{float(row['mnq_mbo_t_imbalance']):.3f} | {float(row['nq_t_imbalance']):.3f} | "
            f"{float(row['mnq_fill_ratio']):.3f} | {float(row['mnq_ask_cancel_share']):.3f} | "
            f"{float(row['mnq_cancel_over_add']):.3f} | {float(row['nq_over_mnq_notional']):.3f} | "
            f"{row['peak_hypothesis']} | {row['drop_hypothesis']} |"
        )
    summ = diagnostics.get("summary", {})
    if isinstance(summ, Mapping) and summ:
        lines += [
            "",
            "Day scan (fill<0.20 and NQ imb>0.20) vs controls:",
            "| group | n | med pts | 10bps | 50bps | 20pt | 40pt |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key in (
            "pattern_windows",
            "pattern_episodes",
            "control",
            "random_control",
            "fill_only",
            "nq_buy_only",
        ):
            row = summ.get(key, {})
            if not isinstance(row, Mapping) or not row:
                continue
            n = row.get("n", 0)
            if n == 0:
                lines.append(f"| {key} | 0 | nan | nan | nan | nan | nan |")
                continue
            lines.append(
                f"| {key} | {n} | {float(row.get('median_drop_points', float('nan'))):.3f} | "
                f"{float(row.get('rate_10bps', float('nan'))):.4f} | "
                f"{float(row.get('rate_50bps', float('nan'))):.4f} | "
                f"{float(row.get('rate_20pt', float('nan'))):.4f} | "
                f"{float(row.get('rate_40pt', float('nan'))):.4f} |"
            )
    lines.append("")
    (out / "TRIPLE_TAPE.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "FILL_RATIO_MAX",
    "LAYER_ID",
    "NQ_IMB_MIN",
    "NQ_IMB_NEAR_ZERO",
    "compare_triple_windows",
    "scan_triple_pattern",
    "write_triple_report",
]
