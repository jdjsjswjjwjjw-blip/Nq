"""تشخيص كسر المدى: Fill_Ratio عند الكسر، وهندسة الوقف. ليست إشارة.

على كل أول طبعة تخترق مدى 30د مكتمل: ``F/(F+C)`` في 5ث و10ث قبل t
(جانب العرض عند كسر صاعد، جانب الطلب عند كسر هابط). النتيجة بعد t:
عاد للمدى قبل امتداد ATR = ``failed``، امتد ATR = ``held``.

الوقف يُعاد تشغيله على نفس دخول المزيج (أصوات التدفق ≥2، بلا صوت الملء)
بقواعد مُعلنة مسبقًا: مدى+ATR، نصف ATR، الطبعة+2/+4 نقاط، ووقت فقط.
ليست overlay وليست قفلًا. أيلول–كانون 2025 لا تُمس. بلا دفتر وبلا لصق.
NQ غير متاح في IDrive ``MES_MBO``.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import EVENT_TS
from nq.research.cross_nq_mnq import MNQ_MULT
from nq.research.failed_break_flow import (
    _ASK,
    _BID,
    _CANCEL,
    _FILL,
    _TRADE,
    FLOW_WINDOW_S,
    HOLD_NS,
    HOLDOUT_START_DATE,
    HYP_FILL_MAX,
    LAYER_ID,
    MIN_VOTES,
    PATH_TICKS,
    _completed_ranges,
    _cum_delta,
    _forward_ticks,
    _path_proxy,
    _ratio,
    _votes,
    iter_idrive_session_files,
    load_idrive_day,
)
from nq.research.failed_breakout import (
    ATR_WINDOW,
    LOOKBACK,
    REWARD_RATIO,
    SESSION_HOUR_END,
    SESSION_HOUR_START,
    _fmt_ts,
    _in_session,
    _json_num,
    _median,
    _tape_prices_to_fixed,
)
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape
from nq.research.opposite_phantom import SECOND_NS
from nq.simulation.fvg import NS_30M, build_ohlcv_bars

DIAG_LAYER: Final = "failed_break_diag"
Side = Literal["short", "long"]
FILL_WINDOWS_S: Final = (5, 10)
PRINT_BUFFERS: Final = (2.0, 4.0)
ATR_MULTS: Final = (1.0, 0.5)
_MAX_ERRORS: Final = 12
_EPS: Final = 1e-12
_INF: Final = 1.0e12

_BREAK_SCHEMA: Final[dict[str, pl.DataType]] = {
    "signal_ts": pl.Int64(),
    "clock": pl.Utf8(),
    "day_id": pl.Utf8(),
    "side": pl.Utf8(),
    "range_high": pl.Float64(),
    "range_low": pl.Float64(),
    "atr": pl.Float64(),
    "entry": pl.Float64(),
    "fill_ratio_5": pl.Float64(),
    "fill_ratio_10": pl.Float64(),
    "ask_hit_5": pl.Float64(),
    "bid_hit_5": pl.Float64(),
    "outcome": pl.Utf8(),
    "hyp_lt_020_5": pl.Boolean(),
    "votes_flow": pl.Int64(),
    "cvd_delta": pl.Float64(),
    "t_imbalance": pl.Float64(),
    "leak_pts": pl.Float64(),
}

_STOP_SCHEMA: Final[dict[str, pl.DataType]] = {
    "signal_ts": pl.Int64(),
    "clock": pl.Utf8(),
    "day_id": pl.Utf8(),
    "side": pl.Utf8(),
    "entry": pl.Float64(),
    "rule": pl.Utf8(),
    "sl": pl.Float64(),
    "tp": pl.Float64(),
    "exit": pl.Float64(),
    "exit_reason": pl.Utf8(),
    "pnl_pts": pl.Float64(),
    "pnl_usd": pl.Float64(),
    "mae_pts": pl.Float64(),
    "risk_pts": pl.Float64(),
}


def _empty() -> dict[str, Any]:
    return {
        "layer": DIAG_LAYER,
        "parent_layer": LAYER_ID,
        "not_fb_lock": True,
        "not_overlay": True,
        "not_mbo_book": True,
        "not_path_head_on_ticks": True,
        "nq_tape": "unavailable_idrive_mnq_only",
        "holdout_start": HOLDOUT_START_DATE,
        "hyp_fill_max": HYP_FILL_MAX,
        "fill_windows_s": list(FILL_WINDOWS_S),
        "print_buffers_pts": list(PRINT_BUFFERS),
        "atr_mults": list(ATR_MULTS),
        "ask_hit_share": "F_ask/(F_ask+C_ask); high = wall consumed not pulled",
        "n_days": 0,
        "n_skipped_holdout": 0,
        "n_skipped_error": 0,
        "n_breaks": 0,
        "n_failed": 0,
        "n_held": 0,
        "n_unresolved": 0,
        "n_fill_finite_5": 0,
        "median_fill_5_failed": float("nan"),
        "median_fill_5_held": float("nan"),
        "share_lt_020_failed": float("nan"),
        "share_lt_020_held": float("nan"),
        "n_base_trades": 0,
        "errors": [],
    }


def _action_cum(
    events: pl.DataFrame, action: str, side: str
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    chunk = (
        events.filter((pl.col("action") == action) & (pl.col("side") == side))
        .sort(EVENT_TS)
        .with_columns(pl.col("size").cast(pl.Float64).cum_sum().alias("cum"))
    )
    if chunk.height == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    return (
        np.asarray(chunk[EVENT_TS].to_numpy(), dtype=np.int64),
        np.asarray(chunk["cum"].to_numpy(), dtype=np.float64),
    )


def _window_amt(stamps: NDArray[np.int64], cum: NDArray[np.float64], t: int, win_ns: int) -> float:
    if stamps.size == 0 or cum.size == 0:
        return 0.0
    hi = int(np.searchsorted(stamps, t, side="right"))
    lo = int(np.searchsorted(stamps, t - win_ns, side="left"))
    return _cum_delta(cum, lo, hi)


def _hit_share(
    f_ts: NDArray[np.int64],
    f_cum: NDArray[np.float64],
    c_ts: NDArray[np.int64],
    c_cum: NDArray[np.float64],
    t: int,
    win_ns: int,
) -> float:
    filled = _window_amt(f_ts, f_cum, t, win_ns)
    canceled = _window_amt(c_ts, c_cum, t, win_ns)
    return _ratio(filled, filled + canceled)


def _break_outcome(
    *,
    side: str,
    rh: float,
    rl: float,
    atr: float,
    ts: NDArray[np.int64],
    px: NDArray[np.float64],
    start_i: int,
    hold_ns: int,
) -> str:
    deadline = int(ts[start_i]) + int(hold_ns)
    fail_lv = rh if side == "short" else rl
    hold_lv = rh + atr if side == "short" else rl - atr
    for i in range(start_i + 1, int(ts.size)):
        if int(ts[i]) > deadline:
            break
        price = float(px[i])
        if side == "short":
            if price <= fail_lv:
                return "failed"
            if price >= hold_lv:
                return "held"
        else:
            if price >= fail_lv:
                return "failed"
            if price <= hold_lv:
                return "held"
    return "unresolved"


def _mae(
    *,
    side: str,
    entry: float,
    ts: NDArray[np.int64],
    px: NDArray[np.float64],
    start_i: int,
    hold_ns: int,
) -> float:
    deadline = int(ts[start_i]) + int(hold_ns)
    mae = 0.0
    for i in range(start_i + 1, int(ts.size)):
        if int(ts[i]) > deadline:
            break
        price = float(px[i])
        mae = max(mae, price - entry) if side == "short" else max(mae, entry - price)
    return float(mae)


def _share_lt(values: list[float], cap: float) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return float("nan")
    return float(sum(1 for x in finite if x < cap)) / float(len(finite))


def _stop_sl(rule: str, *, side: str, rh: float, rl: float, atr: float, entry: float) -> float:
    if rule == "range_plus_atr":
        return rh + atr if side == "short" else rl - atr
    if rule == "range_plus_half_atr":
        half = 0.5 * atr
        return rh + half if side == "short" else rl - half
    if rule == "print_plus_2":
        return entry + 2.0 if side == "short" else entry - 2.0
    if rule == "print_plus_4":
        return entry + 4.0 if side == "short" else entry - 4.0
    if rule == "time_only":
        return _INF if side == "short" else -_INF
    raise ValueError(f"unknown stop rule {rule}")


def scan_tick_diagnostics(  # noqa: PLR0912, PLR0915
    trades: pl.DataFrame,
    mbo: pl.DataFrame | None = None,
    *,
    lookback: int = LOOKBACK,
    atr_window: int = ATR_WINDOW,
    path_ticks: int = PATH_TICKS,
    min_votes: int = MIN_VOTES,
    hyp_fill_max: float = HYP_FILL_MAX,
    hold_ns: int = HOLD_NS,
    reward_ratio: float = REWARD_RATIO,
    point_value: float = MNQ_MULT,
    session_start: int = SESSION_HOUR_START,
    session_end: int = SESSION_HOUR_END,
    skip_open: bool = True,
    day_id: str = "",
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """كل كسر + Fill_Ratio، ثم وقف بديل على دخول المزيج فقط. ليست إشارة."""

    diag = _empty()
    tape = prepare_trades_tape(trades).filter(pl.col("action") == _TRADE)
    tape, unit = _tape_prices_to_fixed(tape)
    diag["tape_price_unit"] = unit
    m30 = build_ohlcv_bars(tape, interval_ns=NS_30M)
    if tape.height == 0 or m30.height == 0:
        return (
            pl.DataFrame(schema=_BREAK_SCHEMA),
            pl.DataFrame(schema=_STOP_SCHEMA),
            diag,
        )
    priced = tape.with_columns((pl.col("price").cast(pl.Float64) * PRICE_SCALE).alias("px"))
    ts = np.asarray(priced[EVENT_TS].to_numpy(), dtype=np.int64)
    px = np.asarray(priced["px"].to_numpy(), dtype=np.float64)
    size = np.asarray(priced["size"].to_numpy(), dtype=np.float64)
    side_arr = priced["side"].to_list()
    signed = np.array(
        [float(sz) if s == _BID else -float(sz) for s, sz in zip(side_arr, size, strict=True)],
        dtype=np.float64,
    )
    cvd = np.cumsum(signed)
    buy_c = np.cumsum(np.where(signed > 0, signed, 0.0))
    sell_c = np.cumsum(np.where(signed < 0, -signed, 0.0))
    m30_starts, range_high, range_low, atr = _completed_ranges(
        m30, lookback=lookback, atr_window=atr_window
    )
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    f_ask_ts = c_ask_ts = f_bid_ts = c_bid_ts = empty_i
    f_ask = c_ask = f_bid = c_bid = empty_f
    if mbo is not None and mbo.height:
        events = prepare_mbo_events(mbo).filter(pl.col("action").is_in([_FILL, _CANCEL]))
        if events.height:
            f_ask_ts, f_ask = _action_cum(events, _FILL, _ASK)
            c_ask_ts, c_ask = _action_cum(events, _CANCEL, _ASK)
            f_bid_ts, f_bid = _action_cum(events, _FILL, _BID)
            c_bid_ts, c_bid = _action_cum(events, _CANCEL, _BID)
    win5 = 5 * SECOND_NS
    win10 = 10 * SECOND_NS
    flow_win = int(FLOW_WINDOW_S) * SECOND_NS
    breaks: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    census_bar: set[int] = set()
    traded_bar: set[int] = set()
    n = int(ts.size)
    for i in range(n):
        stamp = int(ts[i])
        if not _in_session(
            stamp, session_start=session_start, session_end=session_end, skip_open=skip_open
        ):
            continue
        bar = int(stamp // NS_30M * NS_30M)
        if bar in traded_bar:
            continue
        j = int(np.searchsorted(m30_starts, bar, side="left"))
        if j >= int(m30_starts.size) or int(m30_starts[j]) != bar:
            continue
        rh = float(range_high[j])
        rl = float(range_low[j])
        atr_j = float(atr[j])
        if (
            not math.isfinite(rh)
            or not math.isfinite(rl)
            or not math.isfinite(atr_j)
            or atr_j <= _EPS
        ):
            continue
        price = float(px[i])
        upside = price > rh
        downside = price < rl
        if upside == downside:
            continue
        side: Side = "short" if upside else "long"
        lo = int(np.searchsorted(ts, stamp - flow_win, side="left"))
        cvd_d = _cum_delta(cvd, lo, i + 1)
        b = _cum_delta(buy_c, lo, i + 1)
        s = _cum_delta(sell_c, lo, i + 1)
        imb = _ratio(b - s, b + s)
        ask5 = _hit_share(f_ask_ts, f_ask, c_ask_ts, c_ask, stamp, win5)
        bid5 = _hit_share(f_bid_ts, f_bid, c_bid_ts, c_bid, stamp, win5)
        ask10 = _hit_share(f_ask_ts, f_ask, c_ask_ts, c_ask, stamp, win10)
        bid10 = _hit_share(f_bid_ts, f_bid, c_bid_ts, c_bid, stamp, win10)
        fill5 = ask5 if side == "short" else bid5
        fill10 = ask10 if side == "short" else bid10
        p_path = _path_proxy(px, i, side=side, ticks=int(path_ticks))
        votes_flow = _votes(
            side=side,
            p_path=p_path,
            cvd_delta=cvd_d,
            imb=imb,
            fill_ratio=float("nan"),
            hyp_fill_max=float(hyp_fill_max),
        )
        leak = abs((rh if side == "short" else rl) - price)
        if bar not in census_bar:
            census_bar.add(bar)
            outcome = _break_outcome(
                side=side, rh=rh, rl=rl, atr=atr_j, ts=ts, px=px, start_i=i, hold_ns=int(hold_ns)
            )
            hyp5 = bool(math.isfinite(fill5) and fill5 < float(hyp_fill_max))
            breaks.append(
                {
                    "signal_ts": stamp,
                    "clock": _fmt_ts(stamp),
                    "day_id": day_id,
                    "side": side,
                    "range_high": rh,
                    "range_low": rl,
                    "atr": atr_j,
                    "entry": price,
                    "fill_ratio_5": fill5,
                    "fill_ratio_10": fill10,
                    "ask_hit_5": ask5,
                    "bid_hit_5": bid5,
                    "outcome": outcome,
                    "hyp_lt_020_5": hyp5,
                    "votes_flow": int(votes_flow),
                    "cvd_delta": cvd_d,
                    "t_imbalance": imb,
                    "leak_pts": leak,
                }
            )
        if votes_flow < int(min_votes):
            continue
        base_sl = rh + atr_j if side == "short" else rl - atr_j
        if side == "short" and price >= base_sl:
            continue
        if side == "long" and price <= base_sl:
            continue
        mae = _mae(side=side, entry=price, ts=ts, px=px, start_i=i, hold_ns=int(hold_ns))
        rules = (
            "range_plus_atr",
            "range_plus_half_atr",
            "print_plus_2",
            "print_plus_4",
            "time_only",
        )
        for rule in rules:
            sl = _stop_sl(rule, side=side, rh=rh, rl=rl, atr=atr_j, entry=price)
            risk = abs(base_sl - price) if rule == "time_only" else abs(sl - price)
            if risk <= _EPS:
                continue
            if side == "short":
                tp = price - risk * float(reward_ratio)
            else:
                tp = price + risk * float(reward_ratio)
            exit_px, reason = _forward_ticks(
                side=side, sl=sl, tp=tp, ts=ts, px=px, start_i=i, hold_ns=int(hold_ns)
            )
            pnl = (price - exit_px) if side == "short" else (exit_px - price)
            stops.append(
                {
                    "signal_ts": stamp,
                    "clock": _fmt_ts(stamp),
                    "day_id": day_id,
                    "side": side,
                    "entry": price,
                    "rule": rule,
                    "sl": sl,
                    "tp": tp,
                    "exit": exit_px,
                    "exit_reason": reason,
                    "pnl_pts": pnl,
                    "pnl_usd": pnl * float(point_value),
                    "mae_pts": mae,
                    "risk_pts": risk,
                }
            )
        traded_bar.add(bar)
    diag["n_breaks"] = len(breaks)
    table_b = (
        pl.DataFrame(breaks, schema=_BREAK_SCHEMA) if breaks else pl.DataFrame(schema=_BREAK_SCHEMA)
    )
    table_s = (
        pl.DataFrame(stops, schema=_STOP_SCHEMA) if stops else pl.DataFrame(schema=_STOP_SCHEMA)
    )
    return table_b, table_s, _pack_diag(table_b, table_s, diag)


def _pack_diag(breaks: pl.DataFrame, stops: pl.DataFrame, diag: dict[str, Any]) -> dict[str, Any]:
    diag["n_breaks"] = breaks.height
    if breaks.height:
        outcomes = breaks["outcome"].to_list()
        diag["n_failed"] = sum(1 for x in outcomes if x == "failed")
        diag["n_held"] = sum(1 for x in outcomes if x == "held")
        diag["n_unresolved"] = sum(1 for x in outcomes if x == "unresolved")
        fill = [float(x) for x in breaks["fill_ratio_5"].to_list()]
        out = breaks["outcome"].to_list()
        failed = [f for f, o in zip(fill, out, strict=True) if o == "failed"]
        held = [f for f, o in zip(fill, out, strict=True) if o == "held"]
        diag["n_fill_finite_5"] = sum(1 for x in fill if math.isfinite(x))
        diag["median_fill_5_failed"] = _median(failed)
        diag["median_fill_5_held"] = _median(held)
        diag["share_lt_020_failed"] = _share_lt(failed, HYP_FILL_MAX)
        diag["share_lt_020_held"] = _share_lt(held, HYP_FILL_MAX)
        diag["share_lt_020_all"] = _share_lt(fill, HYP_FILL_MAX)
    if stops.height:
        base = stops.filter(pl.col("rule") == "range_plus_atr")
        diag["n_base_trades"] = base.height
        by_rule: dict[str, dict[str, Any]] = {}
        for rule in stops["rule"].unique().to_list():
            chunk = stops.filter(pl.col("rule") == rule)
            usd = [float(x) for x in chunk["pnl_usd"].to_list()]
            pts = [float(x) for x in chunk["pnl_pts"].to_list()]
            reasons = chunk["exit_reason"].to_list()
            by_rule[str(rule)] = {
                "n": chunk.height,
                "gross_usd": float(sum(usd)),
                "median_pts": _median(pts),
                "n_sl": sum(1 for r in reasons if r == "sl"),
                "n_tp": sum(1 for r in reasons if r == "tp"),
                "n_time": sum(1 for r in reasons if r == "time"),
            }
        diag["stops"] = by_rule
        if base.height:
            diag["median_mae_pts"] = _median([float(x) for x in base["mae_pts"].to_list()])
    return diag


def scan_year_idrive_diag(
    mbo_root: Path | str,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
    point_value: float = MNQ_MULT,
    min_votes: int = MIN_VOTES,
    log: Callable[[str], None] | None = None,
    **tick_kwargs: Any,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """ملفات IDrive اليومية قبل holdout. ``T``+``F``/``C`` يومًا بيوم."""

    diag = _empty()
    files = iter_idrive_session_files(mbo_root)
    break_tables: list[pl.DataFrame] = []
    stop_tables: list[pl.DataFrame] = []
    n_days = 0
    n_hold = 0
    n_err = 0
    errors: list[str] = []
    for day_id, path in files:
        if day_id >= holdout_start:
            n_hold += 1
            continue
        if log is not None:
            log(f"day {day_id} {path.name}")
        try:
            trades, mbo = load_idrive_day(path, with_mbo=True)
            br, st, _day = scan_tick_diagnostics(
                trades,
                mbo,
                point_value=point_value,
                min_votes=min_votes,
                day_id=day_id,
                **tick_kwargs,
            )
        except (ValueError, OSError) as exc:
            n_err += 1
            if len(errors) < _MAX_ERRORS:
                errors.append(f"{day_id}: {exc}")
            continue
        n_days += 1
        if br.height:
            break_tables.append(br)
        if st.height:
            stop_tables.append(st)
    stacked_b = (
        pl.concat(break_tables, how="vertical")
        if break_tables
        else pl.DataFrame(schema=_BREAK_SCHEMA)
    )
    stacked_s = (
        pl.concat(stop_tables, how="vertical") if stop_tables else pl.DataFrame(schema=_STOP_SCHEMA)
    )
    packed = _pack_diag(stacked_b, stacked_s, diag)
    packed["n_days"] = n_days
    packed["n_files"] = len(files)
    packed["n_skipped_holdout"] = n_hold
    packed["n_skipped_error"] = n_err
    packed["errors"] = errors
    packed["holdout_start"] = holdout_start
    packed["point_value_usd"] = float(point_value)
    return stacked_b, stacked_s, packed


def write_failed_break_diag_report(
    breaks: pl.DataFrame,
    stops: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if breaks.height:
        breaks.write_parquet(out / "break_census.parquet")
    if stops.height:
        stops.write_parquet(out / "stop_replay.parquet")
    packed = {k: _json_num(v) if isinstance(v, float) else v for k, v in dict(diagnostics).items()}
    (out / "summary.json").write_text(
        json.dumps(packed, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Failed-break diagnostics (Fill_Ratio + stop geometry)",
        "",
        "Fill_Ratio is F/(F+C) in 5s/10s ending at the break print, against the break "
        "(ask on upside, bid on downside). Outcome after t: returned to the range before "
        "ATR extension = failed; reached ATR extension = held. Stop replay uses the same "
        "flow-vote entries (fill vote excluded) and is not a lock. "
        "The 0.94 path head is not used. NQ tape is not in IDrive MES_MBO. "
        f"Holdout {diagnostics.get('holdout_start')} is not scanned.",
        "",
        f"days={diagnostics.get('n_days')} holdout_skipped={diagnostics.get('n_skipped_holdout')} "
        f"breaks={diagnostics.get('n_breaks')} failed={diagnostics.get('n_failed')} "
        f"held={diagnostics.get('n_held')} unresolved={diagnostics.get('n_unresolved')} "
        f"fill_finite_5={diagnostics.get('n_fill_finite_5')} "
        f"median_fill_failed={diagnostics.get('median_fill_5_failed')} "
        f"median_fill_held={diagnostics.get('median_fill_5_held')} "
        f"share_lt_0.20_failed={diagnostics.get('share_lt_020_failed')} "
        f"share_lt_0.20_held={diagnostics.get('share_lt_020_held')}.",
        "",
        "## Stop replay (same entries, not a lock)",
        "",
        "| rule | n | gross_usd | median_pts | n_sl | n_tp | n_time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stops_map = diagnostics.get("stops") or {}
    if isinstance(stops_map, dict):
        for rule, row in sorted(stops_map.items()):
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {rule} | {row.get('n')} | {row.get('gross_usd')} | {row.get('median_pts')} | "
                f"{row.get('n_sl')} | {row.get('n_tp')} | {row.get('n_time')} |"
            )
    if not isinstance(stops_map, dict) or not stops_map:
        lines.append("| — | — | — | — | — | — | — |")
    (out / "FAILED_BREAK_DIAG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


__all__ = [
    "DIAG_LAYER",
    "scan_tick_diagnostics",
    "scan_year_idrive_diag",
    "write_failed_break_diag_report",
]
