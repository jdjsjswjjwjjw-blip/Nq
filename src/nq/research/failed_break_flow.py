"""فشل كسر مبكر أثناء الشمعة: دخول بسعر الطبعة الحالية، بلا انتظار الإغلاق.

عندما تخترق طبعة ``T`` مدى الشموع المكتملة، تُقاس قوة استمرار الكسر من تدفق
معروف عند تلك اللحظة فقط: وكيل مسار (آخر K تيكات)، ΔCVD، اختلال T،
وFill_Ratio من ``F``/``C`` إن وُجد MBO. لا يُستدعى رأس ``y_path_further_beyond``
على التيك — ذلك الرأس على إعدادات المزاد. على براميل السنة 30ث يُستخدم
``p_y_path_further_beyond`` OOF إن وُجد، وإلا مزيج السحب/الدلتا.

الدخول = سعر الطبعة (أو إغلاق 30ث)، ليس ``range_high``. الوقف = المستوى + ATR
ماضٍ. ليست overlay وليست إشارة مقفلة. أيلول–كانون 2025 لا تُمس كـ holdout.
مسار IDrive ``MES_MBO_YYYY_MM`` يُمسح يومًا بيوم من ``T`` فقط (اختياريًا ``F``/``C``
لنسبة الملء)، بلا دفتر وبلا لصق.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nq.contracts.mbo import PRICE_SCALE, MboAction, MboSide
from nq.contracts.temporal import AVAILABILITY_TS, EVENT_TS
from nq.research.cross_nq_mnq import MNQ_MULT
from nq.research.failed_breakout import (
    ATR_WINDOW,
    HOLD_BARS,
    LOOKBACK,
    REWARD_RATIO,
    SESSION_HOUR_END,
    SESSION_HOUR_START,
    _fmt_ts,
    _in_session,
    _json_num,
    _median,
    _ohlc_exit,
    _tape_prices_to_fixed,
)
from nq.research.mbo_sequence_mlp import HOLDOUT_START_DATE
from nq.research.mbo_trade_overlap import prepare_mbo_events, prepare_trades_tape
from nq.research.opposite_phantom import SECOND_NS
from nq.simulation.common import BUCKET_START
from nq.simulation.fvg import NS_30M, build_ohlcv_bars

LAYER_ID = "failed_break_flow"
FLOW_WINDOW_S: Final = 5
PATH_TICKS: Final = 8
MIN_VOTES: Final = 2
HYP_FILL_MAX: Final = 0.20
P_PATH_FADE: Final = 0.5
HOLD_NS: Final = HOLD_BARS * NS_30M
BARS_30S: Final = 60
Side = Literal["short", "long"]
_TRADE = MboAction.TRADE.value
_FILL = MboAction.FILL.value
_CANCEL = MboAction.CANCEL.value
_BID = MboSide.BID.value
_ASK = MboSide.ASK.value
_EPS: Final = 1e-12
_FIXED_POINT_PRICE: Final = 1_000_000.0
_IDRIVE_DAY: Final = re.compile(r"glbx-mdp3-(\d{8})")
_MAX_IDRIVE_ERRORS: Final = 12

_TRADE_SCHEMA: Final[dict[str, pl.DataType]] = {
    "signal_ts": pl.Int64(),
    "clock": pl.Utf8(),
    "side": pl.Utf8(),
    "range_high": pl.Float64(),
    "range_low": pl.Float64(),
    "entry": pl.Float64(),
    "sl": pl.Float64(),
    "tp": pl.Float64(),
    "exit": pl.Float64(),
    "exit_reason": pl.Utf8(),
    "p_path": pl.Float64(),
    "cvd_delta": pl.Float64(),
    "t_imbalance": pl.Float64(),
    "fill_ratio": pl.Float64(),
    "votes": pl.Int64(),
    "risk_pts": pl.Float64(),
    "pnl_pts": pl.Float64(),
    "pnl_usd": pl.Float64(),
    "leak_pts": pl.Float64(),
    "source": pl.Utf8(),
    "day_id": pl.Utf8(),
}


def _to_points(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    med = float(np.median(np.abs(finite)))
    if med >= _FIXED_POINT_PRICE:
        return np.asarray(arr * PRICE_SCALE, dtype=np.float64)
    return arr


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _empty_diag(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "layer": LAYER_ID,
        "not_fb_lock": True,
        "not_overlay": True,
        "not_mbo_book": True,
        "not_path_head_on_ticks": True,
        "fill_rule": "print_at_signal",
        "sl_rule": "range_plus_atr",
        "path_rule": "tick_or_sl_first",
        "holdout_start": HOLDOUT_START_DATE,
        "n_breaks": 0,
        "n_early_fail": 0,
        "n_skipped_holdout": 0,
        "n_trades": 0,
        "n_win": 0,
        "n_loss": 0,
        "n_sl": 0,
        "n_tp": 0,
        "n_time_exit": 0,
        "n_with_p_path": 0,
        "gross_pnl_usd": 0.0,
        "win_rate": float("nan"),
        "median_pnl_pts": float("nan"),
        "median_leak_pts": float("nan"),
    }
    base.update(overrides)
    return base


def _pack_trades(table: pl.DataFrame, diag: dict[str, Any]) -> dict[str, Any]:
    diag["n_trades"] = table.height
    if table.height == 0:
        return diag
    pnl = [float(x) for x in table["pnl_pts"].to_list()]
    usd = [float(x) for x in table["pnl_usd"].to_list()]
    leak = [float(x) for x in table["leak_pts"].to_list()]
    reasons = table["exit_reason"].to_list()
    pcol = [float(x) for x in table["p_path"].to_list()]
    diag["n_win"] = sum(1 for x in pnl if x > 0)
    diag["n_loss"] = sum(1 for x in pnl if x < 0)
    diag["n_sl"] = sum(1 for r in reasons if r == "sl")
    diag["n_tp"] = sum(1 for r in reasons if r == "tp")
    diag["n_time_exit"] = sum(1 for r in reasons if r == "time")
    diag["n_with_p_path"] = sum(1 for x in pcol if math.isfinite(x))
    diag["gross_pnl_usd"] = float(sum(usd))
    diag["win_rate"] = float(diag["n_win"]) / float(table.height)
    diag["median_pnl_pts"] = _median(pnl)
    diag["median_leak_pts"] = _median(leak)
    return diag


def _cum_delta(cum: NDArray[np.float64], lo: int, hi: int) -> float:
    if hi <= lo:
        return 0.0
    end = float(cum[hi - 1])
    start = float(cum[lo - 1]) if lo > 0 else 0.0
    return end - start


def _path_proxy(px: NDArray[np.float64], i: int, *, side: Side, ticks: int) -> float:
    if i < 1:
        return float("nan")
    start = max(1, i - ticks + 1)
    cont = 0
    n = 0
    for j in range(start, i + 1):
        n += 1
        up = float(px[j]) > float(px[j - 1])
        if side == "short" and up:
            cont += 1
        if side == "long" and not up:
            cont += 1
    return float(cont) / float(n) if n else float("nan")


def _completed_ranges(
    m30: pl.DataFrame,
    *,
    lookback: int,
    atr_window: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    work = m30.sort(BUCKET_START)
    starts = np.asarray(work[BUCKET_START].to_numpy(), dtype=np.int64)
    high = np.asarray(work["h"].to_numpy(), dtype=np.float64)
    low = np.asarray(work["l"].to_numpy(), dtype=np.float64)
    rng = np.asarray(work["range"].to_numpy(), dtype=np.float64)
    n = int(starts.size)
    range_high = np.full(n, np.nan, dtype=np.float64)
    range_low = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)
    lb = max(1, int(lookback))
    aw = max(1, int(atr_window))
    for j in range(n):
        lo = j - lb
        if lo < 0:
            continue
        range_high[j] = float(np.max(high[lo:j]))
        range_low[j] = float(np.min(low[lo:j]))
        a0 = j - aw
        if a0 >= 0:
            atr[j] = float(np.mean(rng[a0:j]))
    return starts, range_high, range_low, atr


def _tick_exit(side: Side, sl: float, tp: float, px: float) -> tuple[float | None, str | None]:
    if side == "short":
        if px >= sl:
            return float(px), "sl"
        if px <= tp:
            return float(px), "tp"
        return None, None
    if px <= sl:
        return float(px), "sl"
    if px >= tp:
        return float(px), "tp"
    return None, None


def _forward_ticks(
    *,
    side: Side,
    sl: float,
    tp: float,
    ts: NDArray[np.int64],
    px: NDArray[np.float64],
    start_i: int,
    hold_ns: int,
) -> tuple[float, str]:
    deadline = int(ts[start_i]) + int(hold_ns)
    last_px = float(px[start_i])
    for i in range(start_i + 1, int(ts.size)):
        if int(ts[i]) > deadline:
            break
        last_px = float(px[i])
        hit, reason = _tick_exit(side, sl, tp, last_px)
        if hit is not None and reason is not None:
            return hit, reason
    return last_px, "time"


def _votes(
    *,
    side: Side,
    p_path: float,
    cvd_delta: float,
    imb: float,
    fill_ratio: float,
    hyp_fill_max: float,
) -> int:
    votes = 0
    if math.isfinite(p_path) and p_path < P_PATH_FADE:
        votes += 1
    if side == "short" and cvd_delta < 0:
        votes += 1
    if side == "long" and cvd_delta > 0:
        votes += 1
    if side == "short" and math.isfinite(imb) and imb < 0:
        votes += 1
    if side == "long" and math.isfinite(imb) and imb > 0:
        votes += 1
    if math.isfinite(fill_ratio) and fill_ratio < hyp_fill_max:
        votes += 1
    return votes


def scan_tick_early_fail(  # noqa: PLR0912, PLR0915
    trades: pl.DataFrame,
    mbo: pl.DataFrame | None = None,
    *,
    lookback: int = LOOKBACK,
    atr_window: int = ATR_WINDOW,
    flow_s: int = FLOW_WINDOW_S,
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
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """أول طبعة تخترق المدى → مزيج تدفق عند t → دخول بسعر الطبعة. ليست إشارة."""

    diag = _empty_diag(
        lookback=int(lookback),
        atr_window=int(atr_window),
        flow_s=int(flow_s),
        path_ticks=int(path_ticks),
        min_votes=int(min_votes),
        hyp_fill_max=float(hyp_fill_max),
        point_value_usd=float(point_value),
        source="tick",
    )
    tape = prepare_trades_tape(trades)
    tape = tape.filter(pl.col("action") == _TRADE)
    tape, unit = _tape_prices_to_fixed(tape)
    diag["tape_price_unit"] = unit
    diag["n_tape"] = int(tape.height)
    m30 = build_ohlcv_bars(tape, interval_ns=NS_30M)
    diag["n_30m_bars"] = int(m30.height)
    if tape.height == 0 or m30.height == 0:
        return pl.DataFrame(schema=_TRADE_SCHEMA), diag
    priced = tape.with_columns((pl.col("price").cast(pl.Float64) * PRICE_SCALE).alias("px"))
    ts = np.asarray(priced[EVENT_TS].to_numpy(), dtype=np.int64)
    px = np.asarray(priced["px"].to_numpy(), dtype=np.float64)
    size = np.asarray(priced["size"].to_numpy(), dtype=np.float64)
    side_arr = priced["side"].to_list()
    signed = np.array(
        [float(sz) if s == _BID else -float(sz) for s, sz in zip(side_arr, size, strict=True)],
        dtype=np.float64,
    )
    buy = np.where(signed > 0, signed, 0.0)
    sell = np.where(signed < 0, -signed, 0.0)
    cvd = np.cumsum(signed)
    buy_c = np.cumsum(buy)
    sell_c = np.cumsum(sell)
    m30_starts, range_high, range_low, atr = _completed_ranges(
        m30, lookback=lookback, atr_window=atr_window
    )
    f_ts = np.zeros(0, dtype=np.int64)
    f_ask = np.zeros(0, dtype=np.float64)
    c_ask = np.zeros(0, dtype=np.float64)
    if mbo is not None and mbo.height:
        events = prepare_mbo_events(mbo).filter(pl.col("action").is_in([_FILL, _CANCEL]))
        if events.height:
            f_ask_s = (
                events.filter((pl.col("action") == _FILL) & (pl.col("side") == _ASK))
                .sort(EVENT_TS)
                .with_columns(pl.col("size").cast(pl.Float64).cum_sum().alias("cum"))
            )
            c_ask_s = (
                events.filter((pl.col("action") == _CANCEL) & (pl.col("side") == _ASK))
                .sort(EVENT_TS)
                .with_columns(pl.col("size").cast(pl.Float64).cum_sum().alias("cum"))
            )
            empty_i = np.zeros(0, dtype=np.int64)
            empty_f = np.zeros(0, dtype=np.float64)
            f_ts = (
                np.asarray(f_ask_s[EVENT_TS].to_numpy(), dtype=np.int64)
                if f_ask_s.height
                else empty_i
            )
            f_ask = (
                np.asarray(f_ask_s["cum"].to_numpy(), dtype=np.float64)
                if f_ask_s.height
                else empty_f
            )
            c_ts = (
                np.asarray(c_ask_s[EVENT_TS].to_numpy(), dtype=np.int64)
                if c_ask_s.height
                else empty_i
            )
            c_ask = (
                np.asarray(c_ask_s["cum"].to_numpy(), dtype=np.float64)
                if c_ask_s.height
                else empty_f
            )
        else:
            c_ts = np.zeros(0, dtype=np.int64)
    else:
        c_ts = np.zeros(0, dtype=np.int64)
    win = int(flow_s) * SECOND_NS
    n = int(ts.size)
    rows: list[dict[str, Any]] = []
    used_bar: set[int] = set()
    n_breaks = 0
    n_early = 0
    for i in range(n):
        stamp = int(ts[i])
        if not _in_session(
            stamp, session_start=session_start, session_end=session_end, skip_open=skip_open
        ):
            continue
        bar = int(stamp // NS_30M * NS_30M)
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
        if bar in used_bar:
            continue
        n_breaks += 1
        side: Side = "short" if upside else "long"
        lo = int(np.searchsorted(ts, stamp - win, side="left"))
        cvd_d = _cum_delta(cvd, lo, i + 1)
        b = _cum_delta(buy_c, lo, i + 1)
        s = _cum_delta(sell_c, lo, i + 1)
        imb = _ratio(b - s, b + s)
        fill_ratio = float("nan")
        if f_ts.size and c_ts.size:
            f_hi = int(np.searchsorted(f_ts, stamp, side="right"))
            f_lo = int(np.searchsorted(f_ts, stamp - win, side="left"))
            c_hi = int(np.searchsorted(c_ts, stamp, side="right"))
            c_lo = int(np.searchsorted(c_ts, stamp - win, side="left"))
            f_w = _cum_delta(f_ask, f_lo, f_hi) if f_ask.size else 0.0
            c_w = _cum_delta(c_ask, c_lo, c_hi) if c_ask.size else 0.0
            fill_ratio = _ratio(f_w, f_w + c_w)
        p_path = _path_proxy(px, i, side=side, ticks=int(path_ticks))
        votes = _votes(
            side=side,
            p_path=p_path,
            cvd_delta=cvd_d,
            imb=imb,
            fill_ratio=fill_ratio,
            hyp_fill_max=float(hyp_fill_max),
        )
        if votes < int(min_votes):
            continue
        n_early += 1
        entry = price
        if side == "short":
            sl = rh + atr_j
            leak = abs(rh - entry)
        else:
            sl = rl - atr_j
            leak = abs(rl - entry)
        if side == "short" and entry >= sl:
            continue
        if side == "long" and entry <= sl:
            continue
        risk = abs(sl - entry)
        if risk <= _EPS:
            continue
        if side == "short":
            tp = entry - risk * float(reward_ratio)
        else:
            tp = entry + risk * float(reward_ratio)
        exit_px, reason = _forward_ticks(
            side=side, sl=sl, tp=tp, ts=ts, px=px, start_i=i, hold_ns=int(hold_ns)
        )
        pnl = (entry - exit_px) if side == "short" else (exit_px - entry)
        used_bar.add(bar)
        rows.append(
            {
                "signal_ts": stamp,
                "clock": _fmt_ts(stamp),
                "side": side,
                "range_high": rh,
                "range_low": rl,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit": exit_px,
                "exit_reason": reason,
                "p_path": p_path,
                "cvd_delta": cvd_d,
                "t_imbalance": imb,
                "fill_ratio": fill_ratio,
                "votes": int(votes),
                "risk_pts": risk,
                "pnl_pts": pnl,
                "pnl_usd": pnl * float(point_value),
                "leak_pts": leak,
                "source": "tick",
                "day_id": day_id,
            }
        )
    diag["n_breaks"] = n_breaks
    diag["n_early_fail"] = n_early
    table = pl.DataFrame(rows, schema=_TRADE_SCHEMA) if rows else pl.DataFrame(schema=_TRADE_SCHEMA)
    return table, _pack_trades(table, diag)


def _blended_flow_votes(
    *,
    side: Side,
    p_path: float,
    delta: float,
    withdraw: float,
    fail: float,
) -> int:
    votes = 0
    if math.isfinite(p_path) and p_path < P_PATH_FADE:
        votes += 1
    if side == "short" and delta < 0:
        votes += 1
    if side == "long" and delta > 0:
        votes += 1
    if withdraw > 0:
        votes += 1
    if fail > 0:
        votes += 1
    return votes


def scan_blended_early_fail(  # noqa: PLR0912, PLR0915
    blended: pl.DataFrame,
    oof: pl.DataFrame | None = None,
    *,
    lookback: int = LOOKBACK,
    atr_window: int = ATR_WINDOW,
    min_votes: int = MIN_VOTES,
    hold_bars_30s: int = HOLD_BARS * BARS_30S,
    reward_ratio: float = REWARD_RATIO,
    point_value: float = MNQ_MULT,
    session_start: int = SESSION_HOUR_START,
    session_end: int = SESSION_HOUR_END,
    skip_open: bool = True,
    day_id: str = "",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """براميل 30ث مكتملة: أول اختراق للمدى ثم ``p_path`` OOF أو مزيج تدفق. دخول = الإغلاق."""

    diag = _empty_diag(
        lookback=int(lookback),
        min_votes=int(min_votes),
        point_value_usd=float(point_value),
        source="blended_30s",
        fill_rule="bar_close_at_signal",
    )
    need = [AVAILABILITY_TS, "high", "low", "close"]
    if blended.height == 0 or any(c not in blended.columns for c in need):
        return pl.DataFrame(schema=_TRADE_SCHEMA), diag
    work = blended.sort(AVAILABILITY_TS)
    if oof is not None and oof.height and "p_y_path_further_beyond" in oof.columns:
        pcol = oof.select(AVAILABILITY_TS, "p_y_path_further_beyond")
        work = work.join(pcol, on=AVAILABILITY_TS, how="left")
    elif "p_y_path_further_beyond" not in work.columns:
        work = work.with_columns(pl.lit(None, dtype=pl.Float64).alias("p_y_path_further_beyond"))
    ts = np.asarray(work[AVAILABILITY_TS].to_numpy(), dtype=np.int64)
    high = _to_points(np.asarray(work["high"].to_numpy(), dtype=np.float64))
    low = _to_points(np.asarray(work["low"].to_numpy(), dtype=np.float64))
    close = _to_points(np.asarray(work["close"].to_numpy(), dtype=np.float64))
    delta = (
        np.asarray(work["vp_of_delta"].to_numpy(), dtype=np.float64)
        if "vp_of_delta" in work.columns
        else np.zeros(close.size, dtype=np.float64)
    )
    withdraw = (
        np.asarray(work["lf_liquidity_withdrawal"].to_numpy(), dtype=np.float64)
        if "lf_liquidity_withdrawal" in work.columns
        else np.zeros(close.size, dtype=np.float64)
    )
    fail = (
        np.asarray(work["path_change_fail"].to_numpy(), dtype=np.float64)
        if "path_change_fail" in work.columns
        else np.zeros(close.size, dtype=np.float64)
    )
    p_arr = np.asarray(work["p_y_path_further_beyond"].to_numpy(), dtype=np.float64)
    buckets = ts // NS_30M
    uniq = np.unique(buckets)
    n_b = int(uniq.size)
    b_high = np.zeros(n_b, dtype=np.float64)
    b_low = np.zeros(n_b, dtype=np.float64)
    for k, b in enumerate(uniq):
        mask = buckets == b
        b_high[k] = float(np.max(high[mask]))
        b_low[k] = float(np.min(low[mask]))
    b_range = b_high - b_low
    lb = max(1, int(lookback))
    aw = max(1, int(atr_window))
    rows: list[dict[str, Any]] = []
    n_breaks = 0
    n_early = 0
    used: set[int] = set()
    n = int(ts.size)
    for i in range(n):
        stamp = int(ts[i])
        if not _in_session(
            stamp, session_start=session_start, session_end=session_end, skip_open=skip_open
        ):
            continue
        b = int(buckets[i])
        k = int(np.searchsorted(uniq, b))
        if k < lb:
            continue
        rh = float(np.max(b_high[k - lb : k]))
        rl = float(np.min(b_low[k - lb : k]))
        a0 = k - aw
        if a0 < 0:
            continue
        atr_j = float(np.mean(b_range[a0:k]))
        if atr_j <= _EPS:
            continue
        upside = float(high[i]) > rh
        downside = float(low[i]) < rl
        if upside == downside:
            continue
        if b in used:
            continue
        n_breaks += 1
        side: Side = "short" if upside else "long"
        p_path = float(p_arr[i])
        votes = _blended_flow_votes(
            side=side,
            p_path=p_path,
            delta=float(delta[i]),
            withdraw=float(withdraw[i]),
            fail=float(fail[i]),
        )
        if votes < int(min_votes):
            continue
        n_early += 1
        entry = float(close[i])
        sl = rh + atr_j if side == "short" else rl - atr_j
        if side == "short" and entry >= sl:
            continue
        if side == "long" and entry <= sl:
            continue
        risk = abs(sl - entry)
        if risk <= _EPS:
            continue
        if side == "short":
            tp = entry - risk * float(reward_ratio)
        else:
            tp = entry + risk * float(reward_ratio)
        leak = abs((rh if side == "short" else rl) - entry)
        last_i = min(n - 1, i + int(hold_bars_30s))
        exit_px = entry
        reason = "time"
        for j in range(i + 1, last_i + 1):
            hit, why = _ohlc_exit(
                side=side,
                sl=sl,
                tp=tp,
                o=float(close[j - 1]),
                h=float(high[j]),
                low_px=float(low[j]),
                is_fill_bar=False,
            )
            if hit is not None and why is not None:
                exit_px, reason = hit, why
                break
            exit_px = float(close[j])
        pnl = (entry - exit_px) if side == "short" else (exit_px - entry)
        used.add(b)
        rows.append(
            {
                "signal_ts": stamp,
                "clock": _fmt_ts(stamp),
                "side": side,
                "range_high": rh,
                "range_low": rl,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit": exit_px,
                "exit_reason": reason,
                "p_path": p_path,
                "cvd_delta": float(delta[i]),
                "t_imbalance": float("nan"),
                "fill_ratio": float("nan"),
                "votes": int(votes),
                "risk_pts": risk,
                "pnl_pts": pnl,
                "pnl_usd": pnl * float(point_value),
                "leak_pts": leak,
                "source": "blended_30s",
                "day_id": day_id,
            }
        )
    diag["n_breaks"] = n_breaks
    diag["n_early_fail"] = n_early
    diag["n_30s_bars"] = n
    table = pl.DataFrame(rows, schema=_TRADE_SCHEMA) if rows else pl.DataFrame(schema=_TRADE_SCHEMA)
    return table, _pack_trades(table, diag)


def scan_year_blended(
    year_dir: Path | str,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
    point_value: float = MNQ_MULT,
    min_votes: int = MIN_VOTES,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """أيام ``auction_behavior_year`` قبل holdout فقط. لا لصق MBO عبر الأيام."""

    root = Path(year_dir)
    diag = _empty_diag(
        source="year_blended",
        point_value_usd=float(point_value),
        min_votes=int(min_votes),
    )
    if not root.is_dir():
        return pl.DataFrame(schema=_TRADE_SCHEMA), diag
    days = sorted(p for p in root.iterdir() if p.is_dir() and p.name[:4].isdigit())
    tables: list[pl.DataFrame] = []
    n_days = 0
    n_skip = 0
    n_hold = 0
    n_breaks = 0
    n_early = 0
    for day in days:
        if day.name >= holdout_start:
            n_hold += 1
            continue
        blended_path = day / "blended.parquet"
        if not blended_path.is_file():
            n_skip += 1
            continue
        blended = pl.read_parquet(blended_path)
        oof_path = day / "oof_predictions.parquet"
        oof = pl.read_parquet(oof_path) if oof_path.is_file() else None
        table, day_diag = scan_blended_early_fail(
            blended, oof, point_value=point_value, min_votes=min_votes, day_id=day.name
        )
        n_days += 1
        n_breaks += int(day_diag.get("n_breaks") or 0)
        n_early += int(day_diag.get("n_early_fail") or 0)
        if table.height:
            tables.append(table)
    stacked = pl.concat(tables, how="vertical") if tables else pl.DataFrame(schema=_TRADE_SCHEMA)
    diag["n_days"] = n_days
    diag["n_skipped_missing"] = n_skip
    diag["n_skipped_holdout"] = n_hold
    diag["holdout_start"] = holdout_start
    diag["n_breaks"] = n_breaks
    diag["n_early_fail"] = n_early
    return stacked, _pack_trades(stacked, diag)


def _idrive_rank(path: Path) -> int:
    name = path.name
    if name.endswith(".continuous.clean.parquet") and ".mbo." not in name:
        return 0
    if ".mbo.continuous.clean.parquet" in name:
        return 1
    return 2


def iter_idrive_session_files(mbo_root: Path | str) -> list[tuple[str, Path]]:
    """``MES_MBO_YYYY_MM/glbx-mdp3-YYYYMMDD*.parquet`` يومًا واحدًا لكل تاريخ."""

    root = Path(mbo_root)
    by_day: dict[str, Path] = {}
    if not root.is_dir():
        return []
    for path in sorted(root.glob("MES_MBO_*/glbx-mdp3-*.parquet")):
        match = _IDRIVE_DAY.search(path.name)
        if match is None:
            continue
        ymd = match.group(1)
        day_id = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        prev = by_day.get(day_id)
        if prev is None or _idrive_rank(path) < _idrive_rank(prev):
            by_day[day_id] = path
    return [(day, by_day[day]) for day in sorted(by_day)]


def _load_idrive_day(path: Path, *, with_mbo: bool) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    lf = pl.scan_parquet(path)
    names = lf.collect_schema().names()
    if "action" not in names:
        return lf.collect(), None
    action = pl.col("action").cast(pl.Utf8).str.to_uppercase()
    keep = [_TRADE, _FILL, _CANCEL] if with_mbo else [_TRADE]
    frame = lf.filter(action.is_in(keep)).collect()
    trades = frame.filter(pl.col("action").cast(pl.Utf8).str.to_uppercase() == _TRADE)
    if not with_mbo:
        return trades, None
    mbo = frame.filter(pl.col("action").cast(pl.Utf8).str.to_uppercase().is_in([_FILL, _CANCEL]))
    return trades, mbo


def scan_year_idrive_tick(
    mbo_root: Path | str,
    *,
    holdout_start: str = HOLDOUT_START_DATE,
    point_value: float = MNQ_MULT,
    min_votes: int = MIN_VOTES,
    with_mbo: bool = False,
    log: Callable[[str], None] | None = None,
    **tick_kwargs: Any,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """ملفات IDrive اليومية قبل holdout. ``T`` يومًا بيوم، بلا لصق MBO."""

    diag = _empty_diag(
        source="idrive_tick",
        point_value_usd=float(point_value),
        min_votes=int(min_votes),
        with_mbo=bool(with_mbo),
        fill_rule="print_at_signal",
    )
    files = iter_idrive_session_files(mbo_root)
    tables: list[pl.DataFrame] = []
    n_days = 0
    n_hold = 0
    n_err = 0
    n_breaks = 0
    n_early = 0
    errors: list[str] = []
    units: list[str] = []
    for day_id, path in files:
        if day_id >= holdout_start:
            n_hold += 1
            continue
        if log is not None:
            log(f"day {day_id} {path.name}")
        try:
            trades, mbo = _load_idrive_day(path, with_mbo=with_mbo)
            table, day_diag = scan_tick_early_fail(
                trades,
                mbo,
                point_value=point_value,
                min_votes=min_votes,
                day_id=day_id,
                **tick_kwargs,
            )
        except (ValueError, OSError) as exc:
            n_err += 1
            if len(errors) < _MAX_IDRIVE_ERRORS:
                errors.append(f"{day_id}: {exc}")
            continue
        n_days += 1
        n_breaks += int(day_diag.get("n_breaks") or 0)
        n_early += int(day_diag.get("n_early_fail") or 0)
        unit = day_diag.get("tape_price_unit")
        if isinstance(unit, str):
            units.append(unit)
        if table.height:
            tables.append(table)
    stacked = pl.concat(tables, how="vertical") if tables else pl.DataFrame(schema=_TRADE_SCHEMA)
    diag["n_days"] = n_days
    diag["n_files"] = len(files)
    diag["n_skipped_holdout"] = n_hold
    diag["n_skipped_error"] = n_err
    diag["errors"] = errors
    diag["holdout_start"] = holdout_start
    diag["n_breaks"] = n_breaks
    diag["n_early_fail"] = n_early
    diag["tape_price_units"] = sorted(set(units))
    return stacked, _pack_trades(stacked, diag)


def write_failed_break_flow_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "failed_break_flow_trades.parquet")
    packed = {k: _json_num(v) if isinstance(v, float) else v for k, v in dict(diagnostics).items()}
    (out / "summary.json").write_text(
        json.dumps(packed, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Early failed break (in-bar flow / p_path OOF)",
        "",
        "Fill is the print (or 30s close) at the signal, not `range_high`. "
        "Tick path uses last-K continuation + CVD + T imbalance + optional Fill_Ratio. "
        "Year blended uses OOF `p_y_path_further_beyond` when present. "
        "IDrive MES_MBO days are scanned one parquet at a time (T prints; no book). "
        "The 0.94 path head is not a live tick overlay. Not a lock. "
        f"Holdout {diagnostics.get('holdout_start')} is not scanned.",
        f"source={diagnostics.get('source')} trades={diagnostics.get('n_trades')} "
        f"breaks={diagnostics.get('n_breaks')} early={diagnostics.get('n_early_fail')} "
        f"with_p_path={diagnostics.get('n_with_p_path')} "
        f"holdout_skipped={diagnostics.get('n_skipped_holdout')} "
        f"gross_usd={diagnostics.get('gross_pnl_usd')} "
        f"median_pnl_pts={diagnostics.get('median_pnl_pts')} "
        f"median_leak_pts={diagnostics.get('median_leak_pts')}.",
        "",
        "| day | clock | side | entry | p_path | votes | exit | reason | pnl_pts | leak_pts |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    shown = table.head(50) if table.height else table
    for row in shown.iter_rows(named=True):
        p = float(row["p_path"])
        p_s = "nan" if not math.isfinite(p) else f"{p:.3f}"
        lines.append(
            f"| {row['day_id']} | {row['clock']} | {row['side']} | {row['entry']:.2f} | "
            f"{p_s} | {row['votes']} | {row['exit']:.2f} | {row['exit_reason']} | "
            f"{row['pnl_pts']:.2f} | {row['leak_pts']:.2f} |"
        )
    if table.height == 0:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    (out / "FAILED_BREAK_FLOW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


__all__ = [
    "HOLDOUT_START_DATE",
    "LAYER_ID",
    "iter_idrive_session_files",
    "scan_blended_early_fail",
    "scan_tick_early_fail",
    "scan_year_blended",
    "scan_year_idrive_tick",
    "write_failed_break_flow_report",
]
