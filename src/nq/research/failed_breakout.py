"""كسر فاشل سببي: الملء عند افتتاح الشمعة التالية، بلا تسريب ``range_high``.

الإشارة تُعرف عند إغلاق شمعة 30د (كسر ثم إغلاق داخل المدى). السعر المتاح بعد
ذلك هو افتتاح الشمعة التالية، لا ذروة/قاع الشمعة نفسها. الوقف بعد ذيل شمعة
الإشارة + ATR ماضٍ. داخل الشمعة: إن لامس السعر الوقف والهدف معًا يُحتسب الوقف
أولًا. إن لم يُلمس أي منهما حتى نهاية الأفق يُغلق على الإغلاق.

ليست overlay وليست إشارة مقفلة. ليست أمرًا معلّقًا على ``range_high``.
الهدف السابق ``prior_swing`` من شموع قبل الإشارة، ليس MBO.
يوم شريط واحد: ``prepare_trades_tape`` يرفض لصق الأيام.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nq.contracts.temporal import AVAILABILITY_TS
from nq.research.clock_flow import TZ_NAME
from nq.research.cross_nq_mnq import MNQ_MULT
from nq.research.mbo_trade_overlap import prepare_trades_tape
from nq.simulation.common import BUCKET_END, BUCKET_START
from nq.simulation.fvg import NS_1H, NS_30M, build_ohlcv_bars

LAYER_ID = "failed_breakout_causal"
LOOKBACK: Final = 20
ATR_WINDOW: Final = 20
VOL_WINDOW: Final = 20
SMA_PERIOD: Final = 50
HOLD_BARS: Final = 12
REWARD_RATIO: Final = 2.0
RANGE_MULT: Final = 1.0
VOL_MULT: Final = 1.0
SESSION_HOUR_START: Final = 3
SESSION_HOUR_END: Final = 16
SKIP_OPEN_HOUR: Final = 9
SKIP_OPEN_MINUTE: Final = 30
Side = Literal["short", "long"]
_ET: Final = ZoneInfo(TZ_NAME)
_EPS: Final = 1e-12

_TRADE_SCHEMA: Final[dict[str, pl.DataType]] = {
    "signal_ts": pl.Int64(),
    "fill_ts": pl.Int64(),
    "clock": pl.Utf8(),
    "fill_clock": pl.Utf8(),
    "side": pl.Utf8(),
    "range_high": pl.Float64(),
    "range_low": pl.Float64(),
    "signal_h": pl.Float64(),
    "signal_l": pl.Float64(),
    "signal_c": pl.Float64(),
    "entry": pl.Float64(),
    "sl": pl.Float64(),
    "tp": pl.Float64(),
    "prior_swing": pl.Float64(),
    "exit": pl.Float64(),
    "exit_reason": pl.Utf8(),
    "risk_pts": pl.Float64(),
    "pnl_pts": pl.Float64(),
    "pnl_usd": pl.Float64(),
    "leak_pts": pl.Float64(),
}


def _fmt_ts(ts: int, tz_name: str = TZ_NAME) -> str:
    return dt.datetime.fromtimestamp(ts / 1_000_000_000, tz=ZoneInfo(tz_name)).isoformat()


def _et_hm(ts: int) -> tuple[int, int]:
    stamp = dt.datetime.fromtimestamp(ts / 1_000_000_000, tz=_ET)
    return int(stamp.hour), int(stamp.minute)


def _json_num(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, int):
        return int(value)
    return float(value)


def _median(values: list[float]) -> float:
    finite = [x for x in values if not math.isnan(x)]
    if not finite:
        return float("nan")
    ordered = sorted(finite)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _empty_diag(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "layer": LAYER_ID,
        "not_fb_lock": True,
        "not_hanging_limit": True,
        "not_mbo_book": True,
        "fill_rule": "next_open",
        "sl_rule": "signal_wick_plus_atr",
        "path_rule": "sl_first",
        "point_value_usd": MNQ_MULT,
        "n_30m_bars": 0,
        "n_1h_bars": 0,
        "n_session_bars": 0,
        "n_fb_pattern": 0,
        "n_skipped_sma": 0,
        "n_skipped_no_next": 0,
        "n_skipped_gap_sl": 0,
        "n_skipped_bad_risk": 0,
        "n_trades": 0,
        "n_win": 0,
        "n_loss": 0,
        "n_time_exit": 0,
        "n_sl": 0,
        "n_tp": 0,
        "gross_pnl_usd": 0.0,
        "win_rate": float("nan"),
        "median_pnl_pts": float("nan"),
        "median_leak_pts": float("nan"),
        "sma_filter": "sma50_hourly_min_periods",
    }
    base.update(overrides)
    return base


def _with_past_baselines(
    bars: pl.DataFrame,
    *,
    atr_window: int,
    vol_window: int,
) -> pl.DataFrame:
    work = bars.sort(BUCKET_START)
    if "range" not in work.columns:
        work = work.with_columns((pl.col("h") - pl.col("l")).alias("range"))
    atr_w = max(1, int(atr_window))
    vol_w = max(1, int(vol_window))
    return work.with_columns(
        pl.col("range").shift(1).rolling_mean(window_size=atr_w, min_samples=atr_w).alias("atr"),
        pl.col("volume")
        .shift(1)
        .rolling_mean(window_size=vol_w, min_samples=vol_w)
        .alias("volsma"),
    )


def _hourly_sma(h1: pl.DataFrame, *, period: int) -> pl.DataFrame:
    if period <= 0 or h1.height == 0:
        return pl.DataFrame(schema={AVAILABILITY_TS: pl.Int64(), "sma50": pl.Float64()})
    win = max(1, int(period))
    return (
        h1.sort(BUCKET_START)
        .with_columns(
            pl.col("c").shift(1).rolling_mean(window_size=win, min_samples=win).alias("sma50")
        )
        .select(AVAILABILITY_TS, "sma50")
    )


def _attach_sma(m30: pl.DataFrame, h1: pl.DataFrame, *, sma_period: int) -> pl.DataFrame:
    if sma_period <= 0:
        return m30.with_columns(pl.lit(None, dtype=pl.Float64).alias("sma50"))
    sma = _hourly_sma(h1, period=sma_period)
    if sma.height == 0:
        return m30.with_columns(pl.lit(None, dtype=pl.Float64).alias("sma50"))
    return m30.sort(AVAILABILITY_TS).join_asof(
        sma.sort(AVAILABILITY_TS),
        on=AVAILABILITY_TS,
        strategy="backward",
    )


def _in_session(ts: int, *, session_start: int, session_end: int, skip_open: bool) -> bool:
    hour, minute = _et_hm(ts)
    if hour < session_start or hour >= session_end:
        return False
    return not (skip_open and hour == SKIP_OPEN_HOUR and minute == SKIP_OPEN_MINUTE)


def _ohlc_exit(
    *,
    side: Side,
    sl: float,
    tp: float,
    o: float,
    h: float,
    low_px: float,
    is_fill_bar: bool,
) -> tuple[float | None, str | None]:
    """خروج داخل الشمعة. إن لامس الوقف والهدف: الوقف أولًا. فجوة بعد الملء على الافتتاح."""

    gap_sl = (side == "short" and o >= sl) or (side == "long" and o <= sl)
    gap_tp = (side == "short" and o <= tp) or (side == "long" and o >= tp)
    if not is_fill_bar and gap_sl:
        return float(o), "sl"
    if not is_fill_bar and gap_tp:
        return float(o), "tp"
    hit_sl = (h >= sl) if side == "short" else (low_px <= sl)
    hit_tp = (low_px <= tp) if side == "short" else (h >= tp)
    if hit_sl:
        return float(sl), "sl"
    if hit_tp:
        return float(tp), "tp"
    return None, None


def _simulate_path(
    *,
    side: Side,
    sl: float,
    tp: float,
    o: NDArray[np.float64],
    h: NDArray[np.float64],
    low_px: NDArray[np.float64],
    c: NDArray[np.float64],
    fill_i: int,
    last_i: int,
) -> tuple[float, str]:
    for idx in range(fill_i, last_i + 1):
        px, reason = _ohlc_exit(
            side=side,
            sl=sl,
            tp=tp,
            o=float(o[idx]),
            h=float(h[idx]),
            low_px=float(low_px[idx]),
            is_fill_bar=idx == fill_i,
        )
        if px is not None and reason is not None:
            return px, reason
    return float(c[last_i]), "time"


def simulate_from_bars(  # noqa: PLR0912, PLR0915
    m30: pl.DataFrame,
    h1: pl.DataFrame,
    *,
    lookback: int = LOOKBACK,
    atr_window: int = ATR_WINDOW,
    vol_window: int = VOL_WINDOW,
    sma_period: int = SMA_PERIOD,
    hold_bars: int = HOLD_BARS,
    reward_ratio: float = REWARD_RATIO,
    range_mult: float = RANGE_MULT,
    vol_mult: float = VOL_MULT,
    session_start: int = SESSION_HOUR_START,
    session_end: int = SESSION_HOUR_END,
    skip_open: bool = True,
    point_value: float = MNQ_MULT,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """مسح شموع 30د: نمط كسر فاشل ثم ملء عند افتتاح التالية. ليست إشارة."""

    sma_label = "off_diagnostic_not_original" if sma_period <= 0 else "sma50_hourly_min_periods"
    diag = _empty_diag(
        lookback=int(lookback),
        atr_window=int(atr_window),
        vol_window=int(vol_window),
        sma_period=int(sma_period),
        hold_bars=int(hold_bars),
        reward_ratio=float(reward_ratio),
        range_mult=float(range_mult),
        vol_mult=float(vol_mult),
        session_start=int(session_start),
        session_end=int(session_end),
        skip_open=bool(skip_open),
        point_value_usd=float(point_value),
        sma_filter=sma_label,
        n_1h_bars=int(h1.height),
    )
    if m30.height == 0:
        return pl.DataFrame(schema=_TRADE_SCHEMA), diag
    work = _attach_sma(
        _with_past_baselines(m30, atr_window=atr_window, vol_window=vol_window),
        h1,
        sma_period=sma_period,
    )
    o = np.asarray(work["o"].to_numpy(), dtype=np.float64)
    h = np.asarray(work["h"].to_numpy(), dtype=np.float64)
    low = np.asarray(work["l"].to_numpy(), dtype=np.float64)
    c = np.asarray(work["c"].to_numpy(), dtype=np.float64)
    vol = np.asarray(work["volume"].to_numpy(), dtype=np.float64)
    atr = np.asarray(work["atr"].to_numpy(), dtype=np.float64)
    volsma = np.asarray(work["volsma"].to_numpy(), dtype=np.float64)
    sma = np.asarray(work["sma50"].to_numpy(), dtype=np.float64)
    start_ts = np.asarray(work[BUCKET_START].to_numpy(), dtype=np.int64)
    end_ts = np.asarray(work[BUCKET_END].to_numpy(), dtype=np.int64)
    n = int(o.size)
    lb = max(1, int(lookback))
    hold = max(1, int(hold_bars))
    diag["n_30m_bars"] = n
    n_session = 0
    n_pattern = 0
    n_skipped_sma = 0
    n_skipped_no_next = 0
    n_skipped_gap_sl = 0
    n_skipped_bad_risk = 0
    rows: list[dict[str, Any]] = []
    for j in range(lb, n):
        if not _in_session(
            int(start_ts[j]),
            session_start=session_start,
            session_end=session_end,
            skip_open=skip_open,
        ):
            continue
        n_session += 1
        atr_j = float(atr[j])
        vol_j = float(volsma[j])
        if not math.isfinite(atr_j) or atr_j <= _EPS:
            continue
        if not math.isfinite(vol_j) or vol_j <= _EPS:
            continue
        bar_range = float(h[j] - low[j])
        if range_mult > 0 and bar_range <= atr_j * range_mult:
            continue
        if vol_mult > 0 and float(vol[j]) <= vol_j * vol_mult:
            continue
        range_high = float(np.max(h[j - lb : j]))
        range_low = float(np.min(low[j - lb : j]))
        short_pat = bool(h[j] > range_high and c[j] < range_high)
        long_pat = bool(low[j] < range_low and c[j] > range_low)
        if short_pat == long_pat:
            continue
        n_pattern += 1
        side: Side = "short" if short_pat else "long"
        if sma_period > 0:
            sma_j = float(sma[j])
            if not math.isfinite(sma_j):
                n_skipped_sma += 1
                continue
            if side == "short" and not (c[j] < sma_j):
                n_skipped_sma += 1
                continue
            if side == "long" and not (c[j] > sma_j):
                n_skipped_sma += 1
                continue
        fill_i = j + 1
        if fill_i >= n:
            n_skipped_no_next += 1
            continue
        entry = float(o[fill_i])
        if side == "short":
            sl = float(h[j] + atr_j)
            prior_swing = float(np.min(low[j - lb : j]))
            leak_level = range_high
        else:
            sl = float(low[j] - atr_j)
            prior_swing = float(np.max(h[j - lb : j]))
            leak_level = range_low
        if side == "short" and entry >= sl:
            n_skipped_gap_sl += 1
            continue
        if side == "long" and entry <= sl:
            n_skipped_gap_sl += 1
            continue
        risk = abs(sl - entry)
        if risk <= _EPS:
            n_skipped_bad_risk += 1
            continue
        if side == "short":
            tp = entry - risk * float(reward_ratio)
        else:
            tp = entry + risk * float(reward_ratio)
        last_i = min(n - 1, fill_i + hold - 1)
        exit_px, reason = _simulate_path(
            side=side,
            sl=sl,
            tp=tp,
            o=o,
            h=h,
            low_px=low,
            c=c,
            fill_i=fill_i,
            last_i=last_i,
        )
        pnl_pts = (entry - exit_px) if side == "short" else (exit_px - entry)
        rows.append(
            {
                "signal_ts": int(end_ts[j]),
                "fill_ts": int(start_ts[fill_i]),
                "clock": _fmt_ts(int(start_ts[j])),
                "fill_clock": _fmt_ts(int(start_ts[fill_i])),
                "side": side,
                "range_high": range_high,
                "range_low": range_low,
                "signal_h": float(h[j]),
                "signal_l": float(low[j]),
                "signal_c": float(c[j]),
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "prior_swing": prior_swing,
                "exit": exit_px,
                "exit_reason": reason,
                "risk_pts": risk,
                "pnl_pts": pnl_pts,
                "pnl_usd": pnl_pts * float(point_value),
                "leak_pts": abs(leak_level - entry),
            }
        )
    diag["n_session_bars"] = n_session
    diag["n_fb_pattern"] = n_pattern
    diag["n_skipped_sma"] = n_skipped_sma
    diag["n_skipped_no_next"] = n_skipped_no_next
    diag["n_skipped_gap_sl"] = n_skipped_gap_sl
    diag["n_skipped_bad_risk"] = n_skipped_bad_risk
    table = pl.DataFrame(rows, schema=_TRADE_SCHEMA) if rows else pl.DataFrame(schema=_TRADE_SCHEMA)
    diag["n_trades"] = table.height
    if table.height:
        pnl = [float(x) for x in table["pnl_pts"].to_list()]
        usd = [float(x) for x in table["pnl_usd"].to_list()]
        leak = [float(x) for x in table["leak_pts"].to_list()]
        reasons = table["exit_reason"].to_list()
        diag["n_win"] = sum(1 for x in pnl if x > 0)
        diag["n_loss"] = sum(1 for x in pnl if x < 0)
        diag["n_time_exit"] = sum(1 for r in reasons if r == "time")
        diag["n_sl"] = sum(1 for r in reasons if r == "sl")
        diag["n_tp"] = sum(1 for r in reasons if r == "tp")
        diag["gross_pnl_usd"] = float(sum(usd))
        diag["win_rate"] = float(diag["n_win"]) / float(table.height)
        diag["median_pnl_pts"] = _median(pnl)
        diag["median_leak_pts"] = _median(leak)
    return table, diag


def scan_failed_breakout(
    trades: pl.DataFrame,
    *,
    lookback: int = LOOKBACK,
    atr_window: int = ATR_WINDOW,
    vol_window: int = VOL_WINDOW,
    sma_period: int = SMA_PERIOD,
    hold_bars: int = HOLD_BARS,
    reward_ratio: float = REWARD_RATIO,
    range_mult: float = RANGE_MULT,
    vol_mult: float = VOL_MULT,
    session_start: int = SESSION_HOUR_START,
    session_end: int = SESSION_HOUR_END,
    skip_open: bool = True,
    point_value: float = MNQ_MULT,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """من شريط ``T`` ليوم واحد: شموع 30د/1س ثم المسح السببي. بلا دفتر MBO."""

    tape = prepare_trades_tape(trades)
    m30 = build_ohlcv_bars(tape, interval_ns=NS_30M)
    h1 = build_ohlcv_bars(tape, interval_ns=NS_1H)
    table, diag = simulate_from_bars(
        m30,
        h1,
        lookback=lookback,
        atr_window=atr_window,
        vol_window=vol_window,
        sma_period=sma_period,
        hold_bars=hold_bars,
        reward_ratio=reward_ratio,
        range_mult=range_mult,
        vol_mult=vol_mult,
        session_start=session_start,
        session_end=session_end,
        skip_open=skip_open,
        point_value=point_value,
    )
    diag["n_tape"] = int(tape.height)
    return table, diag


def write_failed_breakout_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "failed_breakout_trades.parquet")
    packed = {k: _json_num(v) if isinstance(v, float) else v for k, v in dict(diagnostics).items()}
    (out / "summary.json").write_text(
        json.dumps(packed, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    sma = diagnostics.get("sma_filter")
    lines = [
        "# Causal failed breakout (next-open fill)",
        "",
        "Signal known at 30m close. Fill is the **next bar open**, not `range_high` / "
        "`range_low`. Stop is the signal wick plus past ATR. Same-bar path: stop before "
        "target. Time stop = last close in the hold window. Prior swing is a label from "
        "bars before the signal, not MBO. Not a lock. Not a hanging limit.",
        f"fill_rule={diagnostics.get('fill_rule')} sl_rule={diagnostics.get('sl_rule')} "
        f"path_rule={diagnostics.get('path_rule')} sma_filter={sma}.",
        f"n_30m={diagnostics.get('n_30m_bars')} n_1h={diagnostics.get('n_1h_bars')} "
        f"session={diagnostics.get('n_session_bars')} pattern={diagnostics.get('n_fb_pattern')} "
        f"skipped_sma={diagnostics.get('n_skipped_sma')} skipped_gap_sl="
        f"{diagnostics.get('n_skipped_gap_sl')} trades={diagnostics.get('n_trades')}.",
        f"win={diagnostics.get('n_win')} loss={diagnostics.get('n_loss')} "
        f"tp={diagnostics.get('n_tp')} sl={diagnostics.get('n_sl')} "
        f"time={diagnostics.get('n_time_exit')} "
        f"gross_usd={diagnostics.get('gross_pnl_usd')} "
        f"median_pnl_pts={diagnostics.get('median_pnl_pts')} "
        f"median_leak_pts={diagnostics.get('median_leak_pts')} "
        f"(phantom |range_level-next_open|).",
        "A single Globex day cannot warm SMA50 hourly (min_periods=50). Zero trades with "
        "the original SMA filter is expected, not a bug. `--sma-bars 0` is a diagnostic, "
        "not the original strategy. Do not treat one-day P&L as an edge.",
        "",
        "| fill | side | entry | sl | tp | exit | reason | pnl_pts | leak_pts |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    shown = table.head(40) if table.height else table
    for row in shown.iter_rows(named=True):
        lines.append(
            f"| {row['fill_clock']} | {row['side']} | {row['entry']:.2f} | {row['sl']:.2f} | "
            f"{row['tp']:.2f} | {row['exit']:.2f} | {row['exit_reason']} | "
            f"{row['pnl_pts']:.2f} | {row['leak_pts']:.2f} |"
        )
    if table.height == 0:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    (out / "FAILED_BREAKOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


__all__ = [
    "ATR_WINDOW",
    "HOLD_BARS",
    "LAYER_ID",
    "LOOKBACK",
    "REWARD_RATIO",
    "SMA_PERIOD",
    "scan_failed_breakout",
    "simulate_from_bars",
    "write_failed_breakout_report",
]
