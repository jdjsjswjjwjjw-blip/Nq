"""محاكي Fair Value Gap (FVG) السببي — من MBO فقط.

يشتق:

* شموع OHLC/V من شريط الصفقات (tape) بنوافذ زمنية سببية.
* مناطق FVG على إطار أعلى (افتراضيًا 1h) متاحة فقط عند ``bucket_end``.
* إشارة Failed FVG / Effort-Without-Result على إطار أدنى (افتراضيًا 30m).

منع التسريب: كل ميزة تحمل ``availability_ts = bucket_end``؛ لا تُستخدم منطقة
FVG قبل اكتمال شمعة التكوين.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from nq.contracts.mbo import PRICE_SCALE
from nq.contracts.temporal import AVAILABILITY_TS
from nq.simulation.common import BUCKET_END, BUCKET_START, add_time_bucket, extract_trades

NS_PER_MIN: Final = 60_000_000_000
NS_30M: Final = 30 * NS_PER_MIN
NS_1H: Final = 60 * NS_PER_MIN

_MIN_FVG_BARS: Final = 3
_DEFAULT_VOL_PRICE_MULT = 1.2
_DEFAULT_VOL_VOLUME_MULT = 1.3
_DEFAULT_FVG_WINDOW_NS = 90 * NS_PER_MIN

SIGNAL_FAIL_BULL_SHORT: Final = -1.0
SIGNAL_FAIL_BEAR_LONG: Final = 1.0

_EMPTY_FVG_SCHEMA: Final[dict[str, pl.DataType]] = {
    "fvg_id": pl.Utf8(),
    "fvg_type": pl.Utf8(),
    "formed_at": pl.Int64(),
    AVAILABILITY_TS: pl.Int64(),
    "fvg_low": pl.Float64(),
    "fvg_high": pl.Float64(),
    "fvg_mid": pl.Float64(),
}


def build_ohlcv_bars(frame: pl.DataFrame, *, interval_ns: int) -> pl.DataFrame:
    """يبني شموع OHLC + حجم/تدفّق من صفقات MBO؛ متاحة عند ``bucket_end``.

    الأعمدة الإضافية (سببية داخل الشمعة المكتملة):
    ``buy_volume``, ``sell_volume``, ``delta`` = buy − sell.
    """
    if interval_ns < 1:
        raise ValueError(f"interval_ns must be >= 1, got {interval_ns}")
    trades = extract_trades(frame)
    empty = {
        BUCKET_START: pl.Int64(),
        BUCKET_END: pl.Int64(),
        AVAILABILITY_TS: pl.Int64(),
        "o": pl.Float64(),
        "h": pl.Float64(),
        "l": pl.Float64(),
        "c": pl.Float64(),
        "volume": pl.Float64(),
        "buy_volume": pl.Float64(),
        "sell_volume": pl.Float64(),
        "delta": pl.Float64(),
        "range": pl.Float64(),
    }
    if trades.height == 0:
        return pl.DataFrame(schema=empty)
    priced = add_time_bucket(trades, interval_ns=interval_ns).with_columns(
        (pl.col("price").cast(pl.Float64) * PRICE_SCALE).alias("px")
    )
    return (
        priced.group_by(BUCKET_START)
        .agg(
            pl.col("px").first().alias("o"),
            pl.col("px").max().alias("h"),
            pl.col("px").min().alias("l"),
            pl.col("px").last().alias("c"),
            pl.col("size").cast(pl.Float64).sum().alias("volume"),
            pl.col("buy_volume").cast(pl.Float64).sum().alias("buy_volume"),
            pl.col("sell_volume").cast(pl.Float64).sum().alias("sell_volume"),
            pl.col(BUCKET_END).first(),
        )
        .sort(BUCKET_START)
        .with_columns(
            (pl.col("h") - pl.col("l")).alias("range"),
            (pl.col("buy_volume") - pl.col("sell_volume")).alias("delta"),
            pl.col(BUCKET_END).alias(AVAILABILITY_TS),
        )
        .select(
            BUCKET_START,
            BUCKET_END,
            AVAILABILITY_TS,
            "o",
            "h",
            "l",
            "c",
            "volume",
            "buy_volume",
            "sell_volume",
            "delta",
            "range",
        )
    )


def detect_h1_fvgs(h1: pl.DataFrame) -> pl.DataFrame:
    """يكتشف FVG على شموع مكتملة؛ ``availability_ts`` = نهاية شمعة التكوين."""
    if h1.height < _MIN_FVG_BARS:
        return pl.DataFrame(schema=_EMPTY_FVG_SCHEMA)

    highs = h1["h"].to_list()
    lows = h1["l"].to_list()
    starts = h1[BUCKET_START].to_list()
    ends = h1[BUCKET_END].to_list()

    rows: list[dict[str, float | int | str]] = []
    for i in range(2, len(highs)):
        formed_at = int(starts[i])
        available_at = int(ends[i])
        if lows[i] > highs[i - 2]:
            fvg_low = float(min(highs[i - 2], lows[i]))
            fvg_high = float(max(highs[i - 2], lows[i]))
            rows.append(
                {
                    "fvg_id": f"{formed_at}_Bull_{fvg_low:.4f}_{fvg_high:.4f}",
                    "fvg_type": "Bull",
                    "formed_at": formed_at,
                    AVAILABILITY_TS: available_at,
                    "fvg_low": fvg_low,
                    "fvg_high": fvg_high,
                    "fvg_mid": (fvg_low + fvg_high) / 2.0,
                }
            )
        if highs[i] < lows[i - 2]:
            fvg_low = float(min(highs[i], lows[i - 2]))
            fvg_high = float(max(highs[i], lows[i - 2]))
            rows.append(
                {
                    "fvg_id": f"{formed_at}_Bear_{fvg_low:.4f}_{fvg_high:.4f}",
                    "fvg_type": "Bear",
                    "formed_at": formed_at,
                    AVAILABILITY_TS: available_at,
                    "fvg_low": fvg_low,
                    "fvg_high": fvg_high,
                    "fvg_mid": (fvg_low + fvg_high) / 2.0,
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema=_EMPTY_FVG_SCHEMA)


def _with_effort_features(m30: pl.DataFrame) -> pl.DataFrame:
    atr = pl.col("range").shift(1).rolling_mean(window_size=20, min_samples=20)
    vol = pl.col("volume").shift(1).rolling_mean(window_size=20, min_samples=20)
    return m30.with_columns(atr.alias("atr20"), vol.alias("volume_sma20"))


def _empty_signal_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            BUCKET_START: pl.Int64(),
            BUCKET_END: pl.Int64(),
            AVAILABILITY_TS: pl.Int64(),
            "o": pl.Float64(),
            "h": pl.Float64(),
            "l": pl.Float64(),
            "c": pl.Float64(),
            "nq_close": pl.Float64(),
            "fail_fvg": pl.Float64(),
            "effort_range_ratio": pl.Float64(),
            "effort_volume_ratio": pl.Float64(),
            "close_vs_mid": pl.Float64(),
            "close_vs_zone": pl.Float64(),
            "fvg_type": pl.Utf8(),
            "fvg_low": pl.Float64(),
            "fvg_high": pl.Float64(),
            "fvg_mid": pl.Float64(),
            "fvg_id": pl.Utf8(),
        }
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid timestamp/price int")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"cannot cast {type(value)!r} to int")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid float feature")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"cannot cast {type(value)!r} to float")


def _base_signal_row(row: dict[str, object]) -> dict[str, float | int | str | None]:
    return {
        BUCKET_START: _as_int(row[BUCKET_START]),
        BUCKET_END: _as_int(row[BUCKET_END]),
        AVAILABILITY_TS: _as_int(row[AVAILABILITY_TS]),
        "o": _as_float(row["o"]),
        "h": _as_float(row["h"]),
        "l": _as_float(row["l"]),
        "c": _as_float(row["c"]),
        "nq_close": _as_float(row["c"]),
        "fail_fvg": 0.0,
        "effort_range_ratio": 0.0,
        "effort_volume_ratio": 0.0,
        "close_vs_mid": 0.0,
        "close_vs_zone": 0.0,
        "fvg_type": None,
        "fvg_low": None,
        "fvg_high": None,
        "fvg_mid": None,
        "fvg_id": None,
    }


def _pick_failed_fvg(
    *,
    close: float,
    high: float,
    low: float,
    signal_time: int,
    fvg_window_ns: int,
    fvg_rows: list[dict[str, object]],
    used: set[str],
) -> tuple[float, dict[str, object] | None]:
    candidates = [
        f
        for f in fvg_rows
        if str(f["fvg_id"]) not in used
        and _as_int(f[AVAILABILITY_TS]) <= signal_time
        and signal_time - _as_int(f[AVAILABILITY_TS]) <= fvg_window_ns
        and high >= _as_float(f["fvg_low"])
        and low <= _as_float(f["fvg_high"])
    ]
    candidates.sort(key=lambda f: _as_int(f[AVAILABILITY_TS]), reverse=True)
    for fvg in candidates:
        fvg_type = str(fvg["fvg_type"])
        fvg_low = _as_float(fvg["fvg_low"])
        fvg_high = _as_float(fvg["fvg_high"])
        fvg_mid = _as_float(fvg["fvg_mid"])
        if fvg_type == "Bull" and (close <= fvg_mid or close < fvg_high):
            return SIGNAL_FAIL_BULL_SHORT, fvg
        if fvg_type == "Bear" and (close >= fvg_mid or close > fvg_low):
            return SIGNAL_FAIL_BEAR_LONG, fvg
    return 0.0, None


def failed_fvg_from_bars(
    h1: pl.DataFrame,
    signal_bars: pl.DataFrame,
    *,
    fvg_window_ns: int = _DEFAULT_FVG_WINDOW_NS,
    vol_price_mult: float = _DEFAULT_VOL_PRICE_MULT,
    vol_volume_mult: float = _DEFAULT_VOL_VOLUME_MULT,
) -> pl.DataFrame:
    """إشارة Failed FVG من شموع مكتملة مسبقًا (سببي — لإعادة استخدام الكاش)."""
    m30 = _with_effort_features(signal_bars)
    fvgs = detect_h1_fvgs(h1)
    if m30.height == 0:
        return _empty_signal_frame()

    fvg_rows: list[dict[str, object]] = fvgs.to_dicts() if fvgs.height > 0 else []
    used: set[str] = set()
    out: list[dict[str, float | int | str | None]] = []

    for row in m30.iter_rows(named=True):
        base = _base_signal_row(row)
        atr20 = row["atr20"]
        vol_sma = row["volume_sma20"]
        candle_range = float(row["range"])
        volume = float(row["volume"])
        close = float(row["c"])
        high = float(row["h"])
        low = float(row["l"])
        signal_time = int(row[AVAILABILITY_TS])

        if atr20 is None or vol_sma is None or not (atr20 > 0 and vol_sma > 0 and candle_range > 0):
            out.append(base)
            continue

        effort_range = candle_range / float(atr20)
        effort_vol = volume / float(vol_sma)
        base["effort_range_ratio"] = effort_range
        base["effort_volume_ratio"] = effort_vol
        if effort_range <= vol_price_mult or effort_vol <= vol_volume_mult:
            out.append(base)
            continue

        signal, chosen = _pick_failed_fvg(
            close=close,
            high=high,
            low=low,
            signal_time=signal_time,
            fvg_window_ns=fvg_window_ns,
            fvg_rows=fvg_rows,
            used=used,
        )
        if chosen is not None:
            used.add(str(chosen["fvg_id"]))
            base["fail_fvg"] = signal
            base["fvg_type"] = str(chosen["fvg_type"])
            base["fvg_low"] = _as_float(chosen["fvg_low"])
            base["fvg_high"] = _as_float(chosen["fvg_high"])
            base["fvg_mid"] = _as_float(chosen["fvg_mid"])
            base["fvg_id"] = str(chosen["fvg_id"])
            base["close_vs_mid"] = close - _as_float(chosen["fvg_mid"])
            if str(chosen["fvg_type"]) == "Bull":
                base["close_vs_zone"] = close - _as_float(chosen["fvg_high"])
            else:
                base["close_vs_zone"] = close - _as_float(chosen["fvg_low"])
        out.append(base)

    return pl.DataFrame(out)


def failed_fvg_features(
    frame: pl.DataFrame,
    *,
    h1_interval_ns: int = NS_1H,
    signal_interval_ns: int = NS_30M,
    fvg_window_ns: int = _DEFAULT_FVG_WINDOW_NS,
    vol_price_mult: float = _DEFAULT_VOL_PRICE_MULT,
    vol_volume_mult: float = _DEFAULT_VOL_VOLUME_MULT,
) -> pl.DataFrame:
    """إطار إشارة Failed FVG سببي من MBO خام."""
    h1 = build_ohlcv_bars(frame, interval_ns=h1_interval_ns)
    signal_bars = build_ohlcv_bars(frame, interval_ns=signal_interval_ns)
    return failed_fvg_from_bars(
        h1,
        signal_bars,
        fvg_window_ns=fvg_window_ns,
        vol_price_mult=vol_price_mult,
        vol_volume_mult=vol_volume_mult,
    )


__all__ = [
    "NS_1H",
    "NS_30M",
    "NS_PER_MIN",
    "build_ohlcv_bars",
    "detect_h1_fvgs",
    "failed_fvg_features",
    "failed_fvg_from_bars",
]
