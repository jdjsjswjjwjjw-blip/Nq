"""ضوضاء عكسية قبل طبعات ``T`` العدوانية: Add ثم Cancel بلا ملء، جانب معاكس.

حدس بحثي على يوم جلسة واحد. ليس حكمًا قانونيًا بالسبوفينج، وليست overlay حيّة،
ولا قناة LSTM. العتبات 3:1 و 5:1 أعداد وصفية فقط، ليست فلتر دخول مثبتًا.

لكل ``T`` فوري (بلا Add سابق) داخل نطاق سعري: أوامر ``order_id`` أُضيفت وأُلغيت
داخل النافذة ``(t-W, t)`` على الجانب المعاكس، بلا ``F``، واختياريًا حجم ≥ ضعف
متوسط Add اليوم. احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import polars as pl

from nq.contracts.mbo import PRICE_SCALE, MboAction, MboSide
from nq.contracts.temporal import EVENT_TS, SEQUENCE
from nq.research.mbo_trade_overlap import prepare_mbo_events

LAYER_ID = "opposite_phantom"
SECOND_NS: Final = 1_000_000_000
DEFAULT_WINDOWS_S: Final = (1, 5, 15, 30)
DEFAULT_TICK_BAND: Final = 4
DEFAULT_SIZE_MULT: Final = 2.0
TICK_POINTS: Final = 0.25
_TICK_FIXED: Final = round(TICK_POINTS / PRICE_SCALE)
_FIXED_POINT_PRICE: Final = 1_000_000.0
_ADD = MboAction.ADD.value
_CANCEL = MboAction.CANCEL.value
_FILL = MboAction.FILL.value
_TRADE = MboAction.TRADE.value
_BID = MboSide.BID.value
_ASK = MboSide.ASK.value
_RATIO_MARKS: Final = (3.0, 5.0)


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _fmean(frame: pl.DataFrame, col: str) -> float:
    val = frame.select(pl.col(col).mean()).item()
    return float("nan") if val is None else float(val)


def _fmedian(frame: pl.DataFrame, col: str) -> float:
    val = frame.select(pl.col(col).median()).item()
    return float("nan") if val is None else float(val)


def _isum(frame: pl.DataFrame, col: str) -> int:
    val = frame.select(pl.col(col).sum()).item()
    return 0 if val is None else int(val)


def infer_tick_size(price: float) -> float:
    """0.25 للنقطة العائمة، ثابت ``PRICE_SCALE`` للعقود القانونية."""

    return float(_TICK_FIXED) if abs(price) >= _FIXED_POINT_PRICE else TICK_POINTS


def _parse_windows(windows_s: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(w) for w in windows_s)
    if not out or any(w <= 0 for w in out):
        raise ValueError("windows_s must be positive")
    return out


def closed_unfilled_orders(mbo: pl.DataFrame) -> pl.DataFrame:
    """أوامر أُغلقت بإلغاء ولم يُسجَّل لها ``F``. أول Add + آخر Cancel."""

    adds = (
        mbo.filter((pl.col("action") == _ADD) & (pl.col("order_id") > 0))
        .sort([EVENT_TS, SEQUENCE])
        .unique(subset=["order_id"], keep="first")
        .select(
            "order_id",
            "side",
            "price",
            pl.col("size").alias("add_size"),
            pl.col(EVENT_TS).alias("add_ts"),
            pl.col(SEQUENCE).alias("add_seq"),
        )
    )
    fills = (
        mbo.filter((pl.col("action") == _FILL) & (pl.col("order_id") > 0))
        .group_by("order_id")
        .agg(pl.col("size").sum().alias("fill_size"))
    )
    cancels = (
        mbo.filter((pl.col("action") == _CANCEL) & (pl.col("order_id") > 0))
        .sort([EVENT_TS, SEQUENCE])
        .unique(subset=["order_id"], keep="last")
        .select(
            "order_id",
            pl.col(EVENT_TS).alias("cancel_ts"),
            pl.col(SEQUENCE).alias("cancel_seq"),
        )
    )
    closed = adds.join(cancels, on="order_id", how="inner").join(fills, on="order_id", how="left")
    return (
        closed.with_columns(pl.col("fill_size").fill_null(0).alias("fill_size"))
        .filter(pl.col("fill_size") <= 0)
        .filter(pl.col("cancel_ts") > pl.col("add_ts"))
        .with_columns((pl.col("cancel_ts") - pl.col("add_ts")).alias("lifetime_ns"))
    )


def immediate_prints(mbo: pl.DataFrame) -> pl.DataFrame:
    """طبعات ``T`` بلا Add سابق لنفس ``order_id`` (عدواني)."""

    first_add = (
        mbo.filter((pl.col("action") == _ADD) & (pl.col("order_id") > 0))
        .group_by("order_id")
        .agg(pl.col(SEQUENCE).min().alias("add_seq"))
    )
    prints = mbo.filter(pl.col("action") == _TRADE)
    tagged = prints.join(first_add, on="order_id", how="left")
    rested = (
        (pl.col("order_id") > 0)
        & pl.col("add_seq").is_not_null()
        & (pl.col("add_seq") < pl.col(SEQUENCE))
    )
    return tagged.filter(~rested).drop("add_seq")


def _opposite_mask() -> pl.Expr:
    return ((pl.col("t_side") == _BID) & (pl.col("side") == _ASK)) | (
        (pl.col("t_side") == _ASK) & (pl.col("side") == _BID)
    )


def _window_pairs(
    prints: pl.DataFrame,
    closed: pl.DataFrame,
    window_s: int,
    tick: float,
    tick_band: int,
) -> pl.DataFrame:
    horizon = int(window_s) * SECOND_NS
    band = float(tick_band) * float(tick)
    tcol = prints.select(
        "t_i",
        pl.col(EVENT_TS).alias("t_ts"),
        pl.col("side").alias("t_side"),
        pl.col("price").alias("t_price"),
    )
    joined = tcol.join_where(
        closed,
        pl.col("add_ts") >= pl.col("t_ts") - horizon,
        pl.col("add_ts") < pl.col("t_ts"),
        pl.col("cancel_ts") < pl.col("t_ts"),
        pl.col("cancel_ts") > pl.col("add_ts"),
    )
    return joined.filter((pl.col("price") - pl.col("t_price")).abs() <= band)


def _agg_side(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return frame.group_by("t_i").agg(
        pl.len().alias(f"{prefix}_n"),
        pl.col("add_size").sum().alias(f"{prefix}_size"),
        pl.col("lifetime_ns").mean().alias(f"{prefix}_life"),
        (pl.col("lifetime_ns") < SECOND_NS).sum().alias(f"{prefix}_instant_n"),
    )


def _per_print(
    prints: pl.DataFrame,
    pairs: pl.DataFrame,
    large_min: float,
) -> pl.DataFrame:
    opposite = pairs.filter(_opposite_mask())
    large = opposite.filter(pl.col("add_size") >= large_min)
    keys = prints.select(
        "t_i",
        pl.col(SEQUENCE).alias("t_seq"),
        pl.col(EVENT_TS).alias("t_ts"),
        pl.col("price").alias("t_price"),
        pl.col("side").alias("t_side"),
        pl.col("size").alias("t_size"),
    )
    out = keys.join(_agg_side(opposite, "opp"), on="t_i", how="left").join(
        _agg_side(large, "large"), on="t_i", how="left"
    )
    return out.with_columns(
        pl.col("opp_n").fill_null(0),
        pl.col("opp_size").fill_null(0),
        pl.col("opp_instant_n").fill_null(0),
        pl.col("large_n").fill_null(0),
        pl.col("large_size").fill_null(0),
        pl.col("large_instant_n").fill_null(0),
    ).with_columns(
        pl.when(pl.col("t_size") > 0)
        .then(pl.col("opp_size") / pl.col("t_size"))
        .otherwise(float("nan"))
        .alias("opp_ratio"),
        pl.when(pl.col("t_size") > 0)
        .then(pl.col("large_size") / pl.col("t_size"))
        .otherwise(float("nan"))
        .alias("large_ratio"),
    )


def _unique_life(frame: pl.DataFrame) -> tuple[int, int, float]:
    if frame.height == 0:
        return 0, 0, float("nan")
    uniq = frame.unique(subset=["order_id"])
    n = uniq.height
    instant = uniq.filter(pl.col("lifetime_ns") < SECOND_NS).height
    return n, instant, _fmean(uniq, "lifetime_ns")


@dataclass(frozen=True, slots=True)
class WindowPhantom:
    """مجمّع نافذة ``W`` قبل طبعات النطاق."""

    window_s: int
    n_t: int
    t_size: int
    n_t_with_opposite: int
    n_t_ratio_ge_3: int
    n_t_ratio_ge_5: int
    mean_phantom_ratio: float
    median_phantom_ratio: float
    mean_large_ratio: float
    median_large_ratio: float
    mean_opposite_n: float
    mean_large_n: float
    mean_lifetime_ns: float
    mean_large_lifetime_ns: float
    instant_cancel_rate: float
    opposite_density_per_s: float
    unique_opposite_oids: int
    unique_large_oids: int
    opposite_size: int
    large_size: int


def _summarize_window(
    window_s: int,
    per_t: pl.DataFrame,
    pairs: pl.DataFrame,
    large_min: float,
) -> WindowPhantom:
    opposite = pairs.filter(_opposite_mask())
    large = opposite.filter(pl.col("add_size") >= large_min)
    n_opp, n_instant, mean_life = _unique_life(opposite)
    n_large, _, mean_large_life = _unique_life(large)
    return WindowPhantom(
        window_s=window_s,
        n_t=per_t.height,
        t_size=_isum(per_t, "t_size"),
        n_t_with_opposite=int((per_t["opp_n"] > 0).sum()),
        n_t_ratio_ge_3=int((per_t["opp_ratio"] >= _RATIO_MARKS[0]).sum()),
        n_t_ratio_ge_5=int((per_t["opp_ratio"] >= _RATIO_MARKS[1]).sum()),
        mean_phantom_ratio=_fmean(per_t, "opp_ratio"),
        median_phantom_ratio=_fmedian(per_t, "opp_ratio"),
        mean_large_ratio=_fmean(per_t, "large_ratio"),
        median_large_ratio=_fmedian(per_t, "large_ratio"),
        mean_opposite_n=_fmean(per_t, "opp_n"),
        mean_large_n=_fmean(per_t, "large_n"),
        mean_lifetime_ns=mean_life,
        mean_large_lifetime_ns=mean_large_life,
        instant_cancel_rate=_ratio(n_instant, n_opp),
        opposite_density_per_s=_ratio(_fmean(per_t, "opp_n"), window_s),
        unique_opposite_oids=n_opp,
        unique_large_oids=n_large,
        opposite_size=_isum(opposite, "add_size") if opposite.height else 0,
        large_size=_isum(large, "add_size") if large.height else 0,
    )


def _empty_windows() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "window_s": pl.Int64(),
            "n_t": pl.Int64(),
            "t_size": pl.Int64(),
            "n_t_with_opposite": pl.Int64(),
            "n_t_ratio_ge_3": pl.Int64(),
            "n_t_ratio_ge_5": pl.Int64(),
            "mean_phantom_ratio": pl.Float64(),
            "median_phantom_ratio": pl.Float64(),
            "mean_large_ratio": pl.Float64(),
            "median_large_ratio": pl.Float64(),
            "mean_opposite_n": pl.Float64(),
            "mean_large_n": pl.Float64(),
            "mean_lifetime_ns": pl.Float64(),
            "mean_large_lifetime_ns": pl.Float64(),
            "instant_cancel_rate": pl.Float64(),
            "opposite_density_per_s": pl.Float64(),
            "unique_opposite_oids": pl.Int64(),
            "unique_large_oids": pl.Int64(),
            "opposite_size": pl.Int64(),
            "large_size": pl.Int64(),
        }
    )


def _window_row(row: WindowPhantom) -> dict[str, Any]:
    return {
        "window_s": row.window_s,
        "n_t": row.n_t,
        "t_size": row.t_size,
        "n_t_with_opposite": row.n_t_with_opposite,
        "n_t_ratio_ge_3": row.n_t_ratio_ge_3,
        "n_t_ratio_ge_5": row.n_t_ratio_ge_5,
        "mean_phantom_ratio": row.mean_phantom_ratio,
        "median_phantom_ratio": row.median_phantom_ratio,
        "mean_large_ratio": row.mean_large_ratio,
        "median_large_ratio": row.median_large_ratio,
        "mean_opposite_n": row.mean_opposite_n,
        "mean_large_n": row.mean_large_n,
        "mean_lifetime_ns": row.mean_lifetime_ns,
        "mean_large_lifetime_ns": row.mean_large_lifetime_ns,
        "instant_cancel_rate": row.instant_cancel_rate,
        "opposite_density_per_s": row.opposite_density_per_s,
        "unique_opposite_oids": row.unique_opposite_oids,
        "unique_large_oids": row.unique_large_oids,
        "opposite_size": row.opposite_size,
        "large_size": row.large_size,
    }


def _score_windows(
    prints: pl.DataFrame,
    closed: pl.DataFrame,
    windows: tuple[int, ...],
    tick: float,
    tick_band: int,
    large_min: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows: list[WindowPhantom] = []
    per_frames: list[pl.DataFrame] = []
    keep = [
        "window_s",
        "t_seq",
        "t_ts",
        "t_price",
        "t_side",
        "t_size",
        "opp_n",
        "opp_size",
        "opp_ratio",
        "large_n",
        "large_size",
        "large_ratio",
        "opp_life",
        "opp_instant_n",
    ]
    for window_s in windows:
        pairs = _window_pairs(prints, closed, window_s, tick, tick_band)
        per_t = _per_print(prints, pairs, large_min)
        per_t = per_t.with_columns(pl.lit(int(window_s), dtype=pl.Int64).alias("window_s"))
        rows.append(_summarize_window(window_s, per_t, pairs, large_min))
        per_frames.append(per_t.select(keep))
    return pl.DataFrame([_window_row(r) for r in rows]), pl.concat(per_frames)


def opposite_phantom(
    mbo: pl.DataFrame,
    *,
    price_lo: float,
    price_hi: float,
    windows_s: Sequence[int] = DEFAULT_WINDOWS_S,
    tick_band: int = DEFAULT_TICK_BAND,
    size_mult: float = DEFAULT_SIZE_MULT,
    mean_add_size: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """يقيس الضوضاء العكسية قبل ``T`` الفوري داخل ``[price_lo, price_hi]``."""

    if price_hi < price_lo:
        raise ValueError("price_hi must be >= price_lo")
    if tick_band < 0:
        raise ValueError("tick_band must be >= 0")
    if size_mult <= 0:
        raise ValueError("size_mult must be > 0")
    windows = _parse_windows(windows_s)
    book = prepare_mbo_events(mbo)
    closed = closed_unfilled_orders(book)
    prints = immediate_prints(book).filter(
        (pl.col("price") >= price_lo) & (pl.col("price") <= price_hi)
    )
    diag_base: dict[str, Any] = {
        "layer": LAYER_ID,
        "price_lo": price_lo,
        "price_hi": price_hi,
        "tick_band": tick_band,
        "size_mult": size_mult,
        "heuristic": ("opposite Add+Cancel, no F, strictly before T; not a legal spoofing call"),
        "not_lstm": True,
        "not_live_overlay": True,
        "windows_s": list(windows),
        "n_closed_unfilled": closed.height,
    }
    if prints.height == 0:
        diag_base.update({"n_t": 0, "note": "no immediate T in band"})
        return _empty_windows(), pl.DataFrame(), diag_base

    prints = prints.with_row_index("t_i")
    sample_px = float(prints.select(pl.col("price").first()).item())
    tick = infer_tick_size(sample_px)
    adds = book.filter((pl.col("action") == _ADD) & (pl.col("order_id") > 0))
    mean_sz = (
        float(mean_add_size)
        if mean_add_size is not None
        else float(adds.select(pl.col("size").mean()).item() or 0.0)
    )
    large_min = mean_sz * float(size_mult)
    window_frame, per_print_frame = _score_windows(
        prints, closed, windows, tick, tick_band, large_min
    )
    diag_base.update(
        {
            "tick": tick,
            "mean_add_size": mean_sz,
            "large_min_size": large_min,
            "n_t": prints.height,
        }
    )
    return window_frame, per_print_frame, diag_base


def write_phantom_report(
    windows: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
    *,
    per_print: pl.DataFrame | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if windows.height:
        windows.write_parquet(out / "opposite_phantom_windows.parquet")
    if per_print is not None and per_print.height:
        per_print.write_parquet(out / "opposite_phantom_prints.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Opposite unfilled cancels before aggressive T",
        "",
        "Heuristic only: Add then Cancel on the opposite side, no F, strictly before T.",
        "Not a legal spoofing judgment. Not a live overlay. Not an LSTM channel.",
        "Ratios 3:1 and 5:1 are descriptive counts, not proven entry thresholds.",
        "",
        "| w | n_T | mean opp/T | med opp/T | mean large/T | >=3 | >=5 | inst | life_ms | large |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in windows.iter_rows(named=True):
        life_ns = row["mean_lifetime_ns"]
        life_s = "nan" if life_ns is None else f"{float(life_ns) / 1e6:.3f}"
        lines.append(
            f"| {row['window_s']} | {row['n_t']} | "
            f"{float(row['mean_phantom_ratio']):.4f} | "
            f"{float(row['median_phantom_ratio']):.4f} | "
            f"{float(row['mean_large_ratio']):.4f} | "
            f"{row['n_t_ratio_ge_3']} | {row['n_t_ratio_ge_5']} | "
            f"{float(row['instant_cancel_rate']):.4f} | {life_s} | "
            f"{row['unique_large_oids']} |"
        )
    lines.append("")
    (out / "OPPOSITE_PHANTOM.md").write_text("\n".join(lines), encoding="utf-8")
    return out


__all__ = [
    "DEFAULT_SIZE_MULT",
    "DEFAULT_TICK_BAND",
    "DEFAULT_WINDOWS_S",
    "LAYER_ID",
    "SECOND_NS",
    "WindowPhantom",
    "closed_unfilled_orders",
    "immediate_prints",
    "infer_tick_size",
    "opposite_phantom",
    "write_phantom_report",
]
