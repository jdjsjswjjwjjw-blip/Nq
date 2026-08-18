"""مطرقة حجم الصفقة × جدار MBP-10. لقطات العمق، بلا إعادة بناء MBO.

العتبات Avg>3 و Ask L1<100 / >200 فرضية من وصف 10:24 وليست قفلًا.
لا تُنسخ تلك الساعات إلى يوم آخر. جدار NQ بعقود NQ (وسيط L1≈2 في 14).
يوم ملف واحد على ``ingest_ts``. ليست overlay وليست إشارة.

احذف الملف + السكربت + الاختبار للإزالة.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from nq.contracts.mbo import MboAction
from nq.contracts.temporal import EVENT_TS, INGEST_TS, SEQUENCE
from nq.research.clock_flow import TZ_NAME, scan_tape_bins
from nq.research.cvd_day_compare import in_rth
from nq.research.mbo_sequence_mlp import assert_single_day_mbo
from nq.research.mbo_trade_overlap import (
    _rename_databento,
    _to_epoch_ns,
    prepare_trades_tape,
)

LAYER_ID = "hammer_wall"
HYP_AVG_SIZE: Final = 3.0
HYP_THIN_L1: Final = 100
HYP_THICK_L1: Final = 200
HYP_FAIL_AVG: Final = 2.0
NQ_PER_MNQ: Final = 10
_TRADE = MboAction.TRADE.value
_ASK_L1 = "ask_sz_00"
_BID_L1 = "bid_sz_00"


def prepare_mbp10(frame: pl.DataFrame) -> pl.DataFrame:
    """لقطات MBP-10 جاهزة. يرفض لصق أيام. لا يبني دفترًا من MBO."""

    work = _rename_databento(frame)
    needed = [EVENT_TS, _ASK_L1, _BID_L1]
    missing = [name for name in needed if name not in work.columns]
    if missing:
        raise ValueError(f"MBP-10 missing columns {missing}")
    work = _to_epoch_ns(work, EVENT_TS)
    if INGEST_TS in work.columns:
        work = _to_epoch_ns(work, INGEST_TS)
    else:
        work = work.with_columns(pl.col(EVENT_TS).alias(INGEST_TS))
    if SEQUENCE not in work.columns:
        work = work.with_columns(pl.arange(0, work.height, dtype=pl.Int64).alias(SEQUENCE))
    work = work.with_columns(
        pl.col(EVENT_TS).cast(pl.Int64).alias(EVENT_TS),
        pl.col(INGEST_TS).cast(pl.Int64).alias(INGEST_TS),
        pl.col(SEQUENCE).cast(pl.Int64).alias(SEQUENCE),
        pl.col(_ASK_L1).cast(pl.Int64).alias("ask_l1"),
        pl.col(_BID_L1).cast(pl.Int64).alias("bid_l1"),
    )
    ask_l3 = pl.col("ask_l1")
    bid_l3 = pl.col("bid_l1")
    for col, alias in (("ask_sz_01", "ask_1"), ("ask_sz_02", "ask_2")):
        if col in work.columns:
            work = work.with_columns(pl.col(col).cast(pl.Int64).alias(alias))
            ask_l3 = ask_l3 + pl.col(alias)
    for col, alias in (("bid_sz_01", "bid_1"), ("bid_sz_02", "bid_2")):
        if col in work.columns:
            work = work.with_columns(pl.col(col).cast(pl.Int64).alias(alias))
            bid_l3 = bid_l3 + pl.col(alias)
    work = work.with_columns(ask_l3.alias("ask_l3"), bid_l3.alias("bid_l3"))
    assert_single_day_mbo(work)
    return work.select(EVENT_TS, SEQUENCE, "ask_l1", "bid_l1", "ask_l3", "bid_l3").sort(
        [EVENT_TS, SEQUENCE]
    )


def _asof_before(
    ts: NDArray[np.int64],
    ask_l1: NDArray[np.int64],
    bid_l1: NDArray[np.int64],
    ask_l3: NDArray[np.int64],
    bid_l3: NDArray[np.int64],
    stamp: int,
) -> dict[str, float | int]:
    if ts.size == 0:
        return {
            "ask_l1": 0,
            "bid_l1": 0,
            "ask_l3": 0,
            "bid_l3": 0,
            "has_snapshot": False,
        }
    idx = int(np.searchsorted(ts, stamp, side="left")) - 1
    if idx < 0:
        return {
            "ask_l1": 0,
            "bid_l1": 0,
            "ask_l3": 0,
            "bid_l3": 0,
            "has_snapshot": False,
        }
    return {
        "ask_l1": int(ask_l1[idx]),
        "bid_l1": int(bid_l1[idx]),
        "ask_l3": int(ask_l3[idx]),
        "bid_l3": int(bid_l3[idx]),
        "has_snapshot": True,
    }


def _avg_size(ts: NDArray[np.int64], size: NDArray[np.int64], start_ts: int, end_ts: int) -> float:
    lo = int(np.searchsorted(ts, start_ts, side="left"))
    hi = int(np.searchsorted(ts, end_ts, side="left"))
    n = hi - lo
    if n <= 0:
        return float("nan")
    return float(size[lo:hi].sum()) / float(n)


def _median(values: list[float]) -> float:
    finite = [x for x in values if not math.isnan(x)]
    if not finite:
        return float("nan")
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def _rate(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _empty_diag(
    *,
    bin_s: int,
    hyp_avg: float,
    hyp_thin: int,
    hyp_thick: int,
) -> dict[str, Any]:
    return {
        "layer": LAYER_ID,
        "bin_s": bin_s,
        "n_bins": 0,
        "n_hyp_pass": 0,
        "n_hyp_fail_recipe": 0,
        "n_ask_thin": 0,
        "n_ask_thick": 0,
        "n_hammer": 0,
        "wall_source": "nq_mbp10_snapshot",
        "hammer_source": "mnq_trades_T",
        "not_mbo_book": True,
        "not_pattern": True,
        "not_hammer_wall_lock": True,
        "hyp_avg": hyp_avg,
        "hyp_thin_l1": hyp_thin,
        "hyp_thick_l1": hyp_thick,
        "nq_per_mnq": NQ_PER_MNQ,
        "not_copy_clocks": True,
        "not_lstm": True,
        "not_live_overlay": True,
    }


def _flag_bin(
    row: Mapping[str, Any],
    wall: Mapping[str, float | int | bool],
    avg: float,
    *,
    hyp_avg: float,
    hyp_thin: int,
    hyp_thick: int,
    hyp_fail_avg: float,
) -> dict[str, Any]:
    ask = int(wall["ask_l1"])
    bid = int(wall["bid_l1"])
    thin = bool(wall["has_snapshot"]) and ask < hyp_thin
    thick = bool(wall["has_snapshot"]) and ask >= hyp_thick
    hammer = (not math.isnan(avg)) and avg > hyp_avg
    light = (not math.isnan(avg)) and avg < hyp_fail_avg
    rec = dict(row)
    rec.update(
        {
            "avg_trade_size": avg,
            "ask_l1": ask,
            "bid_l1": bid,
            "ask_l3": int(wall["ask_l3"]),
            "bid_l3": int(wall["bid_l3"]),
            "ask_l1_mnq_equiv": ask * NQ_PER_MNQ,
            "has_snapshot": bool(wall["has_snapshot"]),
            "hyp_thin_ask": thin,
            "hyp_thick_ask": thick,
            "hyp_hammer": hammer,
            "hyp_light": light,
            "hyp_pass": hammer and thin,
            "hyp_fail_recipe": light and thick,
            "rth": in_rth(str(row["clock"]) if row.get("clock") is not None else None),
        }
    )
    return rec


def _pack_moves(table: pl.DataFrame, flag: str) -> dict[str, Any]:
    sub = table.filter(pl.col(flag)) if table.height else table
    moves5 = [float(x) for x in sub["next_move_5m"].to_list()] if sub.height else []
    moves15 = [float(x) for x in sub["next_move_15m"].to_list()] if sub.height else []
    n_up = sum(1 for x in moves5 if not math.isnan(x) and x > 0)
    return {
        "n": sub.height,
        "median_next5m": _median(moves5),
        "median_next15m": _median(moves15),
        "up_rate_5m": _rate(n_up, sub.height),
    }


def scan_hammer_wall(
    mnq_mbo: pl.DataFrame,
    mnq_trades: pl.DataFrame,
    nq_trades: pl.DataFrame,
    nq_mbp10: pl.DataFrame,
    *,
    bin_s: int = 60,
    tz_name: str = TZ_NAME,
    hyp_avg: float = HYP_AVG_SIZE,
    hyp_thin: int = HYP_THIN_L1,
    hyp_thick: int = HYP_THICK_L1,
    hyp_fail_avg: float = HYP_FAIL_AVG,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """شرائح 60ث: متوسط حجم MNQ Trades + جدار NQ MBP-10 قبل البداية. بلا دفتر MBO."""

    bins, tape_diag = scan_tape_bins(
        mnq_mbo,
        mnq_trades,
        nq_trades,
        bin_s=bin_s,
        tz_name=tz_name,
        inner_s=5,
        label="hammer_wall",
        horizon_s=300,
    )
    book = prepare_mbp10(nq_mbp10)
    tape = prepare_trades_tape(mnq_trades).filter(pl.col("action") == _TRADE).sort(EVENT_TS)
    empty = _empty_diag(bin_s=bin_s, hyp_avg=hyp_avg, hyp_thin=hyp_thin, hyp_thick=hyp_thick)
    if bins.height == 0 or book.height == 0 or tape.height == 0:
        return pl.DataFrame(), empty
    mbp_ts = np.asarray(book[EVENT_TS].to_numpy(), dtype=np.int64)
    ask_l1 = np.asarray(book["ask_l1"].to_numpy(), dtype=np.int64)
    bid_l1 = np.asarray(book["bid_l1"].to_numpy(), dtype=np.int64)
    ask_l3 = np.asarray(book["ask_l3"].to_numpy(), dtype=np.int64)
    bid_l3 = np.asarray(book["bid_l3"].to_numpy(), dtype=np.int64)
    tr_ts = np.asarray(tape[EVENT_TS].to_numpy(), dtype=np.int64)
    tr_sz = np.asarray(tape["size"].to_numpy(), dtype=np.int64)
    rows = [
        _flag_bin(
            row,
            _asof_before(mbp_ts, ask_l1, bid_l1, ask_l3, bid_l3, int(row["start_ts"])),
            _avg_size(tr_ts, tr_sz, int(row["start_ts"]), int(row["end_ts"])),
            hyp_avg=hyp_avg,
            hyp_thin=hyp_thin,
            hyp_thick=hyp_thick,
            hyp_fail_avg=hyp_fail_avg,
        )
        for row in bins.iter_rows(named=True)
    ]
    table = pl.DataFrame(rows)
    empty["n_bins"] = table.height
    empty["n_hyp_pass"] = int(table.filter(pl.col("hyp_pass")).height)
    empty["n_hyp_fail_recipe"] = int(table.filter(pl.col("hyp_fail_recipe")).height)
    empty["n_ask_thin"] = int(table.filter(pl.col("hyp_thin_ask")).height)
    empty["n_ask_thick"] = int(table.filter(pl.col("hyp_thick_ask")).height)
    empty["n_hammer"] = int(table.filter(pl.col("hyp_hammer")).height)
    empty["median_ask_l1"] = _median([float(x) for x in table["ask_l1"].to_list()])
    empty["median_avg_trade_size"] = _median([float(x) for x in table["avg_trade_size"].to_list()])
    empty["n_tape_hyp_cvd"] = tape_diag.get("n_hyp_cvd")
    empty["n_tape_hyp_all_three"] = tape_diag.get("n_hyp_all_three")
    for flag, prefix in (
        ("hyp_pass", "pass"),
        ("hyp_fail_recipe", "fail_recipe"),
        ("hyp_hammer", "hammer"),
    ):
        packed = _pack_moves(table, flag)
        empty[f"{prefix}_n"] = packed["n"]
        empty[f"{prefix}_median_next5m"] = packed["median_next5m"]
        empty[f"{prefix}_median_next15m"] = packed["median_next15m"]
        empty[f"{prefix}_up_rate_5m"] = packed["up_rate_5m"]
    rth = _pack_moves(table.filter(pl.col("hyp_pass") & pl.col("rth")), "hyp_pass")
    empty["pass_rth_n"] = rth["n"]
    empty["pass_rth_median_next5m"] = rth["median_next5m"]
    empty["pass_rth_median_next15m"] = rth["median_next15m"]
    empty["pass_rth_up_rate_5m"] = rth["up_rate_5m"]
    empty["baseline_median_next5m"] = _median([float(x) for x in table["next_move_5m"].to_list()])
    empty["baseline_median_next15m"] = _median([float(x) for x in table["next_move_15m"].to_list()])
    return table, empty


def write_hammer_wall_report(
    table: pl.DataFrame,
    diagnostics: Mapping[str, Any],
    output_dir: Path | str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if table.height:
        table.write_parquet(out / "hammer_wall_bins.parquet")
        table.filter(pl.col("hyp_pass")).write_parquet(out / "hammer_wall_pass.parquet")
    (out / "summary.json").write_text(
        json.dumps(dict(diagnostics), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Hammer × wall: MNQ avg trade size + NQ MBP-10 L1",
        "",
        "Snapshots from MBP-10. No MBO book rebuild. Cuts Avg>3 and Ask L1<100 "
        "are hypothesized, not a lock. Clocks from 10:24/11:51/12:45 are not copied.",
        f"wall={diagnostics.get('wall_source')} hammer={diagnostics.get('hammer_source')}.",
        f"bins={diagnostics.get('n_bins')} median_avg={diagnostics.get('median_avg_trade_size')} "
        f"median_ask_l1={diagnostics.get('median_ask_l1')}.",
        f"ask_thin(<{diagnostics.get('hyp_thin_l1')})={diagnostics.get('n_ask_thin')} "
        f"ask_thick(>={diagnostics.get('hyp_thick_l1')})={diagnostics.get('n_ask_thick')} "
        f"hammer(>{diagnostics.get('hyp_avg')})={diagnostics.get('n_hammer')}.",
        f"pass={diagnostics.get('n_hyp_pass')} "
        f"up_rate_5m={diagnostics.get('pass_up_rate_5m')} "
        f"median_next5m={diagnostics.get('pass_median_next5m')} "
        f"median_next15m={diagnostics.get('pass_median_next15m')}.",
        f"fail_recipe={diagnostics.get('n_hyp_fail_recipe')} "
        f"RTH pass={diagnostics.get('pass_rth_n')} "
        f"RTH up_rate_5m={diagnostics.get('pass_rth_up_rate_5m')}.",
        f"baseline median next5m={diagnostics.get('baseline_median_next5m')} "
        f"next15m={diagnostics.get('baseline_median_next15m')}.",
        "NQ L1 is typically a few contracts; Ask<100 may not bind. Not a signal.",
        "",
        "| clock | avg | ask_l1 | bid_l1 | ask_l3 | pass | next5 | next15 |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    if table.height:
        shown = table.filter(pl.col("hyp_pass"))
        if shown.height == 0:
            shown = table.head(20)
        for row in shown.iter_rows(named=True):
            avg = float(row["avg_trade_size"])
            avg_s = "nan" if math.isnan(avg) else f"{avg:.2f}"
            n5 = float(row["next_move_5m"])
            n15 = float(row["next_move_15m"])
            n5_s = "nan" if math.isnan(n5) else f"{n5:.2f}"
            n15_s = "nan" if math.isnan(n15) else f"{n15:.2f}"
            clock = str(row["clock"])
            hm = clock.split("T", 1)[1][:5] if "T" in clock else clock
            lines.append(
                f"| {hm} | {avg_s} | {row['ask_l1']} | {row['bid_l1']} | "
                f"{row['ask_l3']} | {row['hyp_pass']} | {n5_s} | {n15_s} |"
            )
    (out / "HAMMER_WALL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


__all__ = [
    "HYP_AVG_SIZE",
    "HYP_FAIL_AVG",
    "HYP_THICK_L1",
    "HYP_THIN_L1",
    "LAYER_ID",
    "prepare_mbp10",
    "scan_hammer_wall",
    "write_hammer_wall_report",
]
